# -*- coding: utf-8 -*-
"""
Script for regression using 1D CNN with Optuna optimization and cross-validation.

Created 07 2025

@author: Arkadiusz Leniak
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import gc
import signal
import csv

torch.backends.cudnn.benchmark = True           
torch.backends.cuda.matmul.allow_tf32 = True    
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')  
except AttributeError:
    pass

import torch.nn as nn
import torch.optim as optim
import torch.amp as amp

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import optuna
import mlflow
import mlflow.pytorch  # Import for mlflow.pytorch
from datetime import datetime
import argparse
import optuna.visualization.matplotlib as optuna_viz
import matplotlib.pyplot as plt
import json
from optuna import importance

import logging
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)

# Import MLflow tags from tags_config_pytorch_CNN_1D.py
import tags_config_CNN_1D

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set random seed for reproducibility
SEED = 1988
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Select device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = (device.type == "cuda")

MAX_KSIZE_CAP = 31

class TrialTimeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
    def _handler(self, signum, frame):
        raise TimeoutError(f"Trial exceeded {self.seconds} seconds.")
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Attributes:
        patience (int): How long to wait after last time validation loss improved.
        verbose (bool): If True, prints a message for each validation loss improvement.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        counter (int): Counts epochs with no improvement.
        best_loss (float): Best recorded validation loss.
        early_stop (bool): Whether early stopping was triggered.
    """

    def __init__(self, patience=10, verbose=False, delta=0.0):
        """
        Initialize EarlyStopping.

        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call method to check if validation loss improved.

        Args:
            val_loss (float): Current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'Initial validation loss: {self.best_loss:.6f}')
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased to {self.best_loss:.6f}. Resetting counter.')
        else:
            self.counter += 1
            if self.verbose:
                print(f'No improvement in validation loss for {self.counter} epochs.')
            if self.counter >= self.patience:
                if self.verbose:
                    print('Early stopping triggered.')
                self.early_stop = True


def assert_finite_array(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf. Lower LR or adjust params.")


def assert_finite_loss(loss: torch.Tensor, where: str = ""):
    """
    Jeśli loss nie jest skończony (NaN/Inf) - przycinamy trial (PRUNED),
    zamiast zrzucać całą optymalizację.
    """
    if not torch.isfinite(loss):
        loc = f" in {where}" if where else ""
        raise optuna.TrialPruned(f"Non-finite loss encountered{loc} (NaN/Inf).")


def amp_opt_step(loss: torch.Tensor,
                 optimizer: torch.optim.Optimizer,
                 scaler: amp.GradScaler,
                 model: nn.Module,
                 max_grad_norm: float | None,
                 where: str = ""):
    """
    Backward + unscale + clip_grad_norm_ + step + update dla AMP.
    Wyrzuca błąd, jeśli loss jest nie-skończony.
    """
    assert_finite_loss(loss, where=where)

    scaler.scale(loss).backward()
    # MUSI być przed klipowaniem z AMP:
    scaler.unscale_(optimizer)

    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    scaler.step(optimizer)
    scaler.update()


def make_loader(dataset, batch_size, shuffle=True):
    # Zero-problemowy wariant – żadnych workerów == brak pipe’ów/procesów
    nw = 0
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    # UWAGA: prefetch_factor i persistent_workers NIE ustawiamy przy nw==0
    return torch.utils.data.DataLoader(**kwargs)

def load_data(csv_path, target_column_name='LABEL'):
    """
    Load and preprocess data from a CSV file: coerce numerics, drop NaN/Inf rows.
    """
    try:
        data = pd.read_csv(csv_path)

        # Drop pierwszej kolumny (nazwy próbek)
        data = data.drop(data.columns[0], axis=1)

        # Normalizacja typowych „pustych” tokenów -> NaN
        na_tokens = ['NA','N/A','na','n/a','NaN','nan','NULL','null','Inf','-Inf','inf','-inf']
        data = data.replace(r'^\s*$', np.nan, regex=True).replace(na_tokens, np.nan)

        # Wymuś numeryczne typy (ukryte śmieci -> NaN)
        for c in data.columns:
            data[c] = pd.to_numeric(data[c], errors='coerce')

        # Odrzuć wiersze z brakami (albo tu możesz wstawić imputację, patrz poniżej)
        num_cols = data.columns.tolist()
        med = data[num_cols].median()
        data[num_cols] = data[num_cols].fillna(med)

        # Rozdziel X/y
        if target_column_name not in data.columns:
            raise KeyError(f"Target column '{target_column_name}' not found in {csv_path}.")

        y = data[target_column_name].values.astype(np.float32)
        X = data.drop(columns=[target_column_name]).values.astype(np.float32)

        # Jeszcze jeden bezpiecznik: usuń wiersze niefinitywne (gdyby coś przeszło)
        finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[finite_mask], y[finite_mask]

        return X, y

    except Exception as e:
        print(f"An error occurred while loading data from {csv_path}: {e}")
        sys.exit(1)



def get_optimizer(trial, model_parameters):
    """
    Get the optimizer for the model based on the trial parameters.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        model_parameters (iterable): The model parameters to optimize.

    Returns:
        torch.optim.Optimizer: The selected optimizer.
    """
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adamw'])

    if optimizer_name in ('adam', 'adamw'):
        # mniejszy max LR dla Adam/AdamW – stabilniej z AMP i CNN
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        use_wd = trial.suggest_categorical('use_weight_decay', [False, True])
        weight_decay = 0.0 if not use_wd else trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)

    elif optimizer_name == 'rmsprop':
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        use_wd = trial.suggest_categorical('use_weight_decay', [False, True])
        weight_decay = 0.0 if not use_wd else trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)

    elif optimizer_name == 'sgd':
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 2e-2, log=True)
        use_wd = trial.suggest_categorical('use_weight_decay', [False, True])
        weight_decay = 0.0 if not use_wd else trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)

    if optimizer_name == 'adam':
        beta1 = trial.suggest_float('adam_beta1', 0.85, 0.99)
        beta2 = trial.suggest_float('adam_beta2', 0.95, 0.9999)
        optimizer = optim.Adam(model_parameters, lr=learning_rate,
                            betas=(beta1, beta2), weight_decay=weight_decay, eps=1e-8)
    elif optimizer_name == 'adamw':
        beta1 = trial.suggest_float('adamw_beta1', 0.85, 0.99)
        beta2 = trial.suggest_float('adamw_beta2', 0.95, 0.9999)
        optimizer = optim.AdamW(model_parameters, lr=learning_rate,
                                betas=(beta1, beta2), weight_decay=weight_decay, eps=1e-8, amsgrad=False)
    elif optimizer_name == 'sgd':
        momentum = trial.suggest_float('sgd_momentum', 0.0, 0.99)
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    return optimizer


class Net(nn.Module):
    """
    Convolutional Neural Network (CNN) for regression tasks.

    Args:
        nn.Module (torch.nn.Module): Base class for all neural network modules in PyTorch.

    Attributes:
        conv (torch.nn.Sequential): Convolutional layers.
        fc (torch.nn.Sequential): Fully connected layers.
        regularization (str): Type of regularization ('none', 'l1', or 'l2').
        reg_rate (float): Regularization rate.
    """

    def __init__(self, trial, input_dim):
        """
        Initialize the CNN architecture.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            input_dim (int): Dimensionality of the input features.
        """
        super(Net, self).__init__()

        # Activation function
        activation_name = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'selu'])
        if activation_name == 'relu':
            activation = nn.ReLU()
        elif activation_name == 'tanh':
            activation = nn.Tanh()
        elif activation_name == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name == 'leaky_relu':
            activation = nn.LeakyReLU()
        elif activation_name == 'selu':
            activation = nn.SELU()
        else:
            activation = nn.ReLU()

        self.activation_name = activation_name

        # Regularization
        self.regularization = trial.suggest_categorical('regularization', ['none', 'l1', 'l2'])
        if self.regularization == 'none':
            self.reg_rate = 0.0
        else:
            self.reg_rate = trial.suggest_float('reg_rate', 1e-5, 1e-2, log=True)

        # Dropout rate
        dropout_conv = trial.suggest_float('dropout_conv', 0.0, 0.8)
        dropout_fc   = trial.suggest_float('dropout_fc',   0.0, 0.8)

        # Batch normalization
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        if self.activation_name == 'selu':
            # SELU zakłada self-normalizing – BN zwykle przeszkadza
            use_batch_norm = False

        # Convolutional layers
        num_conv_layers = trial.suggest_int('num_conv_layers', 2, 7)
        conv_layers = []
        in_channels = 1  # Input channels for Conv1d
        input_length = input_dim  # Initial input length

        for i in range(num_conv_layers):
            out_channels = trial.suggest_int(f'num_filters_l{i}', 16, 128, log=True)

            # Padding suggestion
            max_possible_padding = 3  # You can adjust this value
            padding = trial.suggest_int(f'padding_l{i}', 0, max_possible_padding)

            # Calculate maximum kernel_size
            max_kernel_size = input_length + 2 * padding
            max_kernel_size = min(max_kernel_size, MAX_KSIZE_CAP)  # Original maximum kernel_size
            if max_kernel_size < 3:
                max_kernel_size = 3  # Minimum kernel_size

            if max_kernel_size % 2 == 0:
                max_kernel_size -= 1

            # Suggest kernel_size with adjusted maximum
            kernel_size = trial.suggest_int(f'kernel_size_l{i}', 3, max_kernel_size, step=2)

            # Suggest stride
            stride = trial.suggest_int(f'stride_l{i}', 1, 3)

            # Calculate output_length
            numerator = input_length + 2 * padding - (kernel_size - 1) - 1
            if numerator < 0:
                continue  # Skip this hyperparameter combination

            output_length = numerator // stride + 1
            if output_length <= 0:
                continue  # Skip this hyperparameter combination

            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(activation)
            if dropout_conv > 0.0:
                conv_layers.append(nn.Dropout(dropout_conv))
            in_channels = out_channels
            input_length = output_length  # Update input length

        self.conv = nn.Sequential(*conv_layers)

        # Fully connected layers
        num_fc_layers = trial.suggest_int('num_fc_layers', 2, 5)
        fc_layers = []
        in_features = in_channels * input_length

        for i in range(num_fc_layers):
            out_features = trial.suggest_int(f'fc_units_l{i}', 32, 512, log=True)
            fc_layers.append(nn.Linear(in_features, out_features))
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(activation)
            if dropout_fc > 0.0:
                fc_layers.append(nn.Dropout(dropout_fc))
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*fc_layers)

        # Initialize weights
        init_method = trial.suggest_categorical('weight_init', ['xavier', 'kaiming'])
        self.apply(lambda m: self.init_weights(m, init_method))

    def init_weights(self, m, init_method):
        """
        Initialize weights of the model layers.

        Args:
            m (torch.nn.Module): The module (layer) to initialize.
            init_method (str): The initialization method ('xavier', 'kaiming').
        """
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            if self.activation_name == 'selu':
                # LeCun/kaiming fan_in bez nieliniowości – stabilniejsze z SELU
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear', mode='fan_in')
            else:
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Reshape input for Conv1d: (batch_size, channels, length)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# Definition of WrappedModel class
class WrappedModel(nn.Module):
    """
    Wrapped model for evaluation and inference.

    Args:
        nn.Module (torch.nn.Module): Base class for all neural network modules in PyTorch.

    Attributes:
        model (torch.nn.Module): The underlying model.
    """

    def __init__(self, model):
        """
        Initialize the wrapped model.

        Args:
            model (torch.nn.Module): The model to wrap.
        """
        super(WrappedModel, self).__init__()
        self.model = model
        self.model.eval()  # Set model to evaluation mode

    def forward(self, x):
        """
        Forward pass of the wrapped model.

        Args:
            x (torch.Tensor or np.ndarray): Input tensor or numpy array.

        Returns:
            torch.Tensor: Output tensor.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.type(torch.float32)
        with torch.no_grad():
            return self.model(x)


def objective(trial, csv_path):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        csv_path (str): Path to the CSV file.

    Returns:
        float: The value of the objective function (e.g., RMSE).
    """
    # Per-trial hard timeout, np. 45 min = 2700 s
    with TrialTimeout(3600):
        try:
            # Load and preprocess data
            X_train_full, y_train_full = load_data(csv_path, target_column_name='LABEL')

            # Convert to tensor
            X_train_full = X_train_full.astype(np.float32)
            y_train_full = y_train_full.astype(np.float32)

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_train_full, y_train_full, test_size=0.1, random_state=SEED
            )

            # Hyperparameter suggestions
            batch_size = trial.suggest_int('batch_size', 16, 512, log=True)
            epochs = trial.suggest_int('epochs', 50, 500, step=50)
            early_stop_patience = trial.suggest_int('early_stop_patience', 5, 40)
            use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
            clip_grad_value = trial.suggest_float('clip_grad_value', 0.1, 10.0, step=0.1)

            # Scheduler hyper-params
            scheduler_factor   = trial.suggest_float('scheduler_factor', 0.1, 0.9)
            scheduler_patience = trial.suggest_int('scheduler_patience', 2, 15)
            min_lr             = trial.suggest_float('min_lr', 1e-6, 1e-3, log=True)

            # KFold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            rmse_scores = []

            for fold, (train_index, valid_index) in enumerate(kf.split(X_train), 1):
                X_train_fold = X_train[train_index]
                X_valid_fold = X_train[valid_index]
                y_train_fold = y_train[train_index]
                y_valid_fold = y_train[valid_index]

                # Create model
                input_dim = X_train_fold.shape[1]
                model = Net(trial, input_dim).to(device)

                # Loss criterion
                criterion = nn.MSELoss()

                # Optimizer
                optimizer = get_optimizer(trial, model.parameters())

                # Learning rate scheduler
                if use_scheduler:
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min',
                        factor=scheduler_factor,
                        patience=scheduler_patience,
                        min_lr=min_lr
                    )

                # Early stopping
                early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

                # --- AMP scaler + prealloc walidacji (raz na fold) ---
                scaler = amp.GradScaler('cuda', enabled=use_amp)
                X_valid_tensor = torch.as_tensor(X_valid_fold, dtype=torch.float32, device=device)
                y_valid_tensor = torch.as_tensor(y_valid_fold, dtype=torch.float32, device=device)
                # -----------------------------------------------------

                # Training
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_train_fold),
                    torch.from_numpy(y_train_fold)
                )

                loader = make_loader(dataset, batch_size, shuffle=True)

                use_batch_norm_effective = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())

                for epoch in range(epochs):
                    model.train()
                    epoch_loss_sum = 0.0

                    for batch_X, batch_y in loader:
                        batch_X = batch_X.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)

                        # unikamy BN przy batch_size==1
                        if (batch_X.size(0) > 1) or (not use_batch_norm_effective):
                            with amp.autocast('cuda', enabled=use_amp):
                                outputs = model(batch_X).squeeze()
                                loss = criterion(outputs, batch_y)

                                # regularizacja (jak było)
                                if model.regularization == 'l1':
                                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                                    loss = loss + model.reg_rate * l1_norm
                                elif model.regularization == 'l2':
                                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                                    loss = loss + model.reg_rate * l2_norm

                            amp_opt_step(loss, optimizer, scaler, model, clip_grad_value, where="objective/train")

                            # zliczanie do średniej straty na epokę
                            epoch_loss_sum += loss.detach().float().item()
                        else:
                            continue

                    # --- Walidacja bez rekreacji tensorów ---
                    model.eval()
                    with torch.no_grad():
                        with amp.autocast('cuda', enabled=use_amp):
                            y_valid_pred_t = model(X_valid_tensor).squeeze()
                        val_loss_t = torch.nn.functional.mse_loss(y_valid_pred_t, y_valid_tensor)
                    val_loss = float(val_loss_t.detach().cpu())

                    # Scheduler – karmy walidacją (lepsze niż train loss)
                    if use_scheduler:
                        scheduler.step(val_loss)

                    # Early stopping
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        break

                # Evaluation
                model.eval()
                with torch.no_grad(), amp.autocast('cuda', enabled=use_amp):
                    y_pred_t = model(X_valid_tensor).squeeze()
                y_pred = y_pred_t.float().cpu().numpy()
                y_true = y_valid_tensor.cpu().numpy()

                assert_finite_array("Validation predictions (objective)", y_pred)

                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                rmse_scores.append(rmse)

                # Memory management
                del model, loader, dataset
                gc.collect()
                #torch.cuda.empty_cache()

            # Calculate mean RMSE
            final_rmse = np.mean(rmse_scores)

            # Log metric
            mlflow.log_metric("Trial RMSE", final_rmse, step=trial.number)

            # Set trial attributes
            trial.set_user_attr('rmse', final_rmse)

            return final_rmse

        except TimeoutError as te:
            print(f"[Trial timeout] {te}")
            # Timeout traktujemy jako normalną porażkę triala – Optuna i tak to ogarnie
            raise te

        except optuna.TrialPruned as pe:
            # To jest nasz normalny „losowy zły zestaw parametrów”
            print(f"[Trial pruned] {pe}")
            raise

        except ValueError as ve:
            # Na wszelki wypadek: stare ścieżki, gdyby coś jednak rzuciło ValueError
            print(f"[Trial value error] {ve} -> pruning")
            raise optuna.TrialPruned(str(ve))

        except Exception as e:
            # Prawdziwy błąd programistyczny – nie maskujemy
            print(f"An unexpected error occurred in the objective function: {e}")
            raise


def evaluate_model_with_cv(csv_path, trial_params, csv_name):
    """
    Evaluate the model using cross-validation and log metrics and artifacts to MLflow.

    Args:
        csv_path (str): Path to the CSV file.
        trial_params (dict): Dictionary containing the trial parameters.
        csv_name (str): Name of the CSV file (for logging purposes).

    Raises:
        Exception: If any error occurs during model evaluation or logging.
    """
    try:
        # Load data
        X_full, y_full = load_data(csv_path, target_column_name='LABEL')

        # Convert to tensor
        X_full = X_full.astype(np.float32)
        y_full = y_full.astype(np.float32)

        # Initialize K-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

        rmse_scores = []
        mae_scores = []
        r2_scores = []
        pearson_scores = []

        # Lists to collect all true and predicted values
        all_true = []
        all_preds = []

        # Get hyperparameters from trial_params
        params = trial_params

        # Add defaults for missing parameters
        optimizer_name = params.get('optimizer', 'adam')
        learning_rate = params.get('learning_rate', 1e-3)
        sgd_momentum = params.get('sgd_momentum', 0.0)
        use_scheduler = params.get('use_scheduler', False)
        early_stop_patience = params.get('early_stop_patience', 10)
        clip_grad_value = params.get('clip_grad_value', 1.0)
        batch_size = params.get('batch_size', 32)
        epochs = params.get('epochs', 100)
        weight_decay  = params.get('weight_decay', 0.0)
        scheduler_factor   = params.get('scheduler_factor', 0.5)
        scheduler_patience = params.get('scheduler_patience', 5)
        min_lr             = params.get('min_lr', 1e-5)

        for fold, (train_index, test_index) in enumerate(kf.split(X_full), 1):
            X_train, X_test = X_full[train_index], X_full[test_index]
            y_train, y_test = y_full[train_index], y_full[test_index]

            # Create model with best parameters
            input_dim = X_train.shape[1]
            # Note: We need to create a trial object with parameters to pass to Net
            # However, Net requires a trial object. We can create a fake trial object with parameters.
            trial_for_model = optuna.trial.FixedTrial(params)
            model = Net(trial_for_model, input_dim).to(device)

            # Loss criterion
            criterion = nn.MSELoss()

            # Optimizer
            if optimizer_name == 'adam':
                beta1 = params.get('adam_beta1', 0.9)
                beta2 = params.get('adam_beta2', 0.999)
                optimizer = optim.Adam(model.parameters(),
                                    lr=learning_rate,
                                    betas=(beta1, beta2),
                                    weight_decay=weight_decay)

            elif optimizer_name == 'adamw':
                beta1 = params.get('adamw_beta1', 0.9)
                beta2 = params.get('adamw_beta2', 0.999)
                optimizer = optim.AdamW(model.parameters(),
                                        lr=learning_rate,
                                        betas=(beta1, beta2),
                                        weight_decay=weight_decay)

            elif optimizer_name == 'sgd':
                optimizer = optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=sgd_momentum,
                                    weight_decay=weight_decay)

            elif optimizer_name == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

            # Scheduler
            if use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min',
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                    min_lr=min_lr
                )

            # Early stopping
            early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

            # --- AMP scaler + prealloc walidacji (raz na fold) ---
            scaler = amp.GradScaler('cuda', enabled=use_amp)
            X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32, device=device)
            y_test_tensor = torch.as_tensor(y_test, dtype=torch.float32, device=device)
            # -----------------------------------------------------

            # Training

            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_train),
                torch.from_numpy(y_train)
            )

            loader = make_loader(dataset, batch_size, shuffle=True)

            use_batch_norm_effective = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())

            for epoch in range(epochs):
                model.train()
                epoch_loss_sum = 0.0

                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    if (batch_X.size(0) > 1) or (not use_batch_norm_effective):
                        with amp.autocast('cuda', enabled=use_amp):
                            outputs = model(batch_X).squeeze()
                            loss = criterion(outputs, batch_y)

                            if model.regularization == 'l1':
                                l1_norm = sum(p.abs().sum() for p in model.parameters())
                                loss = loss + model.reg_rate * l1_norm
                            elif model.regularization == 'l2':
                                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                                loss = loss + model.reg_rate * l2_norm

                        amp_opt_step(loss, optimizer, scaler, model, clip_grad_value, where="cv/train")

                        epoch_loss_sum += loss.detach().float().item()
                    else:
                        continue

                model.eval()
                with torch.no_grad():
                    with amp.autocast('cuda', enabled=use_amp):
                        y_valid_pred_t = model(X_test_tensor).squeeze()
                    val_loss_t = torch.nn.functional.mse_loss(y_valid_pred_t, y_test_tensor)

                # sprawdź predykcje walidacyjne (te z bieżącej epoki)
                y_valid_pred = y_valid_pred_t.float().cpu().numpy()
                assert_finite_array("Fold validation predictions (CV)", y_valid_pred)

                val_loss = float(val_loss_t.detach().cpu())

                if use_scheduler:
                    scheduler.step(val_loss)

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f'Fold {fold}: Early stopping triggered at epoch {epoch+1}')
                    break


            # Evaluation
            model.eval()
            with torch.no_grad(), amp.autocast('cuda', enabled=use_amp):
                y_test_pred_t = model(X_test_tensor).squeeze()
            y_test_pred = y_test_pred_t.float().cpu().numpy()
                
            assert_finite_array("Fold test predictions (CV)", y_test_pred)

            y_test_true = y_test_tensor.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
            mae  = mean_absolute_error(y_test_true, y_test_pred)
            r2   = r2_score(y_test_true, y_test_pred)
            pearson_corr, _ = pearsonr(y_test_true, y_test_pred)

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            pearson_scores.append(pearson_corr)

            # Collect predictions
            all_true.extend(y_test_true)
            all_preds.extend(y_test_pred)

            # Memory management
            del model, loader, dataset
            gc.collect()
            #torch.cuda.empty_cache()

        # Calculate metrics
        final_rmse = round(np.mean(rmse_scores), 3)
        final_mae = round(np.mean(mae_scores), 3)
        q2 = round(np.mean(r2_scores), 3)
        final_pearson = round(np.mean(pearson_scores), 3)

        rmse_std  = round(np.std(rmse_scores, ddof=1), 3)
        rmse_span = round(np.max(rmse_scores) - np.min(rmse_scores), 3)

        q2_std  = round(np.std(r2_scores, ddof=1), 3)
        q2_span = round(np.max(r2_scores) - np.min(r2_scores), 3)

        # Log metrics
        mlflow.log_metric("RMSE", final_rmse)
        mlflow.log_metric("MAE", final_mae)
        mlflow.log_metric("Q2", q2)
        mlflow.log_metric("Pearson Correlation", final_pearson)
        mlflow.log_metric("RMSE_fold_std",  rmse_std)
        mlflow.log_metric("RMSE_fold_span", rmse_span)
        mlflow.log_metric("Q2_fold_std",  q2_std)
        mlflow.log_metric("Q2_fold_span", q2_span)

        # Save metrics to file
        summary = f"Best parameters:\n"
        for key, value in params.items():
            summary += f"{key}: {value}\n"

        summary += f"\n10CV Metrics:\n"
        summary += f"10CV RMSE: {final_rmse}\n"
        summary += f"10CV MAE: {final_mae}\n"
        summary += f"10CV Q2: {q2}\n"
        summary += f"10CV Pearson Correlation: {final_pearson}\n"
        summary += "\n10CV FoldVariance:\n"
        summary += f"  RMSE std:   {rmse_std}\n"
        summary += f"  RMSE span:  {rmse_span}\n"
        summary += "\n10CV FoldVariance (Q2):\n"
        summary += f"  Q2 std:   {q2_std}\n"
        summary += f"  Q2 span:  {q2_span}\n"

        summary_file_name = f"{csv_name}_summary.txt"
        with open(summary_file_name, 'w') as f:
            f.write(summary)

        mlflow.log_artifact(summary_file_name)

        print(f"\nValidation set evaluation for {csv_name}:")
        print(f"  10CV RMSE: {final_rmse}")
        print(f"  10CV MAE: {final_mae}")
        print(f"  10CV Q2: {q2}")
        print(f"  10CV Pearson Correlation: {final_pearson}")

        # Generate plot
        plt.figure(figsize=(8, 6))
        plt.scatter(all_true, all_preds, alpha=0.6, label='Predicted vs Actual')
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], color='red', linestyle='--', label='Ideal')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values (Cross-Validation)')
        plt.legend()

        # Save plot
        pred_vs_actual_fig_file = f"{csv_name}_pred_vs_actual_cv.png"
        plt.savefig(pred_vs_actual_fig_file)
        mlflow.log_artifact(pred_vs_actual_fig_file)
        plt.close()

        # Save predictions to CSV
        predictions_df = pd.DataFrame({'Actual': all_true, 'Predicted': all_preds})
        predictions_file_name = f"{csv_name}_predictions.csv"
        predictions_df.to_csv(predictions_file_name, index=False)
        mlflow.log_artifact(predictions_file_name)

        print(f"Predictions saved as {predictions_file_name} and logged to MLflow.")

    except Exception as e:
        print(f"An error occurred during model evaluation with cross-validation: {e}")
        raise e


def train_final_model(csv_path, trial_params, csv_name):
    """
    Train the final model on the entire dataset and log the model to MLflow.

    Args:
        csv_path (str): Path to the CSV file.
        trial_params (dict): Dictionary containing the trial parameters.
        csv_name (str): Name of the CSV file (for logging purposes).

    Raises:
        Exception: If any error occurs during final model training or logging.
    """
    try:
        # Load data
        X_full, y_full = load_data(csv_path, target_column_name='LABEL')

        # Convert to tensor
        X_full = X_full.astype(np.float32)
        y_full = y_full.astype(np.float32)

        # >>> Ten sam split 90/10 co w objective (SEED i test_size identyczne)
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.10, random_state=SEED
        )

        # Get best hyperparameters
        params = trial_params

        # Default values for missing parameters
        optimizer_name = params.get('optimizer', 'adam')
        learning_rate = params.get('learning_rate', 1e-3)
        adam_beta1 = params.get('adam_beta1', 0.9)
        adam_beta2 = params.get('adam_beta2', 0.999)
        sgd_momentum = params.get('sgd_momentum', 0.0)
        use_scheduler = params.get('use_scheduler', False)
        early_stop_patience = params.get('early_stop_patience', 10)
        clip_grad_value = params.get('clip_grad_value', 1.0)
        batch_size = params.get('batch_size', 32)
        epochs = params.get('epochs', 100)
        weight_decay  = params.get('weight_decay', 0.0)

        # Create model with best parameters
        input_dim = X_full.shape[1]
        # Again, create a fake trial object
        trial_for_model = optuna.trial.FixedTrial(params)
        model = Net(trial_for_model, input_dim).to(device)

        # Loss criterion
        criterion = nn.MSELoss()

        # Optimizer
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                   betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            beta1 = params.get('adamw_beta1', 0.9)       # <<<
            beta2 = params.get('adamw_beta2', 0.999)     # <<<
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                    betas=(beta1, beta2),
                                    weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler for learning rate reduction on plateau
        if use_scheduler:
            scheduler_factor   = params.get('scheduler_factor', 0.5)
            scheduler_patience = params.get('scheduler_patience', 5)
            min_lr             = params.get('min_lr', 1e-5)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=min_lr
            )

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

        # Training loop

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )

        loader = make_loader(dataset, batch_size, shuffle=True)

        scaler = amp.GradScaler('cuda', enabled=use_amp)
        use_batch_norm_effective = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())

        for epoch in range(epochs):
            model.train()
            epoch_loss_sum = 0.0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if (batch_X.size(0) > 1) or (not use_batch_norm_effective):
                    with amp.autocast('cuda', enabled=use_amp):
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)

                        if model.regularization == 'l1':
                            l1_norm = sum(p.abs().sum() for p in model.parameters())
                            loss = loss + model.reg_rate * l1_norm
                        elif model.regularization == 'l2':
                            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                            loss = loss + model.reg_rate * l2_norm

                    amp_opt_step(loss, optimizer, scaler, model, clip_grad_value, where="final/train")

                    epoch_loss_sum += loss.detach().float().item()
                else:
                    continue

            # Średnia strata na epokę
            epoch_loss_avg = epoch_loss_sum / max(1, len(loader))

            if use_scheduler:
                scheduler.step(epoch_loss_avg)  # tu nie masz walidacji, więc użyj train loss

            # Early stopping na train loss (w finalnym treningu brak walidacji)
            early_stopping(epoch_loss_avg)
            if early_stopping.early_stop:
                print(f'Final model: Early stopping triggered at epoch {epoch+1}')
                break

        model.eval()
        with torch.no_grad(), amp.autocast('cuda', enabled=use_amp):
            y_train_pred_t = model(torch.from_numpy(X_train).to(device)).squeeze()
        y_train_pred = y_train_pred_t.float().cpu().numpy()

        assert_finite_array("Train predictions (final)", y_train_pred)

        # ---------- METRYKI na train (90%) ----------
        rmse_train = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        r2_train   = float(r2_score(y_train, y_train_pred))
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("r2_train",   r2_train)
        print(f"RMSE_train: {rmse_train:.4f} | R2_train: {r2_train:.4f}")
        summary_file_name = f"{csv_name}_summary.txt" 
        with open(summary_file_name, "a") as f:         
            f.write(f"RMSE_train: {rmse_train}\n")
            f.write(f"R2_train:   {r2_train}\n")

        # ---------- TEST na 10% (których Optuna nie widziała) ----------
        X_test_t = torch.from_numpy(X_test).to(device)
        with torch.no_grad(), amp.autocast('cuda', enabled=use_amp):
            y_test_pred_t = model(X_test_t).squeeze()
        y_test_pred = y_test_pred_t.float().cpu().numpy()

        assert_finite_array("Final test predictions", y_test_pred)

        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        mae_test  = float(mean_absolute_error(y_test, y_test_pred))
        r2_test   = float(r2_score(y_test, y_test_pred))

        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("mae_test",  mae_test)
        mlflow.log_metric("r2_test",   r2_test)

        print(f"RMSE_test: {rmse_test:.4f} | MAE_test: {mae_test:.4f} | R2_test: {r2_test:.4f}")

        # dopisz do summary.txt
        with open(summary_file_name, "a") as f:
            f.write(f"RMSE_test: {rmse_test}\n")
            f.write(f"MAE_test:  {mae_test}\n")
            f.write(f"R2_test:   {r2_test}\n")

        # zapisz predykcje z testu jako artefakt
        test_pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_test_pred})
        test_pred_csv = f"{csv_name}_test_predictions.csv"
        test_pred_df.to_csv(test_pred_csv, index=False)
        mlflow.log_artifact(test_pred_csv)


        # ---------- WILLIAMS-plot (FULL TRAIN SET) ----------
        try:
            # 1) leverage
            X_feat = X_train.astype(np.float64)          # (N, p)
            nz = X_feat.std(0) > 1e-12                 # usuń kolumny zerowej wariancji
            X_std = (X_feat[:, nz] - X_feat[:, nz].mean(0)) / X_feat[:, nz].std(0)

            H = X_std @ np.linalg.pinv(X_std.T @ X_std) @ X_std.T
            leverage = np.diag(H)

            # 2) standaryzowane residua
            residuals  = y_train - y_train_pred
            std_resid  = residuals / (rmse_train * np.sqrt(1.0 - leverage + 1e-12))

            # 3) thresholdy
            p, n = X_std.shape[1], X_std.shape[0]
            h_star = 3.0 * (p + 1) / n

            wdf = pd.DataFrame({
                "y_true":        y_train,
                "y_pred":        y_train_pred,
                "residual":      residuals,
                "std_residual":  std_resid,
                "leverage":      leverage,
            })

            full_csv = f"{csv_name}_williams_full.csv"
            out_csv  = f"{csv_name}_williams_outliers.csv"
            wdf.to_csv(full_csv, index=False)
            wdf[(np.abs(std_resid) > 3.0) | (leverage > h_star)].to_csv(out_csv, index=False)

            mlflow.log_artifact(full_csv)
            mlflow.log_artifact(out_csv)
        except Exception as w_exc:
            print(f"[WARN] Williams data not written: {w_exc}")
        # -----------------------------------------------

        # Save final model
        model_file_name = f"{csv_name}_final_model.pth"
        torch.save(model.state_dict(), model_file_name)
        print(f"Final model saved as {model_file_name}")

        # Move model to CPU and set to evaluation mode
        model.to('cpu')
        model.eval()
        model = model.float()

        # Prepare input_example
        input_example = X_full[0:1].astype(np.float32)

        # Create wrapped model
        wrapped_model = WrappedModel(model)

        # Define conda_env
        conda_env = {
            'channels': ['defaults'],
            'dependencies': [
                f'python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
                'pip',
                {
                    'pip': [
                        f'torch=={torch.__version__}',
                        'mlflow',
                    ],
                },
            ],
            'name': 'mlflow-env'
        }

        # Set logging level to DEBUG
        import logging
        logging.getLogger("mlflow").setLevel(logging.DEBUG)

        # Log model to MLflow
        mlflow.pytorch.log_model(
            wrapped_model,
            artifact_path="model",
            conda_env=conda_env,
            input_example=input_example
        )

        print(f"Final model logged to MLflow.")

    except Exception as e:
        print(f"An error occurred during final model training: {e}")
        raise e


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train CNN models on CSV files in a directory.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the directory containing CSV files.')
    parser.add_argument('--experiment_name', type=str, required=False, default='Default', help='MLflow experiment name.')
    parser.add_argument('--n_trials', type=int, required=False, default=1000, help='Number of trials for Optuna hyperparameter optimization')

    args = parser.parse_args()

    csv_directory = args.csv_path
    experiment_name = args.experiment_name

    # Set MLflow experiment name
    mlflow.set_experiment(experiment_name)

    # Check if directory exists
    if not os.path.isdir(csv_directory):
        print(f"The provided path {csv_directory} is not a directory.")
        sys.exit(1)

    # Get list of CSV files
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in directory {csv_directory}")
        sys.exit(1)

    # Iterate over CSV files
    for csv_file in csv_files:
        csv_path = os.path.join(csv_directory, csv_file)
        csv_name = os.path.splitext(csv_file)[0]

        print(f"\nProcessing file: {csv_file}")

        try:
            # Start MLflow run for hyperparameter optimization
            with mlflow.start_run(
                    run_name=f"Optimization_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                    tags=tags_config_CNN_1D.mlflow_tags1
            ) as optuna_run:

                # --- Custom callback to print each finished trial result ---
                def print_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
                    # Bezpieczne formatowanie wartości
                    val = trial.value
                    val_txt = f"{val:.4f}" if isinstance(val, (float, int)) else "NA"
                    state = trial.state.name
                    print(f"[Trial {trial.number:>4}] state={state}  RMSE={val_txt}", flush=True)

                    try:
                        now = datetime.now().isoformat(timespec="seconds")
                        best_val = study.best_value
                        best_num = study.best_trial.number
                        params_json = json.dumps(trial.params, ensure_ascii=False)
                        with open(best_curve_csv, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([now, trial.number, val, best_val, best_num, params_json])
                    except Exception as e:
                        print(f"[WARN] Nie udało się dopisać do {best_curve_csv}: {e}", flush=True)


                # Objective function for study.optimize
                objective_wrapper = lambda tr: objective(tr, csv_path)

                best_curve_csv = f"{csv_name}_best_curve_live.csv"
                # jeśli plik nie istnieje – utwórz z nagłówkiem
                if not os.path.exists(best_curve_csv):
                    with open(best_curve_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "timestamp", "trial_number", "trial_value_rmse",
                            "best_value_so_far_rmse", "best_trial_number_so_far",
                            "params_json"
                        ])

                study = optuna.create_study(direction="minimize")
                study.optimize(
                    objective_wrapper,
                    n_trials=args.n_trials,
                    callbacks=[print_callback],
                    show_progress_bar=True,      # built-in tqdm progress bar
                    catch=(ValueError,)
                )

                #torch.cuda.empty_cache()

                # Log best parameters
                best_trial = study.best_trial  # Rename variable to best_trial
                mlflow.log_metric("best_value", best_trial.value)
                for key, value in best_trial.params.items():
                    mlflow.log_param(key, value)

                # Generate and log parameter importance plot
                importance_ax = optuna_viz.plot_param_importances(study)
                importance_fig = importance_ax.get_figure()
                importance_fig_file = f"{csv_name}_param_importance.png"
                importance_fig.savefig(importance_fig_file)
                mlflow.log_artifact(importance_fig_file)
                plt.close(importance_fig)

                # Save parameter importances
                param_importances = importance.get_param_importances(study)
                param_importances_file = f"{csv_name}_param_importance.json"
                with open(param_importances_file, 'w') as f:
                    json.dump(param_importances, f, indent=4)
                mlflow.log_artifact(param_importances_file)

                mlflow.log_artifact(best_curve_csv)

                # Log metrics
                mlflow.log_metric("RMSE", best_trial.user_attrs.get('rmse', None))

                print(f"Best trial for {csv_file}:")
                print(f"  Value (RMSE): {best_trial.value}")
                print("  Parameters:")
                for key, value in best_trial.params.items():
                    print(f"    {key}: {value}")

            # Store best parameters outside the with block for use in next steps
            best_trial_params = best_trial.params

            # Run MLflow for model evaluation with 10-fold cross-validation
            with mlflow.start_run(
                run_name=f"Evaluation_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_CNN_1D.mlflow_tags2
            ) as evaluation_run:
                # Evaluate model with 10CV
                evaluate_model_with_cv(csv_path, best_trial_params, csv_name)

            # Run MLflow for final model training
            with mlflow.start_run(
                run_name=f"Training_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_CNN_1D.mlflow_tags3
            ) as training_run:
                # Train final model
                train_final_model(csv_path, best_trial_params, csv_name)

        except Exception as e:
            print(f"An error occurred while processing {csv_file}: {e}")
            continue
