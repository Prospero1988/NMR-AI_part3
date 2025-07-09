# -*- coding: utf-8 -*-
"""
Simplified PyTorch script with extended optimization.

Created on Fri Oct  4 10:22:24 2024

@author: Arkadiusz Leniak
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
from sklearn.preprocessing import StandardScaler

from optuna import importance

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import MLflow tags from tags_config_pytorch.py
import tags_config_MLP_1D

# Set random seed for reproducibility
SEED = 88
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Select device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_data(csv_path, target_column_name='LABEL'):
    """
    Load data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        target_column_name (str): Name of the target column.

    Returns:
        tuple: Features and target variable.

    Raises:
        FileNotFoundError: If the file is not found.
        pd.errors.EmptyDataError: If the file is empty.
        KeyError: If the target column is not found.
        Exception: For any other exceptions.
    """
    try:
        data = pd.read_csv(csv_path)
        # Odrzucenie pierwszej kolumny (nazwy próbek)
        data = data.drop(data.columns[0], axis=1)
        y = data[target_column_name].values
        X = data.drop(columns=[target_column_name]).values
        return X, y
    except FileNotFoundError:
        print(f"File {csv_path} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"No data in file {csv_path}.")
        sys.exit(1)
    except KeyError:
        print(f"Target column '{target_column_name}' not found in {csv_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data from {csv_path}: {e}")
        sys.exit(1)


def get_optimizer(trial, model_parameters):
    """
    Get the optimizer based on the trial suggestion.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        model_parameters (iterable): Model parameters to optimize.

    Returns:
        torch.optim.Optimizer: The selected optimizer.
    """
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    if optimizer_name == 'adam':
        # Dodajemy optymalizację beta1 i beta2 dla Adama
        beta1 = trial.suggest_float('adam_beta1', 0.8, 0.9999)
        beta2 = trial.suggest_float('adam_beta2', 0.9, 0.9999)
        optimizer = optim.Adam(model_parameters, lr=learning_rate, betas=(beta1, beta2))
    elif optimizer_name == 'adamw':
        beta1 = trial.suggest_float('adamw_beta1', 0.8, 0.9999)
        beta2 = trial.suggest_float('adamw_beta2', 0.9, 0.9999)
        optimizer = optim.AdamW(model_parameters, lr=learning_rate, betas=(beta1, beta2))
    elif optimizer_name == 'sgd':
        # Dodajemy optymalizację momentum dla SGD
        momentum = trial.suggest_float('sgd_momentum', 0.0, 0.99)
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model_parameters, lr=learning_rate)
    return optimizer


class Net(nn.Module):
    """
    Neural network model.

    Args:
        nn (Module): Inherits from PyTorch nn.Module.

    Attributes:
        activation (callable): Activation function.
        regularization (str): Regularization type ('none', 'l1', 'l2').
        reg_rate (float): Regularization rate.
        model (Sequential): Sequential model containing layers.
    """

    def __init__(self, trial, input_dim):
        """
        Initialize the neural network.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            input_dim (int): Dimensionality of the input data.
        """
        super(Net, self).__init__()

        # Activation function selection
        activation_name = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'selu'])
        self.activation = self.get_activation_function(activation_name)

        # Regularization
        self.regularization = trial.suggest_categorical('regularization', ['none', 'l1', 'l2'])
        self.reg_rate = trial.suggest_float('reg_rate', 1e-5, 1e-2, log=True) if self.regularization != 'none' else 0.0

        # Layer configuration
        num_layers = trial.suggest_int('num_layers', 1, 10)
        units = trial.suggest_int('units', 32, 512, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

        layers_list = []
        in_features = input_dim

        for _ in range(num_layers):
            layers_list.append(nn.Linear(in_features, units))
            if use_batch_norm:
                layers_list.append(nn.BatchNorm1d(units))
            layers_list.append(self.activation)
            if dropout_rate > 0.0:
                layers_list.append(nn.Dropout(dropout_rate))
            in_features = units

        # Output layer
        layers_list.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers_list)

        # Weight initialization
        self.init_method = trial.suggest_categorical('weight_init', ['xavier', 'kaiming', 'normal'])
        self.apply(self.init_weights)

    def get_activation_function(self, name):
        """
        Get the activation function by name.

        Args:
            name (str): Name of the activation function.

        Returns:
            callable: The activation function.

        Raises:
            ValueError: If the activation function is not supported.
        """
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif name == 'selu':
            return nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def init_weights(self, m):
        """
        Initialize weights of the model layers.

        Args:
            m (Module): The layer module.
        """
        if isinstance(m, nn.Linear):
            if self.init_method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif self.init_method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif self.init_method == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)

# Definicja klasy WrappedModel
class WrappedModel(nn.Module):
    """
    Wrapped model for evaluation.

    Args:
        nn (Module): Inherits from PyTorch nn.Module.

    Attributes:
        model (Module): The neural network model.
    """

    def __init__(self, model):
        """
        Initialize the wrapped model.

        Args:
            model (Module): The neural network model.
        """
        super(WrappedModel, self).__init__()
        self.model = model
        self.model.eval()  # Ustawienie modelu w trybie ewaluacji

    def forward(self, x):
        """
        Forward pass of the wrapped model.

        Args:
            x (Tensor or ndarray): Input tensor or numpy array.

        Returns:
            Tensor: Output tensor.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.type(torch.float32)
        with torch.no_grad():
            return self.model(x)

def objective(trial, csv_path):
    """
    Objective function for Optuna trial.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        csv_path (str): Path to the CSV file.

    Returns:
        float: The RMSE metric for the trial.

    Raises:
        Exception: For any exceptions during the trial.
    """
    try:
        # Wczytanie i przetworzenie danych
        X_train_full, y_train_full = load_data(csv_path, target_column_name='LABEL')

        # Konwersja do tensora
        X_train_full = X_train_full.astype(np.float32)
        y_train_full = y_train_full.astype(np.float32)

        # Podział danych na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=SEED
        )

        # Sugestia batch_size i epochs
        batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
        epochs = trial.suggest_int('epochs', 50, 200, step=50)

        # Sugestia liczby epok patience dla Early Stopping
        early_stop_patience = trial.suggest_int('early_stop_patience', 5, 20)

        # Walidacja krzyżowa KFold na zbiorze treningowym
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        rmse_scores = []

        for fold, (train_index, valid_index) in enumerate(kf.split(X_train), 1):
            X_train_fold = X_train[train_index]
            X_valid_fold = X_train[valid_index]
            y_train_fold = y_train[train_index]
            y_valid_fold = y_train[valid_index]

            # Normalizacja danych
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_valid_fold = scaler.transform(X_valid_fold)

            # Tworzenie modelu
            input_dim = X_train_fold.shape[1]
            model = Net(trial, input_dim).to(device)

            # Definicja kryterium straty
            criterion = nn.MSELoss()

            # Definicja optymalizatora
            optimizer = get_optimizer(trial, model.parameters())

            # Definicja schematu zmiany współczynnika uczenia
            use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
            if use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

            # Inicjalizacja EarlyStopping
            early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

            # Trenowanie modelu
            dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32),
                                                     torch.tensor(y_train_fold, dtype=torch.float32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()

                    # Sprawdzenie rozmiaru batcha
                    if batch_X.size(0) > 1:
                        # Jeśli batch ma więcej niż 1 próbkę, wszystko działa normalnie
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs.view(-1), batch_y)
                    else:
                        # Jeśli batch ma tylko 1 próbkę, pomijamy ten batch
                        continue

                    # Dodanie regularizacji
                    if model.regularization == 'l1':
                        l1_norm = sum(p.abs().sum() for p in model.parameters())
                        loss = loss + model.reg_rate * l1_norm
                    elif model.regularization == 'l2':
                        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                        loss = loss + model.reg_rate * l2_norm
                    loss.backward()

                    # Gradient Clipping
                    clip_grad_value = trial.suggest_float('clip_grad_value', 0.1, 1.0, step=0.1)
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

                    optimizer.step()

                if use_scheduler:
                    scheduler.step(loss)

                # Walidacja po każdej epoce
                model.eval()
                with torch.no_grad():
                    X_valid_tensor = torch.tensor(X_valid_fold, dtype=torch.float32).to(device)
                    y_valid_tensor = torch.tensor(y_valid_fold, dtype=torch.float32).to(device)
                    y_valid_pred = model(X_valid_tensor).squeeze().cpu().numpy()
                    y_valid_true = y_valid_tensor.cpu().numpy()
                    val_loss = mean_squared_error(y_valid_true, y_valid_pred)

                # Aktualizacja EarlyStopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break

            # Ocena modelu
            model.eval()
            with torch.no_grad():
                X_valid_tensor = torch.tensor(X_valid_fold, dtype=torch.float32).to(device)
                y_valid_tensor = torch.tensor(y_valid_fold, dtype=torch.float32).to(device)
                y_pred = model(X_valid_tensor).squeeze().cpu().numpy()
                y_true = y_valid_tensor.cpu().numpy()
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            rmse_scores.append(rmse)

            # Zarządzanie pamięcią
            del model
            torch.cuda.empty_cache()

        # Obliczenie średnich metryk
        final_rmse = np.mean(rmse_scores)

        # Logowanie metryki dla bieżącego trialu
        mlflow.log_metric("Trial RMSE", final_rmse, step=trial.number)

        # Logowanie metryk
        trial.set_user_attr('rmse', final_rmse)

        # Zwracanie RMSE jako wartości celu
        return final_rmse

    except Exception as e:
        print(f"Error in objective function: {e}")
        raise e


def evaluate_model_with_cv(csv_path, trial, csv_name):
    """
    Evaluate the model using cross-validation.

    Args:
        csv_path (str): Path to the CSV file.
        trial (optuna.Trial): The Optuna trial object.
        csv_name (str): Base name of the CSV file (for saving results).

    Raises:
        Exception: For any exceptions during evaluation.
    """
    try:
        # Wczytanie danych
        X_full, y_full = load_data(csv_path, target_column_name='LABEL')

        # Konwersja do tensora
        X_full = X_full.astype(np.float32)
        y_full = y_full.astype(np.float32)

        # Inicjalizacja K-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

        rmse_scores = []
        mae_scores = []
        r2_scores = []
        pearson_scores = []

        # Zbiory do zbierania wszystkich prawdziwych i przewidzianych wartości
        all_true = []
        all_preds = []

        # Pobranie hiperparametrów z najlepszego triala
        params = trial.params

        # Dodanie default dla brakujących parametrów
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

        for fold, (train_index, test_index) in enumerate(kf.split(X_full), 1):
            X_train, X_test = X_full[train_index], X_full[test_index]
            y_train, y_test = y_full[train_index], y_full[test_index]

            # Tworzenie modelu z najlepszymi parametrami
            input_dim = X_train.shape[1]
            model = Net(trial, input_dim).to(device)

            # Definicja kryterium straty
            criterion = nn.MSELoss()

            # Definicja optymalizatora
            if optimizer_name == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                       betas=(adam_beta1, adam_beta2))
            elif optimizer_name == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
            elif optimizer_name == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Nieznany optymalizator: {optimizer_name}")

            # Definicja schematu zmiany współczynnika uczenia
            if use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

            # Inicjalizacja EarlyStopping
            early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

            # Trenowanie modelu
            dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                     torch.tensor(y_train, dtype=torch.float32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    # Dodanie regularizacji
                    if model.regularization == 'l1':
                        l1_norm = sum(p.abs().sum() for p in model.parameters())
                        loss = loss + model.reg_rate * l1_norm
                    elif model.regularization == 'l2':
                        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                        loss = loss + model.reg_rate * l2_norm
                    loss.backward()

                    # Gradient Clipping
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

                    optimizer.step()

                if use_scheduler:
                    scheduler.step(loss)

                # Walidacja po każdej epoce
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
                    y_valid_pred = model(X_test_tensor).squeeze().cpu().numpy()
                    y_valid_true = y_test_tensor.cpu().numpy()
                    val_loss = mean_squared_error(y_valid_true, y_valid_pred)

                # Aktualizacja EarlyStopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f'Fold {fold}: Early stopping triggered at epoch {epoch+1}')
                    break

            # Ocena modelu na bieżącym zbiorze testowym
            model.eval()
            with torch.no_grad():
                y_test_pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze().detach().cpu().numpy()
                y_test_true = torch.tensor(y_test, dtype=torch.float32).to(device).cpu().numpy()
                rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
                mae = mean_absolute_error(y_test_true, y_test_pred)
                r2 = r2_score(y_test_true, y_test_pred)
                pearson_corr, _ = pearsonr(y_test_true, y_test_pred)

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            pearson_scores.append(pearson_corr)

            # Zbieranie predykcji i prawdziwych wartości
            all_true.extend(y_test_true)
            all_preds.extend(y_test_pred)

            # Zarządzanie pamięcią
            del model  # Ensure the model is deleted to free memory
            torch.cuda.empty_cache()

        # Obliczenie średnich metryk z K-fold cross-validation
        final_rmse = np.mean(rmse_scores)
        final_mae = np.mean(mae_scores)
        q2 = np.mean(r2_scores)
        final_pearson = np.mean(pearson_scores)

        # Logowanie metryk zbioru testowego
        mlflow.log_metric("RMSE", final_rmse)
        mlflow.log_metric("MAE", final_mae)
        mlflow.log_metric("Q2", q2)
        mlflow.log_metric("Pearson Correlation", final_pearson)

        # Zapisanie metryk i parametrów do pliku txt
        csv_file = os.path.basename(csv_path)
        csv_name = os.path.splitext(csv_file)[0]

        summary = f"Best parameters:\n"
        for key, value in trial.params.items():
            summary += f"{key}: {value}\n"

        summary += f"\n10CV Metrics:\n"
        summary += f"10CV RMSE: {final_rmse}\n"
        summary += f"10CV MAE: {final_mae}\n"
        summary += f"10CV Q2: {q2}\n"
        summary += f"10CV Pearson Correlation: {final_pearson}\n"

        summary_file_name = f"{csv_name}_summary.txt"
        with open(summary_file_name, 'w') as f:
            f.write(summary)

        mlflow.log_artifact(summary_file_name)

        print(f"\nEvaluation on the validation set for {csv_file}:")
        print(f"  10CV RMSE: {final_rmse}")
        print(f"  10CV MAE: {final_mae}")
        print(f"  10CV Q2: {q2}")
        print(f"  10CV Pearson Correlation: {final_pearson}")

        # Generowanie wykresu dla wszystkich foldów
        plt.figure(figsize=(8, 6))
        plt.scatter(all_true, all_preds, alpha=0.6, label='Predicted vs Actual')
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], color='red', linestyle='--', label='Ideal')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values (Cross-Validation)')
        plt.legend()

        # Zapisanie wykresu do pliku PNG
        pred_vs_actual_fig_file = f"{csv_name}_pred_vs_actual_cv.png"
        plt.savefig(pred_vs_actual_fig_file)
        mlflow.log_artifact(pred_vs_actual_fig_file)
        plt.close()

        # Zapisanie wartości all_true i all_preds do pliku CSV
        preds_df = pd.DataFrame({'True_Values': all_true, 'Predicted_Values': all_preds})
        preds_file_name = f"{csv_name}_predictions.csv"
        preds_df.to_csv(preds_file_name, index=False)

        # Logowanie pliku z predykcjami jako artefakt do MLflow
        mlflow.log_artifact(preds_file_name)


    except Exception as e:
        print(f"Error during model evaluation with cross-validation: {e}")
        raise e


def train_final_model(csv_path, trial, csv_name):
    """
    Train the final model on the entire dataset.

    Args:
        csv_path (str): Path to the CSV file.
        trial (optuna.Trial): The Optuna trial object.
        csv_name (str): Base name of the CSV file (for saving the model).

    Raises:
        Exception: For any exceptions during training.
    """
    try:
        # Wczytanie danych
        X_full, y_full = load_data(csv_path, target_column_name='LABEL')

        # Konwersja do tensora
        X_full = X_full.astype(np.float32)
        y_full = y_full.astype(np.float32)

        # Tworzenie modelu z najlepszymi parametrami
        input_dim = X_full.shape[1]
        model = Net(trial, input_dim).to(device)

        # Definicja kryterium straty
        criterion = nn.MSELoss()

        # Definicja optymalizatora
        optimizer = get_optimizer(trial, model.parameters())

        # Pobranie hiperparametrów z triala
        batch_size = trial.params.get('batch_size', 32)
        epochs = trial.params.get('epochs', 100)

        # Trenowanie modelu na całym zbiorze danych
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_full, dtype=torch.float32),
                                                 torch.tensor(y_full, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_train_pred = model(
                torch.tensor(X_full, dtype=torch.float32).to(device)
            ).squeeze().cpu().numpy()

        r2_train = r2_score(y_full, y_train_pred)
        mlflow.log_metric("R2_train", r2_train)
        print(f"R2 on training set: {r2_train:.4f}")

        summary_file_name = f"{csv_name}_summary.txt" 
        with open(summary_file_name, "a") as f:         
            f.write(f"R2_train: {r2_train}\n")

        mlflow.log_artifact(summary_file_name)  

        # Zapisanie modelu
        model_file_name = f"{csv_name}_final_model.pth"
        torch.save(model.state_dict(), model_file_name)
        print(f"Model saved as {model_file_name}")

        # Przeniesienie modelu na CPU i ustawienie w trybie ewaluacji
        model.to('cpu')
        model.eval()
        model = model.float()

        # Przygotowanie przykładu wejściowego
        input_example = X_full[0:1].astype(np.float32)  # numpy.ndarray

        # Stworzenie opakowanego modelu
        wrapped_model = WrappedModel(model)

        # Definicja conda_env lub pip_requirements
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

        # Ustawienie poziomu logowania na DEBUG
        import logging
        logging.getLogger("mlflow").setLevel(logging.DEBUG)

        # Logowanie modelu do MLflow z input_example i conda_env
        mlflow.pytorch.log_model(
            wrapped_model,
            "model",
            conda_env=conda_env,
            input_example=input_example
        )
        print(f"Model logged to MLflow.")

    except Exception as e:
        print(f"Error training final model: {e}")
        raise e


if __name__ == '__main__':
    # Parsowanie argumentów linii poleceń
    parser = argparse.ArgumentParser(description='Train DNN models on CSV files in a directory.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the directory containing CSV files.')
    parser.add_argument('--experiment_name', type=str, required=False, default='Default', help='Experiment name in MLflow.')

    args = parser.parse_args()

    csv_directory = args.csv_path
    experiment_name = args.experiment_name

    # Ustawienie nazwy eksperymentu w MLflow
    mlflow.set_experiment(experiment_name)

    # Sprawdzenie, czy katalog istnieje
    if not os.path.isdir(csv_directory):
        print(f"Provided path {csv_directory} is not a directory.")
        sys.exit(1)

    # Pobranie listy plików CSV
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in directory {csv_directory}")
        sys.exit(1)

    # Iteracja po plikach CSV
    for csv_file in csv_files:
        csv_path = os.path.join(csv_directory, csv_file)
        csv_name = os.path.splitext(csv_file)[0]

        print(f"\nProcessing file: {csv_file}")

        try:
            # Rozpoczęcie runu MLflow dla optymalizacji hiperparametrów
            with mlflow.start_run(
                run_name=f"Optimization_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_MLP_1D.mlflow_tags1
            ) as optuna_run:
                # Definicja funkcji celu z przekazaniem csv_path
                def objective_wrapper(trial):
                    return objective(trial, csv_path)

                study = optuna.create_study(direction='minimize')
                study.optimize(objective_wrapper, n_trials=2000)  # Zwiększono liczbę prób

                torch.cuda.empty_cache()

                # Logowanie najlepszych parametrów
                trial = study.best_trial
                mlflow.log_param("best_value", trial.value)
                for key, value in trial.params.items():
                    mlflow.log_param(key, value)

                # Generowanie wykresu ważności parametrów po zakończeniu optymalizacji
                importance_ax = optuna_viz.plot_param_importances(study)

                # Pobranie obiektu Figure z Axes
                importance_fig = importance_ax.get_figure()

                # Zapisanie wykresu do pliku PNG
                importance_fig_file = f"{csv_name}_param_importance.png"
                importance_fig.savefig(importance_fig_file)

                # Logowanie pliku z wykresem do MLflow
                mlflow.log_artifact(importance_fig_file)

                # Zamknięcie figury, aby zwolnić zasoby
                plt.close(importance_fig)

                # Pobranie ważności hiperparametrów
                param_importances = importance.get_param_importances(study)

                # Zapisanie ważności hiperparametrów do pliku JSON
                param_importances_file = f"{csv_name}_param_importance.json"
                with open(param_importances_file, 'w') as f:
                    json.dump(param_importances, f, indent=4)

                # Logowanie pliku JSON do MLflow
                mlflow.log_artifact(param_importances_file)

                # Logowanie dodatkowych metryk z najlepszego trialu
                mlflow.log_metric("RMSE", trial.user_attrs.get('rmse', None))
                # Dodatkowe metryki mogą być logowane w funkcji evaluate_model_with_cv

                print(f"Best trial for {csv_file}:")
                print(f"  Value (RMSE): {trial.value}")
                print("  Parameters: ")
                for key, value in trial.params.items():
                    print(f"    {key}: {value}")

            # Rozpoczęcie nowego runu MLflow dla oceny modelu z walidacją krzyżową
            with mlflow.start_run(
                run_name=f"Evaluation_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_MLP_1D.mlflow_tags2
            ) as evaluation_run:
                # Ocena modelu z użyciem 10-krotnej walidacji krzyżowej
                evaluate_model_with_cv(csv_path, trial, csv_name)

            # Rozpoczęcie nowego runu MLflow dla treningu finalnego modelu
            with mlflow.start_run(
                run_name=f"Training_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_MLP_1D.mlflow_tags3
            ) as training_run:
                # Trenowanie finalnego modelu
                train_final_model(csv_path, trial, csv_name)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
