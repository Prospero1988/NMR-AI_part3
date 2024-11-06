# -*- coding: utf-8 -*-
"""
Script for regression using 1D CNN with Optuna optimization and cross-validation.

Created on Fri Oct  4 10:22:24 2024

@author: aleniak (modified)
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import optuna
import mlflow
import mlflow.pytorch  # Import dla mlflow.pytorch
from datetime import datetime
import argparse
import optuna.visualization.matplotlib as optuna_viz
import matplotlib.pyplot as plt
import json
from optuna import importance

# Importowanie tagów MLflow z pliku tags_config_pytorch_CNN_1D.py
import tags_config_pytorch_CNN_1D

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ustawienie ziarna losowego dla powtarzalności wyników
SEED = 88
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Wybór urządzenia (GPU lub CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0.0):
        """
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
    try:
        data = pd.read_csv(csv_path)
        # Odrzucenie pierwszej kolumny (nazwy próbek)
        data = data.drop(data.columns[0], axis=1)
        y = data[target_column_name].values
        X = data.drop(columns=[target_column_name]).values
        return X, y
    except FileNotFoundError:
        print(f"Plik {csv_path} nie został znaleziony.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Brak danych w pliku {csv_path}.")
        sys.exit(1)
    except KeyError:
        print(f"Kolumna docelowa '{target_column_name}' nie została znaleziona w {csv_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"Wystąpił błąd podczas wczytywania danych z {csv_path}: {e}")
        sys.exit(1)


def get_optimizer(trial, model_parameters):
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adamw'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    if optimizer_name == 'adam':
        beta1 = trial.suggest_float('adam_beta1', 0.8, 0.9999)
        beta2 = trial.suggest_float('adam_beta2', 0.9, 0.9999)
        optimizer = optim.Adam(model_parameters, lr=learning_rate, betas=(beta1, beta2))
    elif optimizer_name == 'adamw':
        beta1 = trial.suggest_float('adamw_beta1', 0.8, 0.9999)
        beta2 = trial.suggest_float('adamw_beta2', 0.9, 0.9999)
        optimizer = optim.AdamW(model_parameters, lr=learning_rate, betas=(beta1, beta2))
    elif optimizer_name == 'sgd':
        momentum = trial.suggest_float('sgd_momentum', 0.0, 0.99)
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model_parameters, lr=learning_rate)
    return optimizer


class Net(nn.Module):
    def __init__(self, trial, input_dim):
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

        # Regularization
        self.regularization = trial.suggest_categorical('regularization', ['none', 'l1', 'l2'])
        if self.regularization == 'none':
            self.reg_rate = 0.0
        else:
            self.reg_rate = trial.suggest_float('reg_rate', 1e-5, 1e-2, log=True)

        # Dropout rate
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

        # Batch normalization
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])

        # Convolutional layers
        num_conv_layers = trial.suggest_int('num_conv_layers', 2, 5)
        conv_layers = []
        in_channels = 1  # Input channels for Conv1d
        input_length = input_dim  # Initial input length

        for i in range(num_conv_layers):
            out_channels = trial.suggest_int(f'num_filters_l{i}', 16, 128, log=True)
            kernel_size = trial.suggest_int(f'kernel_size_l{i}', 3, 15, step=2)
            stride = trial.suggest_int(f'stride_l{i}', 1, 3)
            padding = trial.suggest_int(f'padding_l{i}', 0, 3)

            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(activation)
            if dropout_rate > 0.0:
                conv_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

            # Calculate new input length after this layer
            input_length = int((input_length + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            if input_length <= 0:
                raise ValueError('Negative or zero input length. Adjust kernel_size, stride, or padding.')

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
            if dropout_rate > 0.0:
                fc_layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*fc_layers)

        # Initialize weights
        init_method = trial.suggest_categorical('weight_init', ['xavier', 'kaiming', 'normal'])
        self.apply(lambda m: self.init_weights(m, init_method))

    def init_weights(self, m, init_method):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_method == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Reshape input for Conv1d: (batch_size, channels, length)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# Definicja klasy WrappedModel
class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model
        self.model.eval()  # Ustawienie modelu w trybie ewaluacji

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.type(torch.float32)
        with torch.no_grad():
            return self.model(x)


def objective(trial, csv_path):
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

        # Sugestia hiperparametrów
        batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
        epochs = trial.suggest_int('epochs', 50, 500, step=50)
        early_stop_patience = trial.suggest_int('early_stop_patience', 5, 20)
        use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
        clip_grad_value = trial.suggest_float('clip_grad_value', 0.1, 3.0, step=0.1)

        # Walidacja krzyżowa KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        rmse_scores = []

        for fold, (train_index, valid_index) in enumerate(kf.split(X_train), 1):
            X_train_fold = X_train[train_index]
            X_valid_fold = X_train[valid_index]
            y_train_fold = y_train[train_index]
            y_valid_fold = y_train[valid_index]

            # Tworzenie modelu
            input_dim = X_train_fold.shape[1]
            model = Net(trial, input_dim).to(device)

            # Kryterium straty
            criterion = nn.MSELoss()

            # Optymalizator
            optimizer = get_optimizer(trial, model.parameters())

            # Schemat zmiany współczynnika uczenia
            if use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

            # Early stopping
            early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

            # Trenowanie
            dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32),
                                                     torch.tensor(y_train_fold, dtype=torch.float32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()

                    # Sprawdzenie rozmiaru batcha dla BatchNorm
                    if batch_X.size(0) > 1:
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs.view(-1), batch_y)
                    else:
                        continue  # Pomijamy, jeśli batch_size == 1

                    # Dodanie regularizacji
                    if model.regularization == 'l1':
                        l1_norm = sum(p.abs().sum() for p in model.parameters())
                        loss = loss + model.reg_rate * l1_norm
                    elif model.regularization == 'l2':
                        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                        loss = loss + model.reg_rate * l2_norm
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

                    optimizer.step()

                if use_scheduler:
                    scheduler.step(loss)

                # Walidacja
                model.eval()
                with torch.no_grad():
                    X_valid_tensor = torch.tensor(X_valid_fold, dtype=torch.float32).to(device)
                    y_valid_tensor = torch.tensor(y_valid_fold, dtype=torch.float32).to(device)
                    y_valid_pred = model(X_valid_tensor).squeeze().cpu().numpy()
                    y_valid_true = y_valid_tensor.cpu().numpy()
                    val_loss = mean_squared_error(y_valid_true, y_valid_pred)

                # Sprawdzenie early stopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break

            # Ocena
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

        # Obliczenie średniego RMSE
        final_rmse = np.mean(rmse_scores)

        # Logowanie metryki
        mlflow.log_metric("Trial RMSE", final_rmse, step=trial.number)

        # Ustawienie atrybutów trialu
        trial.set_user_attr('rmse', final_rmse)

        return final_rmse

    except Exception as e:
        print(f"Wystąpił błąd w funkcji celu: {e}")
        raise e


def evaluate_model_with_cv(csv_path, trial_params, csv_name):
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

        # Pobranie hiperparametrów z trial_params
        params = trial_params

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
            # Uwaga: Musimy stworzyć obiekt trial z parametrami, aby przekazać go do Net
            # Jednakże Net wymaga obiektu trial. Możemy stworzyć fikcyjny obiekt trial z parametrami.
            trial_for_model = optuna.trial.FixedTrial(params)
            model = Net(trial_for_model, input_dim).to(device)

            # Kryterium straty
            criterion = nn.MSELoss()

            # Optymalizator
            if optimizer_name == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                       betas=(adam_beta1, adam_beta2))
            elif optimizer_name == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
            elif optimizer_name == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                        betas=(adam_beta1, adam_beta2))
            else:
                raise ValueError(f"Nieznany optymalizator: {optimizer_name}")

            # Scheduler
            if use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

            # Early stopping
            early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

            # Trenowanie
            dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                     torch.tensor(y_train, dtype=torch.float32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()

                    if batch_X.size(0) > 1:
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                    else:
                        continue  # Pomijamy, jeśli batch_size == 1

                    # Dodanie regularizacji
                    if model.regularization == 'l1':
                        l1_norm = sum(p.abs().sum() for p in model.parameters())
                        loss = loss + model.reg_rate * l1_norm
                    elif model.regularization == 'l2':
                        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                        loss = loss + model.reg_rate * l2_norm
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

                    optimizer.step()

                if use_scheduler:
                    scheduler.step(loss)

                # Walidacja
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
                    y_valid_pred = model(X_test_tensor).squeeze().cpu().numpy()
                    y_valid_true = y_test_tensor.cpu().numpy()
                    val_loss = mean_squared_error(y_valid_true, y_valid_pred)

                # Sprawdzenie early stopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f'Fold {fold}: Early stopping triggered at epoch {epoch+1}')
                    break

            # Ocena
            model.eval()
            with torch.no_grad():
                y_test_pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
                y_test_true = torch.tensor(y_test, dtype=torch.float32).to(device).cpu().numpy()
                rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
                mae = mean_absolute_error(y_test_true, y_test_pred)
                r2 = r2_score(y_test_true, y_test_pred)
                pearson_corr, _ = pearsonr(y_test_true, y_test_pred)

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            pearson_scores.append(pearson_corr)

            # Zbieranie predykcji
            all_true.extend(y_test_true)
            all_preds.extend(y_test_pred)

            # Zarządzanie pamięcią
            del model
            torch.cuda.empty_cache()

        # Obliczenie metryk
        final_rmse = np.mean(rmse_scores)
        final_mae = np.mean(mae_scores)
        final_r2 = np.mean(r2_scores)
        final_pearson = np.mean(pearson_scores)

        # Logowanie metryk
        mlflow.log_metric("RMSE", final_rmse)
        mlflow.log_metric("MAE", final_mae)
        mlflow.log_metric("R2", final_r2)
        mlflow.log_metric("Pearson Correlation", final_pearson)

        # Zapisanie metryk do pliku
        summary = f"Best parameters:\n"
        for key, value in params.items():
            summary += f"{key}: {value}\n"

        summary += f"\n10CV Metrics:\n"
        summary += f"10CV RMSE: {final_rmse}\n"
        summary += f"10CV MAE: {final_mae}\n"
        summary += f"10CV R2: {final_r2}\n"
        summary += f"10CV Pearson Correlation: {final_pearson}\n"

        summary_file_name = f"{csv_name}_summary.txt"
        with open(summary_file_name, 'w') as f:
            f.write(summary)

        mlflow.log_artifact(summary_file_name)

        print(f"\nOcena na zbiorze walidacyjnym dla {csv_name}:")
        print(f"  10CV RMSE: {final_rmse}")
        print(f"  10CV MAE: {final_mae}")
        print(f"  10CV R2: {final_r2}")
        print(f"  10CV Pearson Correlation: {final_pearson}")

        # Generowanie wykresu
        plt.figure(figsize=(8, 6))
        plt.scatter(all_true, all_preds, alpha=0.6, label='Predicted vs Actual')
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], color='red', linestyle='--', label='Ideal')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values (Cross-Validation)')
        plt.legend()

        # Zapisanie wykresu
        pred_vs_actual_fig_file = f"{csv_name}_pred_vs_actual_cv.png"
        plt.savefig(pred_vs_actual_fig_file)
        mlflow.log_artifact(pred_vs_actual_fig_file)
        plt.close()

        # Zapisanie predykcji do CSV
        predictions_df = pd.DataFrame({'Actual': all_true, 'Predicted': all_preds})
        predictions_file_name = f"{csv_name}_predictions.csv"
        predictions_df.to_csv(predictions_file_name, index=False)
        mlflow.log_artifact(predictions_file_name)

        print(f"Predictions saved as {predictions_file_name} and logged to MLflow.")

    except Exception as e:
        print(f"Wystąpił błąd podczas ewaluacji modelu z walidacją krzyżową: {e}")
        raise e


def train_final_model(csv_path, trial_params, csv_name):
    try:
        # Wczytanie danych
        X_full, y_full = load_data(csv_path, target_column_name='LABEL')

        # Konwersja do tensora
        X_full = X_full.astype(np.float32)
        y_full = y_full.astype(np.float32)

        # Pobranie najlepszych hiperparametrów
        params = trial_params

        # Domyślne wartości dla brakujących parametrów
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

        # Tworzenie modelu z najlepszymi parametrami
        input_dim = X_full.shape[1]
        # Uwaga: Ponownie tworzymy fikcyjny obiekt trial
        trial_for_model = optuna.trial.FixedTrial(params)
        model = Net(trial_for_model, input_dim).to(device)

        # Kryterium straty
        criterion = nn.MSELoss()

        # Optymalizator
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                   betas=(adam_beta1, adam_beta2))
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                    betas=(adam_beta1, adam_beta2))
        else:
            raise ValueError(f"Nieznany optymalizator: {optimizer_name}")

        # Scheduler
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        # Early stopping
        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

        # Trenowanie
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_full, dtype=torch.float32),
                                                 torch.tensor(y_full, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                if batch_X.size(0) > 1:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                else:
                    continue  # Pomijamy, jeśli batch_size == 1

                # Dodanie regularizacji
                if model.regularization == 'l1':
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + model.reg_rate * l1_norm
                elif model.regularization == 'l2':
                    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                    loss = loss + model.reg_rate * l2_norm
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

                optimizer.step()

            if use_scheduler:
                scheduler.step(loss)

            # Sprawdzenie early stopping
            early_stopping(loss.item())
            if early_stopping.early_stop:
                print(f'Final model: Early stopping triggered at epoch {epoch+1}')
                break

        # Zapisanie finalnego modelu
        model_file_name = f"{csv_name}_final_model.pth"
        torch.save(model.state_dict(), model_file_name)
        print(f"Final model saved as {model_file_name}")

        # Przeniesienie modelu na CPU i ustawienie w trybie ewaluacji
        model.to('cpu')
        model.eval()
        model = model.float()

        # Przygotowanie input_example
        input_example = X_full[0:1].astype(np.float32)

        # Stworzenie opakowanego modelu
        wrapped_model = WrappedModel(model)

        # Definicja conda_env
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

        # Logowanie modelu do MLflow
        mlflow.pytorch.log_model(
            wrapped_model,
            artifact_path="model",
            conda_env=conda_env,
            input_example=input_example
        )

        print(f"Final model logged to MLflow.")

    except Exception as e:
        print(f"Wystąpił błąd podczas trenowania finalnego modelu: {e}")
        raise e


if __name__ == '__main__':
    # Parsowanie argumentów linii poleceń
    parser = argparse.ArgumentParser(description='Trenowanie modeli CNN na plikach CSV w katalogu.')
    parser.add_argument('--csv_path', type=str, required=True, help='Ścieżka do katalogu zawierającego pliki CSV.')
    parser.add_argument('--experiment_name', type=str, required=False, default='Default', help='Nazwa eksperymentu w MLflow.')
    parser.add_argument('--n_trials', type=int, required=False, default=1000, help='Liczba prób dla optymalizacji hiperparametrów w Optuna')

    args = parser.parse_args()

    csv_directory = args.csv_path
    experiment_name = args.experiment_name

    # Ustawienie nazwy eksperymentu w MLflow
    mlflow.set_experiment(experiment_name)

    # Sprawdzenie, czy katalog istnieje
    if not os.path.isdir(csv_directory):
        print(f"Podana ścieżka {csv_directory} nie jest katalogiem.")
        sys.exit(1)

    # Pobranie listy plików CSV
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"Brak plików CSV w katalogu {csv_directory}")
        sys.exit(1)

    # Iteracja po plikach CSV
    for csv_file in csv_files:
        csv_path = os.path.join(csv_directory, csv_file)
        csv_name = os.path.splitext(csv_file)[0]

        print(f"\nPrzetwarzanie pliku: {csv_file}")

        try:
            # Run MLflow dla optymalizacji hiperparametrów
            with mlflow.start_run(
                run_name=f"Optimization_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_pytorch_CNN_1D.mlflow_tags1
            ) as optuna_run:
                # Wrapper funkcji celu
                def objective_wrapper(trial):
                    return objective(trial, csv_path)

                study = optuna.create_study(direction='minimize')
                study.optimize(objective_wrapper, n_trials=args.n_trials)

                torch.cuda.empty_cache()

                # Logowanie najlepszych parametrów
                best_trial = study.best_trial  # Zmieniamy nazwę zmiennej na best_trial
                mlflow.log_param("best_value", best_trial.value)
                for key, value in best_trial.params.items():
                    mlflow.log_param(key, value)

                # Generowanie i logowanie wykresu ważności parametrów
                importance_ax = optuna_viz.plot_param_importances(study)
                importance_fig = importance_ax.get_figure()
                importance_fig_file = f"{csv_name}_param_importance.png"
                importance_fig.savefig(importance_fig_file)
                mlflow.log_artifact(importance_fig_file)
                plt.close(importance_fig)

                # Zapisanie ważności parametrów
                param_importances = importance.get_param_importances(study)
                param_importances_file = f"{csv_name}_param_importance.json"
                with open(param_importances_file, 'w') as f:
                    json.dump(param_importances, f, indent=4)
                mlflow.log_artifact(param_importances_file)

                # Logowanie metryk
                mlflow.log_metric("RMSE", best_trial.user_attrs.get('rmse', None))

                print(f"Najlepszy trial dla {csv_file}:")
                print(f"  Wartość (RMSE): {best_trial.value}")
                print("  Parametry:")
                for key, value in best_trial.params.items():
                    print(f"    {key}: {value}")

            # Przechowujemy najlepsze parametry poza blokiem with, aby były dostępne w kolejnych etapach
            best_trial_params = best_trial.params

            # Run MLflow dla ewaluacji modelu z 10-krotną walidacją krzyżową
            with mlflow.start_run(
                run_name=f"Evaluation_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_pytorch_CNN_1D.mlflow_tags2
            ) as evaluation_run:
                # Ewaluacja modelu z 10CV
                evaluate_model_with_cv(csv_path, best_trial_params, csv_name)

            # Run MLflow dla treningu finalnego modelu
            with mlflow.start_run(
                run_name=f"Training_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_pytorch_CNN_1D.mlflow_tags3
            ) as training_run:
                # Trenowanie finalnego modelu
                train_final_model(csv_path, best_trial_params, csv_name)

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania {csv_file}: {e}")
            continue
