# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:22:24 2024

@author: aleniak
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr
import optuna
import mlflow
from datetime import datetime

# Importowanie tagów MLflow z pliku tags_config.py
import tags_config_pytorch

# Ustawienie ziarna losowego dla powtarzalności wyników
SEED = 88
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Wybór urządzenia (GPU lub CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(csv_path, target_column_name='LABEL'):
    """
    Wczytuje dane z pliku CSV, odrzuca pierwszą kolumnę (nazwy próbek)
    i dzieli dane na cechy oraz zmienną docelową.

    Parametry:
    - csv_path: Ścieżka do pliku CSV.
    - target_column_name: Nazwa kolumny zawierającej zmienną docelową.

    Zwraca:
    - X: Macierz cech.
    - y: Wektor zmiennej docelowej.
    """
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


class Net(nn.Module):
    def __init__(self, trial, input_dim):
        super(Net, self).__init__()

        # Sugestia hiperparametrów dla funkcji aktywacji
        activation_name = trial.suggest_categorical('activation', ['relu', 'tanh'])
        if activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'tanh':
            self.activation = nn.Tanh()

        # Sugestia hiperparametrów dla regularyzacji
        self.regularization = trial.suggest_categorical('regularization', ['none', 'l1', 'l2'])
        if self.regularization == 'none':
            self.reg_rate = 0.0
        else:
            self.reg_rate = trial.suggest_float('reg_rate', 1e-5, 1e-2, log=True)

        # Sugestia liczby warstw
        n_layers = trial.suggest_int('n_layers', 1, 5)
        layers = []

        in_features = input_dim

        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 4, 512, log=True)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(self.activation)
            dropout_rate = trial.suggest_float(f'dropout_rate_l{i}', 0.0, 0.5)
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        # Warstwa wyjściowa
        layers.append(nn.Linear(in_features, 1))

        # Składanie warstw w Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_optimizer(trial, model_parameters):
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model_parameters, lr=learning_rate)
    return optimizer


def objective(trial, csv_path):
    """
    Funkcja celu dla optymalizacji hiperparametrów Optuna.

    Parametry:
    - trial: Obiekt trial Optuna.
    - csv_path: Ścieżka do pliku CSV.

    Zwraca:
    - final_rmse: Średni błąd RMS modelu w walidacji krzyżowej KFold.
    """
    try:
        # Wczytanie i przetworzenie danych
        X_train_full, y_train_full = load_data(csv_path, target_column_name='LABEL')

        # Podział danych na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=SEED
        )

        # Konwersja do tensora
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # Walidacja krzyżowa KFold na zbiorze treningowym
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        pearson_scores = []

        for train_index, valid_index in kf.split(X_train):
            X_train_fold = X_train[train_index].to(device)
            X_valid_fold = X_train[valid_index].to(device)
            y_train_fold = y_train[train_index].to(device)
            y_valid_fold = y_train[valid_index].to(device)

            # Tworzenie modelu
            input_dim = X_train_fold.shape[1]
            model = Net(trial, input_dim).to(device)

            # Definicja kryterium straty
            criterion = nn.MSELoss()

            # Definicja optymalizatora
            optimizer = get_optimizer(trial, model.parameters())

            # Trenowanie modelu
            epochs = 300
            batch_size = 32
            dataset = torch.utils.data.TensorDataset(X_train_fold, y_train_fold)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in loader:
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
                    optimizer.step()

            # Ocena modelu
            model.eval()
            with torch.no_grad():
                y_pred = model(X_valid_fold).squeeze().cpu().numpy()
                y_true = y_valid_fold.cpu().numpy()
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                pearson_corr, _ = pearsonr(y_true, y_pred)

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            pearson_scores.append(pearson_corr)

            # Zarządzanie pamięcią
            del model
            torch.cuda.empty_cache()

        # Obliczenie średnich metryk
        final_rmse = np.mean(rmse_scores)
        final_mae = np.mean(mae_scores)
        final_r2 = np.mean(r2_scores)
        final_pearson = np.mean(pearson_scores)
        std_mae = np.std(mae_scores)

        # Logowanie metryki dla bieżącego trialu
        mlflow.log_metric("Trial RMSE", final_rmse, step=trial.number)

        # Logowanie metryk
        trial.set_user_attr('rmse', final_rmse)
        trial.set_user_attr('mae', final_mae)
        trial.set_user_attr('r2', final_r2)
        trial.set_user_attr('pearson_corr', final_pearson)
        trial.set_user_attr('std_mae', std_mae)

        # Zwracanie RMSE jako wartości celu
        return final_rmse

    except Exception as e:
        print(f"Wystąpił błąd w funkcji celu: {e}")
        raise e


def train_final_model(csv_path, trial):
    """
    Trenuje finalny model z najlepszymi hiperparametrami z Optuna.

    Parametry:
    - csv_path: Ścieżka do pliku CSV.
    - trial: Najlepszy trial z badania Optuna.
    """
    try:
        # Wczytanie danych
        X_train_full, y_train_full = load_data(csv_path, target_column_name='LABEL')

        # Podział danych na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=SEED
        )

        # Konwersja do tensora
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # Tworzenie modelu z najlepszymi parametrami
        input_dim = X_train.shape[1]
        model = Net(trial, input_dim).to(device)

        # Definicja kryterium straty
        criterion = nn.MSELoss()

        # Definicja optymalizatora
        optimizer_name = trial.params['optimizer']
        learning_rate = trial.params['learning_rate']
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

        # Trenowanie modelu
        epochs = 300
        batch_size = 32
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in loader:
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
                optimizer.step()

        # Ocena modelu na zbiorze testowym
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test).squeeze().cpu().numpy()
            y_test_true = y_test.cpu().numpy()
            test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
            test_mae = mean_absolute_error(y_test_true, y_test_pred)
            test_r2 = r2_score(y_test_true, y_test_pred)
            test_pearson_corr, _ = pearsonr(y_test_true, y_test_pred)

        # Logowanie metryk zbioru testowego
        mlflow.log_metric("Test RMSE", test_rmse)
        mlflow.log_metric("Test MAE", test_mae)
        mlflow.log_metric("Test R2", test_r2)
        mlflow.log_metric("Test Pearson Correlation", test_pearson_corr)

        # Zapisanie finalnego modelu
        csv_file = os.path.basename(csv_path)
        csv_name = os.path.splitext(csv_file)[0]
        model_file_name = f"{csv_name}_optimized_model.pt"
        torch.save(model.state_dict(), model_file_name)
        mlflow.log_artifact(model_file_name)

        # Zapisanie metryk i parametrów do pliku txt
        summary = f"Best parameters:\n"
        for key, value in trial.params.items():
            summary += f"{key}: {value}\n"

        summary += f"\nTest Metrics:\n"
        summary += f"Test RMSE: {test_rmse}\n"
        summary += f"Test MAE: {test_mae}\n"
        summary += f"Test R2: {test_r2}\n"
        summary += f"Test Pearson Correlation: {test_pearson_corr}\n"

        summary_file_name = f"{csv_name}_summary.txt"
        with open(summary_file_name, 'w') as f:
            f.write(summary)

        mlflow.log_artifact(summary_file_name)

        print(f"\nOcena na zbiorze testowym dla {csv_file}:")
        print(f"  Test RMSE: {test_rmse}")
        print(f"  Test MAE: {test_mae}")
        print(f"  Test R2: {test_r2}")
        print(f"  Test Pearson Correlation: {test_pearson_corr}")

    except Exception as e:
        print(f"Wystąpił błąd podczas treningu finalnego modelu: {e}")
        raise e


if __name__ == '__main__':
    # Parsowanie argumentów linii poleceń
    parser = argparse.ArgumentParser(description='Trenowanie modeli DNN na plikach CSV w katalogu.')
    parser.add_argument('--csv_path', type=str, required=True, help='Ścieżka do katalogu zawierającego pliki CSV.')
    parser.add_argument('--experiment_name', type=str, required=False, default='Default', help='Nazwa eksperymentu w MLflow.')

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
            # Rozpoczęcie runu MLflow dla optymalizacji hiperparametrów
            with mlflow.start_run(
                run_name=f"Optimization_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_pytorch.mlflow_tags1
            ) as optuna_run:
                # Definicja funkcji celu z przekazaniem csv_path
                def objective_wrapper(trial):
                    return objective(trial, csv_path)

                study = optuna.create_study(direction='minimize')
                study.optimize(objective_wrapper, n_trials=50)

                # Logowanie najlepszych parametrów
                trial = study.best_trial
                mlflow.log_param("best_value", trial.value)
                for key, value in trial.params.items():
                    mlflow.log_param(key, value)

                # Logowanie dodatkowych metryk z najlepszego trialu
                mlflow.log_metric("RMSE", trial.user_attrs['rmse'])
                mlflow.log_metric("MAE", trial.user_attrs['mae'])
                mlflow.log_metric("R2", trial.user_attrs['r2'])
                mlflow.log_metric("Pearson Correlation", trial.user_attrs['pearson_corr'])
                mlflow.log_metric("MAE Std Dev", trial.user_attrs['std_mae'])

                print(f"Najlepszy trial dla {csv_file}:")
                print(f"  Wartość (RMSE): {trial.value}")
                print("  Parametry: ")
                for key, value in trial.params.items():
                    print(f"    {key}: {value}")

            # Rozpoczęcie nowego runu MLflow dla treningu finalnego modelu
            with mlflow.start_run(
                run_name=f"Training_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_pytorch.mlflow_tags2
            ) as training_run:
                # Trening finalnego modelu
                train_final_model(csv_path, trial)

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania {csv_file}: {e}")
            continue
