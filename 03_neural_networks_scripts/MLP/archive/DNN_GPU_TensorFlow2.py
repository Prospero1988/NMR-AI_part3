# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:22:24 2024

@author: aleniak
"""

import os
import tempfile
import sys
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr
import optuna
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from datetime import datetime
from optuna.integration.mlflow import MLflowCallback
import optuna.visualization.matplotlib as optuna_viz
import matplotlib.pyplot as plt
import json
from optuna import importance

# Importowanie tagów MLflow z pliku tags_config.py
import tags_config_tensor

os.environ['TMPDIR'] = './tmp'

# Ustawienie ziarna losowego dla powtarzalności wyników
SEED = 88
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


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
        y = data[target_column_name].values.astype(np.float32)
        X = data.drop(columns=[target_column_name]).values.astype(np.float32)
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


def get_optimizer(optimizer_name, learning_rate):
    """
    Zwraca optymalizator Keras na podstawie nazwy i współczynnika uczenia.

    Parametry:
    - optimizer_name: Nazwa optymalizatora.
    - learning_rate: Współczynnik uczenia.

    Zwraca:
    - optimizer: Obiekt optymalizatora Keras.
    """
    if optimizer_name == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_model(trial, input_shape):
    """
    Tworzy model Keras Sequential z hiperparametrami sugerowanymi przez Optuna.

    Parametry:
    - trial: Obiekt trial Optuna.
    - input_shape: Kształt danych wejściowych.

    Zwraca:
    - model: Skompilowany model Keras.
    """
    model = keras.Sequential()
    model.add(layers.InputLayer(shape=input_shape))

    # Sugestia hiperparametrów dla funkcji aktywacji
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])

    # Sugestia hiperparametrów dla regularyzacji
    regularization = trial.suggest_categorical(
        'regularization', ['none', 'l1', 'l2']
    )
    if regularization == 'none':
        regularizer = None
    else:
        reg_rate = trial.suggest_float('reg_rate', 1e-5, 1e-2, log=True)
        if regularization == 'l1':
            regularizer = regularizers.l1(reg_rate)
        elif regularization == 'l2':
            regularizer = regularizers.l2(reg_rate)

    # Sugestia liczby warstw
    n_layers = trial.suggest_int('n_layers', 1, 5)

    # Dodawanie warstw modelu
    for i in range(n_layers):
        num_units = trial.suggest_int(f'n_units_l{i}', 4, 512, log=True)
        model.add(
            layers.Dense(
                num_units,
                activation=activation,
                kernel_regularizer=regularizer
            )
        )
        dropout_rate = trial.suggest_float(f'dropout_rate_l{i}', 0.0, 0.5)
        model.add(
            layers.Dropout(
                dropout_rate
            )
        )

    # Warstwa wyjściowa
    model.add(layers.Dense(1))

    # Kompilacja modelu
    optimizer_name = trial.suggest_categorical(
        'optimizer', ['adam', 'sgd', 'rmsprop']
    )
    learning_rate = trial.suggest_float(
        'learning_rate', 1e-5, 1e-2, log=True
    )
    optimizer = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


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

        # Walidacja krzyżowa KFold na zbiorze treningowym
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        pearson_scores = []

        for train_index, valid_index in kf.split(X_train):
            X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
            y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]

            # Tworzenie modelu
            input_shape = (X_train_fold.shape[1],)
            model = create_model(trial, input_shape)

            # Trenowanie modelu
            batch_size = 32
            epochs = 300
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            model.fit(
                X_train_fold,
                y_train_fold,
                validation_data=(X_valid_fold, y_valid_fold),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )

            # Ocena modelu
            y_pred = model.predict(X_valid_fold).flatten()
            rmse = np.sqrt(mean_squared_error(y_valid_fold, y_pred))
            mae = mean_absolute_error(y_valid_fold, y_pred)
            r2 = r2_score(y_valid_fold, y_pred)
            pearson_corr, _ = pearsonr(y_valid_fold, y_pred)

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            pearson_scores.append(pearson_corr)

            # Zarządzanie pamięcią
            tf.keras.backend.clear_session()

        # Obliczenie średnich metryk
        final_rmse = np.mean(rmse_scores)
        final_mae = np.mean(mae_scores)
        final_r2 = np.mean(r2_scores)
        final_pearson = np.mean(pearson_scores)
        std_mae = np.std(mae_scores)

        # Logowanie metryki dla bieżącego trialu
        # mlflow.log_metric("Trial RMSE", final_rmse, step=trial.number)
        # mlflow.log_metric("Trial MAE", final_mae, step=trial.number)
        # mlflow.log_metric("Trial R2", final_r2, step=trial.number)

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

def train_final_model(csv_path, trial, csv_name):
    """
    Trenuje finalny model z najlepszymi hiperparametrami z Optuna z walidacją krzyżową 10CV.

    Parametry:
    - csv_path: Ścieżka do pliku CSV.
    - trial: Najlepszy trial z badania Optuna.
    """
    try:
        # Wczytanie danych
        X, y = load_data(csv_path, target_column_name='LABEL')

        # Tworzenie modelu z najlepszymi parametrami
        input_shape = (X.shape[1],)
        final_model = keras.Sequential()
        final_model.add(layers.InputLayer(shape=input_shape))
        activation = trial.params['activation']

        # Pobranie parametrów regularyzacji
        regularization = trial.params['regularization']
        if regularization == 'none':
            regularizer = None
        else:
            reg_rate = trial.params['reg_rate']
            if regularization == 'l1':
                regularizer = regularizers.l1(reg_rate)
            elif regularization == 'l2':
                regularizer = regularizers.l2(reg_rate)

        # Budowa architektury modelu
        n_layers = trial.params['n_layers']
        for i in range(n_layers):
            num_units = trial.params[f'n_units_l{i}']
            final_model.add(
                layers.Dense(
                    num_units,
                    activation=activation,
                    kernel_regularizer=regularizer
                )
            )
            dropout_rate = trial.params[f'dropout_rate_l{i}']
            final_model.add(layers.Dropout(dropout_rate))
        final_model.add(layers.Dense(1))

        # Kompilacja modelu
        optimizer = get_optimizer(
            trial.params['optimizer'],
            trial.params['learning_rate']
        )
        final_model.compile(optimizer=optimizer, loss='mean_squared_error')

        # 10-krotna walidacja krzyżowa KFold
        kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        pearson_scores = []

        for train_index, valid_index in kf.split(X):
            X_train_fold, X_valid_fold = X[train_index], X[valid_index]
            y_train_fold, y_valid_fold = y[train_index], y[valid_index]

            # Trenowanie modelu dla bieżącego folda
            batch_size = trial.params['batch_size']
            epochs = trial.params['epochs']
            final_model.fit(
                X_train_fold,
                y_train_fold,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0
            )

            # Ocena modelu na zbiorze walidacyjnym
            y_valid_pred = final_model.predict(X_valid_fold).flatten()
            rmse = np.sqrt(mean_squared_error(y_valid_fold, y_valid_pred))
            mae = mean_absolute_error(y_valid_fold, y_valid_pred)
            r2 = r2_score(y_valid_fold, y_valid_pred)
            pearson_corr, _ = pearsonr(y_valid_fold, y_valid_pred)

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            pearson_scores.append(pearson_corr)

        # Obliczenie średnich metryk dla 10CV
        final_rmse = np.mean(rmse_scores)
        final_mae = np.mean(mae_scores)
        final_r2 = np.mean(r2_scores)
        final_pearson = np.mean(pearson_scores)

        # Logowanie średnich metryk zbioru walidacyjnego
        mlflow.log_metric("10CV RMSE", final_rmse)
        mlflow.log_metric("10CV MAE", final_mae)
        mlflow.log_metric("10CV R2", final_r2)
        mlflow.log_metric("10CV Pearson Correlation", final_pearson)

        # Przewidywanie wartości dla całego zbioru danych (treningowy + testowy)
        y_all_pred = final_model.predict(X).flatten()

        # Generowanie wykresu wartości przewidywanych vs rzeczywistych dla całego zbioru
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_all_pred, alpha=0.6, label='Predicted vs Actual')
        plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values (Entire Dataset)')
        plt.legend()

        # Zapisanie wykresu do pliku PNG
        pred_vs_actual_fig_file = f"{csv_name}_pred_vs_actual_entire_dataset.png"
        plt.savefig(pred_vs_actual_fig_file)

        # Logowanie pliku z wykresem do MLflow
        mlflow.log_artifact(pred_vs_actual_fig_file)

        # Zapisanie metryk i parametrów do pliku txt
        csv_file = os.path.basename(csv_path)
        csv_name = os.path.splitext(csv_file)[0]

        summary = f"Best parameters:\n"
        for key, value in trial.params.items():
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

        print(f"\nOcena na zbiorze walidacyjnym dla {csv_file}:")
        print(f"  10CV RMSE: {final_rmse}")
        print(f"  10CV MAE: {final_mae}")
        print(f"  10CV R2: {final_r2}")
        print(f"  10CV Pearson Correlation: {final_pearson}")

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

    # Tworzenie mlflow_callback
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="rmse"
        )
    
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
            # Definicja funkcji celu z przekazaniem csv_path
            def objective_wrapper(trial):
                return objective(trial, csv_path)

            # Optymalizacja z użyciem mlflow_callback
            study = optuna.create_study(direction='minimize')
            study.optimize(objective_wrapper, n_trials=200, callbacks=[mlflow_callback])

            # Logowanie najlepszych parametrów i metryk
            trial = study.best_trial

            # Generowanie wykresu ważności parametrów po zakończeniu optymalizacji
            importance_ax = optuna_viz.plot_param_importances(study)

            # Uzyskanie obiektu figury
            importance_fig = importance_ax.get_figure()

            # Zapisanie wykresu do pliku PNG
            importance_fig_file = f"{csv_name}_param_importance.png"
            importance_fig.savefig(importance_fig_file)

            # Logowanie pliku z wykresem do MLflow
            mlflow.log_artifact(importance_fig_file)

            # Zamknięcie figury, aby zwolnić zasoby
            importance_fig.clf()

            # Pobranie ważności hiperparametrów
            param_importances = importance.get_param_importances(study)

            # Zapisanie ważności hiperparametrów do pliku JSON
            param_importances_file = f"{csv_name}_param_importance.json"
            with open(param_importances_file, 'w') as f:
                json.dump(param_importances, f, indent=4)

            # Logowanie pliku JSON do MLflow
            mlflow.log_artifact(param_importances_file)

            # for key, value in trial.params.items():
                
            #    mlflow.log_param(key, value)

            # mlflow.log_metric("RMSE", trial.value)
            # mlflow.log_metric("MAE", trial.user_attrs['mae'])
            # mlflow.log_metric("R2", trial.user_attrs['r2'])
            # mlflow.log_metric("Pearson Correlation", trial.user_attrs['pearson_corr'])
            # mlflow.log_metric("MAE Std Dev", trial.user_attrs['std_mae'])

            print(f"Najlepszy trial dla {csv_file}:")
            print(f"  Wartość (RMSE): {trial.value}")
            print("  Parametry: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

            # Rozpoczęcie nowego runu MLflow dla treningu finalnego modelu
            with mlflow.start_run(
                run_name=f"Training_{csv_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=tags_config_tensor.mlflow_tags2
            ) as training_run:
                # Trening finalnego modelu
                train_final_model(csv_path, trial, csv_name)

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania {csv_file}: {e}")
            continue
