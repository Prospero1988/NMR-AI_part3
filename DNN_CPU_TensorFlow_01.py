# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:22:24 2024

@author: aleniak
"""

import optuna
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import mlflow
import mlflow.tensorflow
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

def load_data(csv_path, target_column=0):
    data = pd.read_csv(csv_path)
    y = data.iloc[:, target_column].values
    X = data.drop(data.columns[target_column], axis=1).values
    return X, y

def get_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def create_model(trial, input_shape):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])

    # Preprocessing layers
    model.add(layers.Dense(256, activation=activation))
    model.add(layers.Dropout(trial.suggest_float('dropout_rate_1', 0.0, 0.5)))
    model.add(layers.Dense(256, activation=activation))
    model.add(layers.Dropout(trial.suggest_float('dropout_rate_2', 0.0, 0.5)))

    # Hidden layers
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(64, activation=activation))
    model.add(layers.Dense(32, activation=activation))
    model.add(layers.Dense(16, activation=activation))

    # Output layer
    model.add(layers.Dense(1))

    # Compile model
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    model.compile(optimizer=get_optimizer(optimizer, learning_rate), loss='mean_squared_error')

    return model

def objective(trial):
    # Load and preprocess data
    X, y = load_data('data.csv')
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

        # Create model
        input_shape = (X_train.shape[1],)
        model = create_model(trial, input_shape)

        # Train model
        batch_size = trial.suggest_int('batch_size', 16, 64)
        epochs = trial.suggest_int('epochs', 10, 100)
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)

        # Evaluate model
        y_pred = model.predict(X_valid).flatten()
        mse = mean_squared_error(y_valid, y_pred)
        mse_scores.append(mse)

    # Return RMSE
    return np.sqrt(np.mean(mse_scores))

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    with mlflow.start_run(run_name=f"data.csv_optimization_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") as run:
        study.optimize(objective, n_trials=50)

        # Log best parameters
        trial = study.best_trial
        mlflow.log_param("best_value", trial.value)
        for key, value in trial.params.items():
            mlflow.log_param(key, value)

        # Evaluate model with best parameters using KFold cross-validation
        X, y = load_data('data.csv')
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        final_mse_scores = []

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

            # Create model with best parameters
            input_shape = (X_train.shape[1],)
            model = keras.Sequential()
            model.add(layers.InputLayer(input_shape=input_shape))
            activation = trial.params['activation']

            model.add(layers.Dense(256, activation=activation))
            model.add(layers.Dropout(trial.params['dropout_rate_1']))
            model.add(layers.Dense(256, activation=activation))
            model.add(layers.Dropout(trial.params['dropout_rate_2']))
            model.add(layers.Dense(128, activation=activation))
            model.add(layers.Dense(64, activation=activation))
            model.add(layers.Dense(32, activation=activation))
            model.add(layers.Dense(16, activation=activation))
            model.add(layers.Dense(1))
            model.compile(optimizer=get_optimizer(trial.params['optimizer'], trial.params['learning_rate']), loss='mean_squared_error')

            # Train model
            batch_size = trial.params['batch_size']
            epochs = trial.params['epochs']
            callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
            model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)

            # Evaluate model
            y_pred = model.predict(X_valid).flatten()
            mse = mean_squared_error(y_valid, y_pred)
            final_mse_scores.append(mse)

        final_rmse = np.sqrt(np.mean(final_mse_scores))
        mlflow.log_metric("Final RMSE", final_rmse)

        # Train final model on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_shape = (X_scaled.shape[1],)
        final_model = keras.Sequential()
        final_model.add(layers.InputLayer(input_shape=input_shape))
        activation = trial.params['activation']

        final_model.add(layers.Dense(256, activation=activation))
        final_model.add(layers.Dropout(trial.params['dropout_rate_1']))
        final_model.add(layers.Dense(256, activation=activation))
        final_model.add(layers.Dropout(trial.params['dropout_rate_2']))
        final_model.add(layers.Dense(128, activation=activation))
        final_model.add(layers.Dense(64, activation=activation))
        final_model.add(layers.Dense(32, activation=activation))
        final_model.add(layers.Dense(16, activation=activation))
        final_model.add(layers.Dense(1))
        final_model.compile(optimizer=get_optimizer(trial.params['optimizer'], trial.params['learning_rate']), loss='mean_squared_error')

        # Train final model
        batch_size = trial.params['batch_size']
        epochs = trial.params['epochs']
        callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
        final_model.fit(X_scaled, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)

        # Save final model
        mlflow.tensorflow.log_model(final_model, "final_model")

        print("Best trial:")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
