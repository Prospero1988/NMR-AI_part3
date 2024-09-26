# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 07:38:56 2024

@author: aleniak
"""

import setuptools
import warnings
import os
import pandas as pd
import joblib
import sys
import logging
from logging.handlers import RotatingFileHandler
import optuna
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import optuna.visualization as vis
import argparse
import subprocess
from mlflow.models.signature import infer_signature

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Import KFold from sklearn for indices
from sklearn.model_selection import KFold
import numpy as np

def setup_logging(logger_name, log_file):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Log formatting
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Logging initialized.")
    return logger

def check_input_directory(input_directory):
    if not os.path.exists(input_directory):
        logging.error(f"Directory {input_directory} does not exist.")
        sys.exit(1)

    csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

    if not csv_files:
        logging.error("No CSV files found in the input directory.")
        sys.exit(1)

    logging.info(f"Found {len(csv_files)} CSV file(s) in the directory.")
    return csv_files

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values  # Return as NumPy array

def log_search_space():
    search_space = {
        'num_layers': ('int', 1, 5),
        'units': ('int', 32, 512),
        'activation': ('categorical', ['relu', 'tanh', 'sigmoid']),
        'dropout_rate': ('float', 0.0, 0.5),
        'optimizer': ('categorical', ['adam', 'rmsprop', 'sgd']),
        'learning_rate': ('float', 1e-5, 1e-1, 'log'),
        'batch_size': ('int', 16, 128),
        'epochs': ('int', 10, 100),
    }
    mlflow.log_dict(search_space, 'hyperparameter_search_space.json')

def log_environment():
    try:
        # Export conda environment
        conda_env_file = 'conda_environment.yml'
        subprocess.call(['conda', 'env', 'export', '--no-builds', '-f', conda_env_file])
        mlflow.log_artifact(conda_env_file)
    except Exception as e:
        logging.warning(f"Could not log conda environment: {e}")
        # If conda is not available, try logging pip packages
        try:
            pip_req_file = 'requirements.txt'
            subprocess.call('pip freeze > requirements.txt', shell=True)
            mlflow.log_artifact(pip_req_file)
        except Exception as e:
            logging.warning(f"Could not log pip requirements: {e}")

def optimize_hyperparameters(X, y, logger, csv_file):
    logger.info("Starting hyperparameter optimization using TensorFlow/Keras.")

    # Log hyperparameter search space
    log_search_space()

    def objective(trial):
        params = {
            'num_layers': trial.suggest_int('num_layers', 1, 5),
            'units': trial.suggest_int('units', 32, 512),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'batch_size': trial.suggest_int('batch_size', 16, 128),
            'epochs': trial.suggest_int('epochs', 10, 100),
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=69)
        rmse_scores = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Build the model
            model = keras.Sequential()
            input_shape = X_train.shape[1]

            for i in range(params['num_layers']):
                if i == 0:
                    model.add(layers.Dense(params['units'], activation=params['activation'], input_dim=input_shape))
                else:
                    model.add(layers.Dense(params['units'], activation=params['activation']))
                if params['dropout_rate'] > 0.0:
                    model.add(layers.Dropout(params['dropout_rate']))

            model.add(layers.Dense(1))  # Output layer

            # Compile the model
            optimizer_name = params['optimizer']
            learning_rate = params['learning_rate']
            if optimizer_name == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

            model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Train the model
            history = model.fit(X_train, y_train,
                                validation_data=(X_val, y_val),
                                epochs=params['epochs'],
                                batch_size=params['batch_size'],
                                callbacks=[early_stopping],
                                verbose=0)

            # Evaluate the model
            val_mse = model.evaluate(X_val, y_val, verbose=0)[0]
            val_rmse = np.sqrt(val_mse)

            rmse_scores.append(val_rmse)

        mean_rmse = np.mean(rmse_scores)

        trial.set_user_attr('num_boost_round', 0)  # Placeholder

        # Log RMSE
        mlflow.log_metric('rmse', mean_rmse, step=trial.number)

        # Log hyperparameters as metrics
        numeric_params = {
            'num_layers': params['num_layers'],
            'units': params['units'],
            'dropout_rate': params['dropout_rate'],
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'epochs': params['epochs'],
        }
        for param_name, param_value in numeric_params.items():
            mlflow.log_metric(param_name, param_value, step=trial.number)

        # Log categorical parameters
        categorical_params = {
            'activation': params['activation'],
            'optimizer': params['optimizer'],
        }
        for param_name, param_value in categorical_params.items():
            mlflow.log_param(f"{param_name}_{trial.number}", param_value)

        # Collect trial data
        trial_data = {
            'trial_number': trial.number,
            'rmse': mean_rmse,
        }
        trial_data.update(params)

        if not hasattr(optimize_hyperparameters, 'trial_data_list'):
            optimize_hyperparameters.trial_data_list = []
        optimize_hyperparameters.trial_data_list.append(trial_data)

        return mean_rmse

    # Initialize trial data list
    optimize_hyperparameters.trial_data_list = []

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.NopPruner()
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna logs

    study.optimize(objective, n_trials=100, n_jobs=1)  # Set n_jobs=1 to avoid conflicts

    best_trial_number = study.best_trial.number
    logger.info(f"Best trial number: {best_trial_number}")

    best_params = study.best_params
    num_boost_round = study.best_trial.user_attrs.get('num_boost_round', 0)

    # Log best hyperparameters as parameters in MLflow
    mlflow.log_params(best_params)

    # Log hyperparameter importance
    param_importances = optuna.importance.get_param_importances(study)
    logger.info(f"Hyperparameter importances: {param_importances}")

    # Try to generate and save the importance plot
    try:
        fig = vis.plot_param_importances(study)
        param_importances_file = f"param_importances_{os.path.splitext(csv_file)[0]}.html"
        fig.write_html(param_importances_file)
        # Log the plot as an artifact
        mlflow.log_artifact(param_importances_file)
    except Exception as e:
        logger.warning(f"Could not generate hyperparameter importance plot: {e}")
        param_importances_file = None  # Set to None if plot is not generated

    # Save the study for further analysis
    study_file = f"optuna_study_{os.path.splitext(csv_file)[0]}.pkl"
    joblib.dump(study, study_file)
    mlflow.log_artifact(study_file)

    # Save trial data to CSV and log as artifact
    trial_data_df = pd.DataFrame(optimize_hyperparameters.trial_data_list)
    trial_data_csv = f"{os.path.splitext(csv_file)[0]}_trial_data.csv"
    trial_data_df.to_csv(trial_data_csv, index=False)
    mlflow.log_artifact(trial_data_csv)

    return best_params, num_boost_round, param_importances, param_importances_file, study

def train_final_model(X, y, best_params):
    # Build the model
    model = keras.Sequential()
    input_shape = X.shape[1]

    for i in range(best_params['num_layers']):
        if i == 0:
            model.add(layers.Dense(best_params['units'], activation=best_params['activation'], input_dim=input_shape))
        else:
            model.add(layers.Dense(best_params['units'], activation=best_params['activation']))
        if best_params['dropout_rate'] > 0.0:
            model.add(layers.Dropout(best_params['dropout_rate']))

    model.add(layers.Dense(1))  # Output layer

    # Compile the model
    optimizer_name = best_params['optimizer']
    learning_rate = best_params['learning_rate']
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X, y,
                        epochs=best_params['epochs'],
                        batch_size=best_params['batch_size'],
                        callbacks=[early_stopping],
                        verbose=0)

    return model, history

def evaluate_model(model, X, y):
    y_pred = model.predict(X).flatten()

    # Compute metrics
    abs_error = np.abs(y - y_pred)
    per_instance_data = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred,
        'Absolute Error': abs_error
    })

    # Overall metrics
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(abs_error)

    # Compute MAPE safely
    non_zero_mask = y != 0
    if np.any(non_zero_mask):
        mape_values = np.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask]) * 100
        mape_values = np.nan_to_num(mape_values, nan=0.0, posinf=0.0, neginf=0.0)
        mape = np.mean(mape_values)
    else:
        mape = float('inf')

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = (1 - ss_res / ss_tot)

    # Pearson correlation coefficient
    if y.size > 1:
        pearson_corr = np.corrcoef(y, y_pred)[0, 1]
    else:
        pearson_corr = float('nan')

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Pearson': pearson_corr,
    }

    return metrics, per_instance_data

def save_model(model, file_name, logger):
    model.save(file_name)
    logger.info(f"Model saved to {file_name}")

def save_metrics_and_params(metrics, params, file_name, logger):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(f"Model Status: Optimized new model\n\n")
        f.write("Optimal Hyperparameters:\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\nMetrics:\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"MAE: {metrics['MAE']:.4f}\n")
        f.write(f"MAPE: {metrics['MAPE']:.4f}\n")
        f.write(f"R2: {metrics['R2']:.4f}\n")
        f.write(f"Pearson Correlation Coefficient: {metrics['Pearson']:.4f}\n")
    logger.info(f"Metrics and hyperparameters saved to {file_name}")

def log_feature_importances(model, feature_names):
    # DNN models do not provide feature importances in the same way
    logging.info("DNN model does not provide feature importances.")
    mlflow.log_param('feature_importances', 'Not available for DNN')

def log_learning_curve(history):
    # Generate learning curve plot if possible
    if history is not None:
        plt.figure()
        plt.plot(history.history['loss'], label='train_loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt_file = 'learning_curve.png'
        plt.savefig(plt_file)
        plt.close()
        mlflow.log_artifact(plt_file)
    else:
        logging.info("No learning curve data to plot.")
        mlflow.log_param('learning_curve', 'Not available')

def process_file(csv_file, input_directory):
    try:
        data = load_data(os.path.join(input_directory, csv_file))

        y = data[:, 0]
        X = data[:, 1:]

        # Start parent MLflow run
        with mlflow.start_run(run_name=f"Run_{csv_file}"):
            # Add custom tags
            mlflow.set_tag('author', 'aleniak')
            mlflow.set_tag('organization', 'Celon')
            mlflow.set_tag('project', 'NMR')
            mlflow.set_tag('module', 'SPEC-AI')
            mlflow.set_tag('property', 'logD')
            mlflow.set_tag('paper', '3rd paper')

            # Log environment
            log_environment()

            # Initialize logger
            logger = setup_logging('Optimization', f'optimization_{csv_file}.log')
            logger.info(f"Starting processing for {csv_file}")

            # Hyperparameter optimization as a nested run
            with mlflow.start_run(run_name=f"Optimization_{csv_file}", nested=True):
                logger.info(f"Starting optimization on {csv_file}")
                best_params, num_boost_round, param_importances, param_importances_file, study = optimize_hyperparameters(
                    X, y, logger, csv_file
                )
                # Log hyperparameter importances
                mlflow.log_dict(param_importances, 'param_importances.json')
                if param_importances_file:
                    mlflow.log_artifact(param_importances_file)
                logger.info(f"Finished optimization on {csv_file}")

            # Training and evaluating the final model as a nested run
            with mlflow.start_run(run_name=f"Final_Model_{csv_file}", nested=True):
                logger.info(f"Starting final model training on {csv_file}")
                # Log the best hyperparameters
                mlflow.log_params(best_params)

                # Train the model
                model, history = train_final_model(X, y, best_params)

                # Evaluate the model
                metrics, per_instance_data = evaluate_model(model, X, y)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Save per-instance data
                per_instance_data_file = f"{os.path.splitext(csv_file)[0]}_per_instance_data.csv"
                per_instance_data.to_csv(per_instance_data_file, index=False)
                mlflow.log_artifact(per_instance_data_file)

                # Save the model and log as artifact
                model_file_name = f"{os.path.splitext(csv_file)[0]}_optimized_model.h5"
                save_model(model, model_file_name, logger)
                mlflow.log_artifact(model_file_name)

                # Log the model to MLflow using mlflow.tensorflow
                input_example = pd.DataFrame(X[:5])
                signature = infer_signature(input_example, model.predict(X[:5]))

                mlflow.tensorflow.log_model(
                    tf_saved_model_dir=model_file_name,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )

                # Save metrics and parameters
                metrics_file_name = f"{os.path.splitext(csv_file)[0]}_optimized_metrics_and_params.txt"
                save_metrics_and_params(metrics, best_params, metrics_file_name, logger)
                mlflow.log_artifact(metrics_file_name)

                # Log feature importances
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
                log_feature_importances(model, feature_names)

                # Log learning curve
                log_learning_curve(history)

                logger.info(
                    f"Metrics for {csv_file}: RMSE: {metrics['RMSE']:.4f}, "
                    f"MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.4f}, "
                    f"R2: {metrics['R2']:.4f}, Pearson: {metrics['Pearson']:.4f}"
                )
            logger.info(f"Finished processing for {csv_file}")

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        # End any active runs before logging exception
        if mlflow.active_run():
            mlflow.end_run()
        # Use a unique key for the exception to avoid conflicts
        with mlflow.start_run(run_name=f"Error_{csv_file}", nested=True):
            mlflow.log_param(f'Exception_{csv_file}', str(e))

def main():
    # Handle command-line arguments
    parser = argparse.ArgumentParser(description='DNN Hyperparameter Optimization with MLflow and Optuna')
    parser.add_argument('input_directory', type=str, help='Path to the input directory containing CSV files')
    parser.add_argument('--experiment_name', type=str, default='Default', help='Name of the MLflow experiment')
    args = parser.parse_args()

    input_directory = args.input_directory
    experiment_name = args.experiment_name

    # Set the MLflow experiment
    mlflow.set_experiment(experiment_name)

    csv_files = check_input_directory(input_directory)

    for csv_file in csv_files:
        process_file(csv_file, input_directory)
        # Ensure that each run is properly ended
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    main()
