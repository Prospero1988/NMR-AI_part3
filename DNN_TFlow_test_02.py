# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 07:38:56 2024

@author: aleniak
"""

import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '3' to suppress more messages

# Logging configuration
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('optuna').setLevel(logging.WARNING)
logging.getLogger('mlflow').setLevel(logging.WARNING)

import pandas as pd
import joblib
import sys
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
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Import KFold from sklearn for final evaluation
from sklearn.model_selection import KFold

# Import TensorFlow Probability for Pearson correlation
import tensorflow_probability as tfp

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs Available: {gpus}")
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("No GPUs available.")


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

    csv_files = sorted([f for f in os.listdir(input_directory) if f.endswith('.csv')])

    if not csv_files:
        logging.error("No CSV files found in the input directory.")
        sys.exit(1)

    logging.info(f"Found {len(csv_files)} CSV file(s) in the directory.")
    return csv_files

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.values  # Convert to NumPy array
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    return data

def log_search_space():
    search_space = {
        'num_layers': ('int', 2, 20),
        'units': ('int', 32, 2048),
        'activation': ('categorical', ['relu', 'tanh', 'sigmoid']),
        'dropout_rate': ('float', 0.0, 0.5),
        'optimizer': ('categorical', ['adam', 'rmsprop', 'sgd']),
        'learning_rate': ('float', 1e-5, 1e-1, 'log'),
        'batch_size': ('int', 32, 512),
        'epochs': ('int', 10, 200),
    }
    mlflow.log_dict(search_space, 'hyperparameter_search_space.json')

def log_environment():
    try:
        # Export conda environment
        conda_env_file = 'conda_environment.yml'
        result = subprocess.run(['conda', 'env', 'export', '--no-builds', '-f', conda_env_file], check=True)
        mlflow.log_artifact(conda_env_file)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Could not log conda environment: {e}")
        # If conda is not available, try logging pip packages
        try:
            pip_req_file = 'requirements.txt'
            result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, check=True)
            with open(pip_req_file, 'w') as f:
                f.write(result.stdout.decode())
            mlflow.log_artifact(pip_req_file)
        except subprocess.CalledProcessError as e:
            logging.warning(f"Could not log pip requirements: {e}")

def optimize_hyperparameters(X, y, logger, csv_file):
    logger.info("Starting hyperparameter optimization using TensorFlow/Keras.")

    # Log hyperparameter search space
    log_search_space()

    def objective(trial):
        params = {
            'num_layers': trial.suggest_int('num_layers', 1, 10),
            'units': trial.suggest_int('units', 32, 1024),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'batch_size': trial.suggest_int('batch_size', 16, 256),
            'epochs': trial.suggest_int('epochs', 10, 100),
        }

        # Build the model
        model = keras.Sequential()
        input_shape = X.shape[1]
        
        # Add an Input layer first
        model.add(layers.Input(shape=(input_shape,)))

        for i in range(params['num_layers']):
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

        # Split the data into training and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X.numpy(), y.numpy(), test_size=0.2, random_state=trial.number
        )

        # Convert data to tf.data.Dataset
        batch_size = params['batch_size']
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Train the model without using validation_split
        history = model.fit(
            train_dataset,
            epochs=params['epochs'],
            validation_data=val_dataset,
            callbacks=[early_stopping],
            verbose=0
        )

        # Get the validation RMSE
        val_mse = history.history['val_mean_squared_error'][-1]
        val_rmse = tf.sqrt(val_mse).numpy().item()

        # Log RMSE
        mlflow.log_metric('rmse', val_rmse, step=trial.number)

        # Log numeric hyperparameters as metrics
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

        # Map categorical parameters to numeric values for logging
        activation_mapping = {'relu': 0, 'tanh': 1, 'sigmoid': 2}
        optimizer_mapping = {'adam': 0, 'rmsprop': 1, 'sgd': 2}

        mapped_categorical_params = {
            'activation': activation_mapping[params['activation']],
            'optimizer': optimizer_mapping[params['optimizer']],
        }
        for param_name, param_value in mapped_categorical_params.items():
            mlflow.log_metric(param_name, param_value, step=trial.number)

        # Collect trial data
        trial_data = {
            'trial_number': trial.number,
            'rmse': val_rmse,
        }
        trial_data.update(params)

        if not hasattr(optimize_hyperparameters, 'trial_data_list'):
            optimize_hyperparameters.trial_data_list = []
        optimize_hyperparameters.trial_data_list.append(trial_data)

        return val_rmse

    # Initialize trial data list
    optimize_hyperparameters.trial_data_list = []

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna logs

    study.optimize(objective, n_trials=2000, n_jobs=1)

    best_trial_number = study.best_trial.number
    logger.info(f"Best trial number: {best_trial_number}")

    best_params = study.best_params

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

    # Ensure the return statement is present
    return best_params, param_importances, param_importances_file, study

def train_final_model(X, y, best_params):
    # KFold cross-validation with 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=69)
    metrics_list = []
    history_list = []

    fold = 1
    for train_index, test_index in kf.split(X):
        print(f"Training on fold {fold}...")
        X_train_fold = tf.gather(X, train_index)
        X_test_fold = tf.gather(X, test_index)
        y_train_fold = tf.gather(y, train_index)
        y_test_fold = tf.gather(y, test_index)

        # Build the model
        model = keras.Sequential()
        input_shape = X_train_fold.shape[1]
        
        # Add an Input layer first
        model.add(layers.Input(shape=(input_shape,)))
        
        for i in range(best_params['num_layers']):
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Convert data to tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_fold, y_test_fold))

        # Shuffle, batch, and prefetch
        batch_size = best_params['batch_size']
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=best_params['epochs'],
            callbacks=[early_stopping],
            verbose=0
        )

        # Evaluate the model
        metrics, per_instance_data = evaluate_model(model, X_test_fold, y_test_fold)

        metrics_list.append(metrics)
        history_list.append(history)
        fold += 1

    # Compute average metrics using TensorFlow
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = tf.constant([metric[key] for metric in metrics_list], dtype=tf.float32)
        avg_metrics[key] = tf.reduce_mean(values).numpy().item()

    return model, history_list, avg_metrics

def evaluate_model(model, X, y):
    y_pred = model.predict(X).flatten()

    # Convert y and y_pred to tensors if not already
    y = tf.convert_to_tensor(y)
    y_pred = tf.convert_to_tensor(y_pred)

    # Compute metrics using TensorFlow
    abs_error = tf.abs(y - y_pred)
    per_instance_data = pd.DataFrame({
        'Actual': y.numpy(),
        'Predicted': y_pred.numpy(),
        'Absolute Error': abs_error.numpy()
    })

    # Overall metrics
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_pred))).numpy().item()
    mae = tf.reduce_mean(abs_error).numpy().item()

    # Compute MAPE safely
    epsilon = 1e-7  # Small value to avoid division by zero
    mape_values = tf.abs((y - y_pred) / tf.maximum(tf.abs(y), epsilon)) * 100
    mape = tf.reduce_mean(mape_values).numpy().item()

    ss_tot = tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
    ss_res = tf.reduce_sum(tf.square(y - y_pred))
    r2 = (1 - ss_res / ss_tot).numpy().item()

    # Pearson correlation coefficient
    if y.shape[0] > 1:
        pearson_corr = tfp.stats.correlation(y, y_pred, sample_axis=0, event_axis=None)
        pearson_corr = pearson_corr.numpy().item()
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

def log_learning_curve(history_list):
    if history_list:
        plt.figure()
        for history in history_list:
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

        X = data[:, 1:]
        y = data[:, 0]

        # Start parent MLflow run
        with mlflow.start_run(run_name=f"Run_{csv_file}"):
            # Add custom tags
            mlflow.set_tag('author', 'aleniak')
            mlflow.set_tag('organization', 'Celon')
            mlflow.set_tag('project', 'NMR')
            mlflow.set_tag('module', 'SPEC-AI')
            mlflow.set_tag('property', 'logD')
            mlflow.set_tag('paper', '3rd paper')
            mlflow.set_tag('technology', 'DNN Keras')
            # Log environment
            log_environment()

            # Initialize logger
            logger = setup_logging('Optimization', f'optimization_{csv_file}.log')
            logger.info(f"Starting processing for {csv_file}")

            # Hyperparameter optimization as a nested run
            with mlflow.start_run(run_name=f"Optimization_{csv_file}", nested=True):
                logger.info(f"Starting optimization on {csv_file}")
                best_params, param_importances, param_importances_file, study = optimize_hyperparameters(
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
                model, history_list, metrics = train_final_model(X, y, best_params)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log the model to MLflow using mlflow.keras
                # Prepare input example
                #input_example = X[:5].numpy()  # Convert TensorFlow tensor to NumPy array
                #input_example = pd.DataFrame(input_example)
                
                # Predict using the model (ensure input is in the correct format)
                #predictions = model.predict(input_example)
                
                # Infer signature
                #signature = infer_signature(input_example, predictions)
                
                # Log the model
                mlflow.keras.log_model(
                    model=model,
                    artifact_path="model",
                    #signature=signature,
                    #input_example=input_example
                )


                # Save metrics and parameters
                metrics_file_name = f"{os.path.splitext(csv_file)[0]}_optimized_metrics_and_params.txt"
                save_metrics_and_params(metrics, best_params, metrics_file_name, logger)
                mlflow.log_artifact(metrics_file_name)

                # Log feature importances
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
                log_feature_importances(model, feature_names)

                # Log learning curve
                log_learning_curve(history_list)

                logger.info(
                    f"Metrics for {csv_file}: RMSE: {metrics['RMSE']:.4f}, "
                    f"MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.4f}, "
                    f"R2: {metrics['R2']:.4f}, Pearson: {metrics['Pearson']:.4f}"
                )
            logger.info(f"Finished processing for {csv_file}")

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        # Log exception in MLflow
        mlflow.log_param('Exception', str(e))
    finally:
        # Ensure that each run is properly ended
        if mlflow.active_run():
            mlflow.end_run()

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
