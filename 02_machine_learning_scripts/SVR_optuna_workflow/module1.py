import setuptools  # Ensure setuptools is imported first
import time
import os
import sys
import argparse
import logging
import pandas as pd
import cupy as cp
import optuna
import mlflow
import json
import subprocess
from logging.handlers import RotatingFileHandler
from cuml.svm import SVR
from sklearn.model_selection import KFold
import tags_config
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as optuna_visualization

# Initialize basic logging
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='SVR Hyperparameter Optimization with MLflow and Optuna')
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
    # Drop the non-numeric column (if any)
    data = data.iloc[:, 1:]  # Exclude MOLECULE_NAME or similar
    return cp.asarray(data.values)  # Convert remaining data to CuPy array

def log_search_space():
    search_space = {
        'C': ('float', 1e-5, 1e6, 'log'),  # Rozszerzony zakres C
        'epsilon': ('float', 1e-7, 10.0, 'log'),  # Rozszerzony zakres epsilon
        'kernel': ('categorical', ['rbf']),  # Kernel pozostaje 'rbf'
        'tol': ('float', 1e-6, 1e-1, 'log'),  # Rozszerzony zakres tol
        'max_iter': ('int', -1),  # Max iterations
        'gamma_type': ('categorical', ['scale', 'auto', 'float']),
        'gamma': ('float', 1e-9, 1e3, 'log'),  # Rozszerzony zakres gamma
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
    # Detect the number of GPUs
    num_gpus = cp.cuda.runtime.getDeviceCount()
    logger.info(f"Number of GPUs available: {num_gpus}")

    # Initialize trial data list
    optimize_hyperparameters.trial_data_list = []

    def objective(trial):
        # Assign GPU ID based on trial number
        gpu_id = trial.number % num_gpus
        cp.cuda.Device(gpu_id).use()  # Set the current GPU

        params = {
            'C': trial.suggest_float('C', 1e-5, 100.0, log=True),  # Regularization parameter
            'epsilon': trial.suggest_float('epsilon', 1e-7, 1.0, log=True),  # Epsilon in the epsilon-SVR model
            'kernel': trial.suggest_categorical('kernel', ['rbf']),  # Kernel type
            'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True),  # Tolerance for stopping criterion
            'max_iter': -1  # Max iterations
        }

        # Handle 'gamma'
        gamma_type = trial.suggest_categorical('gamma_type', ['scale', 'auto', 'float'])
        if gamma_type == 'float':
            params['gamma'] = trial.suggest_float('gamma', 1e-5, 100.0, log=True)
        else:
            params['gamma'] = gamma_type
        # Store gamma_type for logging
        trial.set_user_attr('gamma_type', gamma_type)

        # Perform K-Fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=69)
        rmse_scores = []
        for fold, (train_index, val_index) in enumerate(kf.split(X.get())):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = SVR(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            # Compute RMSE using CuPy
            rmse = cp.sqrt(cp.mean((y_val - y_pred) ** 2)).item()
            rmse_scores.append(rmse)

        mean_rmse = cp.mean(cp.asarray(rmse_scores)).item()

        # Log RMSE
        mlflow.log_metric('rmse', mean_rmse, step=trial.number)

        return mean_rmse  # Optimize based on RMSE

    try:
        # Log hyperparameter search space
        log_search_space()

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.NopPruner()
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna logs

        # Set n_jobs=1 to avoid GPU conflicts
        study.optimize(objective, n_trials=5000, n_jobs=1)

        # After optimization, get GPU ID from the best trial
        best_trial_number = study.best_trial.number
        gpu_id = best_trial_number % num_gpus
        logger.info(f"Best trial number: {best_trial_number}, Best GPU ID: {gpu_id}")

        best_params = study.best_trial.params
        gamma_type = study.best_trial.user_attrs.get('gamma_type')
        if gamma_type == 'float':
            best_params['gamma'] = best_params['gamma']
        else:
            best_params['gamma'] = gamma_type

        # Log best hyperparameters as parameters in MLflow
        mlflow.log_params(best_params)
        mlflow.log_param('gamma_type', gamma_type)

        # Save the best hyperparameters to a JSON file
        best_params_file = f"best_hyperparameters_{os.path.splitext(csv_file)[0]}.json"
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f)
        mlflow.log_artifact(best_params_file)

        # Save trial data to CSV and log as artifact
        trials_df = study.trials_dataframe()
        trial_data_csv = f"{os.path.splitext(csv_file)[0]}_trial_data.csv"
        trials_df.to_csv(trial_data_csv, index=False)
        mlflow.log_artifact(trial_data_csv)

        return best_params, study

    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        raise  # Re-raise the exception

def process_file(csv_file, input_directory):
    try:
        data = load_data(os.path.join(input_directory, csv_file))

        y = data[:, 0]
        X = data[:, 1:]

        # Start MLflow run
        with mlflow.start_run(run_name=f"Optimization_{csv_file}_{int(time.time())}"):

            # Set tags from the external file
            for tag_name, tag_value in tags_config.mlflow_tags1.items():
                mlflow.set_tag(tag_name, tag_value)

            # Log environment
            log_environment()

            # Initialize logger
            logger = setup_logging('Optimization', f'optimization_{csv_file}.log')
            logger.info(f"Starting optimization for {csv_file}")

            # Hyperparameter optimization
            best_params, study = optimize_hyperparameters(X, y, logger, csv_file)
            logger.info(f"Finished optimization for {csv_file}")

            fig = optuna_visualization.plot_param_importances(study)
            fig_name = f'hyperparameter_importance_{os.path.splitext(csv_file)[0]}.png'

            # Retrieve the Figure object from the Axes
            figure = fig.get_figure()
            figure.savefig(fig_name)
            mlflow.log_artifact(fig_name)
            plt.close(figure)


    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        # End any active runs before logging exception
        if mlflow.active_run():
            mlflow.end_run()
        # Use a unique key for the exception to avoid conflicts
        with mlflow.start_run(run_name=f"Error_{csv_file}_{int(time.time())}", nested=True):
            mlflow.log_param(f'Exception_{csv_file}', str(e))

if __name__ == "__main__":
    main()
