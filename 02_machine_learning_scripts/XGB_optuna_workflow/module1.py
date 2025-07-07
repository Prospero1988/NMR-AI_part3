# hyperparameter_optimization.py

import os
import pandas as pd
import xgboost as xgb
import cupy as cp
import sys
import logging
from logging.handlers import RotatingFileHandler
import optuna
import mlflow
import mlflow.xgboost
import argparse
import subprocess
import time
import json
import tags_config
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as optuna_visualization

def setup_logging(logger_name, log_file):
    """
    Initialize and configure logging to both file and console.

    Args:
        logger_name (str): Name of the logger.
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("Logging initialized.")
    return logger

def check_input_directory(input_directory):
    """
    Check if the input directory exists and contains CSV files.

    Args:
        input_directory (str): Path to the input directory.

    Returns:
        list: Sorted list of CSV filenames.

    Raises:
        SystemExit: If the directory does not exist or contains no CSV files.
    """
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
    """
    Load data from a CSV file, dropping the first column (e.g., MOLECULE_NAME).

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        cp.ndarray: Data as a CuPy array.
    """
    data = pd.read_csv(file_path)
    data = data.iloc[:, 1:]  # Exclude MOLECULE_NAME if present
    return cp.asarray(data.values)  # Convert to CuPy array

def log_search_space():
    """
    Log the hyperparameter search space as a dictionary artifact in MLflow.
    """
    search_space = {
        'max_depth': ('int', 1, 30),
        'learning_rate': ('float', 1e-5, 0.5, 'log'),
        'subsample': ('float', 0.1, 1.0),
        'colsample_bytree': ('float', 0.1, 1.0),
        'colsample_bylevel': ('float', 0.1, 1.0),
        'colsample_bynode': ('float', 0.1, 1.0),
        'min_child_weight': ('float', 0.1, 10.0),
        'gamma': ('float', 0.0, 5.0),
        'reg_lambda': ('float', 1e-8, 100.0, 'log'),
        'reg_alpha': ('float', 1e-8, 100.0, 'log'),
        'max_delta_step': ('float', 0.0, 10.0),
        'grow_policy': ('categorical', ['depthwise', 'lossguide']),
    }
    mlflow.log_dict(search_space, 'hyperparameter_search_space.json')

def log_environment():
    """
    Log the current conda environment or pip requirements as an MLflow artifact.
    """
    try:
        conda_env_file = 'conda_environment.yml'
        subprocess.call(['conda', 'env', 'export', '--no-builds', '-f', conda_env_file])
        mlflow.log_artifact(conda_env_file)
    except Exception as e:
        logging.warning(f"Could not log conda environment: {e}")
        try:
            pip_req_file = 'requirements.txt'
            subprocess.call('pip freeze > requirements.txt', shell=True)
            mlflow.log_artifact(pip_req_file)
        except Exception as e:
            logging.warning(f"Could not log pip requirements: {e}")

def optimize_hyperparameters(X, y, logger, csv_file):
    """
    Optimize XGBoost hyperparameters using Optuna and log results to MLflow.

    Args:
        X (cp.ndarray): Feature matrix.
        y (cp.ndarray): Target vector.
        logger (logging.Logger): Logger instance.
        csv_file (str): Name of the CSV file being processed.

    Returns:
        tuple: (best_params, num_boost_round, study)
    """
    num_gpus = cp.cuda.runtime.getDeviceCount()
    logger.info(f"Number of GPUs available: {num_gpus}")
    log_search_space()

    def objective(trial):
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: Mean RMSE from cross-validation.
        """
        gpu_id = trial.number % num_gpus
        cp.cuda.Device(gpu_id).use()
        params = {
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist',
            'device': f'cuda:{gpu_id}',
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
            'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 10.0),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        }

        dtrain = xgb.DMatrix(X, label=y)
        num_boost_round = 5000
        early_stopping_rounds = 30

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=5,
            metrics='rmse',
            early_stopping_rounds=early_stopping_rounds,
            seed=69,
            verbose_eval=False,
            as_pandas=True,
        )

        mean_rmse = cv_results['test-rmse-mean'].iloc[-1]
        trial.set_user_attr('num_boost_round', len(cv_results))
        mlflow.log_metric('rmse', mean_rmse, step=trial.number)
        return mean_rmse

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=1000, n_jobs=-1)

    best_trial = study.best_trial
    best_params = best_trial.params
    num_boost_round = best_trial.user_attrs['num_boost_round']

    best_params.update({
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'device': f'cuda:{best_trial.number % num_gpus}',
    })

    mlflow.log_params(best_params)
    mlflow.log_metric('num_boost_round', num_boost_round)
    return best_params, num_boost_round, study

def process_file(csv_file, input_directory):
    """
    Process a single CSV file: load data, optimize hyperparameters, and log results.

    Args:
        csv_file (str): Name of the CSV file.
        input_directory (str): Path to the input directory.
    """
    try:
        logger = setup_logging('Optimization', f'optimization_{csv_file}.log')
        data = load_data(os.path.join(input_directory, csv_file))
        y = data[:, 0]
        X = data[:, 1:]

        run_name = f"Optimization_{csv_file}_{int(time.time())}"
        with mlflow.start_run(run_name=run_name):
            
            # Set tags from the external file
            for tag_name, tag_value in tags_config.mlflow_tags1.items():
                mlflow.set_tag(tag_name, tag_value)
            
            log_environment()
            best_params, num_boost_round, study = optimize_hyperparameters(X, y, logger, csv_file)
            best_params['num_boost_round'] = num_boost_round

            # Save best_params to JSON
            best_params_file = f"best_params_{os.path.splitext(csv_file)[0]}.json"
            with open(best_params_file, 'w') as f:
                json.dump(best_params, f)
            mlflow.log_artifact(best_params_file)

            fig = optuna_visualization.plot_param_importances(study)
            fig_name = f"hyperparameter_importance_{os.path.splitext(csv_file)[0]}.png"

            # Retrieve the Figure object from the Axes
            figure = fig.get_figure()
            figure.savefig(fig_name)
            mlflow.log_artifact(fig_name)
            plt.close(figure)

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        mlflow.log_param('Exception', str(e))

def main():
    """
    Main function for hyperparameter optimization with XGBoost and Optuna.
    Parses command-line arguments, sets up MLflow experiment, and processes each CSV file.
    """
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization with XGBoost and Optuna')
    parser.add_argument('input_directory', type=str, help='Path to the input directory containing CSV files')
    parser.add_argument('--experiment_name', type=str, default='Default', help='Name of the MLflow experiment')
    args = parser.parse_args()

    input_directory = args.input_directory
    experiment_name = args.experiment_name
    mlflow.set_experiment(experiment_name)

    csv_files = check_input_directory(input_directory)
    for csv_file in csv_files:
        process_file(csv_file, input_directory)

if __name__ == "__main__":
    main()
