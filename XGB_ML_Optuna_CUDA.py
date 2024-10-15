# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 07:38:56 2024

@author: aleniak
"""

import setuptools
import warnings
import os
import pandas as pd
import xgboost as xgb
import cupy as cp
import numpy as np
import joblib
import sys
import logging
from logging.handlers import RotatingFileHandler
import optuna
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import optuna.visualization as vis
import argparse
import subprocess
from mlflow.models.signature import infer_signature

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
    # Drop the non-numeric column
    data = data.iloc[:, 1:]  # Exclude MOLECULE_NAME
    return cp.asarray(data.values)  # Convert remaining data to CuPy array

def log_search_space():
    search_space = {
        'max_depth': ('int', 1, 20),
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

    # Log hyperparameter search space
    log_search_space()

    def objective(trial):
        # Assign GPU ID based on trial number
        gpu_id = trial.number % num_gpus
        cp.cuda.Device(gpu_id).use()  # Set the current GPU
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
            metrics='rmse',  # Only RMSE
            early_stopping_rounds=early_stopping_rounds,
            seed=69,
            verbose_eval=False,
            as_pandas=True,
        )

        mean_rmse = cv_results['test-rmse-mean'].iloc[-1]

        trial.set_user_attr('num_boost_round', len(cv_results))

        # Log RMSE
        mlflow.log_metric('rmse', mean_rmse, step=trial.number)

        # Log hyperparameters as metrics
        # Numeric hyperparameters
        numeric_params = {
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'colsample_bylevel': params['colsample_bylevel'],
            'colsample_bynode': params['colsample_bynode'],
            'min_child_weight': params['min_child_weight'],
            'gamma': params['gamma'],
            'reg_lambda': params['reg_lambda'],
            'reg_alpha': params['reg_alpha'],
            'max_delta_step': params['max_delta_step'],
        }
        for param_name, param_value in numeric_params.items():
            mlflow.log_metric(param_name, param_value, step=trial.number)

        # Categorical hyperparameters
        categorical_params = {
            'grow_policy': params['grow_policy'],
        }
        # Map categorical values to numeric codes
        grow_policy_mapping = {'depthwise': 0, 'lossguide': 1}
        mlflow.log_metric('grow_policy', grow_policy_mapping[categorical_params['grow_policy']], step=trial.number)

        # Collect trial data
        trial_data = {
            'trial_number': trial.number,
            'rmse': mean_rmse,
        }
        trial_data.update(params)

        # Append trial data to the list
        if not hasattr(optimize_hyperparameters, 'trial_data_list'):
            optimize_hyperparameters.trial_data_list = []
        optimize_hyperparameters.trial_data_list.append(trial_data)

        return mean_rmse  # Optimize based on RMSE

    # Initialize trial data list
    optimize_hyperparameters.trial_data_list = []
 
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.NopPruner()
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna logs

    # Set n_jobs=1 to avoid GPU conflicts
    study.optimize(objective, n_trials=1000, n_jobs=-1)

    # After optimization, get gpu_id from the best trial
    best_trial_number = study.best_trial.number
    gpu_id = best_trial_number % num_gpus
    logger.info(f"Best trial number: {best_trial_number}, Best GPU ID: {gpu_id}")

    best_params = study.best_params
    best_params.update({
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'device': f'cuda:{gpu_id}',
    })

    num_boost_round = study.best_trial.user_attrs['num_boost_round']

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

def train_final_model(X, y, best_params, num_boost_round):
    dtrain = xgb.DMatrix(X, label=y)
    evals_result = {}
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train')],
        evals_result=evals_result,
        verbose_eval=False
    )
    return model, evals_result

def evaluate_model(model, X, y):
    dtrain = xgb.DMatrix(X, label=y)
    y_pred = model.predict(dtrain)
    y_pred = cp.asarray(y_pred)  # Convert y_pred to CuPy array

    # Compute metrics using CuPy
    abs_error = cp.abs(y - y_pred)
    per_instance_data = pd.DataFrame({
        'Actual': cp.asnumpy(y),
        'Predicted': cp.asnumpy(y_pred),
        'Absolute Error': cp.asnumpy(abs_error)
    })

    # Overall metrics
    rmse = cp.sqrt(cp.mean((y - y_pred) ** 2)).item()
    mae = cp.mean(abs_error).item()

    # Compute MAPE safely
    # Create a mask for non-zero actual values to avoid division by zero
    non_zero_mask = y != 0
    if cp.any(non_zero_mask):
        mape_values = cp.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask]) * 100
        # Handle any potential NaN or Inf values
        mape_values = cp.nan_to_num(mape_values, nan=0.0, posinf=0.0, neginf=0.0)
        mape = cp.mean(mape_values).item()
    else:
        mape = float('inf')  # If all actual values are zero, MAPE is undefined

    ss_tot = cp.sum((y - cp.mean(y)) ** 2)
    ss_res = cp.sum((y - y_pred) ** 2)
    r2 = (1 - ss_res / ss_tot).item()

    # Pearson correlation coefficient
    # Ensure that there are at least two samples to compute correlation
    if y.size > 1:
        pearson_corr = cp.corrcoef(y, y_pred)[0, 1].item()
    else:
        pearson_corr = float('nan')  # Not defined for single sample

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Pearson': pearson_corr,
    }

    return metrics, per_instance_data

def save_model(model, file_name, logger):
    joblib.dump(model, file_name)
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
    importances = model.get_score(importance_type='gain')
    # Map feature indices to names
    importances = {feature_names[int(k[1:])]: v for k, v in importances.items()}
    # Log as MLflow artifact
    importance_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    importance_csv = 'feature_importances.csv'
    importance_df.to_csv(importance_csv, index=False)
    mlflow.log_artifact(importance_csv)
    # Plot and log the feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.gca().invert_yaxis()  # So that the most important feature is at the top
    plt.savefig('feature_importances.png')
    mlflow.log_artifact('feature_importances.png')

def log_learning_curve(evals_result):
    plt.figure()
    epochs = len(evals_result['train']['rmse'])
    x_axis = range(0, epochs)
    plt.plot(x_axis, evals_result['train']['rmse'], label='Train')
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Boosting Round')
    plt.title('XGBoost Training Curve')
    plt.savefig('learning_curve.png')
    mlflow.log_artifact('learning_curve.png')

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
                model, evals_result = train_final_model(X, y, best_params, num_boost_round)

                # Evaluate the model
                metrics, per_instance_data = evaluate_model(model, X, y)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Save per-instance data
                per_instance_data_file = f"{os.path.splitext(csv_file)[0]}_per_instance_data.csv"
                per_instance_data.to_csv(per_instance_data_file, index=False)
                mlflow.log_artifact(per_instance_data_file)

                # Save the model as a pickle file and log as artifact
                model_file_name = f"{os.path.splitext(csv_file)[0]}_optimized_model.pkl"
                save_model(model, model_file_name, logger)
                mlflow.log_artifact(model_file_name)

                # Log the model to MLflow in standard format with signature and input example
                input_example = pd.DataFrame(cp.asnumpy(X[:5]))
                signature = infer_signature(input_example, model.predict(xgb.DMatrix(X[:5])))
                mlflow.xgboost.log_model(
                    model,
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
                log_learning_curve(evals_result)

                logger.info(
                    f"Metrics for {csv_file}: RMSE: {metrics['RMSE']:.4f}, "
                    f"MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.4f}, "
                    f"R2: {metrics['R2']:.4f}, Pearson: {metrics['Pearson']:.4f}"
                )
            logger.info(f"Finished processing for {csv_file}")

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        mlflow.log_param('Exception', str(e))

def main():
    # Handle command-line arguments
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Optimization with MLflow and Optuna')
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

if __name__ == "__main__":
    main()
