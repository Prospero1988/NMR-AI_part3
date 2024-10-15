# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 07:38:56 2024

@author: aleniak
"""

import setuptools
import warnings
import os
import pandas as pd
import cupy as cp
import joblib
import sys
import logging
from logging.handlers import RotatingFileHandler
import optuna
import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt
import optuna.visualization as vis
import argparse
import subprocess
from mlflow.models.signature import infer_signature
from cuml.svm import SVR
from sklearn.model_selection import KFold

# Initialize basic logging
logging.basicConfig(level=logging.INFO)

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
        'C': ('float', 1e-5, 100.0, 'log'),  # Regularization parameter. Inverse of regularization strength.
        'epsilon': ('float', 1e-5, 1.0, 'log'),  # Epsilon in the epsilon-SVR model.
 #      'kernel': ('categorical', ['linear', 'poly', 'rbf', 'sigmoid']),  # Specifies the kernel type.
        'kernel': ('categorical', ['rbf']),  # Specifies the kernel type.
        'degree': ('int', 2, 5),  # Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
        'gamma': ('categorical', ['scale', 'auto']),  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        'coef0': ('float', 0.0, 1.0),  # Independent term in kernel function. Only for 'poly' and 'sigmoid'.
        'tol': ('float', 1e-5, 1e-1, 'log'),  # Tolerance for stopping criterion.
        'max_iter': ('int', -1, 1000),  # Limit on iterations within solver, or -1 for no limit.
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
            'epsilon': trial.suggest_float('epsilon', 1e-5, 1.0, log=True),  # Epsilon in the epsilon-SVR model
            'kernel': trial.suggest_categorical('kernel', ['rbf']),  # Kernel type
            'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True),  # Tolerance for stopping criterion
            'max_iter': trial.suggest_int('max_iter', -1, 1000),  # Max iterations
        }

        # Handle 'degree' and 'coef0' only if kernel is 'poly'
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)  # Degree of polynomial kernel
            params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)  # Independent term in kernel function
        else:
            params['degree'] = 3  # Default value
            params['coef0'] = 0.0  # Default value

        # Handle 'gamma' directly
        if params['kernel'] in ['rbf', 'poly', 'sigmoid']:
            gamma_type = trial.suggest_categorical('gamma_type', ['scale', 'auto', 'float'])
            if gamma_type == 'float':
                params['gamma'] = trial.suggest_float('gamma', 1e-5, 100.0, log=True)
            else:
                params['gamma'] = gamma_type
            # Store gamma_type for logging, but do not include it in params
            trial.set_user_attr('gamma_type', gamma_type)
        else:
            params['gamma'] = 'scale'  # Default value for kernels that don't use 'gamma'
            trial.set_user_attr('gamma_type', 'scale')

        # Remove 'gamma_type' from params before passing to SVR
        params.pop('gamma_type', None)

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

        # Log numeric hyperparameters as metrics
        numeric_params = {k: v for k, v in params.items() if isinstance(v, (int, float))}
        for param_name, param_value in numeric_params.items():
            mlflow.log_metric(param_name, param_value, step=trial.number)

        # Map categorical parameters to numeric codes and log as metrics
        kernel_mapping = {'linear': 0, 'poly': 1, 'rbf': 2, 'sigmoid': 3}
        gamma_type_mapping = {'scale': 0, 'auto': 1, 'float': 2}

        kernel_numeric = kernel_mapping.get(params['kernel'], -1)
        mlflow.log_metric('kernel', kernel_numeric, step=trial.number)

        gamma_type = trial.user_attrs.get('gamma_type')
        gamma_type_numeric = gamma_type_mapping.get(gamma_type, -1)
        mlflow.log_metric('gamma_type', gamma_type_numeric, step=trial.number)

        # Collect trial data
        trial_data = {
            'trial_number': trial.number,
            'rmse': mean_rmse,
        }
        trial_data.update(params)
        trial_data['gamma_type'] = gamma_type  # Add gamma_type to trial data

        # Append trial data to the list
        optimize_hyperparameters.trial_data_list.append(trial_data)

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
        study.optimize(objective, n_trials=1000, n_jobs=-1)

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

        # Remove 'gamma_type' from best_params if present
        best_params.pop('gamma_type', None)

        # Log best hyperparameters as parameters in MLflow
        mlflow.log_params(best_params)
        mlflow.log_param('gamma_type', gamma_type)

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

        # Return the results
        return best_params, param_importances, param_importances_file, study

    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        raise  # Re-raise the exception

# The rest of the code remains unchanged

def train_final_model(X, y, best_params):
    model = SVR(**best_params)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred = cp.asarray(y_pred)  # Ensure y_pred is a CuPy array

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
    # SVR does not have feature importances
    logging.info("SVR model does not provide feature importances.")
    # Optionally, log this info in MLflow
    mlflow.log_param('feature_importances', 'Not available for SVR')

def log_learning_curve(evals_result):
    # SVR does not have evals_result or learning curve data
    logging.info("SVR model does not provide learning curve data.")
    # Optionally, log this info in MLflow
    mlflow.log_param('learning_curve', 'Not available for SVR')

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

            # Hyperparameter optimization
            logger.info(f"Starting optimization on {csv_file}")
            best_params, param_importances, param_importances_file, study = optimize_hyperparameters(
                X, y, logger, csv_file
            )
            # Log hyperparameter importances
            mlflow.log_dict(param_importances, 'param_importances.json')
            if param_importances_file:
                mlflow.log_artifact(param_importances_file)
            logger.info(f"Finished optimization on {csv_file}")

            # Training and evaluating the final model
            logger.info(f"Starting final model training on {csv_file}")
            # Log the best hyperparameters
            mlflow.log_params(best_params)

            # Train the model
            model = train_final_model(X, y, best_params)

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
            signature = infer_signature(input_example, cp.asnumpy(model.predict(X[:5])))

            class CumlModelWrapper(mlflow.pyfunc.PythonModel):

                def __init__(self, model):
                    self.model = model

                def predict(self, context, model_input):
                    import cupy as cp
                    X = cp.asarray(model_input.values)
                    y_pred = self.model.predict(X)
                    return cp.asnumpy(y_pred)

            cuml_model_wrapper = CumlModelWrapper(model)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=cuml_model_wrapper,
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
            log_learning_curve(None)

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

if __name__ == "__main__":
    main()
