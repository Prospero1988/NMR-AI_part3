import setuptools  # Ensure setuptools is imported first
import warnings
import os
import pandas as pd
import xgboost as xgb
import numpy as np
import sys
import logging
from logging.handlers import RotatingFileHandler
import mlflow
import mlflow.xgboost
import argparse
import time
import json
import joblib
from mlflow.models.signature import infer_signature
import warnings

# Suppress the MLflow integer column warning
warnings.filterwarnings("ignore", message=".*integer column.*", category=UserWarning)

# Suppress the setuptools/distutils warning
warnings.filterwarnings("ignore", message="Setuptools is replacing distutils")

def setup_logging(logger_name, log_file):
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
    data = data.iloc[:, 1:]  # Exclude MOLECULE_NAME if present
    return data.values  # Return as NumPy array

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
    dtrain = xgb.DMatrix(X)
    y_pred = model.predict(dtrain)
    abs_error = np.abs(y - y_pred)
    per_instance_data = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred,
        'Absolute Error': abs_error
    })

    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(abs_error)
    mae_std = np.std(abs_error)

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot

    if y.size > 1:
        pearson_corr = np.corrcoef(y, y_pred)[0, 1]
    else:
        pearson_corr = float('nan')

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAE StDev': mae_std,
        'R2': r2,
        'Pearson': pearson_corr,
    }

    return metrics, per_instance_data

def save_metrics_and_params(metrics, params, file_name, logger):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(f"Model Status: Trained new model\n\n")
        f.write("Hyperparameters:\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\nMetrics:\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"MAE: {metrics['MAE']:.4f}\n")
        f.write(f"MAE Standard Deviation: {metrics['MAE StDev']:.4f}\n")
        f.write(f"R2: {metrics['R2']:.4f}\n")
        f.write(f"Pearson Correlation Coefficient: {metrics['Pearson']:.4f}\n")
    logger.info(f"Metrics and hyperparameters saved to {file_name}")

def process_file(csv_file, input_directory):
    try:
        logger = setup_logging('Training', f'training_{csv_file}.log')
        data = load_data(os.path.join(input_directory, csv_file))
        y = data[:, 0]
        X = data[:, 1:]

        # Read best hyperparameters
        best_params_file = f"best_params_{os.path.splitext(csv_file)[0]}.json"
        with open(best_params_file, 'r') as f:
            best_params = json.load(f)

        num_boost_round = best_params.pop('num_boost_round', 100)
        # Adjust parameters for CPU training
        best_params.pop('device', None)
        best_params.pop('gpu_id', None)
        best_params['tree_method'] = 'hist'  # Use CPU hist method

        run_name = f"Training_{csv_file}_{int(time.time())}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('stage', 'training')
            model, evals_result = train_final_model(X, y, best_params, num_boost_round)
            metrics, per_instance_data = evaluate_model(model, X, y)

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # Save per-instance data
            per_instance_data_file = f"{os.path.splitext(csv_file)[0]}_per_instance_data.csv"
            per_instance_data.to_csv(per_instance_data_file, index=False)
            mlflow.log_artifact(per_instance_data_file)

            # Save model
            model_file_name = f"{os.path.splitext(csv_file)[0]}_trained_model.pkl"
            joblib.dump(model, model_file_name)
            mlflow.log_artifact(model_file_name)

            # Save the model to MLflow
            input_example = pd.DataFrame(X[:5])
            signature = infer_signature(input_example, model.predict(xgb.DMatrix(X[:5])))
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # Save metrics and parameters to file
            metrics_file_name = f"{os.path.splitext(csv_file)[0]}_trained_metrics_and_params.txt"
            save_metrics_and_params(metrics, best_params, metrics_file_name, logger)
            mlflow.log_artifact(metrics_file_name)

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        mlflow.log_param('Exception', str(e))

def main():
    parser = argparse.ArgumentParser(description='Training with XGBoost')
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
