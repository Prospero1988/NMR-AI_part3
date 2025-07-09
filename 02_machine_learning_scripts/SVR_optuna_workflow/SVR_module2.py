# module2.py
import setuptools  # Ensure setuptools is imported first
import time
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow
import json
import subprocess
from logging.handlers import RotatingFileHandler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from mlflow.models.signature import infer_signature
import warnings
import SVR_tags_config
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.model_selection import KFold  # Import KFold for cross-validation

# Suppress the MLflow integer column warning
warnings.filterwarnings("ignore", message=".*integer column.*", category=UserWarning)

# Suppress the setuptools/distutils warning
warnings.filterwarnings("ignore", message="Setuptools is replacing distutils")

# Initialize basic logging
logging.basicConfig(level=logging.INFO)

def main():
    """
    Main function for SVR model training with MLflow and 10-fold cross-validation.
    Parses command-line arguments, sets up MLflow experiment, and processes each CSV file.
    """
    parser = argparse.ArgumentParser(
        description='SVR Model Training with MLflow and 10-fold Cross-Validation'
    )
    parser.add_argument(
        'input_directory',
        type=str,
        help='Path to the input directory containing CSV files'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='Default',
        help='Name of the MLflow experiment'
    )
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
    """
    Set up logging to both file and console.

    Args:
        logger_name (str): Name of the logger.
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
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
        pd.DataFrame: Data as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    # Drop the non-numeric column (if any), e.g., MOLECULE_NAME
    data = data.iloc[:, 1:]
    return data  # Return as pandas DataFrame

def train_final_model(X, y, best_params):
    """
    Train an SVR model on the provided data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        best_params (dict): Hyperparameters for SVR.

    Returns:
        SVR: Trained SVR model.
    """
    model = SVR(**best_params)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """
    Evaluate the model and compute metrics.

    Args:
        model (SVR): Trained SVR model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        tuple: (metrics dict, per-instance DataFrame)
    """
    y_pred = model.predict(X)

    # Compute metrics using NumPy
    abs_error = np.abs(y - y_pred)
    per_instance_data = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred,
        'Absolute Error': abs_error
    })

    # Overall metrics
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mae_std = np.std(abs_error)
    r2 = r2_score(y, y_pred)

    # Pearson correlation coefficient
    if y.size > 1:
        pearson_corr, _ = pearsonr(y, y_pred)
    else:
        pearson_corr = float('nan')  # Not defined for single sample

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAE StDev': mae_std,
        'R2': r2,
        'Pearson': pearson_corr,
    }

    return metrics, per_instance_data

def save_model(model, file_name, logger):
    """
    Save the trained model to a file.

    Args:
        model (SVR): Trained SVR model.
        file_name (str): Path to save the model.
        logger (logging.Logger): Logger instance.
    """
    joblib.dump(model, file_name)
    logger.info(f"Model saved to {file_name}")

def save_metrics_and_params(avg_metrics, params, file_name, logger, fold_metrics_list=None, final_metrics=None):
    """
    Save metrics and hyperparameters to a text file.

    Args:
        avg_metrics (dict): Average metrics over folds.
        params (dict): Model hyperparameters.
        file_name (str): Path to save the file.
        logger (logging.Logger): Logger instance.
        fold_metrics_list (list, optional): List of per-fold metrics.
        final_metrics (dict, optional): Metrics for the final model.
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(f"Model Status: Trained new model with 10-fold Cross-Validation\n\n")
        f.write("Hyperparameters:\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\nAverage Metrics over folds:\n")
        for metric_name, metric_value in avg_metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
        if fold_metrics_list is not None:
            f.write("\nPer-Fold Metrics:\n")
            for i, metrics in enumerate(fold_metrics_list, 1):
                f.write(f"\nFold {i} Metrics:\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
        if final_metrics is not None:
            f.write("\nMetrics for Final Model Trained on Full Data:\n")
            for metric_name, metric_value in final_metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
    logger.info(f"Metrics and hyperparameters saved to {file_name}")

def process_file(csv_file, input_directory):
    """
    Process a single CSV file: load data, train and evaluate model, log results.

    Args:
        csv_file (str): Name of the CSV file.
        input_directory (str): Path to the input directory.
    """
    try:
        data = load_data(os.path.join(input_directory, csv_file))

        # Ensure data is a pandas DataFrame with proper columns
        # Assuming the target variable is in the first column after MOLECULE_NAME
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values

        # Load best hyperparameters from JSON file
        best_params_file = f"best_hyperparameters_{os.path.splitext(csv_file)[0]}.json"
        if not os.path.exists(best_params_file):
            logging.error(f"Best hyperparameters file not found: {best_params_file}")
            return

        with open(best_params_file, 'r') as f:
            best_params = json.load(f)

        # Adjust hyperparameters for scikit-learn's SVR
        best_params = adjust_hyperparameters(best_params)

        # Start MLflow run
        with mlflow.start_run(run_name=f"Training_{csv_file}_{int(time.time())}"):
            # Set tags from the external file
            for tag_name, tag_value in SVR_tags_config.mlflow_tags2.items():
                mlflow.set_tag(tag_name, tag_value)

            # Log environment
            log_environment()

            # Initialize logger
            logger = setup_logging('Training', f'training_{csv_file}.log')
            logger.info(f"Starting training for {csv_file}")

            # Log best hyperparameters
            mlflow.log_params(best_params)

            # Set up KFold cross-validation
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            fold_metrics_list = []
            per_instance_data_list = []
            y_pred_all = np.zeros_like(y)
            fold = 1

            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Train the model on the training set
                model = train_final_model(X_train, y_train, best_params)

                # Evaluate the model on the validation set
                metrics, per_instance_data = evaluate_model(model, X_val, y_val)

                # Update the overall predictions
                y_pred_all[val_index] = model.predict(X_val)

                # Append per-instance data
                per_instance_data_list.append(per_instance_data)

                # Log per-fold metrics
                logger.info(f"Fold {fold} metrics: {metrics}")
                mlflow.log_metrics({f'Fold_{fold}_{k}': v for k, v in metrics.items()})
                fold_metrics_list.append(metrics)

                fold += 1

            # Compute average metrics over folds
            avg_metrics = {}
            for key in fold_metrics_list[0]:
                avg_metrics[key] = np.mean([m[key] for m in fold_metrics_list])

            # Add overall metrics
            avg_metrics['Q2'] = avg_metrics.pop('R2')
            mlflow.log_metric("Q2", avg_metrics['Q2'])

            # Combine per-instance data
            per_instance_data = pd.concat(per_instance_data_list, ignore_index=True)

            # Generate and log plots for cross-validation predictions
            generate_and_log_plots(y, y_pred_all, csv_file, suffix='_cv')

            # Log average metrics to MLflow
            mlflow.log_metrics(avg_metrics)

            # Save per-instance data
            per_instance_data_file = f"{os.path.splitext(csv_file)[0]}_per_instance_data.csv"
            per_instance_data.to_csv(per_instance_data_file, index=False)
            mlflow.log_artifact(per_instance_data_file)

            # Train final model on the full dataset
            final_model = train_final_model(X, y, best_params)

            # Evaluate final model
            final_metrics, final_per_instance_data = evaluate_model(final_model, X, y)

            # Generate and log plots for final model predictions
            y_pred_final = final_model.predict(X)
            generate_and_log_plots(y, y_pred_final, csv_file, suffix='_final')

            # Log final model metrics
            mlflow.log_metrics({f'Final_{k}': v for k, v in final_metrics.items()})

            # Save the final model trained on the full data
            model_file_name = f"{os.path.splitext(csv_file)[0]}_trained_model.pkl"
            save_model(final_model, model_file_name, logger)
            mlflow.log_artifact(model_file_name)

            # Log the model to MLflow in standard format with signature and input example
            input_example = pd.DataFrame(X[:5])
            signature = infer_signature(input_example, final_model.predict(X[:5]))

            # Convert integer columns to float64 to handle potential missing values in the future
            X_df = pd.DataFrame(X)
            X_df = X_df.astype({col: 'float64' for col in X_df.select_dtypes('int').columns})

            # Now log the model with the updated data types
            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # Save metrics and parameters
            metrics_file_name = f"{os.path.splitext(csv_file)[0]}_trained_metrics_and_params.txt"
            save_metrics_and_params(avg_metrics, best_params, metrics_file_name, logger, fold_metrics_list, final_metrics)
            mlflow.log_artifact(metrics_file_name)

            logger.info(
                f"Average Metrics for {csv_file}: RMSE: {avg_metrics['RMSE']:.4f}, "
                f"Q2: {avg_metrics['Q2']:.4f}, Pearson: {avg_metrics['Pearson']:.4f}, "
                f"MAE: {avg_metrics['MAE']:.4f}, MAE StDev: {avg_metrics['MAE StDev']:.4f}"
            )

            logger.info(
                f"Final Model Metrics for {csv_file}: RMSE: {final_metrics['RMSE']:.4f}, "
                f"R2_train: {final_metrics['R2']:.4f}, Pearson: {final_metrics['Pearson']:.4f}, "
                f"MAE: {final_metrics['MAE']:.4f}, MAE StDev: {final_metrics['MAE StDev']:.4f}"
            )

            logger.info(f"Finished training for {csv_file}")

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        # End any active runs before logging exception
        if mlflow.active_run():
            mlflow.end_run()
        # Use a unique key for the exception to avoid conflicts
        with mlflow.start_run(run_name=f"Error_{csv_file}_{int(time.time())}", nested=True):
            mlflow.log_param(f'Exception_{csv_file}', str(e))

def adjust_hyperparameters(best_params):
    """
    Adjust hyperparameters to be compatible with scikit-learn's SVR.

    Args:
        best_params (dict): Hyperparameters from optimization.

    Returns:
        dict: Adjusted hyperparameters.
    """
    # Handle 'gamma_type' and set 'gamma' accordingly
    gamma_type = best_params.pop('gamma_type', 'scale')  # Default to 'scale' if not present
    if gamma_type == 'float':
        # 'gamma' is already set in best_params
        pass
    else:
        # Set 'gamma' to 'scale' or 'auto'
        best_params['gamma'] = gamma_type
    # Remove any parameters not supported by scikit-learn's SVR
    # Check scikit-learn version
    import sklearn
    from packaging import version
    sk_version = sklearn.__version__
    if version.parse(sk_version) < version.parse('0.24'):
        # Remove 'max_iter' if scikit-learn version is less than 0.24
        best_params.pop('max_iter', None)
    # Else, 'max_iter' is supported and can be kept

    return best_params

def log_environment():
    """
    Log the current conda environment or pip requirements as an MLflow artifact.
    """
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

def generate_and_log_plots(y_true, y_pred, csv_file, suffix=''):
    """
    Generate and log plots for actual vs predicted and residuals.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted values.
        csv_file (str): Name of the CSV file.
        suffix (str): Suffix for plot filenames.
    """
    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values{suffix}')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    actual_vs_predicted_plot = f"{os.path.splitext(csv_file)[0]}_actual_vs_predicted{suffix}.png"
    plt.savefig(actual_vs_predicted_plot)
    mlflow.log_artifact(actual_vs_predicted_plot)
    plt.close()

    # Plot Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Values{suffix}')
    plt.axhline(0, color='r', linestyle='--')
    plt.tight_layout()
    residuals_plot = f"{os.path.splitext(csv_file)[0]}_residuals{suffix}.png"
    plt.savefig(residuals_plot)
    mlflow.log_artifact(residuals_plot)
    plt.close()


if __name__ == "__main__":
    main()
