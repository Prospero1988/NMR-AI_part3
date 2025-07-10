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
import XGB_tags_config
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.model_selection import KFold  # Import KFold for cross-validation

# Suppress the MLflow integer column warning
warnings.filterwarnings("ignore", message=".*integer column.*", category=UserWarning)

# Suppress the setuptools/distutils warning
warnings.filterwarnings("ignore", message="Setuptools is replacing distutils")

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
        np.ndarray: Data as a NumPy array.
    """
    data = pd.read_csv(file_path)
    data = data.iloc[:, 1:]  # Exclude MOLECULE_NAME if present
    return data.values  # Return as NumPy array

def train_final_model(X, y, best_params, num_boost_round):
    """
    Train an XGBoost model on the provided data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        best_params (dict): Hyperparameters for XGBoost.
        num_boost_round (int): Number of boosting rounds.

    Returns:
        tuple: (trained model, evals_result dictionary)
    """
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
    """
    Evaluate the model and compute metrics.

    Args:
        model (Booster): Trained XGBoost model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        tuple: (metrics dict, per-instance DataFrame, predictions)
    """
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)
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

    return metrics, per_instance_data, y_pred

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
    Trenuje model XGBoost z 10-CV, zapisuje Q2, R2_train i pliki Williamsa.
    """
    try:
        logger = setup_logging('Training', f'training_{csv_file}.log')
        data   = load_data(os.path.join(input_directory, csv_file))
        y      = data[:, 0]
        X      = data[:, 1:]

        # ---------- load best hyperparameters --------------------
        best_params_file = f"best_params_{os.path.splitext(csv_file)[0]}.json"
        with open(best_params_file, "r") as fp:
            best_params = json.load(fp)

        num_boost_round = best_params.pop("num_boost_round", 100)
        best_params.pop("device", None)   # CPU
        best_params.pop("gpu_id", None)
        best_params["tree_method"] = "hist"

        run_name = f"Training_{csv_file}_{int(time.time())}"
        with mlflow.start_run(run_name=run_name):

            for k, v in XGB_tags_config.mlflow_tags2.items():
                mlflow.set_tag(k, v)

            # ---------- 10-fold CV ------------------------------------------
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            fold_metrics, y_pred_all = [], np.zeros_like(y)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model, _ = train_final_model(X_tr, y_tr, best_params, num_boost_round)
                m, _, y_hat = evaluate_model(model, X_val, y_val)

                y_pred_all[val_idx] = y_hat
                fold_metrics.append(m)

                mlflow.log_metrics({f"Fold{fold}_{k}": v for k, v in m.items()})

            # --- averages (Q2 is the average R2 of the folds) ------------------------
            avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
            Q2  = avg.pop("R2")   
            avg["Q2"] = Q2
            mlflow.log_metric("Q2", Q2)

            #CV plots
            generate_and_log_plots(y, y_pred_all, csv_file, suffix="_cv")

            # write fold metrics + averages
            avg_metrics_file = f"{os.path.splitext(csv_file)[0]}_cv_metrics.json"
            with open(avg_metrics_file, "w") as fp:
                json.dump({"average": avg, "per_fold": fold_metrics}, fp, indent=2)
            mlflow.log_artifact(avg_metrics_file)

            # ---------- training on whole set -----------------------------
            final_model, _ = train_final_model(X, y, best_params, num_boost_round)
            fin_metrics, _, y_pred_final = evaluate_model(final_model, X, y)

            # R2_train = R2 of the final model on the training set
            R2_train = fin_metrics["R2"]
            mlflow.log_metric("R2_train", R2_train)

            generate_and_log_plots(y, y_pred_final, csv_file, suffix="_final")

            # ---------- Williams for the final model ------------------------
            compute_and_log_williams(X, y, y_pred_final, csv_file)

            # ---------- model & artifact recording ----------------------------
            model_file = f"{os.path.splitext(csv_file)[0]}_final_model.pkl"
            joblib.dump(final_model, model_file)
            mlflow.log_artifact(model_file)

            input_example = pd.DataFrame(X[:5])
            signature     = infer_signature(input_example, y_pred_final[:5])
            mlflow.xgboost.log_model(
                final_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # metrics txt
            metrics_txt = f"{os.path.splitext(csv_file)[0]}_metrics_overview.txt"
            save_metrics_and_params(avg, best_params, metrics_txt,
                                    logger, fold_metrics, fin_metrics)
            mlflow.log_artifact(metrics_txt)

            logger.info(
                f"Averaged RMSE: {avg['RMSE']:.4f}, Q2: {Q2:.4f}, "
                f"Pearson: {avg['Pearson']:.4f}"
            )
            logger.info(
                f"Final - RMSE: {fin_metrics['RMSE']:.4f}, "
                f"R2_train: {R2_train:.4f}, Pearson: {fin_metrics['Pearson']:.4f}"
            )

        logger.info(f"Finished processing {csv_file}")

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        mlflow.log_param("Exception", str(e))
    

def main():
    """
    Main function for training with XGBoost and 10-fold Cross-Validation.
    Parses command-line arguments, sets up MLflow experiment, and processes each CSV file.
    """
    parser = argparse.ArgumentParser(description='Training with XGBoost and 10-fold Cross-Validation')
    parser.add_argument('input_directory', type=str, help='Path to the input directory containing CSV files')
    parser.add_argument('--experiment_name', type=str, default='Default', help='Name of the MLflow experiment')
    args = parser.parse_args()

    input_directory = args.input_directory
    experiment_name = args.experiment_name
    mlflow.set_experiment(experiment_name)

    csv_files = check_input_directory(input_directory)
    for csv_file in csv_files:
        process_file(csv_file, input_directory)

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

# ---------- Williams plot (leverage & standardised residuals) ----------------
def compute_and_log_williams(
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        csv_file: str
    ):
    """
    Saves two files:
    • <name>_williams_full.csv - all samples
    • <name>_williams_outliers.csv - only observations exceeding thresholds

    We upload both files to MLflow as artifacts.
    """
    residuals   = y_true - y_pred
    mse         = np.mean(residuals ** 2)
    std_resid   = residuals / np.sqrt(mse)

    X_centered  = X - X.mean(0, keepdims=True)
    X_var_mask  = X_centered.std(0) > 1e-12
    X_use       = X_centered[:, X_var_mask]
    H           = X_use @ np.linalg.pinv(X_use.T @ X_use) @ X_use.T
    leverage    = np.diag(H)

    p, n        = X_use.shape[1], X_use.shape[0]
    h_star      = 3 * (p + 1) / n         
    out_mask    = (np.abs(std_resid) > 3) | (leverage > h_star)

    base        = os.path.splitext(csv_file)[0]
    full_path   = f"{base}_williams_full.csv"
    out_path    = f"{base}_williams_outliers.csv"

    full_df = pd.DataFrame({
        "y_true"       : y_true,
        "y_pred"       : y_pred,
        "residual"     : residuals,
        "std_residual" : std_resid,
        "leverage"     : leverage,
    })
    full_df.to_csv(full_path, index=False)
    full_df[out_mask].to_csv(out_path, index=False)

    mlflow.log_artifact(full_path)
    mlflow.log_artifact(out_path)


if __name__ == "__main__":
    main()
