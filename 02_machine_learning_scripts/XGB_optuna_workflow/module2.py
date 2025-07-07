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
import tags_config
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.model_selection import KFold  # Import KFold for cross-validation

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
            
            # Set tags from the external file
            for tag_name, tag_value in tags_config.mlflow_tags2.items():
                mlflow.set_tag(tag_name, tag_value)
    
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
                model, evals_result = train_final_model(X_train, y_train, best_params, num_boost_round)

                # Evaluate the model on the validation set
                metrics, per_instance_data, y_pred = evaluate_model(model, X_val, y_val)

                # Update the overall predictions
                y_pred_all[val_index] = y_pred

                # Append per-instance data
                per_instance_data_list.append(per_instance_data)

                # Log per-fold metrics
                logger.info(f"Fold {fold} metrics: {metrics}")
                mlflow.log_metrics({f'Fold_{fold}_{k}': v for k, v in metrics.items()})
                fold_metrics_list.append(metrics)

                fold +=1

            # Compute average metrics over folds
            avg_metrics = {}
            for key in fold_metrics_list[0]:
                avg_metrics[key] = np.mean([m[key] for m in fold_metrics_list])

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
            final_model, evals_result = train_final_model(X, y, best_params, num_boost_round)

            # Evaluate final model
            final_metrics, final_per_instance_data, y_pred_final = evaluate_model(final_model, X, y)

            # Generate and log plots for final model predictions
            generate_and_log_plots(y, y_pred_final, csv_file, suffix='_final')

            # Log final model metrics
            mlflow.log_metrics({f'Final_{k}': v for k, v in final_metrics.items()})

            # Save the final model trained on the full data
            model_file_name = f"{os.path.splitext(csv_file)[0]}_trained_model.pkl"
            joblib.dump(final_model, model_file_name)
            mlflow.log_artifact(model_file_name)

            # Save the model to MLflow
            input_example = pd.DataFrame(X[:5])
            signature = infer_signature(input_example, y_pred_final[:5])
            mlflow.xgboost.log_model(
                final_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # Save metrics and parameters to file
            metrics_file_name = f"{os.path.splitext(csv_file)[0]}_trained_metrics_and_params.txt"
            save_metrics_and_params(avg_metrics, best_params, metrics_file_name, logger, fold_metrics_list, final_metrics)
            mlflow.log_artifact(metrics_file_name)

            logger.info(
                f"Average Metrics for {csv_file}: RMSE: {avg_metrics['RMSE']:.4f}, "
                f"R2: {avg_metrics['R2']:.4f}, Pearson: {avg_metrics['Pearson']:.4f}, "
                f"MAE: {avg_metrics['MAE']:.4f}, MAE StDev: {avg_metrics['MAE StDev']:.4f}"
            )

            logger.info(
                f"Final Model Metrics for {csv_file}: RMSE: {final_metrics['RMSE']:.4f}, "
                f"R2: {final_metrics['R2']:.4f}, Pearson: {final_metrics['Pearson']:.4f}, "
                f"MAE: {final_metrics['MAE']:.4f}, MAE StDev: {final_metrics['MAE StDev']:.4f}"
            )

        logger.info(f"Finished processing {csv_file}")

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        mlflow.log_param('Exception', str(e))

def main():
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
