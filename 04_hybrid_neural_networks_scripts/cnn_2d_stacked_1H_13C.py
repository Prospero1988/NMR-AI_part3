#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D CNN regression script for NMR data (¹H + ¹³C),
with data format:
(N, 1, 2, 200), i.e., 1 channel, height=2, width=200.

During Optuna optimization, uses 3-fold CV; final evaluation uses 10-fold CV.
Logging with MLflow, local results directory, Optuna trials history,
hyperparameter importance, error plot, and RMSE vs. trial ID plot.
"""

import argparse
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

import optuna
import mlflow
import mlflow.pytorch

from optuna.importance import get_param_importances
import optuna.visualization.matplotlib as optuna_viz

# ---------------------------------------------------------------------------------
# MLflow tags (optional)
# ---------------------------------------------------------------------------------
MLFLOW_TAGS = {
    "property": "CHI logD",
    "model": "CNN_2D_Stacked_Vectors",
    "predictor": "1H, 13C"
}

# =================================================================================
# Dataset
# =================================================================================
class NMRDataset(Dataset):
    """
    Dataset for NMR data with shape (N, 1, 2, 200).
    """
    def __init__(self, nmr_data, labels):
        self.nmr_data = nmr_data  # (N, 1, 2, 200)
        self.labels = labels      # (N,)

    def __len__(self):
        return self.nmr_data.size(0)

    def __getitem__(self, idx):
        return self.nmr_data[idx], self.labels[idx]


def load_nmr_data(path_1h, path_13c):
    """
    Loads and merges ¹H and ¹³C data into a single tensor of shape (N, 1, 2, 200).
    - "height" = 2 (row 1: ¹H, row 2: ¹³C)
    - "width" = 200
    - 1 channel (channel=1)
    """
    df_1h = pd.read_csv(path_1h)
    df_13c = pd.read_csv(path_13c)

    df_1h.columns = ["MOLECULE_NAME", "LABEL"] + [f"h_{i}" for i in range(df_1h.shape[1] - 2)]
    df_13c.columns = ["MOLECULE_NAME", "LABEL"] + [f"c_{i}" for i in range(df_13c.shape[1] - 2)]

    merged = pd.merge(df_1h, df_13c, on=["MOLECULE_NAME", "LABEL"], how="inner")

    h_cols = [c for c in merged.columns if c.startswith("h_")]
    c_cols = [c for c in merged.columns if c.startswith("c_")]

    x_h = merged[h_cols].values  # (N, 200)
    x_c = merged[c_cols].values  # (N, 200)

    # After stacking: (N, 2, 200)
    x_nmr = np.stack([x_h, x_c], axis=1)

    # Change to (N, 1, 2, 200) => 1 channel, height = 2, width = 200
    x_nmr = x_nmr[:, np.newaxis, :, :]  # add channel dimension at position 1

    y = merged["LABEL"].values
    return x_nmr, y

# =================================================================================
# CNN Model - alternative data shape
# =================================================================================
class CNN2DAlt(nn.Module):
    """
    2D CNN model for NMR data with alternative data shape.
    """
    def __init__(self, trial):
        super(CNN2DAlt, self).__init__()

        num_conv_layers = trial.suggest_int("cnn_num_layers", 1, 6)
        kernel_size_w = trial.suggest_int("cnn_kernel_size", 3, 13, step=2)
        dropout_cnn = trial.suggest_float("cnn_dropout", 0.1, 0.6, step=0.1)
        batch_norm_on = trial.suggest_categorical("cnn_batch_norm", [True, False])

        conv_layers = []
        in_channels = 1

        # First layer: kernel=(2, kernel_size_w)
        out_channels = trial.suggest_int(f"cnn_out_channels_l0", 8, 256, log=True)
        conv_layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(2, kernel_size_w),
                stride=1,
                padding=0
            )
        )
        conv_layers.append(nn.SiLU(inplace=True))
        if batch_norm_on:
            conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.Dropout2d(p=dropout_cnn))

        in_channels = out_channels

        # Remaining layers: kernel=(1, kernel_size_w)
        for i in range(1, num_conv_layers):
            out_channels = trial.suggest_int(f"cnn_out_channels_l{i}", 8, 256, log=True)

            conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, kernel_size_w),
                    stride=1,
                    padding=0
                )
            )
            conv_layers.append(nn.SiLU(inplace=True))
            if batch_norm_on:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.Dropout2d(p=dropout_cnn))

            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Linear after global pooling
        linear_out = trial.suggest_int("cnn_linear_out", 16, 256, log=True)
        self.fc_conv = nn.Linear(in_channels, linear_out)

        out_dim = trial.suggest_int("final_hidden_dim", 16, 512, log=True)
        dropout_final = trial.suggest_float("final_dropout", 0.0, 0.6, step=0.1)
        self.final_layers = nn.Sequential(
            nn.Linear(linear_out, out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_final),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x):
        # x: (B, 1, 2, W)
        x = self.conv(x)
        # After first layer: (B, out_channels, 1, W'), after each next: (B, out_channels, 1, W''')

        # Global average pooling over width
        x = torch.mean(x, dim=3)  # => (B, out_channels, 1)

        x = x.squeeze(2)  # => (B, out_channels)
        x = self.fc_conv(x)  # => (B, linear_out)
        out = self.final_layers(x)
        return out.squeeze(1)

# =================================================================================
# Utility functions
# =================================================================================
def set_seed(seed=1988):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """
    Get available device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for x_nmr, y in loader:
        x_nmr = x_nmr.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x_nmr)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(y)
    return running_loss / len(loader.dataset)

def validate(model, loader, loss_fn, device):
    """
    Validate the model.
    """
    model.eval()
    y_true = []
    y_pred = []
    val_loss = 0.0
    with torch.no_grad():
        for x_nmr, y in loader:
            x_nmr = x_nmr.to(device)
            y = y.to(device)

            pred = model(x_nmr)
            loss = loss_fn(pred, y)
            val_loss += loss.item() * len(y)

            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return val_loss / len(loader.dataset), rmse, y_true, y_pred

def create_model_and_optimizer(trial):
    """
    Create a CNN2DAlt model and optimizer based on Optuna trial parameters.
    """
    model = CNN2DAlt(trial)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSProp"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    return model, optimizer

# =================================================================================
# 10-fold Cross-Validation
# =================================================================================
def cross_validate(model_func, dataset, device, batch_size=64, n_folds=10, epochs=50):
    """
    Perform n-fold cross-validation and return metrics and predictions.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))

    y_true_all = []
    y_pred_all = []
    fold_indices_all = []

    rmse_list = []
    mae_list = []
    r2_list = []
    pearson_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model, optimizer = model_func()
        model.to(device)
        loss_fn = nn.MSELoss()

        best_val_rmse = float("inf")
        patience_counter = 0
        max_patience = 5

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_rmse, y_true_fold, y_pred_fold = validate(model, val_loader, loss_fn, device)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                break

        _, val_rmse, y_true_fold, y_pred_fold = validate(model, val_loader, loss_fn, device)

        y_true_all.append(y_true_fold)
        y_pred_all.append(y_pred_fold)
        fold_indices_all.append(np.full_like(y_true_fold, fill_value=fold_idx, dtype=np.int32))

        rmse_list.append(val_rmse)
        mae_list.append(mean_absolute_error(y_true_fold, y_pred_fold))
        r2_list.append(r2_score(y_true_fold, y_pred_fold))
        pearson_list.append(pearsonr(y_true_fold, y_pred_fold)[0])

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    fold_indices_all = np.concatenate(fold_indices_all)

    results = {
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "pearson_mean": float(np.mean(pearson_list)),
        "pearson_std": float(np.std(pearson_list)),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "fold_indices_all": fold_indices_all
    }
    return results

# =================================================================================
# Objective (3CV) - CNNOnly (alt data shape)
# =================================================================================
def objective(trial, dataset, device):
    """
    Objective function for Optuna hyperparameter optimization using 3-fold cross-validation.
    Returns the average RMSE across folds.
    """
    batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
    max_epochs = 50

    kf = KFold(n_splits=3, shuffle=True, random_state=123)
    indices = np.arange(len(dataset))

    rmse_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model, optimizer = create_model_and_optimizer(trial)
        model.to(device)
        loss_fn = nn.MSELoss()

        best_val_rmse = float("inf")
        patience_counter = 0
        max_patience = 5

        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            _, val_rmse, _, _ = validate(model, val_loader, loss_fn, device)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                break

        rmse_scores.append(best_val_rmse)

    avg_rmse = float(np.mean(rmse_scores))
    trial.report(avg_rmse, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return avg_rmse

# =================================================================================
# main()
# =================================================================================
def main():
    """
    Main function for running the 2D CNN regression pipeline on NMR data.
    Handles argument parsing, logging, data loading, Optuna optimization,
    and final evaluation with 10-fold cross-validation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_1h", required=True, help="Path to 1H NMR input CSV file.")
    parser.add_argument("--path_13c", required=True, help="Path to 13C NMR input CSV file.")
    parser.add_argument("--experiment_name", default="CNNOnly_AltData_Experiment", help="MLflow experiment name.")
    parser.add_argument("--n_trials", default=20, type=int, help="Number of Optuna trials.")
    parser.add_argument("--epochs_10cv", default=50, type=int, help="Epochs for 10-fold CV.")
    args = parser.parse_args()

    set_seed(1988)
    device = get_device()

    # --------------------------------------------------------------------------
    # Results directory and logger
    # --------------------------------------------------------------------------
    filename_1h = os.path.basename(args.path_1h)
    prefix = filename_1h.split("_")[0]
    res_dir = prefix + "-results-cnnonly-alt"
    os.makedirs(res_dir, exist_ok=True)

    log_file = os.path.join(res_dir, "script_cnnonly_alt.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Script started (CNNOnly, alt). Results directory: %s", res_dir)

    try:
        logger.info("Running on device: %s", device)
        X_nmr, y = load_nmr_data(args.path_1h, args.path_13c)
        # Now X_nmr has shape (N, 1, 2, 200)

        # Create tensors
        X_nmr_t = torch.from_numpy(X_nmr).float()  # already (N,1,2,200)
        y_t = torch.from_numpy(y).float()

        dataset = NMRDataset(X_nmr_t, y_t)

        mlflow.set_experiment(args.experiment_name)
        study = optuna.create_study(direction="minimize")
        logger.info("Starting study.optimize (n_trials=%d)", args.n_trials)
        study.optimize(lambda tr: objective(tr, dataset, device), n_trials=args.n_trials)

        best_params = study.best_params
        best_value = study.best_value
        logger.info("Optuna finished. Best parameters: %s, RMSE=%f", best_params, best_value)

        with mlflow.start_run(run_name="CNNOnly_10CV_AltData") as run:
            run_tags = dict(MLFLOW_TAGS)
            run_tags["file"] = prefix 
            mlflow.set_tags(run_tags)
            mlflow.log_params(best_params)
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("epochs_10cv", args.epochs_10cv)

            # (1) Trials history
            df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            csv_optuna = os.path.join(res_dir, "optuna_trials_cnnonly_alt.csv")
            df_trials.to_csv(csv_optuna, index=False)
            mlflow.log_artifact(csv_optuna)

            # RMSE vs Trial ID plot
            plot_trials_path = os.path.join(res_dir, "optuna_trials_rmse_cnnonly_alt.png")
            fig = plt.figure()
            plt.plot(df_trials["number"], df_trials["value"], marker="o", linestyle="-")
            plt.xlabel("Trial ID")
            plt.ylabel("RMSE (3CV)")
            plt.title("Optuna: RMSE vs Trial ID (CNNOnly, alt)")
            plt.grid(True)
            plt.savefig(plot_trials_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot_trials_path)
            plt.close(fig)

            # (2) Hyperparameter importance
            param_importances = get_param_importances(study)
            json_path = os.path.join(res_dir, "param_importances_cnnonly_alt.json")
            with open(json_path, "w") as f:
                json.dump(param_importances, f, indent=2)
            mlflow.log_artifact(json_path)

            fig_imp = optuna_viz.plot_param_importances(study)
            fig_real = fig_imp.figure
            fig_imp_path = os.path.join(res_dir, "param_importances_cnnonly_alt.png")
            fig_real.savefig(fig_imp_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(fig_imp_path)
            plt.close(fig_real)

            # (3) 10CV with best hyperparameters
            def best_model_func():
                class FrozenTrialStub:
                    def suggest_int(self, name, low, high, step=None, log=False):
                        return best_params[name]
                    def suggest_float(self, name, low, high, step=None, log=False):
                        return best_params[name]
                    def suggest_categorical(self, name, choices):
                        return best_params[name]

                trial_stub = FrozenTrialStub()
                model = CNN2DAlt(trial_stub)
                optimizer_name = best_params["optimizer"]
                lr = best_params["lr"]
                if optimizer_name == "Adam":
                    opt = optim.Adam(model.parameters(), lr=lr)
                elif optimizer_name == "SGD":
                    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                else:
                    opt = optim.RMSprop(model.parameters(), lr=lr)
                return model, opt

            logger.info("Starting 10CV with parameters: %s", best_params)
            results = cross_validate(
                model_func=best_model_func,
                dataset=dataset,
                device=device,
                batch_size=best_params["batch_size"],
                n_folds=10,
                epochs=args.epochs_10cv
            )

            rmse_mean = results["rmse_mean"]
            rmse_std = results["rmse_std"]
            mae_mean = results["mae_mean"]
            mae_std = results["mae_std"]
            r2_mean = results["r2_mean"]
            r2_std = results["r2_std"]
            pearson_mean = results["pearson_mean"]
            pearson_std = results["pearson_std"]

            y_true_all = results["y_true_all"]
            y_pred_all = results["y_pred_all"]
            logger.info("10CV finished. RMSE=%.4f±%.4f, MAE=%.4f±%.4f",
                        rmse_mean, rmse_std, mae_mean, mae_std)

            # Log metrics to MLflow
            mlflow.log_metric("rmse_mean_10cv", rmse_mean)
            mlflow.log_metric("rmse_std_10cv", rmse_std)
            mlflow.log_metric("mae_mean_10cv", mae_mean)
            mlflow.log_metric("mae_std_10cv", mae_std)
            mlflow.log_metric("r2_mean_10cv", r2_mean)
            mlflow.log_metric("r2_std_10cv", r2_std)
            mlflow.log_metric("pearson_mean_10cv", pearson_mean)
            mlflow.log_metric("pearson_std_10cv", pearson_std)

            # Save metrics
            metrics_path = os.path.join(res_dir, "metrics_cnnonly_alt.csv")
            with open(metrics_path, "w") as f:
                f.write("metric,mean,std\n")
                f.write(f"rmse,{rmse_mean},{rmse_std}\n")
                f.write(f"mae,{mae_mean},{mae_std}\n")
                f.write(f"r2,{r2_mean},{r2_std}\n")
                f.write(f"pearson,{pearson_mean},{pearson_std}\n")
            mlflow.log_artifact(metrics_path)

            # Save hyperparameters
            hyperparams_path = os.path.join(res_dir, "hyperparams_cnnonly_alt.csv")
            with open(hyperparams_path, "w") as f:
                f.write("param,value\n")
                for k, v in best_params.items():
                    f.write(f"{k},{v}\n")
            mlflow.log_artifact(hyperparams_path)

            # Save predictions
            fold_indices_all = results["fold_indices_all"]
            df_cv = pd.DataFrame({
                "fold": fold_indices_all,
                "y_true": y_true_all,
                "y_pred": y_pred_all
            })
            cv_csv_path = os.path.join(res_dir, "cv_predictions_cnnonly_alt.csv")
            df_cv.to_csv(cv_csv_path, index=False)
            mlflow.log_artifact(cv_csv_path)

            # Plot y_true vs y_pred
            plot_path = os.path.join(res_dir, "real_vs_pred_plot_cnnonly_alt.png")
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true_all, y_pred_all, alpha=0.5)
            plt.xlabel("True y")
            plt.ylabel("Predicted y")
            plt.title("CNNOnly (alt data): Real vs. Pred (10CV)")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot_path)
            plt.close()

            # Plot absolute error
            error_plot = os.path.join(res_dir, "error_plot_cnnonly_alt.png")
            plt.figure(figsize=(6, 6))
            abs_error = np.abs(y_true_all - y_pred_all)
            plt.scatter(y_true_all, abs_error, alpha=0.5)
            plt.xlabel("True y")
            plt.ylabel("|y_true - y_pred|")
            plt.title("Absolute error vs. y_true (10CV) - CNNOnly alt")
            plt.savefig(error_plot, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(error_plot)
            plt.close()

            # (4) Final model training on the whole dataset
            final_model, final_optimizer = best_model_func()
            final_model.to(device)
            loss_fn = nn.MSELoss()
            full_loader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)

            best_rmse_full = float("inf")
            patience_counter = 0
            max_patience = 5
            for epoch in range(args.epochs_10cv):
                train_loss = train_one_epoch(final_model, full_loader, final_optimizer, loss_fn, device)
                _, full_rmse, _, _ = validate(final_model, full_loader, loss_fn, device)
                if full_rmse < best_rmse_full:
                    best_rmse_full = full_rmse
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= max_patience:
                    break

            mlflow.pytorch.log_model(final_model, artifact_path="model_cnnonly_alt")

            mlflow.end_run()

        logger.info("Script (CNNOnly, alt) finished successfully.")

    except Exception as e:
        logger.exception("Error during script execution (CNNOnly, alt).")
        sys.exit(1)


if __name__ == "__main__":
    main()