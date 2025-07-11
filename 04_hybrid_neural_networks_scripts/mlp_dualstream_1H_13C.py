#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP Dual-Stream regression script for NMR data (¹H + ¹³C).
Uses 3-fold CV during Optuna optimization, and 10-fold CV for final evaluation.
Logs to MLflow, saves local results, Optuna trial history, hyperparameter importance,
error plots, and RMSE vs. trial ID plots.
All messages and errors are logged to a file in the results directory.
"""

import argparse
import os
import sys
import json
import logging
import math
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

from pathlib import Path


# ---------------------------------------------------------------------------------
# MLflow tags (optional)
# ---------------------------------------------------------------------------------
MLFLOW_TAGS = {
    "property": "CHI logD",
    "model": "MLP Dual",
    "predictor": "1Hx13C"
}

# =================================================================================
# Dataset & I/O
# =================================================================================
class NMRDataset(Dataset):
    """
    Dataset for NMR data with shape (N, 2, 200).
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def load_nmr_data(path_1h: str, path_13c: str):
    """
    Loads and merges ¹H and ¹³C data into a single tensor of shape (N, 2, 200).
    Returns: mol_names (pd.Series), x_nmr (N,2,200), y (N,)
    """
    df_h = pd.read_csv(path_1h)
    df_c = pd.read_csv(path_13c)
    # Preserve duplicates using cumcount
    df_h["_dup"] = df_h.groupby("MOLECULE_NAME").cumcount()
    df_c["_dup"] = df_c.groupby("MOLECULE_NAME").cumcount()
    df_h.columns = ["MOLECULE_NAME", "LABEL"] + [f"h_{i}" for i in range(df_h.shape[1] - 3)] + ["_dup"]
    df_c.columns = ["MOLECULE_NAME", "LABEL"] + [f"c_{i}" for i in range(df_c.shape[1] - 3)] + ["_dup"]
    merged = pd.merge(df_h, df_c, on=["MOLECULE_NAME", "LABEL", "_dup"], how="inner")
    h_cols = [c for c in merged.columns if c.startswith("h_")]
    c_cols = [c for c in merged.columns if c.startswith("c_")]
    x_h = merged[h_cols].values.astype(np.float32)
    x_c = merged[c_cols].values.astype(np.float32)
    x_nmr = np.stack([x_h, x_c], axis=1)  # (N,2,200)
    y = merged["LABEL"].values
    mol_names = merged["MOLECULE_NAME"]
    return mol_names, x_nmr, y

# =================================================================================
# Cross-Attention block
# =================================================================================
class CrossAttentionBlock(nn.Module):
    """
    Multihead cross-attention block for MLP streams.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, q, kv):
        # q, kv: (B,1,D)
        out, _ = self.attn(q, kv, kv)
        return out.squeeze(1)  # -> (B,D)

# =================================================================================
# MLP Dual-Stream (+ optional Cross-Attention)
# =================================================================================
class MLPDual(nn.Module):
    """
    Dual-stream MLP for NMR data (¹H and ¹³C) with optional cross-attention.
    """
    def __init__(self, trial: optuna.Trial, in_dim: int):
        super().__init__()

        assert in_dim is not None and in_dim > 0, "`in_dim` cannot be None/0"

        act = nn.SiLU()
        num_layers = trial.suggest_int("mlp_num_layers", 1, 6)
        dropout = trial.suggest_float("mlp_dropout", 0.0, 0.6, step=0.1)
        embed_dim = trial.suggest_int("embed_dim", 16, 512, log=True)
        self.use_ca = trial.suggest_categorical("use_cross_attention", [True, False])
        num_heads = trial.suggest_categorical("ca_heads", [1, 2, 4, 8])

        # Ensure embed_dim is divisible by num_heads if using cross-attention
        if self.use_ca and embed_dim % num_heads != 0:
            embed_dim = math.ceil(embed_dim / num_heads) * num_heads

        def make_stream(prefix: str, in_dim: int):
            layers = []
            for i in range(num_layers):
                out_dim = trial.suggest_int(f"{prefix}_h_l{i}", 32, 1024, log=True)
                layers += [nn.Linear(in_dim, out_dim), act, nn.Dropout(dropout)]
                in_dim = out_dim
            layers += [nn.Linear(in_dim, embed_dim), act]
            return nn.Sequential(*layers)

        self.stream_h = make_stream("h", in_dim)
        self.stream_c = make_stream("c", in_dim)

        if self.use_ca:
            self.ca_h_on_c = CrossAttentionBlock(embed_dim, num_heads)
            self.ca_c_on_h = CrossAttentionBlock(embed_dim, num_heads)

        final_dim = trial.suggest_int("final_hidden_dim", 32, 512, log=True)
        final_drop = trial.suggest_float("final_dropout", 0.0, 0.6, step=0.1)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, final_dim), act,
            nn.Dropout(final_drop),
            nn.Linear(final_dim, 1)
        )

    def forward(self, x):
        # x: (B,2,200)
        h = self.stream_h(x[:, 0, :])  # (B,D)
        c = self.stream_c(x[:, 1, :])  # (B,D)
        if self.use_ca:
            h = self.ca_h_on_c(h.unsqueeze(1), c.unsqueeze(1))  # (B,D)
            c = self.ca_c_on_h(c.unsqueeze(1), h.unsqueeze(1))  # (B,D)
        merged = torch.cat([h, c], dim=1)
        return self.head(merged).squeeze(1)

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
    mae_list  = []
    q2_list = []
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
        q2_list.append(r2_score(y_true_fold, y_pred_fold))
        pearson_list.append(pearsonr(y_true_fold, y_pred_fold)[0])

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    fold_indices_all = np.concatenate(fold_indices_all)

    results = {
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std":  float(np.std(rmse_list)),
        "mae_mean":  float(np.mean(mae_list)),
        "mae_std":   float(np.std(mae_list)),
        "q2_mean":   float(np.mean(q2_list)),
        "q2_std":    float(np.std(q2_list)),
        "pearson_mean": float(np.mean(pearson_list)),
        "pearson_std": float(np.std(pearson_list)),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "fold_indices_all": fold_indices_all
    }
    return results

# =================================================================================
# Objective (3CV) - MLPDual
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

        model, optimizer = create_model_and_optimizer(trial, in_dim=dataset.X.shape[-1])
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
# Model and optimizer factory for MLPDual
# =================================================================================
def create_model_and_optimizer(trial, in_dim):
    """
    Create an MLPDual model and optimizer based on Optuna trial parameters.
    """
    model = MLPDual(trial, in_dim=in_dim)
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
# Main
# =================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_1h", required=True)
    parser.add_argument("--path_13c", required=True)
    parser.add_argument("--experiment_name", default="MLPDual_Experiment")
    parser.add_argument("--n_trials", default=20, type=int)
    parser.add_argument("--epochs_10cv", default=50, type=int)
    args = parser.parse_args()

    set_seed(1988)
    device = get_device()

    # ------------------------------------------------------------------------------
    # Results directory and logger
    # ------------------------------------------------------------------------------
    filename_1h = os.path.basename(args.path_1h)
    prefix = filename_1h.split("_")[0]
    res_dir = prefix + "-results-mlpdual"
    os.makedirs(res_dir, exist_ok=True)

    print(f"Starting experiment {prefix}")

    log_file = os.path.join(res_dir, "script_mlpdual.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Script started (MLPDual). Results directory: %s", res_dir)

    try:
        logger.info("Running on device: %s", device)
        mol_names, X_nmr, y = load_nmr_data(args.path_1h, args.path_13c)
        X_nmr_t = torch.from_numpy(X_nmr).float()
        y_t = torch.from_numpy(y).float()

        dataset = NMRDataset(X_nmr_t, y_t)

        mlflow.set_experiment(args.experiment_name)
        study = optuna.create_study(direction="minimize")
        logger.info("Starting study.optimize (n_trials=%d)", args.n_trials)
        study.optimize(lambda tr: objective(tr, dataset, device), n_trials=args.n_trials)

        best_params = study.best_params
        best_value = study.best_value
        logger.info("Optuna finished. Best parameters: %s, RMSE=%f", best_params, best_value)

        with mlflow.start_run(run_name="MLPDual_10CV") as run:
            run_tags = dict(MLFLOW_TAGS)
            run_tags["file"] = prefix
            mlflow.set_tags(run_tags)
            mlflow.log_params(best_params)
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("epochs_10cv", args.epochs_10cv)

            # (1) Trials history
            df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            csv_optuna = os.path.join(res_dir, "optuna_trials_mlpdual.csv")
            df_trials.to_csv(csv_optuna, index=False)
            mlflow.log_artifact(csv_optuna)

            # RMSE vs Trial ID plot
            plot_trials_path = os.path.join(res_dir, "optuna_trials_rmse_mlpdual.png")
            fig = plt.figure()
            plt.plot(df_trials["number"], df_trials["value"], marker="o", linestyle="-")
            plt.xlabel("Trial ID")
            plt.ylabel("RMSE (3CV)")
            plt.title("Optuna: RMSE vs Trial ID (MLPDual)")
            plt.grid(True)
            plt.savefig(plot_trials_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot_trials_path)
            plt.close(fig)

            # (2) Hyperparameter importance
            param_importances = get_param_importances(study)
            json_path = os.path.join(res_dir, "param_importances_mlpdual.json")
            with open(json_path, "w") as f:
                json.dump(param_importances, f, indent=2)
            mlflow.log_artifact(json_path)

            fig_imp = optuna_viz.plot_param_importances(study)
            fig_real = fig_imp.figure
            fig_imp_path = os.path.join(res_dir, "param_importances.png")
            fig_real.savefig(fig_imp_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(fig_imp_path)
            plt.close(fig_real)

            # (3) 10CV
            def best_model_func():
                class FrozenTrialStub:
                    def suggest_int(self, name, low, high, step=None, log=False):
                        return best_params[name]
                    def suggest_float(self, name, low, high, step=None, log=False):
                        return best_params[name]
                    def suggest_categorical(self, name, choices):
                        return best_params[name]

                trial_stub = FrozenTrialStub()
                model = MLPDual(trial_stub, in_dim=dataset.X.shape[-1])
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
            q2_mean = results["q2_mean"]
            q2_std  = results["q2_std"]
            pearson_mean = results["pearson_mean"]
            pearson_std = results["pearson_std"]

            y_true_all = results["y_true_all"]
            y_pred_all = results["y_pred_all"]
            logger.info("10CV finished. RMSE=%.4f±%.4f, MAE=%.4f±%.4f",
                        rmse_mean, rmse_std, mae_mean, mae_std)

            mlflow.log_metric("rmse_mean_10cv", rmse_mean)
            mlflow.log_metric("rmse_std_10cv", rmse_std)
            mlflow.log_metric("mae_mean_10cv", mae_mean)
            mlflow.log_metric("mae_std_10cv", mae_std)
            mlflow.log_metric("q2_mean_10cv", q2_mean)
            mlflow.log_metric("q2_std_10cv",  q2_std)
            mlflow.log_metric("pearson_mean_10cv", pearson_mean)
            mlflow.log_metric("pearson_std_10cv", pearson_std)

            # Save metrics
            metrics_path = os.path.join(res_dir, "metrics_mlpdual.csv")
            with open(metrics_path, "w") as f:
                f.write("metric,mean,std\n")
                f.write(f"rmse,{rmse_mean},{rmse_std}\n")
                f.write(f"mae,{mae_mean},{mae_std}\n")
                f.write(f"q2,{q2_mean},{q2_std}\n")
                f.write(f"pearson,{pearson_mean},{pearson_std}\n")
            mlflow.log_artifact(metrics_path)

            # Save hyperparameters
            hyperparams_path = os.path.join(res_dir, "hyperparams_mlpdual.csv")
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
            cv_csv_path = os.path.join(res_dir, "cv_predictions_mlpdual.csv")
            df_cv.to_csv(cv_csv_path, index=False)
            mlflow.log_artifact(cv_csv_path)

            # Plot y_true vs y_pred
            plot_path = os.path.join(res_dir, "real_vs_pred_plot_mlpdual.png")
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true_all, y_pred_all, alpha=0.5)
            plt.xlabel("True y")
            plt.ylabel("Predicted y")
            plt.title("MLPDual: Real vs. Pred (10CV)")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot_path)
            plt.close()

            # Plot absolute error
            error_plot = os.path.join(res_dir, "error_plot_mlpdual.png")
            plt.figure(figsize=(6, 6))
            abs_error = np.abs(y_true_all - y_pred_all)
            plt.scatter(y_true_all, abs_error, alpha=0.5)
            plt.xlabel("True y")
            plt.ylabel("|y_true - y_pred|")
            plt.title("Absolute error vs. y_true (10CV) - MLPDual")
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

            mlflow.pytorch.log_model(final_model, artifact_path="model_mlpdual")

            # ---------------- Williams plot CSVs ----------------
            final_model.eval()
            with torch.no_grad():
                y_pred = final_model(X_nmr_t.to(device)).cpu().numpy()

            residuals = y - y_pred
            std_resid = (residuals - residuals.mean()) / residuals.std()
            out_thr = 3

            X_feat = X_nmr.reshape(X_nmr.shape[0], -1)  # (N,400)
            nz_mask = X_feat.std(0) > 1e-10
            X_std = (X_feat[:, nz_mask] - X_feat[:, nz_mask].mean(0)) / X_feat[:, nz_mask].std(0)

            H = X_std @ np.linalg.pinv(X_std.T @ X_std) @ X_std.T
            leverage = np.diag(H)
            lev_thr = 3 * X_std.shape[1] / X_std.shape[0]

            williams_df = pd.DataFrame({
                "MOLECULE_NAME": mol_names,
                "y_true": y,
                "y_pred": y_pred,
                "residual": residuals,
                "std_residual": std_resid,
                "leverage": leverage
            })

            full_csv = os.path.join(res_dir, f"{prefix}_williams_full.csv")
            williams_df.to_csv(full_csv, index=False)
            mlflow.log_artifact(full_csv)

            outliers_df = williams_df[(np.abs(std_resid) > out_thr) | (leverage > lev_thr)]
            out_csv = os.path.join(res_dir, f"{prefix}_williams_outliers.csv")
            outliers_df.to_csv(out_csv, index=False)
            mlflow.log_artifact(out_csv)

            # -------- Metrics on full training set --------
            rmse_train = float(np.sqrt(mean_squared_error(y, y_pred)))
            r2_train   = float(r2_score(y, y_pred))
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("r2_train",   r2_train)

            with open(os.path.join(res_dir, "metrics_mlpdual_final.csv"), "w") as f:
                f.write("metric,value\n")
                f.write(f"rmse_train,{rmse_train}\n")
                f.write(f"r2_train,{r2_train}\n")
            mlflow.log_artifact(os.path.join(res_dir, "metrics_mlpdual_final.csv"))

            mlflow.end_run()

        logger.info("Script (MLPDual) finished successfully.")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception("Error during script execution (MLPDual).")
        sys.exit(1)

if __name__ == "__main__":
    main()