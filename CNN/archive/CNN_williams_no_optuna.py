#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN 1D regression WITHOUT hyperparameter optimisation (Optuna removed).

---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------
python CNN_1D_pytorch_no_optuna.py CHI_026_FP_RDKit.csv

The script expects a sibling summary file produced earlier by the HP search:
    CHI_026_FP_RDKit_summary.txt
---------------------------------------------------------------------------

Created  : 02-Jul-2025
Author   : aleniak (edited for no-Optuna version)
Licence  : MIT
"""

# ------------------------- IMPORTS ----------------------------------------
import argparse
import ast
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, Tuple, List, Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# ------------------------- GLOBALS & SEED ---------------------------------
SEED = 88
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int = SEED) -> None:
    """Reproducibility helper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# ------------------------- UTILITIES --------------------------------------


def cast_value(val: str) -> Any:
    """Clever cast str->bool/int/float/str using ast.literal_eval where possible."""
    val = val.strip()
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def parse_summary_file(path: str) -> Dict[str, Any]:
    """
    Parse *_summary.txt produced by Optuna.

    Expected structure:
        Best parameters:
        batch_size: 74
        epochs: 300
        ...
        (blank line)
        10CV Metrics:
        ...

    Returns dict with params.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Summary file {path} not found.")

    params_block: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as fh:
        in_params = False
        for line in fh:
            line = line.strip()
            if not line:
                if in_params:
                    break  # End of parameter block
                continue
            if line.lower().startswith("best parameters"):
                in_params = True
                continue
            if in_params and ":" in line:
                key, val = line.split(":", maxsplit=1)
                params_block[key.strip()] = cast_value(val)
    if not params_block:
        raise ValueError(f"No hyper-parameters found in {path}")
    return params_block


def load_data(csv_path: str, target_column_name: str = "LABEL") -> Tuple[np.ndarray, np.ndarray]:
    """Load csv and split X/Y."""
    df = pd.read_csv(csv_path)
    if target_column_name not in df.columns:
        raise KeyError(f"{target_column_name} column not in csv.")
    y = df[target_column_name].values
    X = df.drop(columns=[target_column_name]).values
    return X.astype(np.float32), y.astype(np.float32)


def get_activation(name: str) -> nn.Module:
    """Map activation name to nn.Module."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'.")


def init_weights(module: nn.Module, method: str = "xavier_uniform") -> None:
    """Apply weight initialisation."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        if method == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif method == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif method == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.05)
        else:
            raise ValueError(f"Unsupported weight_init '{method}'.")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


# ------------------------- MODEL ------------------------------------------


class CNN1DRegressor(nn.Module):
    """Very flexible 1D-CNN built from hp dict."""

    def __init__(self, input_len: int, hp: Dict[str, Any]):
        super().__init__()

        self.hp = hp

        activation = get_activation(hp.get("activation", "relu"))
        use_bn = hp.get("use_batch_norm", False)
        dropout_p = hp.get("dropout_rate", 0.0)

        conv_layers: List[nn.Module] = []
        in_channels = 1  # We treat fingerprint bits/features as 'channels' == 1
        seq_len = input_len

        num_conv = hp.get("num_conv_layers", 1)

        for i in range(num_conv):
            num_filters = hp.get(f"num_filters_l{i}", 32)
            k_size = hp.get(f"kernel_size_l{i}", 5)
            stride = hp.get(f"stride_l{i}", 1)
            padding = hp.get(f"padding_l{i}", 0)

            conv_layers.append(nn.Conv1d(in_channels, num_filters, k_size, stride, padding))
            if use_bn:
                conv_layers.append(nn.BatchNorm1d(num_filters))
            conv_layers.append(activation)
            if dropout_p > 0:
                conv_layers.append(nn.Dropout(dropout_p))
            in_channels = num_filters

            # Update seq_len after conv (approximate: floor((L + 2P − K)/S) + 1)
            seq_len = int(np.floor((seq_len + 2 * padding - k_size) / stride + 1))

        self.conv = nn.Sequential(*conv_layers)

        # Flatten size after convs
        flat_len = in_channels * seq_len

        # FC layers
        num_fc = hp.get("num_fc_layers", 1)
        fc_layers: List[nn.Module] = []
        in_feats = flat_len
        for i in range(num_fc):
            out_feats = hp.get(f"fc_units_l{i}", 128)
            fc_layers.append(nn.Linear(in_feats, out_feats))
            fc_layers.append(activation)
            if dropout_p > 0:
                fc_layers.append(nn.Dropout(dropout_p))
            in_feats = out_feats
        fc_layers.append(nn.Linear(in_feats, 1))

        self.fc = nn.Sequential(*fc_layers)

        # Apply weight initialisation
        weight_init = hp.get("weight_init", "xavier_uniform")
        self.apply(lambda m: init_weights(m, weight_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (batch, features) -> reshape to (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out.squeeze(1)


# ------------------------- TRAINING LOOPS ---------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    clip_grad: float = None,
) -> float:
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def eval_model(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    preds_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            running_loss += loss.item() * xb.size(0)
            preds_all.append(preds.cpu().numpy())
            y_all.append(yb.cpu().numpy())
    preds_concat = np.concatenate(preds_all)
    y_concat = np.concatenate(y_all)
    return running_loss / len(loader.dataset), preds_concat, y_concat


# ------------------------- METRICS ----------------------------------------


def calc_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pearson}


# ------------------------- CROSS-VALIDATION -------------------------------


def evaluate_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    hp: Dict[str, Any],
    csv_name: str,
) -> Dict[str, float]:
    """10-fold CV, returns mean metrics."""
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics: List[Dict[str, float]] = []

    batch_size = hp.get("batch_size", 64)
    clip_grad = hp.get("clip_grad_value", None)
    lr = hp.get("learning_rate", 1e-3)
    epochs = hp.get("epochs", 100)
    es_patience = hp.get("early_stop_patience", 15)
    use_scheduler = hp.get("use_scheduler", False)

    criterion = nn.MSELoss()

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        valid_ds = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        model = CNN1DRegressor(input_len=X.shape[1], hp=hp).to(DEVICE)
        optimizer_name = hp.get("optimizer", "adam").lower()
        if optimizer_name == "adam":
            beta1 = hp.get("adam_beta1", 0.9)
            beta2 = hp.get("adam_beta2", 0.999)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        else:
            raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")

        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        else:
            scheduler = None

        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, clip_grad)
            val_loss, val_preds, val_true = eval_model(model, valid_loader, criterion)

            if scheduler:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_snapshot = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if es_patience and epochs_no_improve >= es_patience:
                    break

        # Restore best weights
        model.load_state_dict(best_snapshot)
        _, preds_best, true_best = eval_model(model, valid_loader, criterion)
        metrics = calc_reg_metrics(true_best, preds_best)
        fold_metrics.append(metrics)

        print(f"[{csv_name}] Fold {fold:02d} | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    # Average metrics
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    print(f"\n[{csv_name}] 10-CV MEAN  " + " | ".join(f"{k}: {v:.4f}" for k, v in mean_metrics.items()))
    return mean_metrics


# ------------------------- FINAL TRAINING ---------------------------------


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    hp: Dict[str, Any],
    csv_name: str,
    mlflow_run,
) -> None:
    """Train on full data and log to MLflow."""
    batch_size = hp.get("batch_size", 64)
    clip_grad = hp.get("clip_grad_value", None)
    lr = hp.get("learning_rate", 1e-3)
    epochs = hp.get("epochs", 100)
    es_patience = hp.get("early_stop_patience", 15)
    use_scheduler = hp.get("use_scheduler", False)

    train_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    model = CNN1DRegressor(input_len=X.shape[1], hp=hp).to(DEVICE)

    optimizer_name = hp.get("optimizer", "adam").lower()
    if optimizer_name == "adam":
        beta1 = hp.get("adam_beta1", 0.9)
        beta2 = hp.get("adam_beta2", 0.999)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")

    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    else:
        scheduler = None

    best_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, clip_grad)

        if scheduler:
            scheduler.step(train_loss)

        # No hold-out set — simply track training loss for early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            best_snapshot = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if es_patience and epochs_no_improve >= es_patience:
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{csv_name}] Epoch {epoch:03d}/{epochs} | Train-MSE: {train_loss:.4f}")

    model.load_state_dict(best_snapshot)
    mlflow.pytorch.log_model(model, artifact_path="model")
    print(f"[{csv_name}] Final model logged to MLflow.")


# ------------------------- MAIN -------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D-CNN without Optuna.")
    parser.add_argument("csv_file", type=str, help="Input CSV (features + LABEL).")
    args = parser.parse_args()

    csv_path = args.csv_file
    if not os.path.isfile(csv_path):
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    summary_path = os.path.join(
        os.path.dirname(csv_path),
        f"{csv_name}_summary.txt",
    )

    hp = parse_summary_file(summary_path)
    print(f"Loaded {len(hp)} hyper-parameters from '{summary_path}'.")

    # Log basic HP to console
    print(json.dumps(hp, indent=2))

    # ------------------------------------------------------------------ #
    #                       DATA & METRICS                               #
    # ------------------------------------------------------------------ #
    X, y = load_data(csv_path, target_column_name="LABEL")

    with mlflow.start_run(run_name=f"{csv_name}_no_optuna_{datetime.now():%Y%m%d_%H%M%S}") as run:
        # Log parameters
        mlflow.log_params(hp)
        mlflow.log_param("device", str(DEVICE))

        # -------------------- 10-fold CV ---------------------------
        cv_metrics = evaluate_with_cv(X, y, hp, csv_name)
        mlflow.log_metrics({f"CV_{k}": v for k, v in cv_metrics.items()})

        # -------------------- Final training ----------------------
        train_final_model(X, y, hp, csv_name, run)

        # -------------------- Metrics artefact --------------------
        metrics_path = f"{csv_name}_cv_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(cv_metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)

    print("\nAll done! 🚀")


# ------------------------- ENTRY POINT ------------------------------------
if __name__ == "__main__":
    main()
