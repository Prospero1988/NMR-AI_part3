#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLPDual – simplified: no Optuna, training on the full dataset +
artifact <prefix>_williams.csv (MOLECULE_NAME, y_actual, y_pred).

Author: aleniak, 2025-07-02
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

# -----------------------------------------------------------------------------
# Constants / MLflow tags
# -----------------------------------------------------------------------------
MLFLOW_TAGS = {
    "property": "CHI logD",
    "model": "MLP Dual",
    "predictor": "1Hx13C"
}

SEED = 88
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
class NMRDataset(Dataset):
    """
    PyTorch Dataset for NMR data.

    Args:
        nmr_tensor (torch.Tensor): NMR data of shape (N, 2, 200).
        labels (torch.Tensor): Target values of shape (N,).
    """
    def __init__(self, nmr_tensor: torch.Tensor, labels: torch.Tensor):
        self.nmr = nmr_tensor   # (N, 2, 200)
        self.y = labels         # (N,)

    def __len__(self):
        return self.nmr.size(0)

    def __getitem__(self, idx):
        return self.nmr[idx], self.y[idx]


def load_nmr_data(path_1h: str, path_13c: str):
    """
    Merge ¹H and ¹³C files by MOLECULE_NAME (preserving replicates),
    returns: mol_names, x_nmr (N,2,200), y.

    Args:
        path_1h (str): Path to 1H CSV.
        path_13c (str): Path to 13C CSV.

    Returns:
        tuple: (molecule_names, x_nmr, y)
    """
    df_h = pd.read_csv(path_1h)
    df_c = pd.read_csv(path_13c)

    # 1. Number duplicates
    df_h["_dup_id"] = df_h.groupby("MOLECULE_NAME").cumcount()
    df_c["_dup_id"] = df_c.groupby("MOLECULE_NAME").cumcount()

    # 2. Set uniform feature headers
    df_h.columns = ["MOLECULE_NAME", "LABEL"] + \
                   [f"h_{i}" for i in range(df_h.shape[1] - 3)] + ["_dup_id"]
    df_c.columns = ["MOLECULE_NAME", "LABEL"] + \
                   [f"c_{i}" for i in range(df_c.shape[1] - 3)] + ["_dup_id"]

    # 3. Merge by name, label, and dup_id
    merged = pd.merge(
        df_h,
        df_c,
        on=["MOLECULE_NAME", "LABEL", "_dup_id"],
        how="inner",
        suffixes=("_1H", "_13C")
    )

    # 4. Build matrix (N, 2, 200)
    h_cols = [c for c in merged.columns if c.startswith("h_")]
    c_cols = [c for c in merged.columns if c.startswith("c_")]

    x_h = merged[h_cols].values.astype(np.float32)
    x_c = merged[c_cols].values.astype(np.float32)
    x_nmr = np.stack([x_h, x_c], axis=1)  # (N, 2, 200)
    y = merged["LABEL"].values.astype(np.float32)

    return merged["MOLECULE_NAME"], x_nmr, y

# -----------------------------------------------------------------------------
# Hyperparameters from CSV
# -----------------------------------------------------------------------------
def parse_hyperparams_csv(path: str) -> Dict[str, Any]:
    """
    Reads hyperparams_mlpdual.csv: columns 'param,value'.

    Args:
        path (str): Path to the CSV file.

    Returns:
        dict: Dictionary of hyperparameters.
    """
    hp = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        key, val = str(row["param"]), str(row["value"])
        if val.lower() in {"true", "false"}:
            hp[key] = val.lower() == "true"
        else:
            try:
                hp[key] = float(val) if "." in val or "e" in val.lower() else int(val)
            except ValueError:
                hp[key] = val
    return hp

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MLPDualFixed(nn.Module):
    """
    Dual-stream MLP with parameters loaded from a dictionary.

    Args:
        hp (Dict[str, Any]): Hyperparameters dictionary.
    """
    def __init__(self, hp: Dict[str, Any]):
        super().__init__()
        act = nn.SiLU()

        num_layers = hp.get("mlp_num_layers", 2)
        dropout = hp.get("mlp_dropout", 0.0)

        def build_stream(prefix: str):
            layers = []
            in_dim = 200
            for i in range(num_layers):
                out_dim = hp.get(f"{prefix}_hidden_dim_l{i}", 128)
                layers += [nn.Linear(in_dim, out_dim), act, nn.Dropout(dropout)]
                in_dim = out_dim
            return nn.Sequential(*layers), in_dim

        self.stream_h, h_out = build_stream("h")
        self.stream_c, c_out = build_stream("c")

        embed_dim = hp.get("embed_dim", 64)
        self.fc_h_embed = nn.Linear(h_out, embed_dim)
        self.fc_c_embed = nn.Linear(c_out, embed_dim)

        final_dim = hp.get("final_hidden_dim", 128)
        final_dropout = hp.get("final_dropout", 0.0)
        self.final_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, final_dim),
            act,
            nn.Dropout(final_dropout),
            nn.Linear(final_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 2, 200).

        Returns:
            torch.Tensor: Output tensor of shape (B,)
        """
        h = self.stream_h(x[:, 0, :])
        c = self.stream_c(x[:, 1, :])
        combined = torch.cat([self.fc_h_embed(h), self.fc_c_embed(c)], dim=1)
        return self.final_layers(combined).squeeze(1)

# -----------------------------------------------------------------------------
# Wrapper for MLflow logging
# -----------------------------------------------------------------------------
class WrappedModel(nn.Module):
    """
    Wrapper for PyTorch model for MLflow logging.

    Args:
        model (nn.Module): Trained PyTorch model.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        """
        Forward pass for MLflow model.

        Args:
            x (np.ndarray or torch.Tensor): Input data.

        Returns:
            torch.Tensor: Model predictions.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        with torch.no_grad():
            return self.model(x.float())

# -----------------------------------------------------------------------------
# Training + Williams artifact
# -----------------------------------------------------------------------------
def train_model(paths: Dict[str, str], hp: Dict[str, Any], prefix: str):
    """
    Train the dual-stream MLP model and create Williams CSV artifact.

    Args:
        paths (Dict[str, str]): Dictionary with keys "1h" and "13c" for CSV paths.
        hp (Dict[str, Any]): Hyperparameters.
        prefix (str): Prefix for output files.
    """
    mol_names, x_nmr, y = load_nmr_data(paths["1h"], paths["13c"])

    mlflow.set_tags({**MLFLOW_TAGS, "file": prefix})
    mlflow.log_params(hp)

    ds = NMRDataset(torch.tensor(x_nmr), torch.tensor(y))
    dl = DataLoader(ds, batch_size=hp.get("batch_size", 64), shuffle=True)

    model = MLPDualFixed(hp).to(DEVICE)
    loss_fn = nn.MSELoss()

    opt_name = hp.get("optimizer", "Adam").lower()
    lr = hp.get("lr", 1e-3)
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    epochs = hp.get("epochs", 100)
    clip_val = hp.get("clip_grad_value", 1.0)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_val)
            optimizer.step()
            losses.append(loss.item())
        mlflow.log_metric("train_loss", np.mean(losses), step=epoch)

    # -------- log model ----------
    model.cpu().float()
    wrapped = WrappedModel(model)
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "pip",
            {"pip": [f"torch=={torch.__version__}", "mlflow"]},
        ],
        "name": "mlflow-env",
    }
    mlflow.pytorch.log_model(wrapped, "model", conda_env=conda_env, input_example=x_nmr[:1])

    # -------- Williams artifact ----------
    with torch.no_grad():
        y_pred = model(torch.tensor(x_nmr)).numpy()
    df_will = pd.DataFrame(
        {"MOLECULE_NAME": mol_names, "y_actual": y, "y_pred": y_pred}
    )
    will_path = f"{prefix}_williams.csv"
    df_will.to_csv(will_path, index=False)
    mlflow.log_artifact(will_path)
    print(f"Williams CSV saved and logged: {will_path}")

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    """
    Main function for MLPDual training without Optuna.
    Parses command-line arguments, loads hyperparameters, and trains the model.
    """
    parser = argparse.ArgumentParser(description="MLPDual – training without Optuna.")
    parser.add_argument("--path_1h", required=True, help="CSV with 1H spectrum.")
    parser.add_argument("--path_13c", required=True, help="CSV with 13C spectrum.")
    parser.add_argument("--experiment_name", required=True, help="MLflow experiment name.")
    args = parser.parse_args()

    # ---------------- results directory + hyperparameters ----------------
    filename_1h = os.path.basename(args.path_1h)
    prefix = filename_1h.split("_")[0]
    res_dir = f"{prefix}-results-mlpdual"
    hyper_csv = os.path.join(res_dir, "hyperparams_mlpdual.csv")

    if not os.path.exists(hyper_csv):
        print(f"Missing {hyper_csv} – no hyperparameters found.")
        sys.exit(1)

    hp = parse_hyperparams_csv(hyper_csv)

    # ---------------- MLflow run ----------------
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"{prefix}_MLPDual_no_optuna"):
        try:
            train_model(
                paths={"1h": args.path_1h, "13c": args.path_13c},
                hp=hp,
                prefix=prefix,
            )
        except Exception as exc:
            logging.exception("Error during MLPDual training.")
            raise exc

if __name__ == "__main__":
    main()
