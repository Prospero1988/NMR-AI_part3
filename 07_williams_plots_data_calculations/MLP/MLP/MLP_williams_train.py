# -*- coding: utf-8 -*-
"""
MLP (no-Optuna) – trenowanie + artefakt _williams.csv

Użycie:
python mlp_train_williams.py \
       --csv_path ./data_dir \
       --experiment_name MLP_no_optuna
"""

import os
import sys
import argparse
import logging
import random
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

# -------------------------------------------------------
# SEED & DEVICE
# -------------------------------------------------------
SEED = 88
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# DATA
# -------------------------------------------------------
def load_data(csv_path: str, target_column: str = "LABEL"):
    """Zwraca X, y jako np.ndarray (float32)."""
    df = pd.read_csv(csv_path)
    df_no_name = df.drop(df.columns[0], axis=1)  # odrzucamy MOLECULE_NAME z X
    y = df_no_name[target_column].values.astype(np.float32)
    X = df_no_name.drop(columns=[target_column]).values.astype(np.float32)
    return X, y

# -------------------------------------------------------
# PARAMS PARSER
# -------------------------------------------------------
def parse_params(summary_path: str) -> Dict[str, Any]:
    """Parsuje blok 'Best parameters:' z *_summary.txt."""
    params = {}
    in_block = False
    with open(summary_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line == "Best parameters:":
                in_block = True
                continue
            if not in_block:
                continue
            if line == "" or line.endswith(":"):
                # pusty wiersz lub nagłówek następnej sekcji
                if line == "10CV Metrics:":
                    break
                continue
            key, val = line.split(":", 1)
            val = val.strip()
            if val.lower() in {"true", "false"}:
                val = val.lower() == "true"
            else:
                try:
                    val = float(val) if "." in val or "e" in val.lower() else int(val)
                except ValueError:
                    pass
            params[key.strip()] = val
    if not params:
        raise ValueError(f"No hyperparameters found in {summary_path}")
    return params

# -------------------------------------------------------
# MLP MODEL
# -------------------------------------------------------
class MLP(nn.Module):
    """
    Flexible MLP built from a hyperparameter dictionary.

    Args:
        hp (Dict[str, Any]): Hyperparameters dictionary.
        input_dim (int): Number of input features.
    """
    def __init__(self, hp: Dict[str, Any], input_dim: int):
        super().__init__()

        # ---------- activation ----------
        act_name = hp.get("activation", "relu").lower()
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "selu": nn.SELU(),
        }
        activation = activations.get(act_name, nn.ReLU())

        # ---------- regularization ----------
        self.regularization = hp.get("regularization", "none")
        self.reg_rate = hp.get("reg_rate", 1e-5) if self.regularization != "none" else 0.0

        dropout = hp.get("dropout_rate", 0.0)
        use_bn = hp.get("use_batch_norm", False)
        num_layers = hp.get("num_layers", 2)
        units = hp.get("units", 128)

        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, units))
            if use_bn:
                layers.append(nn.BatchNorm1d(units))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = units
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

        # weight init
        init_method = hp.get("weight_init", "xavier").lower()
        self.apply(lambda m: self._init_weights(m, init_method))

    @staticmethod
    def _init_weights(m: nn.Module, method: str):
        if isinstance(m, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif method == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# -------------------------------------------------------
# WRAPPER (do mlflow.pytorch)
# -------------------------------------------------------
class WrappedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        with torch.no_grad():
            return self.model(x.float())

# -------------------------------------------------------
# TRAINING + WILLIAMS CSV
# -------------------------------------------------------
def train_model(csv_path: str, params: Dict[str, Any], csv_name: str, mlflow_tags):
    with mlflow.start_run(run_name=csv_name):
        mlflow.set_tags(mlflow_tags)

        # -------- data ----------
        X, y = load_data(csv_path)
        mlflow.log_param("csv_file", csv_name)
        mlflow.log_params(params)

        # -------- model ----------
        model = MLP(params, input_dim=X.shape[1]).to(DEVICE)
        criterion = nn.MSELoss()

        opt_name = params.get("optimizer", "adam").lower()
        lr = params.get("learning_rate", 1e-3)
        if opt_name == "adam":
            beta1, beta2 = params.get("adam_beta1", 0.9), params.get("adam_beta2", 0.999)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        elif opt_name == "sgd":
            momentum = params.get("sgd_momentum", 0.0)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif opt_name == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif opt_name == "adamw":
            beta1, beta2 = params.get("adamw_beta1", 0.9), params.get("adamw_beta2", 0.999)
            optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2))
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        use_sched = params.get("use_scheduler", False)
        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
            if use_sched
            else None
        )

        batch_size = params.get("batch_size", 32)
        epochs = params.get("epochs", 100)
        clip_val = params.get("clip_grad_value", 1.0)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X), torch.tensor(y)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # -------- training loop ----------
        for epoch in range(epochs):
            model.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)

                # regularisation
                if model.regularization == "l1":
                    loss += model.reg_rate * sum(p.abs().sum() for p in model.parameters())
                elif model.regularization == "l2":
                    loss += model.reg_rate * sum(p.pow(2).sum() for p in model.parameters())

                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_val)
                optimizer.step()
                losses.append(loss.item())
            if use_sched:
                scheduler.step(np.mean(losses))
            mlflow.log_metric("train_loss", np.mean(losses), step=epoch)

        # -------- save & log model ----------
        model_file = f"{csv_name}_model.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Model saved as {model_file}")

        model.cpu().float()
        wrapped = WrappedModel(model)
        logging.getLogger("mlflow").setLevel(logging.DEBUG)
        conda_env = {
            "channels": ["defaults"],
            "dependencies": [
                f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "pip",
                {"pip": [f"torch=={torch.__version__}", "mlflow"]},
            ],
            "name": "mlflow-env",
        }
        input_example = X[:1].astype(np.float32)
        mlflow.pytorch.log_model(wrapped, "model", conda_env=conda_env, input_example=input_example)
        print("Model logged to MLflow")

        # -------- williams csv --------
        try:
            full_df = pd.read_csv(csv_path)
            mol_names = full_df.iloc[:, 0]  # MOLECULE_NAME kol. #0
            with torch.no_grad():
                y_pred = model(torch.tensor(X)).squeeze().numpy()
            williams_df = pd.DataFrame(
                {"MOLECULE_NAME": mol_names, "y_actual": full_df["LABEL"], "y_pred": y_pred}
            )
            williams_file = f"{csv_name}_williams.csv"
            williams_df.to_csv(williams_file, index=False)
            mlflow.log_artifact(williams_file)
            print(f"Williams CSV saved & logged: {williams_file}")
        except Exception as exc:
            print(f"Could not create/log Williams CSV: {exc}")

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MLP training (no Optuna) + Williams artefact."
    )
    parser.add_argument("--csv_path", required=True, help="Dir with CSV files.")
    parser.add_argument("--experiment_name", required=True, help="MLflow experiment.")
    args = parser.parse_args()

    # -------- mlflow experiment ----------
    mlflow.set_experiment(args.experiment_name)

    csv_dir = args.csv_path
    if not os.path.isdir(csv_dir):
        print(f"{csv_dir} is not a directory.")
        sys.exit(1)

    # -------- tags config ----------
    tags_cfg = os.path.join(os.getcwd(), "tags_config_pytorch.py")
    if not os.path.exists(tags_cfg):
        print(f"Missing tags config: {tags_cfg}")
        sys.exit(1)
    spec = __import__("importlib").util.spec_from_file_location("tags_config_pytorch", tags_cfg)
    tags_module = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(tags_module)
    mlflow_tags2 = tags_module.mlflow_tags2

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files in {csv_dir}")
        sys.exit(1)

    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        csv_name = os.path.splitext(csv_file)[0]
        summary_path = os.path.join(os.getcwd(), f"{csv_name}_summary.txt")
        if not os.path.exists(summary_path):
            print(f"Missing summary file for {csv_file} – skipping.")
            continue
        params = parse_params(summary_path)
        print(f"\n=== {csv_file} ===")
        try:
            train_model(csv_path, params, csv_name, mlflow_tags2)
        except Exception as err:
            print(f"Error with {csv_file}: {err}")

if __name__ == "__main__":
    main()
