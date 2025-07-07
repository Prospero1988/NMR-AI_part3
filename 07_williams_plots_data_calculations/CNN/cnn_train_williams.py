# -*- coding: utf-8 -*-
"""
Simplified script to train 1-D CNN models using hyper-parameters from *_summary.txt
and to create an additional artifact: <input>_williams.csv with columns
MOLECULE_NAME, y_actual, y_pred.

Usage
-----
python cnn_train_williams.py --csv_path ./data_dir --experiment_name MyExp
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import mlflow
import mlflow.pytorch
import importlib.util
import logging

# -------------------------------------------------------------------------
# SEED & DEVICE
# -------------------------------------------------------------------------
SEED = 88
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------------
def load_data(csv_path, target_column_name="LABEL"):
    """
    Loads data, drops the first column (molecule names) – not needed for training,
    but will be used later for the Williams file.

    Args:
        csv_path (str): Path to the CSV file.
        target_column_name (str): Name of the target column.

    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector.
    """
    data = pd.read_csv(csv_path)
    data_no_name = data.drop(data.columns[0], axis=1)  # everything except MOLECULE_NAME
    y = data_no_name[target_column_name].values
    X = data_no_name.drop(columns=[target_column_name]).values
    return X, y

# -------------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------------
class Net(nn.Module):
    """
    1D Convolutional Neural Network for regression.

    Args:
        params (dict): Dictionary of hyperparameters.
        input_dim (int): Number of input features.
    """
    def __init__(self, params, input_dim):
        super().__init__()

        # ---------- Activation ----------
        act_name = params.get("activation", "relu").lower()
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "selu": nn.SELU(),
        }
        activation = activations.get(act_name, nn.ReLU())

        # ---------- Regularization ----------
        self.regularization = params.get("regularization", "none")
        self.reg_rate = params.get("reg_rate", 1e-5) if self.regularization != "none" else 0.0

        dropout_rate = params.get("dropout_rate", 0.0)
        use_bn = params.get("use_batch_norm", False)

        # ---------- Convolutional layers ----------
        num_conv = params.get("num_conv_layers", 1)
        conv_layers = []
        in_ch = 1
        input_len = input_dim
        for i in range(num_conv):
            out_ch = params.get(f"num_filters_l{i}", 16)
            k = params.get(f"kernel_size_l{i}", 3)
            s = params.get(f"stride_l{i}", 1)
            p = params.get(f"padding_l{i}", 0)
            conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
            if use_bn:
                conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(activation)
            if dropout_rate > 0.0:
                conv_layers.append(nn.Dropout(dropout_rate))
            in_ch = out_ch
            input_len = int((input_len + 2 * p - (k - 1) - 1) / s + 1)
            if input_len <= 0:
                raise ValueError("Negative/zero input length – adjust kernel/stride/padding.")
        self.conv = nn.Sequential(*conv_layers)

        # ---------- Fully connected layers ----------
        num_fc = params.get("num_fc_layers", 2)
        fc_layers = []
        in_feat = in_ch * input_len
        for i in range(num_fc):
            out_feat = params.get(f"fc_units_l{i}", 32)
            fc_layers.append(nn.Linear(in_feat, out_feat))
            if use_bn:
                fc_layers.append(nn.BatchNorm1d(out_feat))
            fc_layers.append(activation)
            if dropout_rate > 0.0:
                fc_layers.append(nn.Dropout(dropout_rate))
            in_feat = out_feat
        fc_layers.append(nn.Linear(in_feat, 1))
        self.fc = nn.Sequential(*fc_layers)

        # ---------- Weight initialization ----------
        init_method = params.get("weight_init", "xavier")
        self.apply(lambda m: self._init_weights(m, init_method))

    @staticmethod
    def _init_weights(m, method):
        """
        Initialize weights for Conv1d and Linear layers.

        Args:
            m (nn.Module): Layer to initialize.
            method (str): Initialization method.
        """
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif method == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.float().unsqueeze(1)  # (N,1,L)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

# -------------------------------------------------------------------------
# PARAM PARSER (summary.txt)
# -------------------------------------------------------------------------
def parse_params_from_summary(summary_path):
    """
    Parse hyperparameters from a summary.txt file.

    Args:
        summary_path (str): Path to the summary.txt file.

    Returns:
        dict: Dictionary of hyperparameters.
    """
    params = {}
    in_block = False
    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "Best parameters:":
                in_block = True
                continue
            if not in_block or line == "" or line.endswith(":"):
                continue
            if line == "10CV Metrics:":
                break
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
    return params

# -------------------------------------------------------------------------
# WRAPPER for mlflow.pytorch
# -------------------------------------------------------------------------
class WrappedModel(nn.Module):
    """
    Wrapper for PyTorch model for MLflow logging.

    Args:
        model (nn.Module): Trained PyTorch model.
    """
    def __init__(self, model):
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

# -------------------------------------------------------------------------
# TRAINING + WILLIAMS CSV CREATION
# -------------------------------------------------------------------------
def train_model(csv_path, params, csv_name, mlflow_tags2):
    """
    Train the CNN model and create Williams CSV artifact.

    Args:
        csv_path (str): Path to the CSV file.
        params (dict): Hyperparameters.
        csv_name (str): Name of the CSV file (without extension).
        mlflow_tags2 (dict): MLflow tags.
    """
    with mlflow.start_run(run_name=csv_name):
        mlflow.set_tags(mlflow_tags2)

        # --------------- DATA -----------------
        X_full, y_full = load_data(csv_path, target_column_name="LABEL")
        mlflow.log_param("csv_file", csv_name)
        mlflow.log_params(params)

        X_full = X_full.astype(np.float32)
        y_full = y_full.astype(np.float32)

        # --------------- MODEL ----------------
        model = Net(params, X_full.shape[1]).to(device)
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
            raise ValueError(f"Unknown optimizer: {opt_name}")

        use_sched = params.get("use_scheduler", False)
        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5) if use_sched else None
        )

        batch_size = params.get("batch_size", 32)
        epochs = params.get("epochs", 100)
        clip_val = params.get("clip_grad_value", 1.0)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_full), torch.tensor(y_full)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --------------- TRAIN ----------------
        for epoch in range(epochs):
            model.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                if xb.size(0) == 1:  # avoid BN on batch=1
                    continue
                pred = model(xb).squeeze()
                loss = criterion(pred, yb)
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

        # --------------- SAVE & LOG MODEL -----
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
        input_example = X_full[:1].astype(np.float32)
        mlflow.pytorch.log_model(wrapped, "model", conda_env=conda_env, input_example=input_example)
        print("Model logged to MLflow")

        # --------------- WILLIAMS CSV ---------
        try:
            original_df = pd.read_csv(csv_path)
            mol_names = original_df.iloc[:, 0]  # original MOLECULE_NAME

            model.eval()
            with torch.no_grad():
                y_pred = model(torch.tensor(X_full)).squeeze().numpy()

            williams_df = pd.DataFrame(
                {"MOLECULE_NAME": mol_names, "y_actual": original_df["LABEL"], "y_pred": y_pred}
            )
            williams_file = f"{csv_name}_williams.csv"
            williams_df.to_csv(williams_file, index=False)
            mlflow.log_artifact(williams_file)
            print(f"Williams CSV saved as {williams_file} and logged to MLflow")
        except Exception as e:
            print(f"Failed to create/log Williams CSV: {e}")

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    """
    Main function to train CNN models (no Optuna) and create _williams CSV artifact.
    Parses command-line arguments, loads hyperparameters, and trains models.
    """
    parser = argparse.ArgumentParser(
        description="Train CNN models (no Optuna) and create _williams CSV artifact."
    )
    parser.add_argument("--csv_path", required=True, help="Directory with CSV files.")
    parser.add_argument("--experiment_name", required=True, help="MLflow experiment name.")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    csv_dir = args.csv_path
    params_dir = os.getcwd()

    if not os.path.isdir(csv_dir):
        print(f"{csv_dir} is not a directory.")
        sys.exit(1)
    if not os.path.isdir(params_dir):
        print(f"{params_dir} is not a directory.")
        sys.exit(1)

    # ---------- MLflow tags -------------
    tags_path = os.path.join(params_dir, "tags_config_pytorch_CNN_1D.py")
    if not os.path.exists(tags_path):
        print(f"Missing tags config: {tags_path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("tags_config_pytorch_CNN_1D", tags_path)
    tags_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tags_module)
    mlflow_tags2 = tags_module.mlflow_tags2

    # ---------- Iterate over CSVs -------
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        sys.exit(1)

    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        csv_name = os.path.splitext(csv_file)[0]
        params_file = os.path.join(params_dir, f"{csv_name}_summary.txt")
        if not os.path.exists(params_file):
            print(f"No summary file for {csv_file} – skipping.")
            continue

        params = parse_params_from_summary(params_file)
        print(f"\n=== Processing {csv_file} ===")
        try:
            train_model(csv_path, params, csv_name, mlflow_tags2)
        except Exception as e:
            print(f"Error with {csv_file}: {e}")

if __name__ == "__main__":
    main()
