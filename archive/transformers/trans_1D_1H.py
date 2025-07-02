#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Przykład: Transformer 1D dla pojedynczego kanału widma (np. tylko ¹H).
Dodatkowo: Positional Encoding.
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
import math

# ---------------------------------------------------------------------------------
# Tagi MLflow (opcjonalnie)
# ---------------------------------------------------------------------------------
MLFLOW_TAGS = {
    "property": "CHI logD",
    "model": "Transformer 1D (Single Channel + PE)",
    "predictor": "1H"
}


# =================================================================================
# Dataset
# =================================================================================
class NMRDataset(Dataset):
    """
    Dataset dla pojedynczego widma: kształt (N, 1, 1, 200).
    """
    def __init__(self, nmr_data, labels):
        self.nmr_data = nmr_data  # (N,1,1,200)
        self.labels = labels      # (N,)

    def __len__(self):
        return self.nmr_data.size(0)

    def __getitem__(self, idx):
        return self.nmr_data[idx], self.labels[idx]


def load_nmr_data_single(path_1h):
    """
    Ładuje dane TYLKO ¹H w formacie (N, 1, 1, 200):
    - Zakładamy, że plik CSV zawiera kolumny:
      [MOLECULE_NAME, LABEL, h_0, h_1, ..., h_199]
    """
    df_1h = pd.read_csv(path_1h)
    df_1h.columns = ["MOLECULE_NAME", "LABEL"] + [f"FEATURE_{i}" for i in range(df_1h.shape[1] - 2)]

    x_h = df_1h[[f"FEATURE_{i}" for i in range(200)]].values  # (N, 200)
    y = df_1h["LABEL"].values

    # Chcemy kształt (N,1,1,200): 1 kanał, "wysokość"=1, "szerokość"=200
    x_h = x_h[:, np.newaxis, np.newaxis, :]  # => (N,1,1,200)
    return x_h, y


# =================================================================================
# Positional Encoding
# =================================================================================
class PositionalEncoding(nn.Module):
    """
    Klasyczna implementacja sinusoidalnego positional encodingu.
    Zakłada, że d_model jest parzyste (lub obsłużymy odpowiednio tablice),
    ale tutaj głównie wymuszamy z zewnątrz parzyste d_model, aby uniknąć kłopotów.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # nie jest to parametr, raczej stała

    def forward(self, x):
        """
        x shape: (B, seq_len, d_model)
        Dodajemy wektor PE do x.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x



# =================================================================================
# Model Transformer 1D (1 kanał + PE)
# =================================================================================
class TransformerRegressorSingle(nn.Module):
    """
    - Wejście: (B,1,1,200) => 
               => .view(B,1,200) => .transpose(1,2) => (B,200,1)
    - Embedding (1 -> d_model)
    - Dodajemy PositionalEncoding
    - TransformerEncoder
    - Global average pooling
    - Warstwa końcowa (regresja)
    """

    def __init__(self, trial):
        super().__init__()

        # ---------------------------
        # Wymuszamy d_model parzyste i wielokrotność nhead
        # ---------------------------
        # 1) losujemy nhead
        nhead = trial.suggest_int("nhead", 1, 8)

        # 2) losujemy d_model z parzystego przedziału [16..256]
        d_model = trial.suggest_int("d_model", 16, 256, step=2)

        # 3) sprawdzamy, czy d_model % nhead == 0
        if d_model % nhead != 0:
            # przerwij tę próbę i przejdź do następnej
            raise optuna.TrialPruned()

        # Pozostałe parametry
        num_layers = trial.suggest_int("num_encoder_layers", 1, 4)
        dim_ff = trial.suggest_int("dim_feedforward", 32, 1024, log=True)
        trans_dropout = trial.suggest_float("transformer_dropout", 0.0, 0.5, step=0.1)

        # Embedding 1 -> d_model
        self.embedding = nn.Linear(1, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=trans_dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Warstwa końcowa
        final_hidden_dim = trial.suggest_int("final_hidden_dim", 16, 512, log=True)
        final_dropout = trial.suggest_float("final_dropout", 0.0, 0.5, step=0.1)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, final_hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=final_dropout),
            nn.Linear(final_hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, 1, 1, 200)
        B = x.size(0)

        # 1) usuwamy "kanał" i "wysokość=1", by uzyskać (B,1,200)
        x = x.view(B, 1, 200)

        # zamieniamy na (B,200,1)
        x = x.transpose(1, 2)  # => (B, 200, 1)

        # 2) embedding (B,200,1) -> (B,200,d_model)
        x = self.embedding(x)

        # 3) dodajemy positional encoding
        x = self.pos_encoding(x)  # (B,200,d_model)

        # 4) transformer encoder => (B,200,d_model)
        x = self.transformer_encoder(x)

        # 5) global average pooling => (B,d_model)
        x = x.mean(dim=1)

        # 6) regressor => (B,1)
        out = self.regressor(x)
        return out.squeeze(1)


# =================================================================================
# Pozostałe funkcje jak train, validate, cross_validate, objective, itp.
# =================================================================================
def set_seed(seed=1988):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, loss_fn, device):
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
    model = TransformerRegressorSingle(trial)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSProp"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    return model, optimizer

def cross_validate(model_func, dataset, device, batch_size=64, n_folds=10, epochs=50):
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

        # Po zakończeniu epok jeszcze raz metryki
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

def objective(trial, dataset, device):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_1h", required=True, help="CSV z danymi 1H do predykcji")
    parser.add_argument("--experiment_name", default="TransformerNMR_Single_1H")
    parser.add_argument("--n_trials", default=20, type=int)
    parser.add_argument("--epochs_10cv", default=50, type=int)
    args = parser.parse_args()

    set_seed(1988)
    device = get_device()

    # Katalog wyników i logger
    filename_1h = os.path.basename(args.path_1h)
    prefix = filename_1h.split("_")[0]
    res_dir = prefix + "-results-transformer-single"
    os.makedirs(res_dir, exist_ok=True)

    log_file = os.path.join(res_dir, "script_transformer_single.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Start skryptu (Transformer 1D, single channel). Katalog wynikowy: %s", res_dir)

    try:
        logger.info("Uruchomienie na urządzeniu: %s", device)

        # Wczytujemy TYLKO 1H
        X_nmr, y = load_nmr_data_single(args.path_1h)
        X_nmr_t = torch.from_numpy(X_nmr).float()  # (N,1,1,200)
        y_t = torch.from_numpy(y).float()
        dataset = NMRDataset(X_nmr_t, y_t)

        mlflow.set_experiment(args.experiment_name)
        study = optuna.create_study(direction="minimize")
        logger.info("Rozpoczynam study.optimize (n_trials=%d)", args.n_trials)
        study.optimize(lambda tr: objective(tr, dataset, device), n_trials=args.n_trials)

        best_params = study.best_params
        best_value = study.best_value
        logger.info("Optuna zakończona. Najlepsze parametry: %s, RMSE=%f", best_params, best_value)

        with mlflow.start_run(run_name="Transformer_10CV_Single") as run:
            run_tags = dict(MLFLOW_TAGS)
            run_tags["file"] = prefix
            mlflow.set_tags(run_tags)
            mlflow.log_params(best_params)
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("epochs_10cv", args.epochs_10cv)

            # (1) Historia triali
            df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            csv_optuna = os.path.join(res_dir, "optuna_trials_transformer_single.csv")
            df_trials.to_csv(csv_optuna, index=False)
            mlflow.log_artifact(csv_optuna)

            # Wykres RMSE vs Trial ID
            plot_trials_path = os.path.join(res_dir, "optuna_trials_rmse_transformer_single.png")
            fig = plt.figure()
            plt.plot(df_trials["number"], df_trials["value"], marker="o", linestyle="-")
            plt.xlabel("Trial ID")
            plt.ylabel("RMSE (3CV)")
            plt.title("Optuna: RMSE vs Trial ID (Transformer single)")
            plt.grid(True)
            plt.savefig(plot_trials_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot_trials_path)
            plt.close(fig)

            # (2) Ważność hiperparametrów
            param_importances = get_param_importances(study)
            json_path = os.path.join(res_dir, "param_importances_transformer_single.json")
            with open(json_path, "w") as f:
                json.dump(param_importances, f, indent=2)
            mlflow.log_artifact(json_path)

            fig_imp = optuna_viz.plot_param_importances(study)
            fig_real = fig_imp.figure
            fig_imp_path = os.path.join(res_dir, "param_importances_transformer_single.png")
            fig_real.savefig(fig_imp_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(fig_imp_path)
            plt.close(fig_real)

            # (3) 10CV z najlepszymi parametrami
            def best_model_func():
                class FrozenTrialStub:
                    def suggest_int(self, name, low, high, step=None, log=False):
                        return best_params[name]
                    def suggest_float(self, name, low, high, step=None, log=False):
                        return best_params[name]
                    def suggest_categorical(self, name, choices):
                        return best_params[name]

                trial_stub = FrozenTrialStub()
                model = TransformerRegressorSingle(trial_stub)
                optimizer_name = best_params["optimizer"]
                lr = best_params["lr"]
                if optimizer_name == "Adam":
                    opt = optim.Adam(model.parameters(), lr=lr)
                elif optimizer_name == "SGD":
                    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                else:
                    opt = optim.RMSprop(model.parameters(), lr=lr)
                return model, opt

            logger.info("Rozpoczynam 10CV z parametrami: %s", best_params)
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
            logger.info("10CV zakończone. RMSE=%.4f±%.4f, MAE=%.4f±%.4f",
                        rmse_mean, rmse_std, mae_mean, mae_std)

            mlflow.log_metric("rmse_mean_10cv", rmse_mean)
            mlflow.log_metric("rmse_std_10cv", rmse_std)
            mlflow.log_metric("mae_mean_10cv", mae_mean)
            mlflow.log_metric("mae_std_10cv", mae_std)
            mlflow.log_metric("r2_mean_10cv", r2_mean)
            mlflow.log_metric("r2_std_10cv", r2_std)
            mlflow.log_metric("pearson_mean_10cv", pearson_mean)
            mlflow.log_metric("pearson_std_10cv", pearson_std)

            # Zapis metrics
            metrics_path = os.path.join(res_dir, "metrics_transformer_single.csv")
            with open(metrics_path, "w") as f:
                f.write("metric,mean,std\n")
                f.write(f"rmse,{rmse_mean},{rmse_std}\n")
                f.write(f"mae,{mae_mean},{mae_std}\n")
                f.write(f"r2,{r2_mean},{r2_std}\n")
                f.write(f"pearson,{pearson_mean},{pearson_std}\n")
            mlflow.log_artifact(metrics_path)

            # hyperparams
            hyperparams_path = os.path.join(res_dir, "hyperparams_transformer_single.csv")
            with open(hyperparams_path, "w") as f:
                f.write("param,value\n")
                for k, v in best_params.items():
                    f.write(f"{k},{v}\n")
            mlflow.log_artifact(hyperparams_path)

            # csv z predykcjami
            fold_indices_all = results["fold_indices_all"]
            df_cv = pd.DataFrame({
                "fold": fold_indices_all,
                "y_true": y_true_all,
                "y_pred": y_pred_all
            })
            cv_csv_path = os.path.join(res_dir, "cv_predictions_transformer_single.csv")
            df_cv.to_csv(cv_csv_path, index=False)
            mlflow.log_artifact(cv_csv_path)

            # Wykres y_true vs y_pred
            plot_path = os.path.join(res_dir, "real_vs_pred_plot_transformer_single.png")
            plt.figure(figsize=(6,6))
            plt.scatter(y_true_all, y_pred_all, alpha=0.5)
            plt.xlabel("Rzeczywiste y")
            plt.ylabel("Przewidywane y")
            plt.title("Transformer (Single 1H + PE): Real vs. Pred (10CV)")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot_path)
            plt.close()

            # Wykres błędu
            error_plot = os.path.join(res_dir, "error_plot_transformer_single.png")
            plt.figure(figsize=(6,6))
            abs_error = np.abs(y_true_all - y_pred_all)
            plt.scatter(y_true_all, abs_error, alpha=0.5)
            plt.xlabel("Rzeczywiste y")
            plt.ylabel("|y_true - y_pred|")
            plt.title("Błąd bezwzględny vs. y_true (10CV) - Transformer Single")
            plt.savefig(error_plot, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(error_plot)
            plt.close()

            # (4) Trening finalny modelu na całym zbiorze
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

            mlflow.pytorch.log_model(final_model, artifact_path="model_transformer_single")

            mlflow.end_run()

        logger.info("Skrypt (Transformer 1D, single channel) zakończył się sukcesem.")

    except Exception as e:
        logger.exception("Błąd w trakcie działania skryptu (Transformer 1D single).")
        sys.exit(1)


if __name__ == "__main__":
    main()
