#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skrypt (CNN + MLP) do regresji na danych NMR (¹H + ¹³C) i fingerprintach ECFP4.
W trakcie optymalizacji (Optuna) używa 3CV, finalnie 10CV. Logowanie w MLflow,
lokalny katalog wyników, historia triali Optuny, ważność hiperparametrów,
wykres błędu i teraz także wykres przebiegu RMSE vs ID triala.
Dodatkowo logujemy wszystkie komunikaty i ewentualne błędy do pliku w katalogu wynikowym.
"""

import argparse
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # żeby uniknąć problemów z backendem w środowisku bez GUI
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
# Tagi MLflow
# ---------------------------------------------------------------------------------
MLFLOW_TAGS = {
    "property": "CHI logD",
    "model": "hybrid 2DCNN-MLP",
    "predictor": "1Hx13C+FP"
}

# =================================================================================
# Dataset
# =================================================================================
class NMRFpDataset(Dataset):
    def __init__(self, nmr_data, fp_data, labels):
        self.nmr_data = nmr_data  # (N, 2, 1, 200)
        self.fp_data = fp_data    # (N, 2048)
        self.labels = labels      # (N,)

    def __len__(self):
        return self.nmr_data.size(0)

    def __getitem__(self, idx):
        return (
            self.nmr_data[idx],
            self.fp_data[idx],
            self.labels[idx]
        )


def load_and_align_data(path_1h, path_13c, path_fp):
    df_1h = pd.read_csv(path_1h)
    df_13c = pd.read_csv(path_13c)
    df_fp = pd.read_csv(path_fp)

    df_1h.columns = ["MOLECULE_NAME", "LABEL"] + [f"h_{i}" for i in range(df_1h.shape[1] - 2)]
    df_13c.columns = ["MOLECULE_NAME", "LABEL"] + [f"c_{i}" for i in range(df_13c.shape[1] - 2)]
    df_fp.columns = ["MOLECULE_NAME", "LABEL"] + [f"fp_{i}" for i in range(df_fp.shape[1] - 2)]

    merged_1h_13c = pd.merge(df_1h, df_13c, on=["MOLECULE_NAME", "LABEL"], how="inner")
    merged_all = pd.merge(merged_1h_13c, df_fp, on=["MOLECULE_NAME", "LABEL"], how="inner")

    h_cols = [c for c in merged_all.columns if c.startswith("h_")]
    c_cols = [c for c in merged_all.columns if c.startswith("c_")]
    fp_cols = [c for c in merged_all.columns if c.startswith("fp_")]

    x_h = merged_all[h_cols].values  # (N, 200)
    x_c = merged_all[c_cols].values  # (N, 200)
    x_fp = merged_all[fp_cols].values
    y = merged_all["LABEL"].values

    x_nmr = np.stack([x_h, x_c], axis=1)  # (N, 2, 200)
    return x_nmr, x_fp, y

# =================================================================================
# Moduły sieciowe
# =================================================================================
class CNNModule(nn.Module):
    def __init__(self, trial):
        super(CNNModule, self).__init__()
        num_conv_layers = trial.suggest_int("cnn_num_layers", 1, 6)
        kernel_size = trial.suggest_int("cnn_kernel_size", 3, 13, step=2)
        dropout_cnn = trial.suggest_float("cnn_dropout", 0.1, 0.6, step=0.1)
        batch_norm_on = trial.suggest_categorical("cnn_batch_norm", [True, False])

        conv_layers = []
        in_channels = 2
        for i in range(num_conv_layers):
            out_channels = trial.suggest_int(f"cnn_out_channels_l{i}", 8, 256, log=True)
            stride = 1
            padding = 0

            conv_layers.append(nn.Conv2d(in_channels, out_channels,
                                         kernel_size=(1, kernel_size),
                                         stride=stride, padding=(0, padding)))
            conv_layers.append(nn.SiLU(inplace=True))
            if batch_norm_on:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.Dropout2d(p=dropout_cnn))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        linear_out = trial.suggest_int("cnn_linear_out", 16, 256, log=True)
        self.fc = nn.Linear(in_channels, linear_out)
        self.out_features = linear_out

    def forward(self, x):
        x = self.conv(x)           # (B, out_channels, 1, W')
        x = torch.mean(x, dim=3)   # global avg pooling -> (B, out_channels, 1)
        x = x.squeeze(2)           # -> (B, out_channels)
        x = self.fc(x)             # -> (B, linear_out)
        return x

class MLPModule(nn.Module):
    def __init__(self, trial):
        super(MLPModule, self).__init__()
        num_layers = trial.suggest_int("mlp_num_layers", 1, 6)
        dropout_mlp = trial.suggest_float("mlp_dropout", 0.0, 0.6, step=0.1)

        layers = []
        in_dim = 2048
        for i in range(num_layers):
            out_dim = trial.suggest_int(f"mlp_hidden_dim_l{i}", 64, 2048, log=True)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_mlp))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.out_features = in_dim

    def forward(self, x):
        return self.mlp(x)

class HybridModel(nn.Module):
    def __init__(self, trial, cnn_module, mlp_module):
        super(HybridModel, self).__init__()
        self.cnn = cnn_module
        self.mlp = mlp_module

        combined_in_dim = self.cnn.out_features + self.mlp.out_features
        out_dim = trial.suggest_int("final_hidden_dim", 16, 512, log=True)
        dropout_final = trial.suggest_float("final_dropout", 0.0, 0.6, step=0.1)

        self.final_layers = nn.Sequential(
            nn.Linear(combined_in_dim, out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_final),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x_nmr, x_fp):
        cnn_out = self.cnn(x_nmr)
        mlp_out = self.mlp(x_fp)
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        out = self.final_layers(combined)
        return out.squeeze(1)

# =================================================================================
# Pomocnicze funkcje treningowe
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
    for x_nmr, x_fp, y in loader:
        x_nmr = x_nmr.to(device)
        x_fp = x_fp.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x_nmr, x_fp)
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
        for x_nmr, x_fp, y in loader:
            x_nmr = x_nmr.to(device)
            x_fp = x_fp.to(device)
            y = y.to(device)

            pred = model(x_nmr, x_fp)
            loss = loss_fn(pred, y)
            val_loss += loss.item() * len(y)

            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return val_loss / len(loader.dataset), rmse, y_true, y_pred

def create_model_and_optimizer(trial):
    cnn_module = CNNModule(trial)
    mlp_module = MLPModule(trial)
    model = HybridModel(trial, cnn_module, mlp_module)

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
# 10CV
# =================================================================================
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
# Objective (3CV)
# =================================================================================
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

# =================================================================================
# main()
# =================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_1h", required=True)
    parser.add_argument("--path_13c", required=True)
    parser.add_argument("--path_fp", required=True)
    parser.add_argument("--experiment_name", default="HybridNMR_FP_Experiment_3Fold")
    parser.add_argument("--n_trials", default=20, type=int)
    parser.add_argument("--epochs_10cv", default=50, type=int)
    args = parser.parse_args()

    set_seed(1988)
    device = get_device()

    # ------------------------------------------------------------------------------
    # Katalog wyników i logger
    # ------------------------------------------------------------------------------
    filename_1h = os.path.basename(args.path_1h)
    prefix = filename_1h.split("_")[0]
    res_dir = prefix + "-results_hybrid"
    os.makedirs(res_dir, exist_ok=True)

    # Ustawiamy logger, żeby logi poszły też do pliku
    log_file = os.path.join(res_dir, "script.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Start skryptu (hybrid). Katalog wynikowy: %s", res_dir)

    try:
        logger.info("Uruchomienie na urządzeniu: %s", device)
        X_nmr, X_fp, y = load_and_align_data(args.path_1h, args.path_13c, args.path_fp)
        X_nmr_t = torch.from_numpy(X_nmr).float().unsqueeze(2)
        X_fp_t = torch.from_numpy(X_fp).float()
        y_t = torch.from_numpy(y).float()
        dataset = NMRFpDataset(X_nmr_t, X_fp_t, y_t)

        mlflow.set_experiment(args.experiment_name)
        study = optuna.create_study(direction="minimize")
        logger.info("Rozpoczynam study.optimize (n_trials=%d)", args.n_trials)
        study.optimize(lambda tr: objective(tr, dataset, device), n_trials=args.n_trials)

        best_params = study.best_params
        best_value = study.best_value
        logger.info("Optuna zakończona. Najlepsze parametry: %s, RMSE=%f", best_params, best_value)

        with mlflow.start_run(run_name="HybridCNN_MLP_Optuna_10CV") as run:
            run_tags = dict(MLFLOW_TAGS)
            run_tags["file"] = prefix 
            mlflow.set_tags(run_tags)
            mlflow.log_params(best_params)
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("epochs_10cv", args.epochs_10cv)

            # (1) Zapis historii triali
            df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            csv_optuna = os.path.join(res_dir, "optuna_trials.csv")
            df_trials.to_csv(csv_optuna, index=False)
            mlflow.log_artifact(csv_optuna)

            # Wykres RMSE vs. Trial ID
            plot_trials_path = os.path.join(res_dir, "optuna_trials_rmse.png")
            fig = plt.figure()
            plt.plot(df_trials["number"], df_trials["value"], marker="o", linestyle="-")
            plt.xlabel("Trial ID")
            plt.ylabel("RMSE (3CV)")
            plt.title("Optuna: RMSE vs Trial ID (hybrid)")
            plt.grid(True)
            plt.savefig(plot_trials_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot_trials_path)
            plt.close(fig)

            # (2) Ważność hiperparametrów
            param_importances = get_param_importances(study)
            json_path = os.path.join(res_dir, "param_importances.json")
            with open(json_path, "w") as f:
                json.dump(param_importances, f, indent=2)
            mlflow.log_artifact(json_path)

            # Poprawka: tutaj 'fig_imp' to Axes, więc pobieramy figure:
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
                model = HybridModel(trial_stub, CNNModule(trial_stub), MLPModule(trial_stub))
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

            # <-- TU DODAŁEMY LOGOWANIE PARAMETRÓW DO MLflow
            mlflow.log_metric("rmse_mean_10cv", rmse_mean)
            mlflow.log_metric("rmse_std_10cv", rmse_std)
            mlflow.log_metric("mae_mean_10cv", mae_mean)
            mlflow.log_metric("mae_std_10cv", mae_std)
            mlflow.log_metric("r2_mean_10cv", r2_mean)
            mlflow.log_metric("r2_std_10cv", r2_std)
            mlflow.log_metric("pearson_mean_10cv", pearson_mean)
            mlflow.log_metric("pearson_std_10cv", pearson_std)

            # Zapis metrics.csv
            metrics_path = os.path.join(res_dir, "metrics.csv")
            with open(metrics_path, "w") as f:
                f.write("metric,mean,std\n")
                f.write(f"rmse,{rmse_mean},{rmse_std}\n")
                f.write(f"mae,{mae_mean},{mae_std}\n")
                f.write(f"r2,{r2_mean},{r2_std}\n")
                f.write(f"pearson,{pearson_mean},{pearson_std}\n")
            mlflow.log_artifact(metrics_path)

            # hyperparams.csv
            hyperparams_path = os.path.join(res_dir, "hyperparams.csv")
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
            cv_csv_path = os.path.join(res_dir, "cv_predictions.csv")
            df_cv.to_csv(cv_csv_path, index=False)
            mlflow.log_artifact(cv_csv_path)

            # Wykres y_true vs y_pred
            plot_path = os.path.join(res_dir, "real_vs_pred_plot.png")
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true_all, y_pred_all, alpha=0.5)
            plt.xlabel("Rzeczywiste y")
            plt.ylabel("Przewidywane y")
            plt.title("Hybrid Model: Real vs. Pred (10CV)")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(plot_path)
            plt.close()

            # Wykres błędu: X = y_true, Y = |y_true-y_pred|
            error_plot = os.path.join(res_dir, "error_plot.png")
            plt.figure(figsize=(6,6))
            abs_error = np.abs(y_true_all - y_pred_all)
            plt.scatter(y_true_all, abs_error, alpha=0.5)
            plt.xlabel("Rzeczywiste y")
            plt.ylabel("|y_true - y_pred|")
            plt.title("Błąd bezwzględny vs. y_true (10CV)")
            plt.savefig(error_plot, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(error_plot)
            plt.close()

            # Trening finalny model na CAŁYM zbiorze
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

            mlflow.pytorch.log_model(final_model, artifact_path="model")

            mlflow.end_run()

        logger.info("Skrypt zakończył się sukcesem.")

    except Exception as e:
        logger.exception("Błąd w trakcie działania skryptu (hybrid).")
        sys.exit(1)


if __name__ == "__main__":
    main()
