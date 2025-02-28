#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skrypt trenujący hybrydową sieć neuronową (CNN + MLP) do regresji na danych NMR (¹H + ¹³C) i fingerprintach ECFP4.
Wykorzystuje:
  - Optuna do optymalizacji hiperparametrów, tym razem na 3-krotnej walidacji,
  - MLflow do logowania,
  - Early Stopping zarówno podczas optymalizacji, jak i w finalnej 10CV.
  
Przykładowe użycie:
python train_hybrid_model.py \
    --path_1h /sciezka/do/danych_1H.csv \
    --path_13c /sciezka/do/danych_13C.csv \
    --path_fp /sciezka/do/danych_fp.csv \
    --experiment_name "MojeEksperymenty" \
    --n_trials 20
"""

import argparse
import os
import numpy as np
import pandas as pd

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
import matplotlib.pyplot as plt

# =================================================================================
# DATALOADER
# =================================================================================
class NMRFpDataset(Dataset):
    """
    Dataset łączący dane NMR (2 kanały: ¹H i ¹³C) + fingerprinty ECFP4 + etykietę (float).
    """
    def __init__(self, nmr_data, fp_data, labels):
        """
        :param nmr_data: torch.Tensor o wymiarach (N, 2, 1, 200)
        :param fp_data:  torch.Tensor o wymiarach (N, 2048)
        :param labels:   torch.Tensor o wymiarach (N,)
        """
        self.nmr_data = nmr_data
        self.fp_data = fp_data
        self.labels = labels

    def __len__(self):
        return self.nmr_data.size(0)

    def __getitem__(self, idx):
        return (
            self.nmr_data[idx],  # (2, 1, 200)
            self.fp_data[idx],   # (2048,)
            self.labels[idx]     # scalar
        )

def load_and_align_data(path_1h, path_13c, path_fp):
    """
    Ładuje i łączy dane po MOLECULE_NAME. Zakładamy, że:
    - CSV ¹H ma: [MOLECULE_NAME, LABEL, feat_1, ..., feat_200]
    - CSV ¹³C ma: [MOLECULE_NAME, LABEL, feat_1, ..., feat_200]
    - CSV FP ma:  [MOLECULE_NAME, LABEL, fp_1, ..., fp_2048]
    Zwraca:
    - X_nmr: numpy array (N, 2, 200)
    - X_fp:  numpy array (N, 2048)
    - y:     numpy array (N,)
    """
    df_1h = pd.read_csv(path_1h)
    df_13c = pd.read_csv(path_13c)
    df_fp = pd.read_csv(path_fp)

    # Nadajmy jednolite nazwy kolumn
    df_1h.columns = ["MOLECULE_NAME", "LABEL"] + [f"h_{i}" for i in range(df_1h.shape[1] - 2)]
    df_13c.columns = ["MOLECULE_NAME", "LABEL"] + [f"c_{i}" for i in range(df_13c.shape[1] - 2)]
    df_fp.columns = ["MOLECULE_NAME", "LABEL"] + [f"fp_{i}" for i in range(df_fp.shape[1] - 2)]

    # Merge po MOLECULE_NAME i LABEL (wewnętrzne przecięcie)
    merged_1h_13c = pd.merge(df_1h, df_13c, on=["MOLECULE_NAME", "LABEL"], how="inner")
    merged_all = pd.merge(merged_1h_13c, df_fp, on=["MOLECULE_NAME", "LABEL"], how="inner")

    # Kolumny z cechami
    h_cols = [col for col in merged_all.columns if col.startswith("h_")]
    c_cols = [col for col in merged_all.columns if col.startswith("c_")]
    fp_cols = [col for col in merged_all.columns if col.startswith("fp_")]

    x_h = merged_all[h_cols].values  # (N, 200)
    x_c = merged_all[c_cols].values  # (N, 200)
    x_fp = merged_all[fp_cols].values  # (N, 2048)
    y = merged_all["LABEL"].values  # (N,)

    # Łączymy ¹H i ¹³C w (N, 2, 200)
    x_nmr = np.stack([x_h, x_c], axis=1)  # (N, 2, 200)

    return x_nmr, x_fp, y


# =================================================================================
# MODELE (CNNModule, MLPModule, HybridModel)
# =================================================================================
class CNNModule(nn.Module):
    """
    Konwolucyjny moduł 2D do analizy widm ¹H i ¹³C.
    Zakładamy wejście: (batch_size, 2, 1, 200).
    """
    def __init__(self, trial):
        super(CNNModule, self).__init__()

        # Hiperparametry z Optuny
        num_conv_layers = trial.suggest_int("cnn_num_layers", 1, 5)
        kernel_size = trial.suggest_int("cnn_kernel_size", 3, 9, step=2)
        dropout_cnn = trial.suggest_float("cnn_dropout", 0.1, 0.5, step=0.1)
        batch_norm_on = trial.suggest_categorical("cnn_batch_norm", [True, False])

        conv_layers = []
        in_channels = 2  # wejściowe: 2 kanały (¹H, ¹³C)
        for i in range(num_conv_layers):
            out_channels = trial.suggest_int(f"cnn_out_channels_l{i}", 8, 256, log=True)
            stride = 1
            padding = 0

            conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                             stride=stride, padding=(0, padding))
            conv_layers.append(conv)
            conv_layers.append(nn.ReLU(inplace=True))

            if batch_norm_on:
                conv_layers.append(nn.BatchNorm2d(out_channels))

            conv_layers.append(nn.Dropout2d(p=dropout_cnn))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Po konwolucjach mamy (B, out_channels, 1, W')
        # Zastosujemy global average pooling po szerokości, a potem warstwę fc
        linear_out = trial.suggest_int("cnn_linear_out", 16, 256, log=True)
        self.fc = nn.Linear(in_channels, linear_out)
        self.out_features = linear_out

    def forward(self, x):
        # x: (B, 2, 1, 200)
        x = self.conv(x)         # (B, out_channels, 1, W')
        x = torch.mean(x, dim=3) # global average pooling -> (B, out_channels, 1)
        x = x.squeeze(2)         # -> (B, out_channels)
        x = self.fc(x)           # -> (B, linear_out)
        return x


class MLPModule(nn.Module):
    """
    Moduł MLP do analizy fingerprintów ECFP4 (wejście: (batch_size, 2048)).
    """
    def __init__(self, trial):
        super(MLPModule, self).__init__()

        num_layers = trial.suggest_int("mlp_num_layers", 1, 5)
        dropout_mlp = trial.suggest_float("mlp_dropout", 0.0, 0.9, step=0.1)

        layers = []
        in_dim = 2048
        for i in range(num_layers):
            out_dim = trial.suggest_int(f"mlp_hidden_dim_l{i}", 64, 2048, log=True)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_mlp))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.out_features = in_dim

    def forward(self, x):
        return self.mlp(x)


class HybridModel(nn.Module):
    """
    Łączy CNNModule (NMR) i MLPModule (FP), a następnie przechodzi przez finalne warstwy do regresji.
    """
    def __init__(self, trial, cnn_module, mlp_module):
        super(HybridModel, self).__init__()
        self.cnn = cnn_module
        self.mlp = mlp_module

        combined_in_dim = self.cnn.out_features + self.mlp.out_features
        out_dim = trial.suggest_int("final_hidden_dim", 16, 512, log=True)
        dropout_final = trial.suggest_float("final_dropout", 0.0, 0.5, step=0.1)

        self.final_layers = nn.Sequential(
            nn.Linear(combined_in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_final),
            nn.Linear(out_dim, 1)  # regresja
        )

    def forward(self, x_nmr, x_fp):
        cnn_out = self.cnn(x_nmr)
        mlp_out = self.mlp(x_fp)
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        out = self.final_layers(combined)
        return out.squeeze(1)  # (B,)


# =================================================================================
# FUNKCJE POMOCNICZE
# =================================================================================
def set_seed(seed=42):
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
        y_pred = model(x_nmr, x_fp)
        loss = loss_fn(y_pred, y)
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
    # Tworzymy moduły CNN i MLP
    cnn_module = CNNModule(trial)
    mlp_module = MLPModule(trial)
    # Tworzymy model hybrydowy
    model = HybridModel(trial, cnn_module, mlp_module)

    # Optymalizator
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSProp"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:  # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    return model, optimizer


# =================================================================================
# WALIDACJA KRZYŻOWA (10-CV) Z EARLY STOPPING
# =================================================================================
def cross_validate(model_func, dataset, device, batch_size=64, n_folds=10, epochs=20):
    """
    :param model_func: funkcja (bez argumentów) -> (model, optimizer)
    :param dataset: torch Dataset z całościowymi danymi
    :param device: cpu/cuda
    :param batch_size: batch size
    :param n_folds: liczba foldów
    :param epochs: maksymalna liczba epok (z early stopping) na fold
    :return: dict z metrykami i y_true_all, y_pred_all
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))

    y_true_all = []
    y_pred_all = []

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

        # Early stopping
        best_val_rmse = float("inf")
        patience_counter = 0
        max_patience = 5  # np. 5 epok bez poprawy

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_rmse, y_true, y_pred = validate(model, val_loader, loss_fn, device)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            # (opcjonalne logi)
            # print(f"[Fold {fold_idx}, Epoch {epoch}] train_loss={train_loss:.4f}, val_rmse={val_rmse:.4f}")

            if patience_counter >= max_patience:
                # Przerywamy trenowanie w tym foldzie
                break

        # Uzyskujemy finalne predykcje
        _, val_rmse, y_true, y_pred = validate(model, val_loader, loss_fn, device)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        rmse_list.append(val_rmse)
        mae_list.append(mean_absolute_error(y_true, y_pred))
        r2_list.append(r2_score(y_true, y_pred))
        pearson_list.append(pearsonr(y_true, y_pred)[0])

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

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
        "y_pred_all": y_pred_all
    }
    return results


# =================================================================================
# CZĘŚCIOWA 3-FOLD CV W TRAKCIE OPTIMIZACJI (OBJECTIVE)
# =================================================================================
def objective(trial, dataset, device):
    """
    Optuna: używamy 3-krotnej walidacji do oceny jakości hiperparametrów.
    W każdym z 3 foldów trenujemy model (od zera) z early stopping i bierzemy RMSE.
    Zwracamy średni RMSE z 3 foldów jako wartość docelową do minimalizacji.
    
    Uwaga: Ponieważ zwracamy tylko jedną wartość (na koniec 3-fold),
    'pruning' staje się de facto jednorazowy (na step=0). 
    """
    batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
    max_epochs = 50  # np. 30 epok (z early stopping)

    # 3-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    indices = np.arange(len(dataset))

    rmse_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Nowy model i optimizer dla każdego folda
        model, optimizer = create_model_and_optimizer(trial)
        model.to(device)
        loss_fn = nn.MSELoss()

        best_val_rmse = float("inf")
        patience_counter = 0
        max_patience = 5

        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            _, val_rmse, _, _ = validate(model, val_loader, loss_fn, device)

            # Klasyczny early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                break

        rmse_scores.append(best_val_rmse)

    avg_rmse = float(np.mean(rmse_scores))

    # Raportujemy do Optuny (tylko 1 step)
    trial.report(avg_rmse, step=0)
    # Sprawdzamy pruning (ale tak naprawdę jest to single-step)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return avg_rmse


# =================================================================================
# GŁÓWNA FUNKCJA SKRYPTU
# =================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_1h", required=True, help="Ścieżka do pliku CSV z danymi ¹H")
    parser.add_argument("--path_13c", required=True, help="Ścieżka do pliku CSV z danymi ¹³C")
    parser.add_argument("--path_fp", required=True, help="Ścieżka do pliku CSV z fingerprintami ECFP4")
    parser.add_argument("--experiment_name", default="HybridNMR_FP_Experiment_3Fold", help="Nazwa eksperymentu w MLflow")
    parser.add_argument("--n_trials", default=20, type=int, help="Liczba prób Optuna")
    parser.add_argument("--epochs_10cv", default=20, type=int, help="Maks. liczba epok do finalnej 10CV (z early stopping)")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Używane urządzenie: {device}")

    # Ładujemy dane i tworzymy dataset
    X_nmr, X_fp, y = load_and_align_data(args.path_1h, args.path_13c, args.path_fp)
    X_nmr_t = torch.from_numpy(X_nmr).float().unsqueeze(2)  # (N, 2, 1, 200)
    X_fp_t = torch.from_numpy(X_fp).float()                 # (N, 2048)
    y_t = torch.from_numpy(y).float()

    dataset = NMRFpDataset(X_nmr_t, X_fp_t, y_t)

    mlflow.set_experiment(args.experiment_name)

    # Optuna: optymalizacja z 3CV
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda tr: objective(tr, dataset, device), n_trials=args.n_trials)

    best_params = study.best_params
    best_value = study.best_value

    print("Najlepsze parametry z Optuna (3CV):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Średnie RMSE (3CV): {best_value:.4f}")

    with mlflow.start_run(run_name="HybridCNN_MLP_Optuna_10CV"):
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("epochs_10cv", args.epochs_10cv)

        # Funkcja, która zamraża parametry z Optuny
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

        # 10CV z early stopping
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

        # Logowanie metryk
        mlflow.log_metric("rmse_mean", rmse_mean)
        mlflow.log_metric("rmse_std", rmse_std)
        mlflow.log_metric("mae_mean", mae_mean)
        mlflow.log_metric("mae_std", mae_std)
        mlflow.log_metric("r2_mean", r2_mean)
        mlflow.log_metric("r2_std", r2_std)
        mlflow.log_metric("pearson_mean", pearson_mean)
        mlflow.log_metric("pearson_std", pearson_std)

        print("\n=== Wyniki 10CV z Early Stopping ===")
        print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
        print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"R2 : {r2_mean:.4f} ± {r2_std:.4f}")
        print(f"Pearson : {pearson_mean:.4f} ± {pearson_std:.4f}")

        # (Opcjonalnie) trenujemy finalny model na całym zbiorze z tymi parametrami
        final_model, final_optimizer = best_model_func()
        final_model.to(device)
        loss_fn = nn.MSELoss()
        full_loader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)

        # Kilka epok z early stopping w skali całego zbioru
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

        # Logujemy finalny model
        mlflow.pytorch.log_model(final_model, artifact_path="model")

        # Wykres Real vs Pred (z 10CV)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_all, y_pred_all, alpha=0.5)
        plt.xlabel("Rzeczywista etykieta")
        plt.ylabel("Przewidywana etykieta")
        plt.title("Rzeczywiste vs. przewidywane (10CV)")
        plot_path = "real_vs_pred_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(plot_path)

        mlflow.end_run()


if __name__ == "__main__":
    main()
