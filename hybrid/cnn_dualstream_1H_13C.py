#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN-Dual-Stream (¹H + ¹³C) z Optuną, pełną integracją MLflow, 3CV → 10CV, finalny
trening i logowanie artefaktów. Architektura:

  • Strumień ¹H : Conv1D-blokᵢ → … → embedding_h
  • Strumień ¹³C: Conv1D-blokᵢ → … → embedding_c
             └────── concat([embedding_h, embedding_c]) → (B, C*, L*)
                         └─ wspólny blok Conv1D (opcjonalny)
                               └─ Flatten → FC → ŷ

Autor: aleniak  | 07-2025
"""

# ---------------------------------------------------------------------------
# Importy
# ---------------------------------------------------------------------------
import argparse, os, sys, json, logging, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

import optuna, mlflow, mlflow.pytorch
from optuna.importance import get_param_importances
import optuna.visualization.matplotlib as opt_viz
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Stałe / tagi
# ---------------------------------------------------------------------------
MLFLOW_TAGS = {
    "property": "CHI logD",
    "model": "CNN Dual",
    "predictor": "1Hx13C"
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NMRDataset(Dataset):
    def __init__(self, nmr_tensor: torch.Tensor, labels: torch.Tensor):
        """
        nmr_tensor: shape (N, 2, 200)
        labels    : shape (N,)
        """
        self.nmr = nmr_tensor
        self.labels = labels
    def __len__(self): return self.nmr.size(0)
    def __getitem__(self, idx):
        return self.nmr[idx], self.labels[idx]

def load_nmr_data(p1h: str, p13c: str) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    """Zwraca: mol_names (pd.Series), x_nmr (N,2,200), y (N,)"""

    df_h  = pd.read_csv(p1h)
    df_c  = pd.read_csv(p13c)

    df_h.columns = ["MOLECULE_NAME", "LABEL"] + [f"h_{i}" for i in range(df_h.shape[1]-2)]
    df_c.columns = ["MOLECULE_NAME", "LABEL"] + [f"c_{i}" for i in range(df_c.shape[1]-2)]

    # parowanie 1-do-1 z zachowaniem duplikatów (cumcount)
    df_h["_dup"] = df_h.groupby("MOLECULE_NAME").cumcount()
    df_c["_dup"] = df_c.groupby("MOLECULE_NAME").cumcount()

    merged = pd.merge(df_h, df_c, on=["MOLECULE_NAME","LABEL","_dup"], how="inner")

    h_cols = [c for c in merged.columns if c.startswith("h_")]
    c_cols = [c for c in merged.columns if c.startswith("c_")]

    x_h = merged[h_cols].values.astype(np.float32)   # (N,200)
    x_c = merged[c_cols].values.astype(np.float32)
    x_nmr = np.stack([x_h, x_c], axis=1)             # (N,2,200)
    y = merged["LABEL"].values.astype(np.float32)

    mol_names = merged["MOLECULE_NAME"]          # <── nowa linia
    return mol_names, x_nmr, y  

# ---------------------------------------------------------------------------
# CNN Dual-Stream
# ---------------------------------------------------------------------------
def conv_block_1d(in_ch: int,
                  out_ch: int,
                  k: int,
                  s: int,
                  p: int,
                  dropout: float,
                  use_bn: bool = True) -> List[nn.Module]:
    layers = [nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
    if use_bn: layers.append(nn.BatchNorm1d(out_ch))
    layers += [nn.SiLU()]
    if dropout > 0.0: layers.append(nn.Dropout(dropout))
    return layers

class CNNDual(nn.Module):
    def __init__(self, trial):
        super().__init__()

        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        use_bn  = trial.suggest_categorical("use_bn", [True, False])

        # ----- ¹H stream --------------------------------------------------
        in_ch = 1
        layers_h = []
        n_conv_h = trial.suggest_int("n_conv_h", 1, 4)
        for i in range(n_conv_h):
            out_ch = trial.suggest_int(f"h_ch_l{i}", 8, 128, log=True)
            k      = trial.suggest_int(f"h_k_l{i}", 3, 15, step=2)
            s      = trial.suggest_int(f"h_s_l{i}", 1, 3)
            p = (k - 1) // 2
            layers_h += conv_block_1d(in_ch, out_ch, k, s, p, dropout, use_bn)
            in_ch = out_ch
        self.stream_h = nn.Sequential(*layers_h,
                                      nn.AdaptiveAvgPool1d(1))  # GAP → (B, C_h, 1)
        self.c_out_h = in_ch

        # ----- ¹³C stream --------------------------------------------------
        in_ch = 1
        layers_c = []
        n_conv_c = trial.suggest_int("n_conv_c", 1, 4)
        for i in range(n_conv_c):
            out_ch = trial.suggest_int(f"c_ch_l{i}", 8, 128, log=True)
            k      = trial.suggest_int(f"c_k_l{i}", 3, 15, step=2)
            s      = trial.suggest_int(f"c_s_l{i}", 1, 3)
            p = (k - 1) // 2
            layers_c += conv_block_1d(in_ch, out_ch, k, s, p, dropout, use_bn)
            in_ch = out_ch
        self.stream_c = nn.Sequential(*layers_c,
                                      nn.AdaptiveAvgPool1d(1))
        self.c_out_c = in_ch

        # --------- embedding FC dla obu -----------------------------------
        embed_dim = trial.suggest_int("embed_dim", 32, 256, log=True)
        self.fc_h = nn.Linear(self.c_out_h, embed_dim)
        self.fc_c = nn.Linear(self.c_out_c, embed_dim)

        # -------- cross-attention ----------
        use_xattn = trial.suggest_categorical("use_xattn", [True, False])
        self.use_xattn = use_xattn
        if use_xattn:
            # projection do przestrzeni Q/K/V (wspólny embed_dim)
            self.q_h = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_c = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_c = nn.Linear(embed_dim, embed_dim, bias=False)

            self.q_c = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_h = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_h = nn.Linear(embed_dim, embed_dim, bias=False)

            self.scale = embed_dim ** -0.5
            merge_dim = 2 * embed_dim   # (h_att || c_att)
        else:
            merge_dim = 2 * embed_dim   # (h_emb || c_emb)

        # ------ wspólna głowa -------------------------------------------
        fc_hidden = trial.suggest_int("fc_hidden", 32, 512, log=True)
        self.head = nn.Sequential(
            nn.Linear(merge_dim, fc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

    def forward(self, x):
        # x: (B,2,200)
        h = self.stream_h(x[:,0,:].unsqueeze(1)).squeeze(-1)   # (B, C_h)
        c = self.stream_c(x[:,1,:].unsqueeze(1)).squeeze(-1)   # (B, C_c)

        h_emb = self.fc_h(h)                                   # (B, embed_dim)
        c_emb = self.fc_c(c)                                   # (B, embed_dim)

        # ── cross-attention lub zwykłe concat ─────────────────────────────
        if self.use_xattn:
            # H → zapytanie,  C → klucz/wartość
            att_hc = torch.softmax(
                (self.q_h(h_emb) * self.k_c(c_emb)).sum(dim=1, keepdim=True) * self.scale,
                dim=-1
            )                               # (B,1)
            h_att = h_emb + att_hc * self.v_c(c_emb)

            # C → zapytanie,  H → klucz/wartość
            att_ch = torch.softmax(
                (self.q_c(c_emb) * self.k_h(h_emb)).sum(dim=1, keepdim=True) * self.scale,
                dim=-1
            )
            c_att = c_emb + att_ch * self.v_h(h_emb)

            merged = torch.cat([h_att, c_att], dim=1)   # (B, 2*embed_dim)
        else:
            merged = torch.cat([h_emb, c_emb], dim=1)   # (B, 2*embed_dim)

        # ── predykcja ─────────────────────────────────────────────────────
        return self.head(merged).squeeze(1)
        # ---------------------------------------------------------------------------
# Narzędzia
# ---------------------------------------------------------------------------
def set_seed(seed=1988):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train(); running=0.0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        running += loss.item()*len(yb)
    return running/len(loader.dataset)

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval(); y_true=[]; y_pred=[]; loss_sum=0.0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss_sum += loss.item()*len(yb)
        y_true.append(yb.cpu().numpy()); y_pred.append(pred.cpu().numpy())
    y_true, y_pred = map(lambda arr: np.concatenate(arr), (y_true, y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return loss_sum/len(loader.dataset), rmse, y_true, y_pred

# ---------------------------------------------------------------------------
# Helpers dla Optuny
# ---------------------------------------------------------------------------
def create_model_and_optimizer(trial: optuna.Trial):
    model = CNNDual(trial)
    opt_name = trial.suggest_categorical("optimizer", ["Adam","SGD","RMSProp"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    if opt_name == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "SGD":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = optim.RMSprop(model.parameters(), lr=lr)
    return model, opt

# ---------------------------------------------------------------------------
# 10-CV
# ---------------------------------------------------------------------------
def cross_validate(model_func, dataset, device,
                   batch_size=64, n_folds=10, epochs=50):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))

    y_true_all=[]; y_pred_all=[]; fold_idx_all=[]
    rmse_l=[]; mae_l=[]; r2_l=[]; pearson_l=[]

    for fold,(tr_idx,val_idx) in enumerate(kf.split(indices)):
        tr_ds = torch.utils.data.Subset(dataset, tr_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader= DataLoader(val_ds,batch_size=batch_size, shuffle=False)

        model,opt = model_func(); model.to(device)
        loss_fn = nn.MSELoss()

        best_rmse=float("inf"); patience=0; max_pat=5
        for ep in range(epochs):
            train_one_epoch(model,tr_loader,opt,loss_fn,device)
            _, rmse,_,_ = validate(model,val_loader,loss_fn,device)
            if rmse < best_rmse:
                best_rmse=rmse; patience=0
            else:
                patience+=1
            if patience>=max_pat: break

        _,rmse,y_t,y_p = validate(model,val_loader,loss_fn,device)
        y_true_all.append(y_t); y_pred_all.append(y_p)
        fold_idx_all.append(np.full_like(y_t, fold, dtype=int))

        rmse_l.append(rmse)
        mae_l.append(mean_absolute_error(y_t,y_p))
        r2_l.append(r2_score(y_t,y_p))
        pearson_l.append(pearsonr(y_t,y_p)[0])

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    fold_idx_all= np.concatenate(fold_idx_all)

    return dict(
        rmse_mean=float(np.mean(rmse_l)), rmse_std=float(np.std(rmse_l)),
        mae_mean=float(np.mean(mae_l)),  mae_std=float(np.std(mae_l)),
        r2_mean=float(np.mean(r2_l)),    r2_std=float(np.std(r2_l)),
        pearson_mean=float(np.mean(pearson_l)), pearson_std=float(np.std(pearson_l)),
        y_true_all=y_true_all, y_pred_all=y_pred_all, fold_indices_all=fold_idx_all
    )

# ---------------------------------------------------------------------------
# Objective (3CV)
# ---------------------------------------------------------------------------
def objective(trial, dataset, device):
    batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
    epochs = 50
    kf = KFold(n_splits=3, shuffle=True, random_state=123)
    indices = np.arange(len(dataset))

    rmses=[]
    for tr_idx,val_idx in kf.split(indices):
        tr_ds = torch.utils.data.Subset(dataset, tr_idx)
        val_ds= torch.utils.data.Subset(dataset, val_idx)
        tr_loader=DataLoader(tr_ds,batch_size=batch_size,shuffle=True)
        val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False)

        model,opt = create_model_and_optimizer(trial)
        model.to(device); loss_fn=nn.MSELoss()

        best_rmse=float("inf"); patience=0; max_pat=5
        for ep in range(epochs):
            train_one_epoch(model,tr_loader,opt,loss_fn,device)
            _, rmse,_,_ = validate(model,val_loader,loss_fn,device)
            if rmse<best_rmse: best_rmse=rmse; patience=0
            else: patience+=1
            if patience>=max_pat: break
        rmses.append(best_rmse)

    avg_rmse=float(np.mean(rmses))
    trial.report(avg_rmse, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return avg_rmse

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--path_1h", required=True)
    parser.add_argument("--path_13c",required=True)
    parser.add_argument("--experiment_name", default="CNNDual_Experiment")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--epochs_10cv", type=int, default=60)
    args=parser.parse_args()

    set_seed(1988); device=get_device()

    # ----- katalog wyników -----
    prefix=os.path.basename(args.path_1h).split("_")[0]
    res_dir=f"{prefix}-results-cnndual"
    os.makedirs(res_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(res_dir,"script_cnndual.log")),
                  logging.StreamHandler(sys.stdout)]
    )
    logger=logging.getLogger(__name__)
    logger.info("Start CNNDual | folder wyników: %s", res_dir)

    try:
        mol_names, X, y = load_nmr_data(args.path_1h, args.path_13c)
        ds=NMRDataset(torch.from_numpy(X), torch.from_numpy(y))
        mlflow.set_experiment(args.experiment_name)

        study=optuna.create_study(direction="minimize")
        logger.info("Optuna: n_trials=%d", args.n_trials)
        study.optimize(lambda tr: objective(tr, ds, device), n_trials=args.n_trials)

        best_params=study.best_params; best_value=study.best_value
        logger.info("Najlepsze parametry: %s | RMSE=%.4f", best_params, best_value)

        # ---------- MLflow run ----------
        with mlflow.start_run(run_name="CNNDual_10CV") as run:
            run_tags=dict(MLFLOW_TAGS); run_tags["file"]=prefix
            mlflow.set_tags(run_tags)
            mlflow.log_params(best_params)
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("epochs_10cv", args.epochs_10cv)

            # --- artefakty Optuny ---
            df_trials=study.trials_dataframe(attrs=("number","value","params","state"))
            df_trials.to_csv(os.path.join(res_dir,"optuna_trials_cnndual.csv"),index=False)
            mlflow.log_artifact(os.path.join(res_dir,"optuna_trials_cnndual.csv"))

            # Wykres przebiegu RMSE
            plt.figure(); plt.plot(df_trials["number"], df_trials["value"],"o-")
            plt.xlabel("Trial ID"); plt.ylabel("RMSE (3CV)")
            plt.title("Optuna RMSE vs Trial ID (CNNDual)"); plt.grid(True)
            plt.savefig(os.path.join(res_dir,"optuna_trials_rmse_cnndual.png"), dpi=300, bbox_inches="tight")
            mlflow.log_artifact(os.path.join(res_dir,"optuna_trials_rmse_cnndual.png"))
            plt.close()

            # Ważności hiperparametrów
            imp=get_param_importances(study)
            with open(os.path.join(res_dir,"param_importances_cnndual.json"),"w") as f:
                json.dump(imp,f,indent=2)
            fig_imp=opt_viz.plot_param_importances(study)
            fig_imp.figure.savefig(os.path.join(res_dir,"param_importances_cnndual.png"),
                                   dpi=300,bbox_inches="tight")
            mlflow.log_artifact(os.path.join(res_dir,"param_importances_cnndual.png"))
            plt.close(fig_imp.figure)

            # ---------- 10-CV ----------
            def best_model_func():
                class FrozenTrial:
                    def __init__(self, params): self.params=params
                    def suggest_int(self, name, *a, **k):   return self.params[name]
                    def suggest_float(self, name, *a, **k): return self.params[name]
                    def suggest_categorical(self, name, choices): return self.params[name]
                tr=FrozenTrial(best_params)
                model=CNNDual(tr)
                opt_name=best_params["optimizer"]; lr=best_params["lr"]
                if opt_name=="Adam":  opt=optim.Adam(model.parameters(), lr=lr)
                elif opt_name=="SGD": opt=optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                else:                opt=optim.RMSprop(model.parameters(), lr=lr)
                return model,opt

            logger.info("Start 10CV")
            cv_res=cross_validate(best_model_func, ds, device,
                                  batch_size=best_params["batch_size"],
                                  n_folds=10, epochs=args.epochs_10cv)

            for k,v in cv_res.items():
                if isinstance(v,float): mlflow.log_metric(f"{k}_10cv", v)
            # zapis tabel/metryk
            pd.DataFrame({
              "fold": cv_res["fold_indices_all"],
              "y_true": cv_res["y_true_all"],
              "y_pred": cv_res["y_pred_all"]
            }).to_csv(os.path.join(res_dir,"cv_predictions_cnndual.csv"),index=False)
            mlflow.log_artifact(os.path.join(res_dir,"cv_predictions_cnndual.csv"))

            # wykresy
            plt.figure(figsize=(6,6))
            plt.scatter(cv_res["y_true_all"], cv_res["y_pred_all"], alpha=0.5)
            plt.xlabel("y true"); plt.ylabel("y pred")
            plt.title("CNN-Dual 10CV: y_true vs y_pred")
            plt.savefig(os.path.join(res_dir,"real_vs_pred_cnndual.png"),dpi=300,bbox_inches="tight")
            mlflow.log_artifact(os.path.join(res_dir,"real_vs_pred_cnndual.png"))
            plt.close()

            plt.figure(figsize=(6,6))
            abs_err=np.abs(cv_res["y_true_all"]-cv_res["y_pred_all"])
            plt.scatter(cv_res["y_true_all"], abs_err, alpha=0.5)
            plt.xlabel("y true"); plt.ylabel("|error|")
            plt.title("Abs error vs y_true (CNNDual 10CV)")
            plt.savefig(os.path.join(res_dir,"error_plot_cnndual.png"),dpi=300,bbox_inches="tight")
            mlflow.log_artifact(os.path.join(res_dir,"error_plot_cnndual.png"))
            plt.close()

            # ---------- final training ----------
            final_model,final_opt = best_model_func(); final_model.to(device)
            loader = DataLoader(ds,batch_size=best_params["batch_size"], shuffle=True)
            loss_fn=nn.MSELoss()
            best_rmse=float("inf"); patience=0; max_pat=5
            for ep in range(args.epochs_10cv):
                train_one_epoch(final_model, loader, final_opt, loss_fn, device)
                _,rmse,_,_ = validate(final_model, loader, loss_fn, device)
                if rmse<best_rmse: best_rmse=rmse; patience=0
                else: patience+=1
                if patience>=max_pat: break

            mlflow.pytorch.log_model(final_model, artifact_path="model_cnndual")

            # ---------- WILLIAMS: pełna tabela + outliers ----------
            final_model.eval()
            with torch.no_grad():
                y_pred = final_model(torch.tensor(X).to(device)).cpu().numpy()

            # ------- obliczenia do Williamsa -------
            residuals = y - y_pred
            std_resid = (residuals - residuals.mean()) / residuals.std()
            out_thr = 3

            # macierz cech do lewara – po prostu sklej widma 1H i 13C
            X_feat = X.reshape(X.shape[0], -1)          # (N, 400)

            # usuwamy kolumny o zerowej wariancji (jak w Twoim skrypcie rysują-cym)
            nz_mask = X_feat.std(axis=0) > 1e-10
            X_feat = (X_feat[:, nz_mask] -
                    X_feat[:, nz_mask].mean(axis=0)) / X_feat[:, nz_mask].std(axis=0)

            H = X_feat @ np.linalg.pinv(X_feat.T @ X_feat) @ X_feat.T
            leverage = np.diag(H)
            lev_thr = 3 * X_feat.shape[1] / X_feat.shape[0]

            # ------- DataFrame -------
            will_df = pd.DataFrame({
                "MOLECULE_NAME": mol_names,
                "y_actual":      y,
                "y_pred":        y_pred,
                "residual":      residuals,
                "std_residual":  std_resid,
                "leverage":      leverage,
            })

            # --- sanity check: musi być tyle samo wierszy co w danych ---
            assert len(will_df) == len(y), \
                f"Mismatch rows: {len(will_df)} vs {len(y)}"

            # zapisz i zaloguj pełną tabelę
            full_path = f"{prefix}_williams_full.csv"
            will_df.to_csv(full_path, index=False)
            mlflow.log_artifact(full_path)

            # filtr outliers
            out_df = will_df[(np.abs(std_resid) > out_thr) | (leverage > lev_thr)].copy()
            out_path = f"{prefix}_williams_outliers.csv"
            out_df.to_csv(out_path, index=False)
            mlflow.log_artifact(out_path)

        logger.info("CNNDual run zakończony OK")

    except Exception as exc:
        logging.exception("Błąd w skrypcie CNNDual"); sys.exit(1)

if __name__=="__main__":
    main()
