# predict_r2_mlp.py
# Użycie:
#   python predict_r2_mlp.py path/to/my_data.csv
import os, sys, re, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import r2_score
from packaging import version

SEED = 88
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- narzędziówka ----------------------------------------
def smart_cast(value: str):
    """zamień str -> bool / int / float, jeśli się da."""
    v = value.strip()
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    try:
        if re.fullmatch(r'-?\d+\.\d*(e[-+]?\d+)?', v, re.I):
            return float(v)
        return int(v)
    except ValueError:
        return v

def read_params(summary_path: str):
    """wyciągnij słownik parametrów z pliku *_summary.txt."""
    params = {}
    grab = False
    with open(summary_path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('Best parameters'):
                grab = True
                continue
            if grab:
                if not line.strip() or line.lstrip().startswith('10CV'):
                    break
                if ':' in line:
                    k, v = line.split(':', 1)
                    params[k.strip()] = smart_cast(v)
    if not params:
        sys.exit(f"❌ Nie znaleziono hiperparametrów w {summary_path}")
    return params
# ---------------------------------------------------------------------------

class MLP1D(nn.Module):
    """Odzwierciedlenie architektury trenowanej w MLP_1D_pytorch.py."""
    def __init__(self, params: dict, input_dim: int):
        super().__init__()
        act = {
            'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01), 'selu': nn.SELU()
        }.get(params.get('activation', 'relu'), nn.ReLU())
        use_bn       = params.get('use_batch_norm', False)
        dropout_rate = float(params.get('dropout_rate', 0.0))
        n_layers     = int(params.get('num_layers', 1))
        units        = int(params.get('units', 128))

        layers = []
        in_f = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_f, units))
            if use_bn:
                layers.append(nn.BatchNorm1d(units))
            layers.append(act)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_f = units
        layers.append(nn.Linear(in_f, 1))
        self.model = nn.Sequential(*layers)

        # inicjalizacja wag (niekonieczna przy wczytywaniu state_dict, ale zachowana)
        init = params.get('weight_init', 'xavier')
        self.apply(lambda m: MLP1D._init_weights(m, init))

    @staticmethod
    def _init_weights(m, how):
        if isinstance(m, nn.Linear):
            if how == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif how == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.05)
            else:  # domyślnie 'xavier'
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):          # x: (B, input_dim)
        return self.model(x).squeeze(1)

def load_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.drop(df.columns[0], axis=1)            # nazwa próbki
    y  = df['LABEL'].values.astype(np.float32)
    X  = df.drop(columns=['LABEL']).values.astype(np.float32)
    return X, y

def main():
    ap = argparse.ArgumentParser(description="Inferencja MLP-1D + R² (parametry z *_summary.txt)")
    ap.add_argument('csv', help='ścieżka do pliku CSV z danymi')
    args = ap.parse_args()

    csv_path = args.csv
    base = os.path.splitext(os.path.basename(csv_path))[0]
    model_path   = f"{base}_final_model.pth"
    summary_path = f"{base}_summary.txt"

    for f in (model_path, summary_path):
        if not os.path.exists(f):
            sys.exit(f"❌ Brak pliku: {f}")

    params = read_params(summary_path)
    X, y   = load_csv(csv_path)
    net    = MLP1D(params, X.shape[1]).to(device)

    # ----------  wczytywanie wag  ----------
    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    else:
        state_dict = torch.load(model_path, map_location=device)

    net.load_state_dict(state_dict)

    net.eval()

    with torch.no_grad():
        y_pred = net(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()

    # --- sprawdźmy, czy w danych są NaN/Inf ----------------------------------
    bad_y      = ~np.isfinite(y)
    bad_pred   = ~np.isfinite(y_pred)
    if bad_y.any() or bad_pred.any():
        print(f"‼️  UWAGA: y -> {bad_y.sum()} nie-finite,  y_pred -> {bad_pred.sum()} nie-finite")
        # Jeśli chcesz po prostu pominąć te punkty w R²:
        mask = np.isfinite(y) & np.isfinite(y_pred)
        y, y_pred = y[mask], y_pred[mask]
    # -------------------------------------------------------------------------
    r2 = r2_score(y, y_pred)

    print(f"✅  R² = {r2:.4f}")

    out = f"{base}_prediction_summary.txt"
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(f"CSV           : {csv_path}\n")
        fh.write(f"Model weights : {model_path}\n")
        fh.write(f"Params file   : {summary_path}\n")
        fh.write(f"R2            : {r2:.6f}\n")
    print(f"📄  Zapisano {out}")

if __name__ == '__main__':
    main()
