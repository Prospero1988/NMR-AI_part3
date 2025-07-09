# predict_r2_cnn_v2.py
# Użycie:
#   python predict_r2_cnn_v2.py path/to/my_data.csv
import os, sys, re, argparse, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import r2_score
from packaging import version

SEED = 88
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- CNN 1D (skrócona wersja – identyczna topologia) -----------------
class Net(nn.Module):
    def __init__(self, params: dict, input_dim: int):
        super().__init__()
        act_name = params.get('activation', 'relu')
        act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(),
               'leaky_relu': nn.LeakyReLU(), 'selu': nn.SELU()}.get(act_name, nn.ReLU())
        dropout = float(params.get('dropout_rate', 0.0))
        use_bn  = params.get('use_batch_norm', False)

        self.regularization = params.get('regularization', 'none')
        self.reg_rate       = float(params.get('reg_rate', 0.0))

        n_conv = int(params.get('num_conv_layers', 2))
        in_c, length = 1, input_dim
        conv = []
        for i in range(n_conv):
            out_c = int(params.get(f'num_filters_l{i}', 32))
            pad   = int(params.get(f'padding_l{i}', 0))
            ksize = int(params.get(f'kernel_size_l{i}', 3))
            stride= int(params.get(f'stride_l{i}', 1))
            conv.append(nn.Conv1d(in_c, out_c, kernel_size=ksize,
                                  stride=stride, padding=pad))
            if use_bn: conv.append(nn.BatchNorm1d(out_c))
            conv.append(act)
            if dropout: conv.append(nn.Dropout(dropout))
            length = (length + 2*pad - (ksize-1) - 1)//stride + 1
            in_c   = out_c
        self.conv = nn.Sequential(*conv)

        n_fc, in_f = int(params.get('num_fc_layers',2)), in_c*length
        fc = []
        for i in range(n_fc):
            out_f = int(params.get(f'fc_units_l{i}',128))
            fc.append(nn.Linear(in_f,out_f))
            if use_bn: fc.append(nn.BatchNorm1d(out_f))
            fc.append(act)
            if dropout: fc.append(nn.Dropout(dropout))
            in_f = out_f
        fc.append(nn.Linear(in_f,1))
        self.fc = nn.Sequential(*fc)

        init = params.get('weight_init','xavier')
        self.apply(lambda m: Net._init(m, init))

    @staticmethod
    def _init(m, how):
        if isinstance(m,(nn.Conv1d,nn.Linear)):
            if how=='kaiming': nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif how=='normal': nn.init.normal_(m.weight,0.,0.05)
            else: nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self,x):
        x=x.unsqueeze(1)
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        return self.fc(x).squeeze(1)
# ---------------------------------------------------------------------------

def smart_cast(val:str):
    """zamień łańcuch na int, float lub bool – jeśli się da."""
    val = val.strip()
    if re.fullmatch(r'True|False', val, re.I):
        return val.lower()=='true'
    try:
        if '.' in val or 'e' in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val  # pozostaje string

def read_params_from_summary(txt_path:str)->dict:
    params={}
    grab=False
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith("Best parameters"):
                grab=True
                continue
            if grab:
                if not line.strip() or line.lstrip().startswith("10CV"):
                    break
                if ':' in line:
                    k,v=line.split(':',1)
                    params[k.strip()]=smart_cast(v)
    if not params:
        sys.exit(f"❌ Nie znaleziono sekcji 'Best parameters' w {txt_path}")
    return params

def load_csv(csv):
    df=pd.read_csv(csv)
    df=df.drop(df.columns[0],axis=1)
    y=df['LABEL'].values.astype(np.float32)
    X=df.drop(columns=['LABEL']).values.astype(np.float32)
    return X,y

def main():
    parser=argparse.ArgumentParser(description="Predykcja CNN-1D + R² (parametry z *_summary.txt)")
    parser.add_argument("csv", help="plik wejściowy CSV")
    args=parser.parse_args()

    csv_path=args.csv
    base=os.path.splitext(os.path.basename(csv_path))[0]
    model_path   = f"{base}_final_model.pth"
    summary_path = f"{base}_summary.txt"

    for f in (model_path, summary_path):
        if not os.path.exists(f):
            sys.exit(f"❌ Brak wymaganego pliku: {f}")

    params=read_params_from_summary(summary_path)
    X,y   =load_csv(csv_path)
    net   = Net(params, X.shape[1]).to(device)

    # ----------  wczytywanie wag  ----------
    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    else:
        state_dict = torch.load(model_path, map_location=device)

    net.load_state_dict(state_dict)

    net.eval()

    with torch.no_grad():
        y_pred = net(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()

    r2 = r2_score(y, y_pred)
    print(f"✅  R² = {r2:.4f}")

    out_file=f"{base}_prediction_summary.txt"
    with open(out_file,'w',encoding='utf-8') as f:
        f.write(f"CSV          : {csv_path}\n")
        f.write(f"Model weights: {model_path}\n")
        f.write(f"Params file  : {summary_path}\n")
        f.write(f"R2           : {r2:.6f}\n")
    print(f"📄  Zapisano {out_file}")

if __name__=="__main__":
    main()
