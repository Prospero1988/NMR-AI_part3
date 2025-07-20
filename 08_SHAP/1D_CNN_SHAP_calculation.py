#!/usr/bin/env python3
"""
Computes SHAP values (GradientExplainer = integrated gradients)
for a 1-D CNN and saves them to a CSV file.
"""

import argparse
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap


# ---------- 1. Parse summary.txt ---------- #
def parse_params(path: Path):
    """Parses hyperparameters from a summary.txt file."""
    params, grab = {}, False
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln == "Best parameters:":
            grab = True
            continue
        if not grab or ln == "" or ln.endswith(":"):
            continue
        if ln.startswith("10CV"):  # end of block
            break
        k, v = [x.strip() for x in ln.split(":", 1)]
        k = re.sub(r"\s*\(.*?\)$", "", k).strip()
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v) if "." not in v else float(v)
            except ValueError:
                pass
        params[k] = v
    return params


# ---------- 2. CNN definition ---------- #
class Net(nn.Module):
    """Simple 1D CNN based on parsed parameters."""

    def __init__(self, p: dict, L_in: int):
        super().__init__()
        act = {
            "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU, "selu": nn.SELU,
        }.get(p.get("activation", "relu").lower(), nn.ReLU)()

        use_bn = p.get("use_batch_norm", False)
        drop = p.get("dropout_rate", 0.0)

        conv, C, L = [], 1, L_in
        for i in range(p.get("num_conv_layers", 1)):
            F = p.get(f"num_filters_l{i}", 16)
            k = p.get(f"kernel_size_l{i}", 3)
            s = p.get(f"stride_l{i}", 1)
            pad = p.get(f"padding_l{i}", 0)
            conv += [nn.Conv1d(C, F, k, s, pad)]
            if use_bn:
                conv += [nn.BatchNorm1d(F)]
            conv += [act]
            if drop:
                conv += [nn.Dropout(drop)]
            C = F
            L = math.floor((L + 2 * pad - (k - 1) - 1) / s + 1)
        self.conv = nn.Sequential(*conv)
        self.flatten = nn.Flatten(1)

        dense, D = [], C * L
        for i in range(p.get("num_fc_layers", 2)):
            out = p.get(f"fc_units_l{i}", 32)
            dense += [nn.Linear(D, out)]
            if use_bn:
                dense += [nn.BatchNorm1d(out)]
            dense += [act]
            if drop:
                dense += [nn.Dropout(drop)]
            D = out
        dense += [nn.Linear(D, 1)]
        self.fc = nn.Sequential(*dense)

    def forward(self, x):
        return self.fc(self.flatten(self.conv(x)))


# ---------- 3. Helper functions ---------- #
def infer_min_len(p, state):
    """Infers minimal input length for CNN from checkpoint."""
    fc_in, out_ch = state["fc.0.weight"].shape[1], p.get(f"num_filters_l{p.get('num_conv_layers', 1)-1}", 16)
    L = fc_in // out_ch
    for i in reversed(range(p.get("num_conv_layers", 1))):
        k = p.get(f"kernel_size_l{i}", 3)
        s = p.get(f"stride_l{i}", 1)
        pad = p.get(f"padding_l{i}", 0)
        L = (L - 1) * s + (k - 1) + 1 - 2 * pad
    return L


def forward_len(L, p):
    """Computes forward-passed length of CNN after all conv layers."""
    for i in range(p.get("num_conv_layers", 1)):
        k = p.get(f"kernel_size_l{i}", 3)
        s = p.get(f"stride_l{i}", 1)
        pad = p.get(f"padding_l{i}", 0)
        L = math.floor((L + 2 * pad - (k - 1) - 1) / s + 1)
    return L


# ---------- 4. CLI ---------- #
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--summary", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--bg", type=int, default=100)
ap.add_argument("--ntest", type=int, default=128)
ap.add_argument("--seed", type=int, default=1988)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# ---------- 5. Data loading ---------- #
df = pd.read_csv(args.input)
id_col = df.columns[0]
label_col = "LABEL" if "LABEL" in df.columns else None

X = df.drop(columns=[id_col] + ([label_col] if label_col else [])).to_numpy(dtype=np.float32)
N, F = X.shape

# ---------- 6. Load model ---------- #
p = parse_params(Path(args.summary))
state = torch.load(args.model, map_location="cpu", weights_only=True)
min_len = infer_min_len(p, state)
fc_expect = state["fc.0.weight"].shape[1]
fc_from_csv = forward_len(F, p) * p.get(f"num_filters_l{p.get('num_conv_layers', 1)-1}", 16)

if fc_from_csv != fc_expect:
    sys.exit(f"\nERROR: CSV → fc_in={fc_from_csv}, but checkpoint has {fc_expect}. Mismatch in preprocessing/params.\n")
elif F != min_len:
    print(f"Warning: Model requires at least {min_len} features, CSV has {F} (OK for GradientExplainer).")

model = Net(p, min_len)
model.load_state_dict(state, strict=True)
model.eval()

# ---------- 7. Sample splitting ---------- #
idx = np.random.permutation(N)
bg_idx = idx[:args.bg]
test_idx = idx[args.bg:args.bg + args.ntest]
X_bg = torch.tensor(X[bg_idx]).unsqueeze(1)
X_test = torch.tensor(X[test_idx]).unsqueeze(1)

# ---------- 8. SHAP via GradientExplainer ---------- #
explainer = shap.GradientExplainer(model, X_bg)
shap_vals = np.asarray(explainer.shap_values(X_test)).reshape(X_test.shape[0], -1)

# ---------- 9. Save to CSV ---------- #
out = pd.DataFrame(
    shap_vals,
    columns=[f"SHAP_{c}" for c in df.columns if c not in (id_col, label_col)]
)
out.insert(0, id_col, df[id_col].iloc[test_idx].values)
if label_col:
    out.insert(1, label_col, df[label_col].iloc[test_idx].values)

out.to_csv(args.output, index=False)
print(f"Saved {out.shape[0]} × {out.shape[1] - 1} → {args.output}")
