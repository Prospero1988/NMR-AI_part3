#!/usr/bin/env python3
"""
Generates a SHAP beeswarm plot (top-20 features) from a CSV file
created by 1D_CNN_SHAP.py.

Usage:
    python plot_beeswarm.py \
        --shap    SHAP_CHIlogD026.csv \
        --input   chilogd026_1H13C.csv \
        --out     beeswarm_top20.png \
        --top     20
"""

import argparse
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--shap", required=True, help="CSV with columns SHAP_*")
parser.add_argument("--input", required=True, help="Original CSV with feature values")
parser.add_argument("--out", default="beeswarm.png", help="Output PNG file")
parser.add_argument("--top", type=int, default=20, help="Number of top features to show")
args = parser.parse_args()

# --- 1. Data loading ---
shap_df = pd.read_csv(args.shap)
feat_df = pd.read_csv(args.input)

# Assume ID column is the first column
id_col = feat_df.columns[0]
shap_cols = [col for col in shap_df.columns if col.startswith("SHAP_")]
feat_names = [col.replace("SHAP_", "") for col in shap_cols]

# --- 2. Merge on ID column ---
merged = shap_df[[id_col] + shap_cols].merge(
    feat_df[[id_col] + feat_names],
    on=id_col,
    how="inner",
    suffixes=("_shap", "")
)

shap_values = merged[shap_cols].values
feature_values = merged[feat_names].values

# --- 3. Top-N features selection ---
mean_abs = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs)[-args.top:][::-1]

shap_values_top = shap_values[:, top_idx]
feature_values_top = feature_values[:, top_idx]
feature_names_top = [feat_names[i] for i in top_idx]

# --- Automatic bucket center labeling ---
ppm_labels = []
bucket_info = {
    '1H': {
        'range': (-1, 14),
        'bucket_size': 0.06,
        'start_idx': 0,
        'end_idx': 249
    },
    '13C': {
        'range': (-10, 230),
        'bucket_size': 0.96,
        'start_idx': 250,
        'end_idx': 499
    }
}

for idx in top_idx:
    if idx <= bucket_info['1H']['end_idx']:
        info = bucket_info['1H']
        ppm_center = info['range'][0] + info['bucket_size'] * (idx + 0.5)
        label = f"¹H {ppm_center:.2f}"
    else:
        info = bucket_info['13C']
        adjusted_idx = idx - info['start_idx']
        ppm_center = info['range'][0] + info['bucket_size'] * (adjusted_idx + 0.5)
        label = f"¹³C {ppm_center:.1f}"
    ppm_labels.append(label)

print(ppm_labels)

# --- 4. Beeswarm plot ---
shap.summary_plot(
    shap_values_top,
    feature_values_top,
    feature_names=ppm_labels,
    max_display=args.top,
    show=False,
    plot_size=(8, 10)
)

# Modify X-axis appearance
main_ax = plt.gcf().axes[0]
main_ax.set_xlabel("SHAP value", fontsize=14, fontweight='bold')
main_ax.tick_params(axis='x', labelsize=14)

# Modify colorbar
colorbar_ax = plt.gcf().axes[-1]  # assume it's the last axis
colorbar_ax.tick_params(labelsize=14)
for label in colorbar_ax.get_yticklabels():
    label.set_fontweight('bold')

colorbar_ax.set_ylabel("Feature value", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(args.out, dpi=100, bbox_inches='tight')
print(f"✓ Saved to {args.out}")
