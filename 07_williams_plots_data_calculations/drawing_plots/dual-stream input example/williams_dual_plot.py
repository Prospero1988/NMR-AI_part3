import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ PLOT SETTINGS (as before) ------------------
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'

# === CONFIGURATION ===
INPUT_FILE_1H     = "chilogd105_1H_ML_input.csv"     # <–– Your 1H file
INPUT_FILE_13C    = "chilogd105_13C_ML_input.csv"    # <–– Your 13C file
PREDICTIONS_FILE  = "chilogd105_williams.csv"        # y_actual & y_pred
OUTPUT_DIR        = "williams_output"
FULL_DATA_FILE    = "williams_full_data.csv"
OUTLIERS_FILE     = "williams_outliers.csv"
PLOT_FILE         = "williams_plot.png"
SHOW_LEGEND = True   # False ⇒ legend will not be shown
SHOW_LABELS_X = True
SHOW_LABELS_Y = True

# --- additional axis control ---
XLIM = (-0.05, 1.05)   # None or (xmin, xmax)
YLIM = (-7.0, 7.0)     # None or (ymin, ymax)

# --------------------------- LOADING ----------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_1h  = pd.read_csv(INPUT_FILE_1H)
df_13c = pd.read_csv(INPUT_FILE_13C)
pred_df = pd.read_csv(PREDICTIONS_FILE)

# -------------------- ID VALIDATION (All three) -----------------------
if not (df_1h.iloc[:, 0].equals(df_13c.iloc[:, 0]) and
        df_1h.iloc[:, 0].equals(pred_df.iloc[:, 0])):
    raise ValueError("MOLECULE_NAME entries do not match across the three files.")

# -------------------- DATA PREPARATION --------------------------------
y_actual = pred_df.iloc[:, 1].values
y_pred   = pred_df.iloc[:, 2].values
residuals = y_actual - y_pred
std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
outlier_threshold = 3

# --- FEATURES: concatenate 1H and 13C ---
# assume: col.0 = MOLECULE_NAME, col.1 = LABEL, rest are features
features_1h  = df_1h.iloc[:, 2:]
features_13c = df_13c.iloc[:, 2:]
X = pd.concat([features_1h, features_13c], axis=1).values

# Remove features with zero variance
stds = X.std(axis=0)
nonzero_std_mask = stds > 1e-10
X = X[:, nonzero_std_mask]

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Leverage (hat matrix)
H = X @ np.linalg.pinv(X.T @ X) @ X.T
leverage = np.diag(H)

# Limits
n, p = X.shape
leverage_limit = 3 * p / n

# --------------------------- CREATE TABLE ------------------------------
full_df = pd.DataFrame({
    "MOLECULE_NAME": df_1h.iloc[:, 0],
    "y_actual": y_actual,
    "y_pred": y_pred,
    "residual": residuals,
    "leverage": leverage,
    "std_residual": std_residuals,
})

# ------------------------ OUTLIER DETECTION ----------------------------
outliers = full_df[
    (np.abs(full_df["std_residual"]) > outlier_threshold) |
    (full_df["leverage"] > leverage_limit)
].copy()

outliers["residual_flag"] = np.where(np.abs(outliers["std_residual"]) > outlier_threshold,
                                     "std_residual", "")
outliers["leverage_flag"] = np.where(outliers["leverage"] > leverage_limit,
                                     "leverage", "")
outliers["note"] = outliers[["residual_flag", "leverage_flag"]].agg(" ".join, axis=1).str.strip()

# -------------------------------- SAVE -----------------------------------
full_df.to_csv(os.path.join(OUTPUT_DIR, FULL_DATA_FILE), index=False)
outliers.to_csv(os.path.join(OUTPUT_DIR, OUTLIERS_FILE), index=False)

# -------------------------------- PLOTTING -------------------------------
plt.figure(figsize=(12, 10))

# --- axis ranges (if provided) ---
if XLIM is not None:
    plt.xlim(*XLIM)
if YLIM is not None:
    plt.ylim(*YLIM)

# Masks for inliers and outliers
outlier_mask = (np.abs(full_df["std_residual"]) > outlier_threshold) | (full_df["leverage"] > leverage_limit)
inlier_mask  = ~outlier_mask

# Points
plt.scatter(full_df.loc[inlier_mask,  "leverage"],
            full_df.loc[inlier_mask,  "std_residual"],
            marker='x', color='orange', s=50, label='Regular points')

plt.scatter(full_df.loc[outlier_mask, "leverage"],
            full_df.loc[outlier_mask, "std_residual"],
            marker='x', color='purple', s=50, label='Outliers')

# Lines
plt.axhline(y=3, color='red', linestyle='--', linewidth=1.5, label='±3 Standardized Residuals')
plt.axhline(y=-3, color='red', linestyle='--', linewidth=1.5)
plt.axvline(x=leverage_limit, color='navy', linestyle='--', linewidth=1.5, label='Leverage Threshold')

# Titles and style
plt.title("Dual MLP with 1H, 13C Datasets at pH 10.5", fontsize=24, fontweight='bold')

if SHOW_LABELS_X:
    plt.xlabel("Leverage", fontsize=22, fontweight='bold')
    plt.xticks(fontsize=20)
else:
    plt.xlabel("")
    ax = plt.gca()
    ax.tick_params(axis='x', which='both', labelbottom=False)


if SHOW_LABELS_Y:
    plt.ylabel("Standardized Residuals", fontsize=22, fontweight='bold')
    plt.yticks(fontsize=20)
else:
    plt.ylabel("")
    ax = plt.gca()
    ax.tick_params(axis='y', which='both', labelleft=False)

plt.grid(False)

# Legend
if SHOW_LEGEND:
    legend = plt.legend(fontsize=16, frameon=True, shadow=False)
    # for text in legend.get_texts():
    #     text.set_fontweight('bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, PLOT_FILE), dpi=100)
plt.show()

