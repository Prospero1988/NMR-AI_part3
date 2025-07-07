# 🧪 NMR-based logD Prediction using Deep Learning

This repository contains three deep learning models for predicting the CHI logD property from **¹H and ¹³C NMR spectra**. The models differ in architecture and input representation strategies, allowing for benchmarking and analysis of different fusion techniques for spectral data.

---

## 📌 Key Features

- Predicts **CHI logD** using **1D NMR spectra** (¹H + ¹³C)
- Supports three architectures: **MLP Dual-Stream**, **CNN Dual-Channel**, and **CNN Stacked-Spectra**
- Full integration with **Optuna** for hyperparameter optimization and **MLflow** for experiment tracking
- Robust evaluation using **3-fold CV** during optimization and **10-fold CV** during final model assessment
- Automatically logs metrics, predictions, plots, and models
- Compatible with both **CPU and CUDA-enabled GPUs**

---

## 🧠 Model Architectures

| Script                             | Architecture               | Input Shape             | Description                                                                 |
|-----------------------------------|----------------------------|--------------------------|-----------------------------------------------------------------------------|
| `mlp_dualstream_1H_13C.py`        | **MLP Dual-Stream**        | `(B, 2, 200)`            | Two separate MLP branches for ¹H and ¹³C, merged into a shared embedding    |
| `cnn_2d_dualchannel_1H_13C.py`    | **CNN 2D Dual-Channel**    | `(B, 2, 1, 200)`         | Treats ¹H and ¹³C as separate channels, similar to RGB image channels       |
| `cnn_2d_stacked_1H_13C.py`        | **CNN 2D Stacked-Spectra** | `(B, 1, 2, 200)`         | Combines ¹H and ¹³C vertically as a 2×200 image, processed spatially        |

> All architectures output a single predicted scalar per sample (`logD`) and include regularization layers (Dropout, BatchNorm) and optional non-linearities (e.g., SiLU).

---

## 📂 Input Data Format

Each model takes two CSV files as input:

- `--path_1h` → CSV file with ¹H NMR spectrum (200 features)
- `--path_13c` → CSV file with ¹³C NMR spectrum (200 features)

Each file must have the following format:
```
MOLECULE_NAME,LABEL,f_0,f_1,...,f_199
```
- `MOLECULE_NAME`: Unique ID of the molecule
- `LABEL`: Target value (CHI logD)
- `f_0` to `f_199`: Intensity values of the NMR spectrum

---

## 🚀 How to Run

Each script can be run independently with the same CLI interface.

Example (MLP Dual-Stream):
```bash
python mlp_dualstream_1H_13C.py \
  --path_1h data/1H.csv \
  --path_13c data/13C.csv \
  --experiment_name MLP_Dual_Experiment \
  --n_trials 50 \
  --epochs_10cv 100
```

Replace the script name to run the CNN variants:
- `cnn_2d_dualchannel_1H_13C.py`
- `cnn_2d_stacked_1H_13C.py`

---

## ⚙️ Dependencies

You can install the required packages via `pip`:

```bash
pip install torch optuna mlflow pandas matplotlib scikit-learn
```

For CUDA support, ensure your environment has the correct PyTorch version with GPU support:
https://pytorch.org/get-started/locally/

---

## 🧪 Training & Evaluation Workflow

Each run includes the following stages:

1. **Data Loading**: Reads and merges ¹H and ¹³C spectra
2. **Optimization**: Optuna performs hyperparameter search using 3-fold CV
3. **Final Evaluation**: Best model is retrained and evaluated using 10-fold CV
4. **MLflow Logging**:
   - All parameters, metrics, and artifacts
   - RMSE, MAE, R², Pearson metrics
   - Trial history and parameter importance plots
   - Predicted vs true values and error scatterplots

Final model is trained on the full dataset with early stopping and logged as an MLflow artifact.

---

## 📊 Evaluation Metrics

For every model, the following are calculated:

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Pearson Correlation Coefficient**

These are logged with means and standard deviations across CV folds.

---

## 📁 Output Directory Structure

Each script creates a result directory based on the input file name and model type:

```
<DATASET>-results-<model_type>/
├── optuna_trials_*.csv
├── metrics_*.csv
├── cv_predictions_*.csv
├── real_vs_pred_plot_*.png
├── error_plot_*.png
├── param_importances_*.json/png
└── model_<type>/ (MLflow model)
```

---

## 🔍 Model Comparison

| Architecture        | Separation Strategy    | Complexity | Best For…                        |
|---------------------|------------------------|------------|----------------------------------|
| MLP Dual-Stream     | Parallel, learned merge | ★★★☆☆      | Clear separation of ¹H / ¹³C     |
| CNN Dual-Channel    | Channel-based fusion   | ★★★★☆      | Learning local spatial patterns  |
| CNN Stacked-Spectra | Spatial stacking       | ★★★★☆      | Capturing cross-signal features |

---

## 🧩 Extension Ideas

- Add third input: **molecular fingerprints (e.g. ECFP4)**
- Switch to **multi-task regression** (e.g., predict multiple ADME properties)
- Use **attention-based architectures** to weigh ¹H/¹³C contribution dynamically
- Implement **residual connections** or **Transformer-based NMR models**

---

## 💬 Credits & Author

Developed by **Arek**, chemist and NMR specialist  
Architecture design and optimization powered by **PyTorch + Optuna + MLflow**

---

> *“The spectra don't lie — they whisper secrets of solubility if you learn to listen.”*
