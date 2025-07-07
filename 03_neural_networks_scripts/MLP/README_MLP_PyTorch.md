
# MLP-Based logD Prediction using PyTorch

This repository contains a PyTorch-based MLP (Multilayer Perceptron) model for predicting CHI logD values using numerical input features. The model is designed for flexibility and includes automated hyperparameter optimization and evaluation.

---

## 📁 Files

- `MLP_GPU_pytorch_simple_2.py`: Main training script using Optuna for hyperparameter tuning and MLflow for tracking.
- `tags_config_pytorch.py`: Helper module for consistent MLflow tag configuration.

---

## 🧠 Model Overview

- Architecture: Multilayer Perceptron with configurable number of layers, units, activation functions, and regularization.
- Optimizers: Supports `Adam`, `SGD`, `RMSProp`, each with tunable learning rate and momentum/betas.
- Regularization: L1, L2, or none, tunable via Optuna.
- Weight initialization: `xavier`, `kaiming`, or `normal`.
- EarlyStopping, learning rate schedulers, and batch normalization are available and optionally tunable.
- Input: Numerical vectors extracted from `.csv` files.

---

## 📊 Input Format

The script expects `.csv` files located in the specified directory, each structured as:

```
ID,F1,F2,...,Fn,LABEL
```

- First column: identifier (ignored)
- Last column: `LABEL` – the target regression value (logD)
- All other columns: numerical features

The number of features is dynamically inferred.

---

## 🚀 Running the Script

```bash
python MLP_GPU_pytorch_simple_2.py \
    --csv_path path/to/csv_directory \
    --experiment_name MLP_logD_experiment
```

### Arguments

- `--csv_path`: Path to the directory containing input `.csv` files.
- `--experiment_name`: Name used in MLflow for the experiment.

---

## 🧪 Training Workflow

1. **Hyperparameter Optimization (Optuna)**  
   Uses 3-fold cross-validation with RMSE as the primary objective.

2. **Cross-Validation Evaluation**  
   The best trial is re-trained and evaluated using 10-fold CV.

3. **Final Training on Full Data**  
   The final model is trained on the full dataset with early stopping.

---

## 📦 Output Artifacts

For each run, the following are generated and logged via MLflow:

- Optimized model (`.pth`)
- Cross-validation predictions (`*_predictions.csv`)
- Summary report (`*_summary.txt`)
- RMSE, MAE, R², and Pearson metrics
- Parameter importance plot (`param_importance.png`)
- Real vs. predicted plot (`pred_vs_actual_cv.png`)

---

## 🏷 MLflow Tags

Tags are defined in `tags_config_pytorch.py` and cover:
- Architecture type
- Target property (`CHIlogD`)
- Stage: `Optuna HP`, `training`, `final training`

Tags enable consistent filtering and grouping in MLflow UI.

---

## 📌 Requirements

- `pytorch`
- `optuna`
- `mlflow`
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`

---

## 🔬 Author

Developed by **aleniak** for NMR-based predictive modeling of physicochemical properties.

