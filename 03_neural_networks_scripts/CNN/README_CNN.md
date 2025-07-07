
# NMR-based logD Prediction — CNN 1D & CNN 2D Architectures (PyTorch)

This repository section contains **two PyTorch-based convolutional neural network models** designed to predict CHI logD values from paired ¹H and ¹³C NMR spectra. The models use either 1D or 2D convolutional processing depending on how spectral inputs are represented and structured.

---

## 🧠 Model Architectures

### 🔹 CNN_1D_pytorch_1.py
- **Architecture**: 1D Convolutional Neural Network
- **Input shape**: `(n_samples, 2, 200)` → two NMR spectra as 1D signals
- **Design**:
  - Treats ¹H and ¹³C spectra as separate channels in 1D space
  - Uses multiple `Conv1d` layers followed by pooling, dropout, and dense layers
  - Optimized via **Optuna** with MLflow tracking
- **Use case**: Ideal when ¹H and ¹³C can be interpreted as dual-channel temporal signals

### 🔹 CNN_2D_pytorch_1.py
- **Architecture**: 2D Convolutional Neural Network
- **Input shape**: `(n_samples, 2, 1, 200)` → 2-channel, 1-row "image" format
- **Design**:
  - Input reshaped to mimic image-like spatial structure
  - Uses `Conv2d`, pooling, dropout, and fully connected layers
  - Also fully integrated with Optuna and MLflow
- **Use case**: Suitable for spatial fusion of ^1H and ^13C data when cross-spectrum patterns matter

---

## 🏷️ Tags Configuration

The following helper files define MLflow tags for different run stages and prediction contexts:

- `tags_config_pytorch_CNN_1D.py`:
  - Tags model as `CNN 1D`, architecture `Pytorch`, property `CHIlogD`
  - Differentiates between hyperparameter optimization (`stage: Optuna HP`) and training (`stage: training`)

- `tags_config_pytorch_CNN_2D.py`:
  - Similar logic but with `CNN 2D` architecture label
  - Enables easy filtering and reproducibility in MLflow UI

You can import these tags into your script with:
```python
from tags_config_pytorch_CNN_1D import mlflow_tags1, mlflow_tags2
```

---

## 🚀 How to Run

```bash
python CNN_1D_pytorch_1.py \
  --csv1 path/to/1H.csv \
  --csv2 path/to/13C.csv \
  --experiment_name NMR_CNN1D_Optuna \
  --n_trials 200

python CNN_2D_pytorch_1.py \
  --csv1 path/to/1H.csv \
  --csv2 path/to/13C.csv \
  --experiment_name NMR_CNN2D_Optuna \
  --n_trials 200
```

---

## 📦 Outputs & Logging

Both scripts generate:
- MLflow run with full metrics and artifacts
- Trained `.pth` model file
- Prediction CSVs with actual vs. predicted logD
- Plots: prediction scatter, error distribution, parameter importance

---

## 🧪 Requirements

```bash
pip install torch optuna mlflow pandas scikit-learn matplotlib
```

---

## 📞 Author

Developed by **aleniak**  
Specialized in NMR-based machine learning and predictive modeling of physicochemical properties.
