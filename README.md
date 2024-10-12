
# Machine Learning Hyperparameter Optimization with CUDA and Optuna

This repository contains two Python scripts designed for optimizing machine learning models using Optuna, MLflow, and CUDA acceleration.

## Files
- `SVR_ML_Optuna_CUDA_v2.py`: This script performs Support Vector Regression (SVR) hyperparameter optimization using Optuna, MLflow for logging, and CUDA for GPU acceleration. It includes data loading, model training, and evaluation routines.
- `XGB_ML_Optuna_CUDA.py`: This script focuses on XGBoost model optimization using Optuna and MLflow, also leveraging CUDA for accelerated training.
- `conda_environment.yml`: A YAML file describing the conda environment required to run the scripts, including necessary dependencies.

### Key Features
- **Hyperparameter Optimization**: Both scripts use Optuna to optimize key model parameters.
- **CUDA Acceleration**: Training is accelerated using NVIDIA GPUs through CuPy for SVR and XGBoost's GPU-based algorithms.
- **MLflow Integration**: All experiments are tracked using MLflow, including logging of parameters, metrics, and artifacts.
- **Cross-Validation**: Both scripts employ cross-validation for evaluating model performance.
- **Result Logging**: Both models log performance metrics such as RMSE, MAE, and R², along with hyperparameter search spaces and importance.
- **Feature Importances and Learning Curves**: XGBoost script logs feature importances and learning curves to MLflow.

## Installation

### Prerequisites
Ensure you have the following installed:
- CUDA-enabled GPU
- Python 3.x
- The required Python packages listed in `conda_environment.yml`

You can create the conda environment by running:
```bash
conda env create -f conda_environment.yml
conda activate optuna_new
```

This will install the necessary packages, including:
- `cupy`, `optuna`, `mlflow`, `pandas`, `joblib`, `scikit-learn`, `matplotlib`, `xgboost`, and `cuml`.

### Environment
Make sure your environment is set up for CUDA, and that the necessary drivers and libraries are installed (e.g., `nvidia-cuda-toolkit`).

## Usage

### Running SVR Optimization (`SVR_ML_Optuna_CUDA_v2.py`)
```bash
python SVR_ML_Optuna_CUDA_v2.py /path/to/your/input_directory --experiment_name 'SVR_Experiment'
```
This script will search for CSV files in the input directory, load the data, and begin optimizing SVR hyperparameters. The results, including optimized models and metrics, will be logged using MLflow.

### Running XGBoost Optimization (`XGB_ML_Optuna_CUDA.py`)
```bash
python XGB_ML_Optuna_CUDA.py /path/to/your/input_directory --experiment_name 'XGBoost_Experiment'
```
Similar to the SVR script, this will load data from CSV files and optimize XGBoost hyperparameters. Results are logged using MLflow.

### Input Files
The input directory should contain CSV files with the following format:
- Column 1: Sample identifier
- Column 2: Target variable (e.g., continuous regression target)
- Columns 3+: Feature vectors

## Model Evaluation

Both scripts perform cross-validation and log various evaluation metrics, including:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Pearson correlation coefficient

The XGBoost script additionally logs:
- Feature importances
- Learning curves

## Logging and Artifacts
MLflow is used to log:
- Hyperparameter search space and importance
- Metrics (RMSE, MAE, R², etc.)
- Model artifacts (e.g., trained models, feature importance plots)

## License
This project is licensed under the MIT License.
