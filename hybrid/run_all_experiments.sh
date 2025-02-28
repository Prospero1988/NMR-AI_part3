#!/bin/bash
# Skrypt uruchamiający eksperymenty dla modeli hybrydowego (CNN+MLP) oraz CNN-only.
# W razie błędu skrypt zatrzyma się (set -e).

set -e

echo "Rozpoczynam eksperymenty dla modelu hybrydowego (CNN+MLP)..."

# Eksperymenty dla hybrydowego modelu (CNN+MLP)
python3 hybrid_04.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --path_fp 'inputs/chilogd026_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_002' --n_trials 8000
python3 hybrid_04.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --path_fp 'inputs/chilogd074_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_002' --n_trials 8000
python3 hybrid_04.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --path_fp 'inputs/chilogd105_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_002' --n_trials 8000

echo "Eksperymenty dla modelu hybrydowego zakończone."

echo "Rozpoczynam eksperymenty dla modelu CNN-only (bez MLP)..."

# Eksperymenty dla modelu CNN-only (bez MLP)
python3 cnn_only_04.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_002' --n_trials 8000
python3 cnn_only_04.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_002' --n_trials 8000
python3 cnn_only_04.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_002' --n_trials 8000

echo "Wszystkie eksperymenty zakończone!"
