#!/bin/bash
# Skrypt uruchamiający eksperymenty dla modeli hybrydowego (CNN+MLP) oraz CNN-only.
# W razie błędu skrypt zatrzyma się (set -e).

# set -e

echo "Rozpoczynam eksperymenty dla modelu hybrydowego (MLP)..."

# Eksperymenty dla hybrydowego modelu (CNN+MLP)
python3 MLPx2_1H_13C.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 2000
python3 MLPx2_1H_13C.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 2000
python3 MLPx2_1H_13C.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 2000

echo "Eksperymenty dla modelu hybrydowego zakończone."
