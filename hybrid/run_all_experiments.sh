#!/bin/bash
# Skrypt uruchamiający eksperymenty dla modeli hybrydowego (CNN+MLP) oraz CNN-only.
# W razie błędu skrypt zatrzyma się (set -e).

# set -e

echo "Rozpoczynam eksperymenty dla modelu hybrydowego (CNN+MLP)..."

# Eksperymenty dla hybrydowego modelu (CNN+MLP)
python3 hybrid_04.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --path_fp 'inputs/chilogd026-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 hybrid_04.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --path_fp 'inputs/chilogd074-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 hybrid_04.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --path_fp 'inputs/chilogd105-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000

echo "Eksperymenty dla modelu hybrydowego zakończone."

echo "Rozpoczynam eksperymenty dla modeli hybrydowych 2xMLP"

# Eksperymenty dla hybrydowego 2xMLP 1H
python3 MLPx2_1H.py --path_nmr 'inputs/chilogd026_1H_ML_input.csv' --path_fp 'inputs/chilogd026-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 MLPx2_1H.py --path_nmr 'inputs/chilogd074_1H_ML_input.csv' --path_fp 'inputs/chilogd074-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 MLPx2_1H.py --path_nmr 'inputs/chilogd105_1H_ML_input.csv' --path_fp 'inputs/chilogd105-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000

echo "Eksperymenty dla modelu hybrydowego 2xMLP 1H zakończone."

# Eksperymenty dla hybrydowego 2xMLP 13C
python3 MLPx2_13C.py --path_nmr 'inputs/chilogd026_13C_ML_input.csv' --path_fp 'inputs/chilogd026-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 MLPx2_13C.py --path_nmr 'inputs/chilogd074_13C_ML_input.csv' --path_fp 'inputs/chilogd074-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 MLPx2_13C.py --path_nmr 'inputs/chilogd105_13C_ML_input.csv' --path_fp 'inputs/chilogd105-RDKIT_FP_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000

echo "Eksperymenty dla modelu hybrydowego 2xMLP 1H zakończone."

#echo "Wszystkie eksperymenty zakończone!"

echo "Rozpoczynam eksperymenty dla modelu CNN-only (bez MLP)..."

# Eksperymenty dla modelu CNN-only (bez MLP)
python3 cnn_only_04.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 cnn_only_04.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 cnn_only_04.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000

echo "Wszystkie eksperymenty zakończone!"
