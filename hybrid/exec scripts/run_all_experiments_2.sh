#!/bin/bash
# Skrypt uruchamiający eksperymenty dla różnych modeli sieci neuronowych dla widm NMR
# W razie błędu skrypt zatrzyma się (set -e).

# set -e

echo "Rozpoczynam eksperymenty dla modelu CNN-only alt, czyli jeden kanał, wysokosć 2"

# Eksperymenty dla modelu CNN-only alt, czyli jeden kanał, wysokosć 2
python3 cnn_only_04_alt.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 cnn_only_04_alt.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 cnn_only_04_alt.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000

echo "Rozpoczynam eksperymenty dla modelu 1D transformer widma 1H"

# Eksperymenty dla modelu modelu 1D transformer widma 1H
python3 trans_1D_1H.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 trans_1D_1H.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 trans_1D_1H.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000

echo "Rozpoczynam eksperymenty dla modelu 1D transformer widma 13C"

# Eksperymenty dla modelu 1D transformer widma 13C
python3 trans_1D_13C.py --path_13c 'inputs/chilogd026_1H_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 trans_1D_13C.py --path_13c 'inputs/chilogd074_1H_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 trans_1D_13C.py --path_13c 'inputs/chilogd105_1H_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000

echo "Rozpoczynam eksperymenty dla modelu CNN-only alt, czyli jeden kanał, wysokosć 2"

# Eksperymenty dla modelu 1D transformer widm [1H,13C] tensor wysokości 2, jeden kanał.
python3 trans_1D_1H-13C.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 trans_1D_1H-13C.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000
python3 trans_1D_1H-13C.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_005' --n_trials 4000


echo "Wszystkie eksperymenty zakończone!"
