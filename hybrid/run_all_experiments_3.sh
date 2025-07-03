#!/bin/bash
# Skrypt uruchamiający eksperymenty dla modeli hybrydowych.

echo "Rozpoczynam eksperymenty dla modelu hybrydowego (CNN Dual-Stream)..."

# Eksperymenty dla hybrydowego modelu (CNN Dual-Stream)
python3 cnn_dualstream_1H_13C.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_006' --n_trials 2000
python3 cnn_dualstream_1H_13C.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_006' --n_trials 2000
python3 cnn_dualstream_1H_13C.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_006' --n_trials 2000

echo "Eksperymenty dla modeli hybrydowych zakończone."
