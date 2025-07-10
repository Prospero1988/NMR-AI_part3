#!/bin/bash
# Script to run experiments for hybrid models.

echo "Starting experiments for the hybrid model (MLP Dual-Stream)..."

# Experiments for the hybrid model (MLP Dual-Stream)
python3 mlp_dualstream_1H_13C.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000
python3 mlp_dualstream_1H_13C.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000
python3 mlp_dualstream_1H_13C.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000

echo "Starting experiments for the hybrid model (CNN Dual-Stream)..."

# Experiments for the hybrid model (CNN Dual-Stream)
python3 cnn_dualstream_1H_13C.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000
python3 cnn_dualstream_1H_13C.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000
python3 cnn_dualstream_1H_13C.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000

echo "Starting experiments for the 2D CNN Stacked Vectors model..."

# Experiments for the 2D CNN Stacked Vectors model
#python3 cnn_2d_stacked_1H_13C.py --path_1h 'inputs/chilogd026_1H_ML_input.csv' --path_13c 'inputs/chilogd026_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000
#python3 cnn_2d_stacked_1H_13C.py --path_1h 'inputs/chilogd074_1H_ML_input.csv' --path_13c 'inputs/chilogd074_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000
#python3 cnn_2d_stacked_1H_13C.py --path_1h 'inputs/chilogd105_1H_ML_input.csv' --path_13c 'inputs/chilogd105_13C_ML_input.csv' --experiment_name 'hybrid_neural_network_001' --n_trials 2000

# All experiments for all models completed.
echo "All experiments for all models completed."