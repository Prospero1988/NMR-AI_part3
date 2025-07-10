#!/bin/bash
# Script to run experiments for neural networks models.

echo "Starting experiments CNN 1D Models"
python3 CNN_1D_pytorch.py --csv_path "./input" --experiment_name "CNN_1D_pytorch" --n_trials 20

echo ""

echo "Starting experiments MLP 1D Models"
python3 MLP_1D_pytorch.py --csv_path "./input" --experiment_name "CNN_1D_pytorch" --n_trials 20

# All experiments for all models completed.
echo ""
echo "All experiments for all models completed."