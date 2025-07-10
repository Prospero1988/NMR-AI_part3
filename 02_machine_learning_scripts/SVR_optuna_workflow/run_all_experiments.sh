#!/bin/bash
# Script to run experiments for neural networks models.

#echo "Starting experiments for XGBoost Models"
#python3 XGB_main.py --csv_path "./input" --experiment_name "CNN_1D_pytorch" --n_trials 20

echo ""

echo "Starting experiments for SVR Models"
python3 SVR_main.py --csv_path "./input" --experiment_name "CNN_1D_pytorch" --n_trials 20

# All experiments for all models completed.
echo ""
echo "All experiments for all models completed."