#!/bin/bash

echo "Starting experiments for the SHAP calculations for 1D CNN models on fused 1H|13C datasets"

# Experiments for the hybrid model (MLP Dual-Stream)
python3 1D_CNN_SHAP.py --input 'inputs/chilogd026_1H13C.csv' --model "models/CNN_1H13C_chilogd026_final_model.pth" --summary "summary/CNN_1H13C_chilogd026_final_summary.txt" --output "SHAP_CHIlogD026.csv" --bg 256 --ntest 512
python3 1D_CNN_SHAP.py --input 'inputs/chilogd074_1H13C.csv' --model "models/CNN_1H13C_chilogd074_final_model.pth" --summary "summary/CNN_1H13C_chilogd074_final_summary.txt" --output "SHAP_CHIlogD074.csv" --bg 256 --ntest 512
python3 1D_CNN_SHAP.py --input 'inputs/chilogd105_1H13C.csv' --model "models/CNN_1H13C_chilogd105_final_model.pth" --summary "summary/CNN_1H13C_chilogd105_final_summary.txt" --output "SHAP_CHIlogD105.csv" --bg 256 --ntest 512

echo "All experiments for all models completed."