import argparse
import torch
import shap
import pandas as pd
import numpy as np
import ast
from CNN_1D_pytorch import CNN
import random

def load_params(params_path):
    with open(params_path, "r") as f:
        line = f.readline().strip()
        return ast.literal_eval(line)

def main():
    parser = argparse.ArgumentParser(description="SHAP generator for CNN1D models")
    parser.add_argument('--input', required=True, help='Path to input CSV file with features')
    parser.add_argument('--model', required=True, help='Path to trained model weights (.pth)')
    parser.add_argument('--params', required=True, help='Path to model parameter file (e.g., summary .txt)')
    parser.add_argument('--output', required=True, help='Path to output CSV file for SHAP values')
    parser.add_argument('--background_size', type=int, default=100, help='Number of random samples to use as SHAP background')
    parser.add_argument('--seed', type=int, default=1988, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Wczytaj dane
    df = pd.read_csv(args.input, sep=None, engine='python')
    X = df.values.astype(np.float32)
    X_tensor = torch.tensor(X)

    # Wczytaj parametry
    params = load_params(args.params)
    cnn_params = {
        'num_filters': params['num_filters'],
        'kernel_size': params['kernel_size'],
        'stride': params['stride'],
        'padding': params['padding'],
        'dropout_rate': params['dropout_rate'],
        'hidden_size': params['hidden_size']
    }

    # Stwórz model
    input_size = X.shape[1]
    model = CNN1D(input_size=input_size, **cnn_params)
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    model.eval()

    # Losowe background z seedem
    random.seed(args.seed)
    idx = random.sample(range(len(X_tensor)), k=min(args.background_size, len(X_tensor)))
    background = X_tensor[idx]

    # SHAP
    explainer = shap.DeepExplainer(model, background.unsqueeze(1))  # [batch, 1, features]
    shap_values = explainer.shap_values(X_tensor.unsqueeze(1))

    # Zapisz SHAP do CSV
    shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_df = pd.DataFrame(shap_array.squeeze(), columns=[f"feat_{i}" for i in range(X.shape[1])])
    shap_df.to_csv(args.output, index=False)
    print(f"SHAP values saved to {args.output}")

if __name__ == "__main__":
    main()
