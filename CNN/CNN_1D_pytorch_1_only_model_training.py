# -*- coding: utf-8 -*-
"""
Simplified script to train CNN models using hyperparameters from JSON files for multiple CSV files in a directory.

Created on (Date)

@author:
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse

# Set random seed for reproducibility
SEED = 88
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Choose device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(csv_path, target_column_name='LABEL'):
    try:
        data = pd.read_csv(csv_path)
        # Drop the first column (sample names) if necessary
        data = data.drop(data.columns[0], axis=1)
        y = data[target_column_name].values
        X = data.drop(columns=[target_column_name]).values
        return X, y
    except Exception as e:
        print(f"Error loading data from {csv_path}: {e}")
        sys.exit(1)

class Net(nn.Module):
    def __init__(self, params, input_dim):
        super(Net, self).__init__()

        activation_name = params.get('activation', 'relu')
        if activation_name == 'relu':
            activation = nn.ReLU()
        elif activation_name == 'tanh':
            activation = nn.Tanh()
        elif activation_name == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name == 'leaky_relu':
            activation = nn.LeakyReLU()
        elif activation_name == 'selu':
            activation = nn.SELU()
        else:
            activation = nn.ReLU()

        self.regularization = params.get('regularization', 'none')
        if self.regularization == 'none':
            self.reg_rate = 0.0
        else:
            self.reg_rate = params.get('reg_rate', 1e-5)

        dropout_rate = params.get('dropout_rate', 0.0)
        use_batch_norm = params.get('use_batch_norm', False)

        # Convolutional layers
        num_conv_layers = params.get('num_conv_layers', 1)
        conv_layers = []
        in_channels = 1  # Input channels for Conv1d
        input_length = input_dim  # Initial input length

        for i in range(num_conv_layers):
            out_channels = params.get(f'num_filters_l{i}', 16)
            kernel_size = params.get(f'kernel_size_l{i}', 3)
            stride = params.get(f'stride_l{i}', 1)
            padding = params.get(f'padding_l{i}', 0)

            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(activation)
            if dropout_rate > 0.0:
                conv_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

            # Calculate new input length after this layer
            input_length = int((input_length + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            if input_length <= 0:
                raise ValueError('Negative or zero input length. Adjust kernel_size, stride, or padding.')

        self.conv = nn.Sequential(*conv_layers)

        # Fully connected layers
        num_fc_layers = params.get('num_fc_layers', 2)
        fc_layers = []
        in_features = in_channels * input_length

        for i in range(num_fc_layers):
            out_features = params.get(f'fc_units_l{i}', 32)
            fc_layers.append(nn.Linear(in_features, out_features))
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(activation)
            if dropout_rate > 0.0:
                fc_layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*fc_layers)

        # Initialize weights
        init_method = params.get('weight_init', 'xavier')
        self.apply(lambda m: self.init_weights(m, init_method))

    def init_weights(self, m, init_method):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_method == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def train_model(csv_path, params, csv_name):
    # Load data
    X_full, y_full = load_data(csv_path, target_column_name='LABEL')

    # Convert to tensors
    X_full = X_full.astype(np.float32)
    y_full = y_full.astype(np.float32)

    # Create model
    input_dim = X_full.shape[1]
    model = Net(params, input_dim).to(device)

    # Loss criterion
    criterion = nn.MSELoss()

    # Optimizer
    optimizer_name = params.get('optimizer', 'adam')
    learning_rate = params.get('learning_rate', 1e-3)

    if optimizer_name == 'adam':
        beta1 = params.get('adam_beta1', 0.9)
        beta2 = params.get('adam_beta2', 0.999)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    elif optimizer_name == 'sgd':
        momentum = params.get('sgd_momentum', 0.0)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adamw':
        beta1 = params.get('adamw_beta1', 0.9)
        beta2 = params.get('adamw_beta2', 0.999)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Scheduler
    use_scheduler = params.get('use_scheduler', False)
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Training parameters
    batch_size = params.get('batch_size', 32)
    epochs = params.get('epochs', 100)
    clip_grad_value = params.get('clip_grad_value', 1.0)

    # Training
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_full, dtype=torch.float32),
                                             torch.tensor(y_full, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            if batch_X.size(0) > 1:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
            else:
                continue  # Skip if batch size is 1

            # Add regularization
            if model.regularization == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + model.reg_rate * l1_norm
            elif model.regularization == 'l2':
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss = loss + model.reg_rate * l2_norm
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)

            optimizer.step()

        if use_scheduler:
            scheduler.step(loss)

    # Save the model
    model_file_name = f"{csv_name}_model.pth"
    torch.save(model.state_dict(), model_file_name)
    print(f"Model saved as {model_file_name}")

def main():
    parser = argparse.ArgumentParser(description='Train CNN models using hyperparameters from JSON files.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the directory containing CSV files.')

    args = parser.parse_args()

    csv_directory = args.csv_path
    params_directory = os.getcwd()

    # Check if directories exist
    if not os.path.isdir(csv_directory):
        print(f"The path {csv_directory} is not a directory.")
        sys.exit(1)

    if not os.path.isdir(params_directory):
        print(f"The path {params_directory} is not a directory.")
        sys.exit(1)

    # Get list of CSV files
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files in directory {csv_directory}")
        sys.exit(1)

    for csv_file in csv_files:
        csv_path = os.path.join(csv_directory, csv_file)
        csv_name = os.path.splitext(csv_file)[0]

        # Find corresponding JSON file
        params_file_name = f"{csv_name}_best_params.json"
        params_file_path = os.path.join(params_directory, params_file_name)

        if not os.path.exists(params_file_path):
            print(f"Hyperparameters JSON file {params_file_name} not found for {csv_file}. Skipping...")
            continue

        # Load hyperparameters
        with open(params_file_path, 'r') as f:
            params = json.load(f)

        print(f"\nProcessing file: {csv_file}")

        try:
            train_model(csv_path, params, csv_name)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

if __name__ == '__main__':
    main()
