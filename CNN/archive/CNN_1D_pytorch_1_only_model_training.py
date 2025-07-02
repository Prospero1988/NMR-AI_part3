# -*- coding: utf-8 -*-
"""
Simplified script to train CNN models using hyperparameters from summary TXT files for multiple CSV files in a directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import mlflow
import mlflow.pytorch
import importlib.util
import logging

# Set random seed for reproducibility
SEED = 88
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
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
        x = x.type(torch.float32)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def parse_params_from_summary(summary_file_path):
    params = {}
    with open(summary_file_path, 'r') as f:
        lines = f.readlines()

    in_params_section = False
    for line in lines:
        line = line.strip()
        if line == 'Best parameters:':
            in_params_section = True
            continue
        if line == '' or line.endswith(':'):
            continue
        if in_params_section:
            if line == '10CV Metrics:':
                break
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Convert value to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        if '.' in value or 'e' in value.lower():
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string if cannot convert
                params[key] = value
    return params

# Now, we define the WrappedModel
class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.type(torch.float32)
        with torch.no_grad():  # Wyłączenie śledzenia gradientów
            return self.model(x)

def train_model(csv_path, params, csv_name, mlflow_tags2):
    with mlflow.start_run(run_name=csv_name):
        # Set tags
        mlflow.set_tags(mlflow_tags2)

        # Load data
        X_full, y_full = load_data(csv_path, target_column_name='LABEL')

        # Log parameters
        mlflow.log_param('csv_file', csv_name)
        mlflow.log_params(params)

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
            epoch_loss = 0
            num_batches = 0
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

                epoch_loss += loss.item()
                num_batches += 1

            if use_scheduler:
                scheduler.step(loss)

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            mlflow.log_metric('train_loss', avg_epoch_loss, step=epoch)

        # Save the model locally (optional)
        model_file_name = f"{csv_name}_model.pth"
        torch.save(model.state_dict(), model_file_name)
        print(f"Model saved as {model_file_name}")

        # Move model to CPU and set to float32
        model.to('cpu')
        model = model.float()

        # Prepare input_example as numpy.ndarray
        input_example = X_full[0:1].astype(np.float32)  # numpy.ndarray

        # Create wrapped model
        wrapped_model = WrappedModel(model)

        # Set logging level to DEBUG
        logging.getLogger("mlflow").setLevel(logging.DEBUG)

        # Define conda_env
        conda_env = {
            'channels': ['defaults'],
            'dependencies': [
                f'python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
                'pip',
                {
                    'pip': [
                        f'torch=={torch.__version__}',
                        'mlflow',
                    ],
                },
            ],
            'name': 'mlflow-env'
        }

        # Log the wrapped model to MLflow
        mlflow.pytorch.log_model(wrapped_model, "model", conda_env=conda_env, input_example=input_example)
        print(f"Model logged to MLflow.")

def main():
    parser = argparse.ArgumentParser(description='Train CNN models using hyperparameters from summary TXT files.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the directory containing CSV files.')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the MLflow experiment.')

    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    csv_directory = args.csv_path
    params_directory = os.getcwd()

    # Check if directories exist
    if not os.path.isdir(csv_directory):
        print(f"The path {csv_directory} is not a directory.")
        sys.exit(1)

    if not os.path.isdir(params_directory):
        print(f"The path {params_directory} is not a directory.")
        sys.exit(1)

    # Import mlflow_tags2 from tags_config_pytorch_CNN_1D.py
    tags_file_path = os.path.join(params_directory, 'tags_config_pytorch_CNN_1D.py')
    if not os.path.exists(tags_file_path):
        print(f"Tags configuration file 'tags_config_pytorch_CNN_1D.py' not found in {params_directory}.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("tags_config_pytorch_CNN_1D", tags_file_path)
    tags_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tags_module)
    mlflow_tags2 = tags_module.mlflow_tags2

    # Get list of CSV files
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files in directory {csv_directory}")
        sys.exit(1)

    for csv_file in csv_files:
        csv_path = os.path.join(csv_directory, csv_file)
        csv_name = os.path.splitext(csv_file)[0]

        # Find corresponding summary TXT file
        params_file_name = f"{csv_name}_summary.txt"
        params_file_path = os.path.join(params_directory, params_file_name)

        if not os.path.exists(params_file_path):
            print(f"Summary TXT file {params_file_name} not found for {csv_file}. Skipping...")
            continue

        # Parse hyperparameters from summary TXT file
        params = parse_params_from_summary(params_file_path)

        print(f"\nProcessing file: {csv_file}")

        try:
            train_model(csv_path, params, csv_name, mlflow_tags2)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

if __name__ == '__main__':
    main()
