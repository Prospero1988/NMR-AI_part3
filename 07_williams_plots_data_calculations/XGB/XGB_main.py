# main_script.py

import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Main script to run hyperparameter optimization and training')
    parser.add_argument('input_directory', type=str, help='Path to the input directory containing CSV files')
    parser.add_argument('--experiment_name', type=str, default='Default', help='Name of the MLflow experiment')
    args = parser.parse_args()

    input_directory = args.input_directory
    experiment_name = args.experiment_name

    # Call training module with williams lots data creation
    subprocess.run(['python', 'XGB_module4.py', input_directory, '--experiment_name', experiment_name])

if __name__ == "__main__":
    main()
