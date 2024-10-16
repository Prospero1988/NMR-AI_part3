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

    # Call hyperparameter optimization module
    subprocess.run(['python', 'module1.py', input_directory, '--experiment_name', experiment_name])

    # Call training module
    subprocess.run(['python', 'module2.py', input_directory, '--experiment_name', experiment_name])

if __name__ == "__main__":
    main()
