# main.py

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Main script to run hyperparameter optimization and model training.')
    parser.add_argument('input_directory', type=str, help='Path to the input directory containing CSV files')
    parser.add_argument('--experiment_name', type=str, default='Default', help='Name of the MLflow experiment')
    args = parser.parse_args()

    input_directory = args.input_directory
    experiment_name = args.experiment_name

    # Call module1.py
    #subprocess.run(['python', 'module1.py', input_directory, '--experiment_name', experiment_name])

    # Call module2.py 10CV evaluation
    subprocess.run(['python', 'module2.py', input_directory, '--experiment_name', experiment_name])

    # Call module3.py LOO evaluation
    subprocess.run(['python', 'module3.py', input_directory, '--experiment_name', experiment_name])

if __name__ == "__main__":
    main()
