# main.py

import argparse
import subprocess

def main():
    """
    Main function to run hyperparameter optimization and model training scripts.

    Parses command-line arguments for input directory and experiment name,
    then calls module1.py for Optuna optimization and module2.py for 10-fold CV evaluation.
    """
    parser = argparse.ArgumentParser(
        description='Main script to run hyperparameter optimization and training'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to the input directory containing CSV files'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='Default',
        help='Name of the MLflow experiment'
    )

    parser.add_argument(
        '--n_trials', 
        type=int, 
        required=False, 
        default=1000, 
        help='Number of trials for Optuna hyperparameter optimization')

    args = parser.parse_args()
    input_directory = args.csv_path
    experiment_name = args.experiment_name
    n_trials = args.n_trials


    # Call module1.py for hyperparameter optimization with Optuna
    subprocess.run(['python', 'SVR_module1.py', input_directory, '--experiment_name', experiment_name, '--n_trials', str(n_trials)])

    # Call module2.py for 10-fold cross-validation evaluation
    subprocess.run(['python', 'SVR_module2.py', input_directory, '--experiment_name', experiment_name])

if __name__ == "__main__":
    # Entry point for the script
    main()
