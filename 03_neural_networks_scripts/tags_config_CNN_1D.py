# tags_config_pytorch.py

mlflow_tags1 = {
    "architecture": "Pytorch",
    "model": "CNN 1D",
    "stage": "Optuna HP",
    "author": "aleniak",
    "opt trials": "1000",
    "property": "logD_7.4"
}

mlflow_tags2 = {
    "architecture": "Pytorch",
    "predictor": "1H|13C",
    "model": "CNN",
    "stage": "evaluation",
    "author": "aleniak",
    "opt trials": "1000",
    "evaluation": "10CV",
    "property": "logD_7.4"
}

mlflow_tags3 = {
    "architecture": "Pytorch",
    "predictor": "1H|13C",
    "model": "CNN",
    "stage": "training",
    "author": "aleniak",
    "opt trials": "1000",
    "evaluation": "10CV",
    "property": "logD_7.4"
}
