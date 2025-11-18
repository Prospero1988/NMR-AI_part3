# tags_config_pytorch.py

mlflow_tags1 = {
    "architecture": "Pytorch",
    "predictor": "1H|13C",
    "model": "CNN",
    "stage": "Optuna HP",
    "author": "aleniak",
    "opt trials": "2000",
    "property": "pKb_RK"
}

mlflow_tags2 = {
    "architecture": "Pytorch",
    "predictor": "1H|13C",
    "model": "CNN",
    "stage": "evaluation",
    "author": "aleniak",
    "opt trials": "2000",
    "evaluation": "10CV",
    "property": "pKb_RK"
}

mlflow_tags3 = {
    "architecture": "Pytorch",
    "predictor": "1H|13C",
    "model": "CNN",
    "stage": "training",
    "author": "aleniak",
    "opt trials": "2000",
    "evaluation": "10CV",
    "property": "pKb_RK"
}
