# tags_config_pytorch.py

mlflow_tags1 = {
    "architecture": "Pytorch",
    "model": "MLP",
    "stage": "Optuna HP",
    "author": "aleniak",
    "opt trials": "2000",
    "property": "CHIlogD"
}

mlflow_tags2 = {
    "architecture": "Pytorch",
    "predictor": "1H_conc_13C",
    "model": "MLP",
    "stage": "evaluation",
    "author": "aleniak",
    "opt trials": "2000",
    "evaluation": "10CV",
    "property": "CHIlogD"
}

mlflow_tags3 = {
    "architecture": "Pytorch",
    "predictor": "1H_conc_13C",
    "model": "MLP",
    "stage": "training",
    "author": "aleniak",
    "opt trials": "2000",
    "evaluation": "10CV",
    "property": "CHIlogD"
}