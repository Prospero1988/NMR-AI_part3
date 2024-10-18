# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:10:04 2024

@author: aleniak
"""

import numpy as np
import torch
import pandas as pd

# Załaduj dane z dwóch plików CSV
data1 = pd.read_csv("data1.csv")  # Zawiera dane 1H
data2 = pd.read_csv("data2.csv")  # Zawiera dane 13C

# Zakładamy, że pierwsza kolumna to etykiety (wartość docelowa)
labels1 = data1.iloc[:, 0].values
labels2 = data2.iloc[:, 0].values

# Sprawdzamy, czy etykiety w obu plikach są takie same
assert np.array_equal(labels1, labels2), "Etykiety w obu plikach muszą być takie same!"

# Pobieramy cechy (bez pierwszej kolumny)
features1 = data1.iloc[:, 1:].values  # Wektory cech dla 1H
features2 = data2.iloc[:, 1:].values  # Wektory cech dla 13C

# Konwersja do tensora o wymiarze 3D: (n_samples, n_features, 2)
# Dodajemy nowy wymiar na końcu i łączymy dane z obu plików wzdłuż tego wymiaru
tensor = np.stack((features1, features2), axis=-1)

# Konwersja do tensora PyTorch
X = torch.tensor(tensor, dtype=torch.float32)
y = torch.tensor(labels1, dtype=torch.float32)  # Etykiety (wartości docelowe)

# Sprawdzenie rozmiaru tensora
print(X.shape)  # Oczekiwane wymiary: (n_samples, n_features, 2)
print(y.shape)  # Oczekiwane wymiary: (n_samples,)
