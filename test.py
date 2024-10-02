# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:48:36 2024

@author: aleniak
"""
# test_gpu.py
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))
