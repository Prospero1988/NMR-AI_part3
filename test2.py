import mlflow
import mlflow.keras
from tensorflow.keras import layers, models
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import TensorSpec

# Definicja prostego modelu
model = models.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(5, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Dane przykładowe
X_dummy = np.random.rand(10, 10).astype(np.float32)
y_dummy = np.random.rand(10, 1).astype(np.float32)

# Trenowanie modelu
model.fit(X_dummy, y_dummy, epochs=1, verbose=0)

# Przygotowanie input_example
input_example = X_dummy[:2]

# Przewidywania
predictions = model.predict(input_example).astype(np.float32)

# Definicja ModelSignature
signature = ModelSignature(
    inputs=TensorSpec(shape=(None, 10), dtype=np.float32),
    outputs=TensorSpec(shape=(None, 1), dtype=np.float32)
)

# Logowanie modelu
mlflow.set_experiment("Test_Experiment")
with mlflow.start_run():
    mlflow.keras.log_model(
        model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )
