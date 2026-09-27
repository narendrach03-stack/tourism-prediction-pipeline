import os
import pandas as pd
import mlflow

# Use file-based tracking to eliminate HTTP server dependency
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlops-training-experiment")

RAW_PATH = "tourism_project/data/tourism.csv"

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Ensure tourism.csv is in tourism_project/data/")

df = pd.read_csv(RAW_PATH)
print("Loaded raw dataset successfully. Shape:", df.shape)

with mlflow.start_run(run_name="data_registration"):
    mlflow.log_param("dataset_path", RAW_PATH)
    mlflow.log_metric("num_rows", df.shape[0])
    mlflow.log_metric("num_cols", df.shape[1])
    print("Logged raw dataset metadata to MLflow.")
