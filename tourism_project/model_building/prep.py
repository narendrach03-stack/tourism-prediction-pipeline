import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

# Set local MLflow tracking server (started by GitHub Actions workflow)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mlops-training-experiment")

# Relative path for GitHub Actions workspace
RAW_PATH = "tourism_project/data/tourism.csv"

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Ensure tourism.csv is in tourism_project/data/")

df = pd.read_csv(RAW_PATH)

# Drop index/identifier columns if present
cols_to_drop = [col for col in ["Unnamed: 0", "CustomerID"] if col in df.columns]
df = df.drop(columns=cols_to_drop)

target_col = "ProdTaken"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save preprocessed splits to workspace root for downstream jobs
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation complete. Saved Xtrain, Xtest, ytrain, ytest CSVs.")

with mlflow.start_run(run_name="data_preparation"):
    mlflow.log_metric("train_samples", Xtrain.shape[0])
    mlflow.log_metric("test_samples", Xtest.shape[0])
