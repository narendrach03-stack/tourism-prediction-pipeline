import os
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlops-training-experiment")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

Xtrain = pd.get_dummies(Xtrain)
Xtest = pd.get_dummies(Xtest)
Xtrain, Xtest = Xtrain.align(Xtest, join='left', axis=1, fill_value=0)

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

with mlflow.start_run(run_name="model_training"):
    model.fit(Xtrain, ytrain)
    
    ypred = model.predict(Xtest)
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, average="weighted")
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(model, model_path)
    
    print(f"Model trained successfully. Accuracy: {acc:.4f}, F1: {f1:.4f}")
