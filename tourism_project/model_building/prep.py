import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
RAW_PATH = "/content/tourism_project/tourism.csv"
df = pd.read_csv(RAW_PATH)

# Drop ID / index columns that have no predictive power
cols_to_drop = [col for col in ["Unnamed: 0", "CustomerID"] if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# Separate features and target variable
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Categorical features (TypeofContact, Occupation, Gender, ProductPitched, MaritalStatus, Designation) 
# are intentionally left as raw strings so downstream transformers (like OneHotEncoder/OrdinalEncoder) 
# in the training pipeline process them seamlessly.

# stratify=y keeps the target class distribution consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save train and test sets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Type values kept as:", sorted(X["Type"].unique()))
