import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, "data", "housing.csv"))

# Drop rows with missing values
df = df.dropna()

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# One-hot encode categorical column
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
dataset_size = len(X_train)

print(f"Dataset size (training samples): {dataset_size}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save metrics to file (for GitHub Actions summary + Docker push decision)
metrics = {
    "rmse": round(rmse, 2),
    "r2": round(r2, 4),
    "dataset_size": dataset_size
}
os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved to metrics/metrics.json")