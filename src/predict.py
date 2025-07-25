import joblib
from sklearn.datasets import fetch_california_housing

# Load the model
model = joblib.load("models/model.joblib")

# Load some test data
X, _ = fetch_california_housing(return_X_y=True)

# Predict on first 5 rows
predictions = model.predict(X[:5])
print("Sample predictions:", predictions)
