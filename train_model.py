"""
VERIDOC - ML model training

This script:
1. Creates/loads the prototype dataset.
2. Trains a Logistic Regression model using gradient descent.
3. Saves the trained model as document_screening_model.json.

Important:
This is a prototype training dataset. For a production system,
replace it with a properly labelled real/synthetic document-image dataset.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATASET = BASE / "document_screening_dataset.csv"
MODEL = BASE / "document_screening_model.json"

FEATURES = [
    "ocr_confidence",
    "authenticity_score",
    "anomaly_score",
    "image_width",
    "image_height",
    "field_completeness",
    "document_number_present",
    "date_present",
    "gender_present",
    "layout_consistency",
]

df = pd.read_csv(DATASET)

X = df[FEATURES].to_numpy(dtype=float)
y = (df["label"].to_numpy() == "genuine").astype(float)

rng = np.random.default_rng(42)
indices = np.arange(len(df))
rng.shuffle(indices)

split = int(len(df) * 0.80)
train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0

X_train_s = (X_train - mean) / std
X_test_s = (X_test - mean) / std

weights = np.zeros(len(FEATURES), dtype=float)
bias = 0.0

for _ in range(5000):
    z = X_train_s @ weights + bias
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    weights -= 0.05 * (X_train_s.T @ (p - y_train)) / len(y_train)
    bias -= 0.05 * float(np.mean(p - y_train))

test_p = 1.0 / (1.0 + np.exp(-np.clip(X_test_s @ weights + bias, -40, 40)))
test_pred = (test_p >= 0.5).astype(int)
accuracy = float(np.mean(test_pred == y_test))

payload = {
    "model_type": "Logistic Regression",
    "algorithm": "Binary logistic regression trained with gradient descent",
    "dataset": "Prototype synthetic/curated screening dataset",
    "dataset_rows": int(len(df)),
    "features": FEATURES,
    "feature_mean": mean.tolist(),
    "feature_std": std.tolist(),
    "weights": weights.tolist(),
    "bias": float(bias),
    "test_accuracy": round(accuracy, 4),
}

MODEL.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("Training complete.")
print(f"Dataset rows: {len(df)}")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"Saved model: {MODEL}")
