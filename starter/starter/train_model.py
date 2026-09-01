# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    save_model,
    train_model,
)

# Add code to load in the data.
DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "census_clean.csv"
)
data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {fbeta:.3f}")
save_model(model, encoder)
print("Model and encoder saved to model/trained_model.pkl and "
      "model/trained_encoder.pkl")
