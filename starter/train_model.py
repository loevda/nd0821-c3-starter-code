# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
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

# Compute slice metrics for the education feature and write them to
# slice_output.txt at the project root (starter/).
X_test_raw = test.drop(columns=["salary"])
y_test_raw = test["salary"]
education_values = sorted(test["education"].unique())
slice_metrics = compute_slice_metrics(
    model,
    encoder,
    lb,
    X_test_raw,
    y_test_raw,
    cat_features,
    "education",
    education_values,
)
SLICE_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "slice_output.txt"
)
with open(SLICE_OUTPUT_PATH, "w", encoding="utf-8") as f:
    for value, (precision, recall, fbeta) in slice_metrics.items():
        f.write(
            f"education = {value} | Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | F1: {fbeta:.3f}\n"
        )
print(f"Slice metrics written to {SLICE_OUTPUT_PATH}")
