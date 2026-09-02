"""FastAPI application for census income prediction.

Exposes:
    GET  /         -> greeting message
    POST /predict  -> model inference on a single census record
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_model

# Model artifacts live in <repo root>/model/. Using __file__ keeps the
# paths correct both locally and on the cloud platform (the app is
# deployed from the repository root).
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "trained_model.pkl"
ENCODER_PATH = MODEL_DIR / "trained_encoder.pkl"

# Must match the categorical features used during training.
CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Continuous features in the exact column order used during training
# (CSV column order with the label column removed).
CONTINUOUS_FEATURES: list[str] = [
    "age",
    "fnlgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# The label binarizer sorts classes alphabetically, so the model's
# binarized outputs map as 0 -> "<=50K" and 1 -> ">50K".
LABELS: list[str] = ["<=50K", ">50K"]

# Load the trained model and encoder once at import time.
model, encoder = load_model()

app = FastAPI(
    title="Census Income Prediction API",
    description=(
        "Predicts whether a census record's salary is <=50K or >50K "
        "using a trained random forest classifier."
    ),
    version="1.0.0",
)


class CensusFeatures(BaseModel):
    """Features of a single census record (salary label excluded).

    Hyphenated CSV column names are handled with Pydantic aliases, so
    the CSV columns themselves are never renamed.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )

    age: int = Field(..., description="Age in years")
    workclass: str = Field(..., description="Work class")
    fnlgt: int = Field(..., description="Final wage")
    education: str = Field(..., description="Highest education level")
    education_num: int = Field(
        ..., alias="education-num", description="Years of education"
    )
    marital_status: str = Field(
        ..., alias="marital-status", description="Marital status"
    )
    occupation: str = Field(..., description="Occupation")
    relationship: str = Field(..., description="Household relationship")
    race: str = Field(..., description="Race")
    sex: str = Field(..., description="Sex")
    capital_gain: int = Field(
        ..., alias="capital-gain", description="Capital gain"
    )
    capital_loss: int = Field(
        ..., alias="capital-loss", description="Capital loss"
    )
    hours_per_week: int = Field(
        ..., alias="hours-per-week", description="Hours worked per week"
    )
    native_country: str = Field(
        ..., alias="native-country", description="Country of birth"
    )


class PredictionResponse(BaseModel):
    """Response body for the inference endpoint."""

    prediction: str = Field(..., description="Predicted salary class")


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a greeting message."""
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CensusFeatures) -> PredictionResponse:
    """Run model inference on a single census record."""
    record: dict[str, object] = features.model_dump(by_alias=True)
    X = pd.DataFrame([record])[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES]
    X_processed, _, _, _ = process_data(
        X,
        categorical_features=CATEGORICAL_FEATURES,
        training=False,
        encoder=encoder,
    )
    pred = inference(model, X_processed)[0]
    return PredictionResponse(prediction=LABELS[int(pred)])

