"""Unit tests for the FastAPI application in main.py.

Covers:
    - GET / returns a greeting (status code and body contents).
    - POST /predict for EACH possible model output (<=50K and >50K).

Run from the repository root:
    python -m pytest tests/ -v
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# A census record the trained model predicts as <=50K.
LOW_INCOME_RECORD = {
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

# A census record the trained model predicts as >50K.
HIGH_INCOME_RECORD = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}


def test_get_root_returns_greeting():
    """GET / must return 200 and a greeting message in the body."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "Welcome" in body["message"]


def test_post_predict_returns_low_income():
    """POST /predict returns 200 and predicts <=50K for a low-income record."""
    response = client.post("/predict", json=LOW_INCOME_RECORD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "<=50K"


def test_post_predict_returns_high_income():
    """POST /predict returns 200 and predicts >50K for a high-income record."""
    response = client.post("/predict", json=HIGH_INCOME_RECORD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == ">50K"
