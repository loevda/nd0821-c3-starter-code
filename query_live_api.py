"""Query the live deployed census prediction API.

Sends one POST request to the /predict endpoint of the deployed API
using the requests module and prints the model inference result and
the HTTP status code.
"""

import requests

LIVE_API_URL = "https://census-prediction-api.onrender.com/predict"

# Matches the Pydantic example in main.py (CensusFeatures).
PAYLOAD: dict[str, object] = {
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


def main() -> None:
    """POST one record to the live API and print the result."""
    response = requests.post(LIVE_API_URL, json=PAYLOAD, timeout=60)
    print("Inference result:", response.json())
    print("HTTP status code:", response.status_code)


if __name__ == "__main__":
    main()
