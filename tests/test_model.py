"""Unit tests for the ML functions in starter.ml.model.

ML outputs are stochastic, so these tests only check that the functions
return the expected types and shapes.

Run from the starter/ directory:
    python -m pytest tests/ -v
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

import starter.ml.model as model_module
from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)


def _make_regression_like_data(n=50, seed=42):
    """Small synthetic dataset: one informative feature plus noise."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 5))
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


def test_train_model_returns_fitted_classifier():
    X, y = _make_regression_like_data()
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    # These attributes only exist after fit().
    assert model.n_features_in_ == 5
    assert model.n_classes_ == 2


def test_inference_returns_array_of_correct_shape():
    X, y = _make_regression_like_data()
    model = train_model(X, y)
    X_test = np.random.default_rng(0).random((10, 5))
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_returns_three_floats():
    y = np.array([0, 1, 0, 1, 1, 0])
    preds = np.array([0, 1, 1, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_save_and_load_model_roundtrip(tmp_path, monkeypatch):
    # Redirect the hardcoded artifact paths so the committed model
    # files in starter/model/ are never touched.
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "m.pkl")
    monkeypatch.setattr(
        model_module, "ENCODER_PATH", tmp_path / "e.pkl"
    )
    X, y = _make_regression_like_data()
    model = train_model(X, y)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array([["a"], ["b"]]))
    save_model(model, encoder)
    assert (tmp_path / "m.pkl").exists()
    assert (tmp_path / "e.pkl").exists()
    loaded_model, loaded_encoder = load_model()
    assert isinstance(loaded_model, RandomForestClassifier)
    assert isinstance(loaded_encoder, OneHotEncoder)
    np.testing.assert_array_equal(
        inference(model, X), inference(loaded_model, X)
    )


def test_compute_slice_metrics_returns_dict_of_metric_tuples():
    rng = np.random.default_rng(42)
    n = 100
    data = pd.DataFrame({
        "education": rng.choice(["low", "high"], size=n),
        "feature": rng.random(n),
        "salary": rng.choice(["<=50K", ">50K"], size=n),
    })
    X_train, y_train, encoder, lb = process_data(
        data, ["education"], "salary", training=True
    )
    model = train_model(X_train, y_train)
    X = data.drop(columns=["salary"])
    y = data["salary"]
    slice_metrics = compute_slice_metrics(
        model,
        encoder,
        lb,
        X,
        y,
        ["education"],
        "education",
        ["low", "high"],
    )
    assert isinstance(slice_metrics, dict)
    assert set(slice_metrics) == {"low", "high"}
    for value in ("low", "high"):
        precision, recall, fbeta = slice_metrics[value]
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fbeta, float)
