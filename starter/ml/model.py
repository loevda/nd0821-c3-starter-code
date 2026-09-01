import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .data import process_data

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "model"
MODEL_PATH = MODEL_DIR / "trained_model.pkl"
ENCODER_PATH = MODEL_DIR / "trained_encoder.pkl"


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, encoder):
    """
    Saves the trained model and categorical encoder to disk.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : OneHotEncoder
        Trained categorical encoder.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)


def load_model():
    """
    Loads the trained model and categorical encoder from disk.

    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : OneHotEncoder
        Trained categorical encoder.
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def compute_slice_metrics(
    model, encoder, lb, X, y, categorical_features, feature_name,
    feature_values,
):
    """
    Computes the model metrics with a categorical feature's value held fixed.

    For each unique value of `feature_name`, the metrics (precision, recall,
    F1) are computed on the subset of the data where the feature equals that
    value.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : OneHotEncoder
        Trained categorical encoder.
    lb : LabelBinarizer
        Trained label binarizer.
    X : pd.DataFrame
        Raw features (label column excluded).
    y : pd.Series
        Raw labels.
    categorical_features : list[str]
        Names of the categorical features.
    feature_name : str
        Categorical feature to slice on.
    feature_values : list
        Unique values of `feature_name`.
    Returns
    -------
    slice_metrics : dict
        Dictionary mapping each feature value to its
        (precision, recall, fbeta) tuple.
    """
    slice_metrics = {}
    for value in feature_values:
        mask = X[feature_name] == value
        X_slice, _, _, _ = process_data(
            X.loc[mask],
            categorical_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        y_slice = lb.transform(y.loc[mask].values).ravel()
        preds = inference(model, X_slice)
        slice_metrics[value] = compute_model_metrics(y_slice, preds)
    return slice_metrics
