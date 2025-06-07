import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    train_model,
    inference,
    compute_model_metrics
)

def test_train_model_output_type():
#Test that train_model returns a RandomForestClassifier instance.
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_inference_shape():
#Test that inference returns predictions of the correct shape.
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]

def test_metrics_return_type():
#Test that compute_model_metrics returns float values for precision, recall, and fbeta.
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
