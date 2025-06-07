import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
    pass


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
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
    return model.predict(X)
    pass


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    pass

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    pass

def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model
):

    slice_df = data[data[column_name] == slice_value]
    if slice_df.empty:
        return None, None, None


    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    preds = inference(model, X_slice)
    precision = precision_score(y_slice, preds, zero_division=1)
    recall = recall_score(y_slice, preds, zero_division=1)
    fbeta = fbeta_score(y_slice, preds, beta=1, zero_division=1)

    return precision, recall, fbeta