import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Set project-relative paths
data_path = os.path.join("data", "census.csv")
model_dir = os.path.join("model")

# Load the data
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Train/test split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
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

# Process training and test data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save the model and encoders
model = train_model(X_train, y_train)

model_path = os.path.join("model", "model.pkl")
encoder_path = os.path.join("model", "encoder.pkl")
lb_path = os.path.join("model", "lb.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

# Load model (to simulate deployment)
model = load_model(model_path)

# Run inference and compute metrics
preds = inference(model, X_test)
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Evaluate performance on categorical slices
slice_output_path = "slice_output.txt"
with open(slice_output_path, "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            if p is not None:
                f.write(f"{col}: {slicevalue}, Count: {count:,}\n")
                f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n\n")