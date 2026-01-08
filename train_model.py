import pandas as pd
import numpy as np
import os
import kagglehub
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ===============================
# Download dataset
# ===============================
DATASET_PATH = kagglehub.dataset_download(
    "itachi9604/disease-symptom-description-dataset"
)

print("Dataset Path:", DATASET_PATH)

# ===============================
# Load datasets
# ===============================
df = pd.read_csv(os.path.join(DATASET_PATH, "dataset.csv"))

# ===============================
# Encode target
# ===============================
label_encoder = LabelEncoder()
df["Disease"] = label_encoder.fit_transform(df["Disease"])

X = df.drop("Disease", axis=1)
y = df["Disease"]

# ===============================
# Encode symptoms (IMPORTANT)
# ===============================
X = X.astype(str)
X_encoded = pd.get_dummies(X)

# ===============================
# Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Train Weighted KNN
# ===============================
model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)

model.fit(X_train, y_train)

# ===============================
# Evaluate
# ===============================
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ===============================
# Save artifacts
# ===============================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X_encoded.columns, open("columns.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("Model & encoders saved successfully")
