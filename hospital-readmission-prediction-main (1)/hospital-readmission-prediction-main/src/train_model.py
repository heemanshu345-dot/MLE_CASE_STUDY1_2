"""
Hospital Readmission Prediction
Logistic Regression with L2 regularization.

Dataset:
  data/diabetic_data.csv
Target:
  1 if readmitted == "<30", else 0 for "NO" or ">30".

Run:
  python src/train_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "diabetic_data.csv")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

df = pd.read_csv(DATA).replace("?", np.nan)
df["target_30day"] = (df["readmitted"] == "<30").astype(int)

drop_cols = [
    "encounter_id", "patient_nbr", "readmitted", "target_30day",
    "weight", "payer_code", "medical_specialty"
]
X = df.drop(columns=drop_cols, errors="ignore")
y = df["target_30day"]

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

clf = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    max_iter=2000,
    class_weight="balanced"
)
pipeline = Pipeline([("preprocess", preprocess), ("model", clf)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)

prob = pipeline.predict_proba(X_test)[:, 1]
pred = (prob >= 0.50).astype(int)

auc_score = roc_auc_score(y_test, prob)
print(f"ROC-AUC: {auc_score:.4f}")
print("\nClassification report:\n", classification_report(y_test, pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

fpr, tpr, _ = roc_curve(y_test, prob)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.3f}")
plt.plot([0,1],[0,1],"--",label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — 30-Day Readmission")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "roc_curve.png"), dpi=180)
plt.close()

precision, recall, _ = precision_recall_curve(y_test, prob)
plt.figure(figsize=(7,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — 30-Day Readmission")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "precision_recall_curve.png"), dpi=180)
plt.close()
