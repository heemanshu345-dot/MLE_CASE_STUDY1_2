import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
SEED = 42

# A compact, reproducible baseline for the large IEEE-CIS dataset.
BASE_FEATURES = [
    "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
]

def normalize_columns(df):
    rename = {c: c.replace("id-", "id_") for c in df.columns if c.startswith("id-")}
    return df.rename(columns=rename)

def load_data(data_dir):
    tx_path = os.path.join(data_dir, "train_transaction.csv")
    id_path = os.path.join(data_dir, "train_identity.csv")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(
            f"{tx_path} not found. Download train_transaction.csv from Kaggle."
        )
    train = pd.read_csv(tx_path)
    print("Transactions:", train.shape)
    if os.path.exists(id_path):
        identity = normalize_columns(pd.read_csv(id_path))
        train = train.merge(identity, on="TransactionID", how="left")
        print("After identity merge:", train.shape)
    return train

def chronological_split(df, fraction=0.20):
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    cut = int(len(df) * (1 - fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def preprocess(train_df, val_df, features):
    cat = [c for c in features if train_df[c].dtype == "object"]
    num = [c for c in features if c not in cat]
    A, B = train_df[features].copy(), val_df[features].copy()

    if cat:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        A[cat] = enc.fit_transform(A[cat].astype("string").fillna("__MISSING__"))
        B[cat] = enc.transform(B[cat].astype("string").fillna("__MISSING__"))

    for c in num:
        A[c] = pd.to_numeric(A[c], errors="coerce")
        B[c] = pd.to_numeric(B[c], errors="coerce")
        med = A[c].median()
        if pd.isna(med):
            med = 0.0
        A[c] = A[c].fillna(med)
        B[c] = B[c].fillna(med)

    return A.astype("float32"), B.astype("float32")

def tune_threshold(y, p):
    rows = []
    for t in np.arange(0.05, 0.951, 0.01):
        pred = (p >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        })
    result = pd.DataFrame(rows)
    best = result.loc[result["f1"].idxmax()]
    return float(best["threshold"]), result

def make_plots(y, p, pred, importance, out_dir):
    top = importance.head(20).sort_values("importance")
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top 20 XGBoost Feature Importances (Gain)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=180)
    plt.close()

    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y, p):.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y, p):.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curve.png"), dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    ap.add_argument("--smote-max-rows", type=int, default=150000)
    ap.add_argument("--smote-k-neighbors", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    np.random.seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.data_dir)
    if "isFraud" not in df:
        raise ValueError("isFraud is missing from train_transaction.csv")

    features = [c for c in BASE_FEATURES if c in df.columns]
    if len(features) < 10:
        raise ValueError("Too few expected features found.")

    train_df, val_df = chronological_split(df, args.validation_fraction)
    y_train = train_df["isFraud"].astype(int).to_numpy()
    y_val = val_df["isFraud"].astype(int).to_numpy()
    X_train, X_val = preprocess(train_df, val_df, features)

    print("Original class counts:", np.bincount(y_train))

    smote = SMOTE(random_state=SEED, k_neighbors=args.smote_k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # SMOTE can be RAM-heavy on this dataset, so cap the resampled experiment.
    if args.smote_max_rows and len(X_res) > args.smote_max_rows:
        rng = np.random.default_rng(SEED)
        idx0 = np.flatnonzero(y_res == 0)
        idx1 = np.flatnonzero(y_res == 1)
        n_each = args.smote_max_rows // 2
        chosen = np.concatenate([
            rng.choice(idx0, min(n_each, len(idx0)), replace=False),
            rng.choice(idx1, min(n_each, len(idx1)), replace=False),
        ])
        rng.shuffle(chosen)
        X_res = X_res.iloc[chosen]
        y_res = y_res[chosen]

    print("Training class counts after SMOTE/cap:", np.bincount(y_res))

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_res, y_res, eval_set=[(X_val, y_val)], verbose=False)

    p = model.predict_proba(X_val)[:, 1]
    threshold, threshold_table = tune_threshold(y_val, p)
    pred = (p >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, p)),
        "pr_auc": float(average_precision_score(y_val, p)),
        "selected_threshold": threshold,
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "validation_rows": int(len(y_val)),
        "train_rows_before_smote": int(len(y_train)),
        "train_rows_after_smote_cap": int(len(y_res)),
        "features": features,
    }

    print("\n=== RESULTS ===")
    for k, v in metrics.items():
        if k != "features":
            print(f"{k}: {v}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_val, pred, digits=4, zero_division=0))

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    threshold_table.to_csv(os.path.join(args.output_dir, "threshold_results.csv"), index=False)
    importance.to_csv(os.path.join(args.output_dir, "feature_importance.csv"), index=False)
    make_plots(y_val, p, pred, importance, args.output_dir)

    print("\nTop features:")
    print(importance.head(15).to_string(index=False))
    print(f"\nSaved results to {args.output_dir}/")

if __name__ == "__main__":
    main()
