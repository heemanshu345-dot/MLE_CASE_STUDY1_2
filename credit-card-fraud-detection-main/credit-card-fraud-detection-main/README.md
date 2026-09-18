# Credit Card Fraud Detection — IEEE-CIS + XGBoost + SMOTE

## Case Study 2
Apply XGBoost to the IEEE-CIS Fraud Detection dataset, handle severe class imbalance with SMOTE, tune the decision threshold, and interpret the model with feature-importance scores.

### Dataset
Source: https://www.kaggle.com/competitions/ieee-fraud-detection/data

The Kaggle dataset contains transaction and identity tables joined by `TransactionID`. The target is `isFraud`. Kaggle lists five competition files and about 1.35 GB of CSV data. The training set contains 590,540 transactions; fraud is only about 3.5%, making class imbalance important.

### Put these files in `data/`
```text
data/
├── train_transaction.csv   # required
├── train_identity.csv      # optional
├── test_transaction.csv   # optional
├── test_identity.csv       # optional
└── sample_submission.csv   # optional
```

**Do not upload the Kaggle CSV files to GitHub.** The repository `.gitignore` excludes them.

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Then:
```bash
pip install -r requirements.txt
```

## Run EDA

```bash
python src/make_eda.py --data-dir data --output-dir outputs
```

## Run the full model

```bash
python src/train_fraud_xgboost.py --data-dir data --output-dir outputs
```

Laptop-friendly run:
```bash
python src/train_fraud_xgboost.py --data-dir data --output-dir outputs --smote-max-rows 120000 --n-estimators 300
```

### Pipeline

```text
IEEE-CIS data
      ↓
merge transaction + identity
      ↓
chronological train/validation split
      ↓
categorical encoding + numeric imputation
      ↓
SMOTE on training only
      ↓
XGBoost
      ↓
validation probabilities
      ↓
threshold search
      ↓
ROC-AUC / PR-AUC / Precision / Recall / F1
      ↓
feature importance + plots
```

SMOTE is deliberately applied **after** the split. Validation data remains untouched.

## Outputs

The training script creates:
- `metrics.json`
- `threshold_results.csv`
- `feature_importance.csv`
- `feature_importance.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`

## Why threshold tuning?

XGBoost produces fraud probabilities. A threshold converts them to decisions:

`probability >= threshold → fraud`

A lower threshold may increase fraud recall but can also increase false positives. The default experiment selects the threshold with the highest validation F1. For a real payment system, the threshold should instead be chosen using actual business costs.

## Evaluation

- **ROC-AUC:** ranking quality across thresholds.
- **PR-AUC:** useful for an imbalanced positive class.
- **Precision:** how many flagged transactions are actually fraud.
- **Recall:** how much fraud is detected.
- **F1:** balance between precision and recall.
- **Confusion matrix:** counts of TP, TN, FP and FN.

## Feature importance

The project exports XGBoost gain-based importance. IEEE-CIS contains anonymized Vesta features, so names such as `Vxxx` do not necessarily have an interpretable business meaning. Importance indicates model usage, not causality.

## GitHub upload

```bash
git init
git add .
git commit -m "Add IEEE-CIS fraud detection with XGBoost and SMOTE"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
