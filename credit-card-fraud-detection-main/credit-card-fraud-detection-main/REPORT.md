# Case Study 2 — Credit Card Fraud Detection

## 1. Problem Statement
Credit-card and e-commerce fraud detection is a binary classification task in which legitimate transactions substantially outnumber fraudulent transactions. The goal is to estimate the probability of fraud and choose an appropriate decision threshold.

## 2. Dataset
The IEEE-CIS Fraud Detection competition provides transaction and identity tables. `TransactionID` links the tables and `isFraud` is the target. The transaction data contains payment-card, address, amount, time, count, delta, match and anonymized Vesta-engineered variables. The identity data contains anonymized device/network/digital-signature variables.

The training set has 590,540 transactions and fraud is approximately 3.5% of observations.

## 3. Methodology
1. Load `train_transaction.csv`.
2. Merge `train_identity.csv` when available.
3. Normalize copied identity names such as `id-01` to `id_01`.
4. Sort by `TransactionDT`.
5. Keep the earliest 80% for training and the latest 20% for validation.
6. Encode categorical variables and impute missing values using training information.
7. Apply SMOTE only to the training set.
8. Train XGBoost.
9. Predict validation probabilities.
10. Search thresholds from 0.05 to 0.95.
11. Select the threshold maximizing validation F1.
12. Export metrics, plots and feature importance.

## 4. Why SMOTE?
Accuracy can be misleading when the minority class is small. SMOTE generates synthetic minority observations from minority-class neighbors. It is used only on the training data to avoid contaminating validation.

## 5. Why XGBoost?
XGBoost uses gradient-boosted decision trees and can capture nonlinear relationships and feature interactions. This is useful for anonymized transaction-risk features.

## 6. Threshold and Fraud Costs
A false negative means fraud is not detected. A false positive means a legitimate transaction is flagged. These errors have different operational consequences. Therefore, the 0.50 threshold should not automatically be treated as optimal.

## 7. Results
Run the code first, then paste the generated values from `outputs/metrics.json`.

| Metric | Result |
|---|---:|
| ROC-AUC | generated after run |
| PR-AUC | generated after run |
| Precision | generated after run |
| Recall | generated after run |
| F1 | generated after run |
| Selected threshold | generated after run |

Do not invent numerical results before executing the experiment.

## 8. Conclusion
The experiment demonstrates an end-to-end imbalanced fraud-detection workflow. SMOTE increases minority-class representation during training, XGBoost models nonlinear transaction patterns, and threshold tuning makes the final fraud decision explicit. ROC-AUC evaluates probability ranking while PR-AUC, precision, recall, F1 and the confusion matrix show minority-class behavior.

## 9. Limitations
- Many IEEE-CIS variables are anonymized.
- SMOTE can require substantial RAM.
- Ordinal encoding is a practical baseline and does not imply a true numeric relationship between categories.
- Validation results are not Kaggle leaderboard results.
- Production deployment would require calibration, drift monitoring and cost-based threshold selection.
