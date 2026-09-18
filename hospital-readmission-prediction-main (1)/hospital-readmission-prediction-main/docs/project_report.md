# Hospital Readmission Prediction — Project Report

## 1. Abstract
This project develops a binary classification model to estimate whether a patient encounter is associated with a hospital readmission within 30 days. Logistic regression with L2 regularization is used because it provides a transparent baseline while controlling coefficient magnitude in a high-dimensional one-hot encoded feature space. The model is evaluated primarily using ROC-AUC and supplemented with confusion-matrix, precision, recall, and threshold analysis.

## 2. Problem Statement
Hospital readmission shortly after discharge can increase healthcare utilization and may indicate that additional discharge planning, follow-up, medication reconciliation, or post-discharge support could be useful. A predictive model can help identify encounters that deserve closer review.

The machine-learning task is:

**Input:** encounter-level demographic, admission, diagnosis, utilization, laboratory/procedure, and medication information.

**Output:** probability of readmission within 30 days.

## 3. Dataset
The supplied dataset contains **101,766 encounters and 50 original columns**. The target variable is `readmitted`, with three values:
- `<30`
- `>30`
- `NO`

For this case study, `<30` is the positive class. `>30` and `NO` are combined into the negative class because the project specifically asks for 30-day readmission.

The supplied `IDS_mapping.csv` file documents ID mappings for admission/discharge/source variables.

## 4. Feature Engineering and Preprocessing
The following were removed:
- `encounter_id`
- `patient_nbr`

These are identifiers rather than predictive clinical features.

The following fields were also excluded from the baseline:
- `weight`
- `payer_code`
- `medical_specialty`

These fields have substantial missingness in the supplied dataset and were excluded to keep the baseline reproducible and focused.

The outcome column `readmitted` is removed from model inputs to prevent target leakage.

### Missing values
The source represents missing values with `?`. They are converted to `NaN`.

- Numeric features → median imputation.
- Categorical features → most-frequent imputation.

### Categorical encoding
Categorical variables, including diagnosis codes, are one-hot encoded. Unknown categories in the test set are ignored.

### Scaling
Numeric features are standardized before logistic regression.

## 5. Model
The model is logistic regression with L2 regularization:

**P(readmission) = sigmoid(b₀ + b₁x₁ + ... + bₙxₙ)**

L2 regularization adds a penalty on large coefficients. This helps control overfitting when many categorical variables are expanded through one-hot encoding.

Configuration:
- penalty: `l2`
- C: `1.0`
- solver: `liblinear`
- max iterations: `2000`
- class weight: `balanced`

## 6. Experimental Setup
Data is split into:
- 80% training
- 20% testing

The split is stratified to preserve the class ratio and uses `random_state=42`.

The test set contains **20,354 encounters**.

## 7. Results
### Primary metric
**ROC-AUC = 0.6426**

ROC-AUC summarizes how well the model ranks positive cases above negative cases across possible thresholds. It does not by itself identify the operating threshold for a clinical workflow.

### Confusion matrix at threshold 0.50
| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 11,894 | 6,189 |
| Actual Positive | 1,024 | 1,247 |

See the generated figures in `results/`.

## 8. Clinical Cost of Errors
### False negatives
A false negative occurs when a patient who will actually be readmitted within 30 days is classified as low risk.

Potential costs include:
- missed opportunity for enhanced discharge planning
- missed follow-up
- delayed recognition of post-discharge needs
- potentially preventable utilization

Because false negatives can have meaningful clinical consequences, a care program may prefer a threshold that provides higher sensitivity.

### False positives
A false positive occurs when a patient is flagged as high risk but does not experience a 30-day readmission.

Potential costs include:
- unnecessary outreach
- additional appointments or calls
- staff workload
- resource allocation to patients who may not need it

### Threshold trade-off
There is no universally correct threshold. If the cost of missing a high-risk patient is considered greater than the cost of additional outreach, the operating threshold can be lowered. This normally increases recall/sensitivity while reducing precision and increasing false positives.

The project includes `src/threshold_analysis.py` to quantify this trade-off.

## 9. Interpretation
Logistic regression coefficients can be inspected to understand which encoded features have stronger positive or negative associations with the predicted log-odds. These associations should not be interpreted as causal effects.

The file `results/top_25_coefficients.csv` contains the 25 largest coefficient magnitudes from the fitted model.

## 10. Limitations
- The dataset is retrospective and historical.
- Direct vital-sign measurements are not comprehensively represented in the supplied data, so the baseline cannot honestly claim to use a full vitals feature set.
- External validation is required before considering use at another hospital.
- Calibration should be evaluated if predicted probabilities will be used for resource allocation.
- Performance may differ across demographic groups and clinical subpopulations.
- Diagnosis-code categories are not equivalent to detailed clinical narratives.
- Class weighting changes the classification behavior and should be considered when interpreting threshold metrics.
- This project is educational and must not be used as a clinical decision tool.

## 11. Conclusion
The project demonstrates an end-to-end, reproducible baseline for 30-day readmission prediction using L2-regularized logistic regression. ROC-AUC provides a threshold-independent evaluation, while threshold analysis highlights the operational trade-off between false negatives and false positives. A real-world clinical deployment would require prospective testing, calibration, external validation, subgroup analysis, and review by qualified clinical and governance teams.

## 12. Reproduction
```bash
pip install -r requirements.txt
python src/train_model.py
python src/threshold_analysis.py
```
