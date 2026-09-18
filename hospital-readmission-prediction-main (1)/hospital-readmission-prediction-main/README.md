# Hospital Readmission Prediction

A machine-learning case study that predicts **30-day hospital readmission** using **logistic regression with L2 regularization**.

## Project objective
Predict whether a patient encounter is followed by a readmission within 30 days.

### Target definition
- `readmitted == "<30"` → **1 (30-day readmission)**
- `readmitted == "NO"` or `">30"` → **0 (not readmitted within 30 days)**

The uploaded dataset contains 101,766 encounters. The positive 30-day readmission rate in this dataset is **11.16%**.

## Dataset
The project uses the UCI/Diabetes 130-US hospitals dataset supplied with this project:
- `data/diabetic_data.csv`
- `data/IDS_mapping.csv`

The dataset includes admission information, demographics, diagnosis codes, utilization history, laboratory/procedure counts, medications, and other encounter-level variables.

> Note: the dataset does not contain a complete set of direct vital-sign measurements. Therefore, the implementation uses the available clinical and utilization variables rather than inventing vitals.

## Model
**Logistic Regression + L2 regularization**

Why logistic regression?
- Produces a probability of 30-day readmission.
- Coefficients can be inspected for direction and magnitude.
- L2 regularization reduces the effect of unstable/high-dimensional features.
- Works well as an interpretable baseline for binary classification.

Preprocessing:
1. Replace `?` with missing values.
2. Median-impute numeric features.
3. Most-frequent-impute categorical features.
4. One-hot encode categorical variables.
5. Standardize numeric variables.
6. Fit L2-regularized logistic regression.
7. Use a stratified 80/20 train/test split (`random_state=42`).

## Evaluation
The primary metric is **ROC-AUC**, because the model produces a risk score and the decision threshold can be changed depending on clinical workflow.

Current run:
- **ROC-AUC: 0.6426**
- Test set: 20,354 encounters
- Positive class: 30-day readmission (`<30`)

See:
- `results/roc_curve.png`
- `results/precision_recall_curve.png`
- `results/confusion_matrix.png`
- `results/classification_report.txt`
- `results/threshold_analysis.csv`
- `results/top_25_coefficients.csv`

## False negatives vs false positives
A **false negative (FN)** is a patient who is actually readmitted within 30 days but is classified as low risk.

Potential clinical implications:
- missed opportunity for discharge planning
- missed follow-up or medication review
- potentially avoidable readmission
- delayed intervention for high-risk patients

A **false positive (FP)** is a patient predicted as high risk who is not readmitted within 30 days.

Potential implications:
- additional follow-up resources
- extra calls/appointments
- increased workload for care teams
- unnecessary interventions for some patients

The appropriate threshold is therefore a **clinical and operational decision**, not simply the threshold that maximizes accuracy. Lowering the threshold generally increases sensitivity/recall but also increases false positives. The included `threshold_analysis.py` script demonstrates this trade-off.

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and evaluate
```bash
python src/train_model.py
```

### 3. Analyze thresholds
```bash
python src/threshold_analysis.py
```

## Project structure
```text
hospital-readmission-prediction/
├── data/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── src/
│   ├── train_model.py
│   └── threshold_analysis.py
├── results/
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── metrics.json
│   ├── threshold_analysis.csv
│   └── top_25_coefficients.csv
├── docs/
│   └── project_report.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Important limitations
1. This is a retrospective machine-learning experiment, not a clinical decision-support system.
2. The dataset is historical and comes from a specific hospital population; performance may not generalize to other hospitals or current patients.
3. Missingness is handled statistically and may itself contain information.
4. Diagnosis codes are represented as categorical features rather than full clinical NLP.
5. ROC-AUC measures ranking ability; it does not establish clinical utility.
6. A real deployment would require external validation, calibration, subgroup/fairness analysis, prospective evaluation, and clinical governance.

## Reproducibility
- Split seed: `42`
- Test size: `20%`
- Logistic regression penalty: `L2`
- Regularization parameter: `C=1.0`
- Solver: `liblinear`
- Class weighting: `balanced`

## License / dataset note
Keep the original dataset attribution and terms associated with the source dataset. Do not treat this repository as a substitute for the dataset provider's licensing/usage requirements.
