# Telecom Customer Churn Prediction & Retention Analytics

Completed tasks: dataset acquisition and inspection, followed by missing-charge imputation, categorical conversion, and duplicate removal.

## This week's feature importance and SHAP analysis

See the [explainability findings](reports/explainability/findings.md) for global drivers, signed contributions, local customer explanations, and interpretation limits.

- [Explanation script](scripts/explain_model.py) and [verification script](scripts/verify_explanations.py)
- [Global importance comparison](reports/explainability/global_importance.csv) and [driver direction](reports/explainability/driver_direction.csv)
- [Customer-level grouped SHAP](reports/explainability/shap_by_customer.csv), [encoded SHAP](reports/explainability/shap_encoded.csv), and [prediction references](reports/explainability/explanation_predictions.csv)
- [Charts and local examples](reports/explainability/)
- [Pinned explanation environment](requirements-explainability-lock.txt)

Install the explanation lock file, then run `python scripts/explain_model.py` and `python scripts/verify_explanations.py`. The existing Random Forest is explained without refitting. Results describe model behavior, not causal effects on actual churn.

## Previous week's machine learning deliverables

Trained Logistic Regression, Decision Tree, and Random Forest with fold-local preprocessing and training-only cross-validation selection. See the [model evaluation report](reports/models/evaluation_report.md) for comparisons and limitations of the previously explored evaluation split.

- [Training script](scripts/train_models.py) and [metric verification](scripts/verify_models.py)
- [Metrics](reports/models/test_metrics.csv), [evaluation predictions](reports/models/test_predictions.csv), and [metadata](reports/models/metadata.json)
- [Saved fitted pipelines](models/)
- [ROC/precision–recall curves](reports/models/evaluation_curves.png) and [confusion matrices](reports/models/confusion_matrices.png)

Run `python scripts/train_models.py`, then `python scripts/verify_models.py`. Training uses raw records and the existing split manifest, with imputation and encoding fitted inside each cross-validation fold. The saved pipelines expect the 19 original predictor columns; convert SeniorCitizen to string as shown in the training script. Load only trusted joblib files and use the package versions recorded in metadata. No model is deployed.

For a local Windows virtual environment, create it with `python -m venv .venv`, install dependencies with `.venv\Scripts\python.exe -m pip install -r requirements.txt`, and run the scripts with `.venv\Scripts\python.exe` in place of `python`.

The tested model environment is pinned in [requirements-models-lock.txt](requirements-models-lock.txt); install that file instead of the broad requirements to reproduce this training run's dependency versions.

## Previous week's exploratory analysis

Churn rates across contract types, payment methods, and tenure bands are documented in the [EDA findings report](reports/eda/findings.md), with segment counts, rates, four charts, and limitations.

- [Reproducible Python EDA](scripts/analyze_churn.py)
- [SQL segment queries](sql/churn_eda.sql)
- [Power BI visual setup](reports/eda/power_bi_eda.md)
- [Contract summary](reports/eda/churn_by_Contract.csv), [payment summary](reports/eda/churn_by_PaymentMethod.csv), [tenure summary](reports/eda/churn_by_TenureBand.csv)
- [Contract/tenure joint breakdown](reports/eda/contract_tenure_breakdown.csv)

Run `python scripts/analyze_churn.py` after cleaning. The script verifies all segment counts and rates against independent SQL aggregations.

## Previous week's cleaning deliverables

See the [cleaning report](reports/cleaning_report.md) for decisions and results.

- [Cleaned dataset](data/processed/telco_churn_cleaned.csv) for Power BI and SQL.
- [Cleaning pipeline](scripts/clean_data.py) and [verification script](scripts/verify_cleaning.py).
- Numeric model inputs: [X_train](data/processed/X_train.csv), [X_test](data/processed/X_test.csv), [y_train](data/processed/y_train.csv), [y_test](data/processed/y_test.csv).
- [Split manifest](data/processed/split_manifest.csv), [imputed-record audit](reports/imputed_records.csv), and [preprocessing metadata](reports/cleaning_metadata.json).

The median and one-hot encoding vocabulary are learned from training records only. The readable dataset retains categorical labels for dashboard usability; the modeling exports contain numeric features and a binary target. The original data and prior inspection outputs remain available.

Start with [the inspection report](reports/initial_inspection.md). Supporting outputs include the feature dictionary, class balance chart and CSV, numeric summary, quality checks, and source checksum in `reports/`.

## Deliverables

| Deliverable | File |
|---|---|
| Original Kaggle dataset | [Raw CSV](data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv) |
| Typed dataset with null charges preserved | [Prepared CSV](data/processed/telco_churn_typed.csv) |
| Findings and modeling implications | [Inspection report](reports/initial_inspection.md) |
| Every column's type, role, and missingness | [Feature dictionary](reports/feature_dictionary.csv) |
| Class counts and percentages | [Balance CSV](reports/class_balance.csv) |
| Class balance visualization | [Chart](reports/class_balance.png) |
| Source checksum and audit metadata | [Provenance](reports/provenance.json) |
| Reproducible acquisition and inspection | [Download script](scripts/download_data.py), [audit script](scripts/inspect_data.py) |
| SQL equivalents of the core audit | [SQL queries](sql/initial_audit.sql) |
| Power BI import settings and measures | [Power BI guide](reports/power_bi_handoff.md) |

The audit found 7,043 customers and 21 columns. Churn is moderately imbalanced: 26.54% Yes versus 73.46% No. Eleven TotalCharges values are blank; there are no duplicate customer IDs. The dataset has no direct complaints or detailed usage measurements.

## Reproduce

Requires Python 3.10+:

```sh
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/inspect_data.py
python scripts/clean_data.py
python scripts/verify_cleaning.py
```

The raw CSV is included, so downloading again is optional. The public Kaggle endpoint requires internet access and may change or require authentication later. Acquisition errors are surfaced rather than silently substituting another dataset. The processed CSV retains all customers and only converts TotalCharges to numeric, preserving nulls. The audit runs independently of the current working directory.

Source: [BlastChar's Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Refer to the source's terms before external redistribution. This repository does not confer a new license on the data.

Completed scope includes acquisition, cleaning, exploratory analysis, model training/evaluation, explainability, and a native Power BI historical-data dashboard. See the [dashboard guide](powerbi/README.md), [editable Power BI project](powerbi/ChurnRisk.pbip), and [Desktop validation](reports/dashboard/validation.md). Live production data and deployment remain subsequent project work.

GitHub repository: [Telecom Customer Churn Prediction & Retention Analytics](https://github.com/hrishihete/Telecom-Customer-Churn-Prediction-Retention-Analytics).
