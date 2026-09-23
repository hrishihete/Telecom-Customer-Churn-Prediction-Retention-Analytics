# Telecom Customer Churn Prediction & Retention Analytics

Completed tasks: dataset acquisition and inspection, followed by missing-charge imputation, categorical conversion, and duplicate removal.

## This week's cleaning deliverables

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

The completed scope includes dataset acquisition, initial inspection, and data cleaning. Model training and a Power BI dashboard remain subsequent project work.

GitHub repository: [Telecom Customer Churn Prediction & Retention Analytics](https://github.com/hrishihete/Telecom-Customer-Churn-Prediction-Retention-Analytics).
