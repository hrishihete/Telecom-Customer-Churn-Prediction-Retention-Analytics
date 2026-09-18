# Telecom Customer Churn Prediction & Retention Analytics

Initial task completed: acquire the Kaggle Telco Customer Churn dataset, inspect feature types and data quality, and identify target imbalance.

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
```

The raw CSV is included, so downloading again is optional. The public Kaggle endpoint requires internet access and may change or require authentication later. Acquisition errors are surfaced rather than silently substituting another dataset. The processed CSV retains all customers and only converts TotalCharges to numeric, preserving nulls. The audit runs independently of the current working directory.

Source: [BlastChar's Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Refer to the source's terms before external redistribution. This repository does not confer a new license on the data.

The assigned scope is dataset acquisition and initial inspection. Model training and a Power BI dashboard remain subsequent project work.

GitHub repository: [Telecom Customer Churn Prediction & Retention Analytics](https://github.com/hrishihete/Telecom-Customer-Churn-Prediction-Retention-Analytics).
