# Weekly deliverable: data cleaning

## Results

| Check | Result |
|---|---:|
| Input records | 7,043 |
| Exact duplicates removed | 0 |
| Output records | 7,043 |
| Missing TotalCharges imputed | 11 |
| Training-set median TotalCharges | 1398.12 |
| Missing cells after cleaning | 0 |
| Encoded predictor columns | 46 |
| Training / test records | 5634 / 1409 |

## Decisions

Whitespace is stripped and numeric fields are parsed strictly. Exact duplicate records are removed after normalization. This dataset has no duplicates to remove. Conflicting records sharing a customer ID raise an error instead of silently choosing a record.

The 11 blank TotalCharges values are imputed using the training-set median. All 11 have zero tenure. Median imputation is an explicit statistical assumption for this assignment, not evidence that these customers actually incurred those charges. A zero-value policy may be more appropriate if the business confirms that zero tenure means no accrued charges. Individual imputed IDs are recorded in `imputed_records.csv`.

Records are split 80/20 with stratification on Churn and seed 42 before fitting the median or categorical vocabulary. The same fitted preprocessing transforms the held-out records. For future cross-validation, fit a fresh `make_preprocessor` inside each training fold using the normalized raw records, rather than reusing the exported training matrix across folds.

All 16 categorical predictors, including SeniorCitizen, are one-hot encoded without dropping categories. No internet service and No phone service remain distinct categories. Unknown categories at inference become all-zero indicators for that feature; monitor unknown values in production. Churn maps No to 0 and Yes to 1. customerID is excluded from predictors. Scaling and resampling are not applied.

## Files and use

- `data/processed/telco_churn_cleaned.csv`: full readable dataset for Power BI/SQL, with filled TotalCharges and original category labels.
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` in the same directory: numeric modeling exports; X and y rows correspond exactly within each split.
- `split_manifest.csv`: customerID, split and zero-based export_row for traceability.
- `cleaning_metadata.json`: fitted median, categories, feature order, seed, and raw checksum.
- `scripts/clean_data.py`: rerunnable cleaning and preprocessing functions.

Raw data and last week's typed dataset are preserved. Churn counts remain No = 5,174 and Yes = 1,869. Imputation does not add real complaint or usage measurements; those remain absent from the source.

For Power BI, follow `power_bi_handoff.md` but import `telco_churn_cleaned.csv` instead. The missing-charge card should now read 0. The 7,043 customer count and 26.54% churn rate remain unchanged. For SQL, import the cleaned file as `telco_customers` and run `sql/initial_audit.sql`; both missing-charge results should now be 0.
