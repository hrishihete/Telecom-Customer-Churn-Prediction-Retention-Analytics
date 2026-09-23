"""Clean Telco records and prepare leakage-safe encoded train/test exports."""
from pathlib import Path
import json
import hashlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def normalize_records(raw):
    df = raw.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip().replace("", pd.NA)
    for col in NUMERIC + ["SeniorCitizen"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if df.customerID.isna().any() or df.customerID.duplicated().any():
        raise ValueError("Missing or conflicting customer IDs require review; records were not arbitrarily discarded.")
    if df.drop(columns="TotalCharges").isna().any().any():
        raise ValueError("Unexpected missing values outside TotalCharges require review.")
    if not df.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Unexpected Churn labels.")
    if not df.SeniorCitizen.isin([0, 1]).all():
        raise ValueError("Unexpected SeniorCitizen codes.")
    if (df[NUMERIC] < 0).any().any():
        raise ValueError("Negative tenure or charges require review.")
    return df, removed


def make_preprocessor(categorical):
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int), categorical),
    ], verbose_feature_names_out=True)


def main():
    source = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    out = ROOT / "data/processed"
    reports = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source)
    df, removed = normalize_records(raw)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df.Churn)
    X = df.drop(columns=["customerID", "Churn"])
    categorical = [c for c in X if c not in NUMERIC]
    # Treat the integer-encoded SeniorCitizen as a categorical variable.
    X["SeniorCitizen"] = X.SeniorCitizen.astype(int).astype(str)
    prep = make_preprocessor(categorical)
    train = prep.fit_transform(X.loc[train_idx])
    test = prep.transform(X.loc[test_idx])
    names = prep.get_feature_names_out().tolist()
    median = float(prep.named_transformers_["numeric"].named_steps["imputer"].statistics_[2])
    cleaned = df.copy()
    missing = cleaned.TotalCharges.isna()
    cleaned["TotalCharges"] = cleaned.TotalCharges.fillna(median)
    cleaned.to_csv(out / "telco_churn_cleaned.csv", index=False)
    for label, idx, matrix in [("train", train_idx, train), ("test", test_idx, test)]:
        pd.DataFrame(matrix, columns=names).to_csv(out / f"X_{label}.csv", index=False)
        df.loc[idx, "Churn"].map({"No": 0, "Yes": 1}).rename("Churn").to_csv(out / f"y_{label}.csv", index=False)
    manifest = pd.DataFrame({"customerID": df.customerID, "split": "train"})
    manifest.loc[test_idx, "split"] = "test"
    # Explicit row ordering links separate X/y exports to customer IDs.
    manifest["export_row"] = -1
    manifest.loc[train_idx, "export_row"] = range(len(train_idx))
    manifest.loc[test_idx, "export_row"] = range(len(test_idx))
    manifest.to_csv(out / "split_manifest.csv", index=False)
    pd.DataFrame({"customerID": df.loc[missing, "customerID"], "TotalCharges_imputed": median,
                  "method": "training-set median"}).to_csv(reports / "imputed_records.csv", index=False)
    categories = {col: list(values) for col, values in zip(categorical, prep.named_transformers_["categorical"].categories_)}
    metadata = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), input_rows=len(raw),
                    output_rows=len(cleaned), duplicate_rows_removed=removed, total_charges_imputed=int(missing.sum()),
                    imputation_method="Median fitted on training records only", total_charges_median=median,
                    train_rows=len(train_idx), test_rows=len(test_idx), random_state=42,
                    encoded_features=len(names), categories=categories, feature_names=names,
                    target_mapping={"No": 0, "Yes": 1}, missing_values_after=int(cleaned.isna().sum().sum()))
    assert not cleaned.isna().any().any()
    assert not cleaned.customerID.duplicated().any()
    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == len(cleaned)
    assert not pd.isna(train).any() and not pd.isna(test).any()
    (reports / "cleaning_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    (reports / "cleaning_report.md").write_text(f"""# Weekly deliverable: data cleaning

## Results

| Check | Result |
|---|---:|
| Input records | {len(raw):,} |
| Exact duplicates removed | {removed} |
| Output records | {len(cleaned):,} |
| Missing TotalCharges imputed | {missing.sum()} |
| Training-set median TotalCharges | {median:.2f} |
| Missing cells after cleaning | {cleaned.isna().sum().sum()} |
| Encoded predictor columns | {len(names)} |
| Training / test records | {len(train_idx)} / {len(test_idx)} |

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
""", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ["categories", "feature_names"]}, indent=2))


if __name__ == "__main__":
    main()
