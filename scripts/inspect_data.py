"""Reproduce the initial Telco dataset audit; run from any working directory."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
(ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SOURCE)
assert df.columns.is_unique and {"customerID", "Churn", "TotalCharges"} <= set(df.columns)
assert set(df.Churn.dropna().unique()) == {"Yes", "No"}
clean = df.copy()
clean["TotalCharges"] = pd.to_numeric(df.TotalCharges.str.strip(), errors="coerce")
numeric = {"tenure", "MonthlyCharges", "TotalCharges"}
rows = []
for col in df:
    s = df[col]
    blank = s.astype("string").str.strip().eq("").sum()
    role = "identifier" if col == "customerID" else "target" if col == "Churn" else "predictor"
    kind = "numeric" if col in numeric else "identifier" if role == "identifier" else "categorical"
    notes = ""
    if col == "SeniorCitizen":
        notes = "Binary category encoded as 0/1; not a continuous measurement."
    elif col == "TotalCharges":
        notes = "Numeric stored as text; whitespace-only values converted to null, not imputed."
    elif col == "tenure":
        notes = "Discrete number of months."
    elif col == "customerID":
        notes = "Exclude from model inputs."
    elif col == "Contract":
        notes = "Contract category has duration ordering; encoding is a modeling choice."
    values = "; ".join(map(str, sorted(s.dropna().unique(), key=str))) if s.nunique() <= 10 else ""
    rows.append(dict(feature=col, role=role, semantic_type=kind, raw_dtype=str(s.dtype),
                     prepared_dtype=str(clean[col].dtype), distinct_values=s.nunique(),
                     raw_nulls=int(s.isna().sum()), blank_strings=int(blank),
                     prepared_nulls=int(clean[col].isna().sum()), categories=values, notes=notes))
dictionary = pd.DataFrame(rows)
dictionary.to_csv(REPORTS / "feature_dictionary.csv", index=False)
counts = df.Churn.value_counts().reindex(["No", "Yes"])
balance = counts.rename_axis("Churn").reset_index(name="customers")
balance["percent"] = balance.customers / len(df) * 100
balance.to_csv(REPORTS / "class_balance.csv", index=False)
clean.describe().to_csv(REPORTS / "numeric_summary.csv")
clean.to_csv(ROOT / "data/processed/telco_churn_typed.csv", index=False)
missing = clean.TotalCharges.isna()
invalid = missing & df.TotalCharges.notna() & df.TotalCharges.str.strip().ne("")
quality = dict(rows=len(df), columns=len(df.columns), predictors=len(df.columns)-2,
               duplicate_rows=int(df.duplicated().sum()), duplicate_customer_ids=int(df.customerID.duplicated().sum()),
               missing_customer_ids=int(df.customerID.isna().sum()), missing_target=int(df.Churn.isna().sum()),
               missing_total_charges=int(missing.sum()), invalid_nonblank_total_charges=int(invalid.sum()),
               missing_total_charges_with_zero_tenure=int((missing & df.tenure.eq(0)).sum()),
               majority_minority_ratio=float(counts.max()/counts.min()))
(REPORTS / "quality_checks.json").write_text(json.dumps(quality, indent=2)+"\n")
provenance = dict(dataset="blastchar/telco-customer-churn", source_page="https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
                  download_url="https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn",
                  file=SOURCE.name, sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  audit_utc=datetime.now(timezone.utc).isoformat(), pandas_version=pd.__version__)
(REPORTS / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(["Retained (No)", "Churned (Yes)"], counts, color=["#227c9d", "#e07848"])
for bar, count in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, count+100, f"{count:,} ({count/len(df):.2%})", ha="center", weight="bold")
ax.set(title="Telco customer churn: target class balance", ylabel="Customers", ylim=(0, counts.max()*1.18))
ax.text(.5, -.16, f"n = {len(df):,} | Retained-to-churned ratio: {counts['No']/counts['Yes']:.2f}:1", transform=ax.transAxes, ha="center")
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(REPORTS / "class_balance.png", dpi=180)
plt.close(fig)
table = "\n".join(f"| {r.feature} | {r.raw_dtype} | {r.semantic_type} | {r.prepared_nulls} |" for r in dictionary.itertuples())
report = f"""# Initial dataset inspection

Source: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), downloaded directly through Kaggle's public dataset endpoint. The raw CSV is preserved; its SHA-256 is recorded in `provenance.json`.

## Shape and target balance

- **{len(df):,} customers, {len(df.columns)} columns**: one identifier, 19 candidate predictors, one target (`Churn`).
- Retained (`No`): **{counts['No']:,} ({counts['No']/len(df):.2%})**.
- Churned (`Yes`): **{counts['Yes']:,} ({counts['Yes']/len(df):.2%})**.
- Majority/minority ratio: **{counts['No']/counts['Yes']:.2f}:1**, a moderate imbalance.
- Always predicting `No` gives **{counts['No']/len(df):.2%} accuracy** and zero recall for churn. Accuracy alone is therefore misleading.

![Class balance](class_balance.png)

## Feature types

Three numeric predictors: `tenure`, `MonthlyCharges`, `TotalCharges`. The remaining 16 predictors are categorical, including integer-encoded `SeniorCitizen`. `customerID` is an identifier and must be excluded from model inputs. `Churn` is the binary categorical target.

| Feature | Inferred raw pandas dtype | Semantic type | Nulls after numeric conversion |
|---|---|---|---:|
{table}

See `feature_dictionary.csv` for categories, cardinality, blank counts, and preparation notes.

## Data quality

- Duplicate rows: **{quality['duplicate_rows']}**; duplicate customer IDs: **{quality['duplicate_customer_ids']}**.
- Missing customer IDs: **{quality['missing_customer_ids']}**; missing targets: **{quality['missing_target']}**.
- `TotalCharges` contains **{missing.sum()} whitespace-only/missing values** and is initially read as text. All **{quality['missing_total_charges_with_zero_tenure']}** missing-charge records have zero tenure. Nonblank unparseable values: **{invalid.sum()}**.
- The prepared export converts `TotalCharges` to numeric and preserves missing values. No rows were removed and no values were imputed. A zero-charge interpretation would require a business assumption; otherwise fit imputation on training data only.
- `No internet service` and `No phone service` are explicit categories, not missing data.

## Implications for the project

The data supports analysis of contract structure, payment method, tenure, charges, and service subscriptions. It contains **no direct complaint counts/text, call records, data-consumption measurements, or observation timestamps**. Service subscription indicators describe products held, not actual usage intensity. Complaint and detailed usage analysis requires another source. Without observation dates, a temporal validation or clearly defined future prediction window cannot be established from this file alone.

For subsequent modeling, use a reproducible stratified train/test split and stratified cross-validation. Fit preprocessing only on training folds. Compare class-weighted models against an unweighted baseline; any resampling must occur only inside training folds. Evaluate churn precision, recall, F1, average precision/PR curves, ROC-AUC, and confusion matrices. Choose an operating threshold using validation data and retention costs, then evaluate once on the held-out test set. Preserve the natural class distribution in evaluation and Power BI reporting.

## Handoff

`data/processed/telco_churn_typed.csv` is ready for import into Power BI or SQL staging. Set `customerID` to text, `tenure`/`SeniorCitizen` to whole numbers, and charges to decimal numbers; treat blank `TotalCharges` as null. No predictive model or dashboard is included in this acquisition-and-inspection task.
"""
(REPORTS / "initial_inspection.md").write_text(report, encoding="utf-8")
print(json.dumps(quality, indent=2))
print(balance.to_string(index=False))
