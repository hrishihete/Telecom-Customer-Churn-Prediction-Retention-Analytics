"""Check cleaning edge cases and the exported data contract."""
import json
import numpy as np
import pandas as pd
from clean_data import ROOT, NUMERIC, normalize_records, make_preprocessor

raw = pd.read_csv(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
normalized, _ = normalize_records(raw)
# Inject duplicates to verify removal even though the source has none.
deduped, count = normalize_records(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))
assert count == 1 and len(deduped) == len(raw)
conflict = raw.iloc[[0]].copy()
conflict["MonthlyCharges"] += 1
try:
    normalize_records(pd.concat([raw, conflict], ignore_index=True))
except ValueError as error:
    assert "conflicting" in str(error)
else:
    raise AssertionError("Conflicting IDs were accepted")
# Held-out extremes must not influence the fitted imputation statistic.
fixture = pd.DataFrame({"tenure": [1, 2, 3], "MonthlyCharges": [10, 20, 30],
                        "TotalCharges": [10, 30, np.nan], "Contract": ["A", "B", "A"]})
processor = make_preprocessor(["Contract"])
processor.fit(fixture)
heldout = fixture.iloc[[2]].copy()
heldout["Contract"] = "unseen"
transformed = processor.transform(heldout)
assert transformed[0, 2] == 20 and (transformed[0, 3:] == 0).all()

out = ROOT / "data/processed"
clean = pd.read_csv(out / "telco_churn_cleaned.csv")
meta = json.loads((ROOT / "reports/cleaning_metadata.json").read_text())
manifest = pd.read_csv(out / "split_manifest.csv")
assert len(clean) == len(normalized) == len(manifest) == 7043
assert not clean.isna().any().any() and clean.customerID.is_unique
pd.testing.assert_frame_equal(clean.drop(columns="TotalCharges"), normalized.drop(columns="TotalCharges"))
observed = normalized.TotalCharges.notna()
assert np.allclose(clean.loc[observed, "TotalCharges"], normalized.loc[observed, "TotalCharges"])
assert (clean.loc[~observed, "TotalCharges"] == meta["total_charges_median"]).all()
train_ids = manifest.loc[manifest.split.eq("train"), "customerID"]
assert normalized.set_index("customerID").loc[train_ids, "TotalCharges"].median() == meta["total_charges_median"]
X = normalized.drop(columns=["customerID", "Churn"])
X["SeniorCitizen"] = X.SeniorCitizen.astype(int).astype(str)
X.index = normalized.customerID
prep = make_preprocessor([c for c in X if c not in NUMERIC])
train_order = manifest[manifest.split.eq("train")].sort_values("export_row").customerID
prep.fit(X.loc[train_order])
for split in ["train", "test"]:
    ids = manifest[manifest.split.eq(split)].sort_values("export_row").customerID
    features = pd.read_csv(out / f"X_{split}.csv")
    labels = pd.read_csv(out / f"y_{split}.csv").Churn
    assert features.columns.tolist() == meta["feature_names"]
    assert np.allclose(features, prep.transform(X.loc[ids]))
    expected = normalized.set_index("customerID").loc[ids, "Churn"].map({"Yes": 1, "No": 0})
    assert np.array_equal(labels, expected)
    assert not features.isna().any().any()
assert clean.Churn.value_counts().to_dict() == {"No": 5174, "Yes": 1869}
print("PASS: duplicate removal, conflicting-ID rejection, training-only median, unknown categories, unchanged observed values, full export reconstruction, and X/y row alignment.")
