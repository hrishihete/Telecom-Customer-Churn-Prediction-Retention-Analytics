"""Verify persisted attribution exports independently of report generation."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "reports/explainability"
meta = json.loads((out / "metadata.json").read_text())
grouped = pd.read_csv(out / "shap_by_customer.csv").set_index("customerID")
encoded = pd.read_csv(out / "shap_encoded.csv").set_index("customerID")
mapping = pd.read_csv(out / "feature_mapping.csv")
pred = pd.read_csv(out / "explanation_predictions.csv").set_index("customerID")
prior = pd.read_csv(ROOT / "reports/models/test_predictions.csv").set_index("customerID")
assert grouped.index.is_unique and len(grouped) == meta["customers"]
assert grouped.index.equals(encoded.index) and grouped.index.equals(pred.index)
assert set(pred.index) == set(prior.index)
assert np.isfinite(grouped.to_numpy()).all() and np.isfinite(encoded.to_numpy()).all()
assert np.allclose(pred.predicted_probability, prior.loc[pred.index, meta["model"] + "_probability"])
assert np.array_equal(pred.actual_churn, prior.loc[pred.index, "actual_churn"])
assert np.allclose(pred.base_probability + grouped.sum(axis=1), pred.predicted_probability, atol=1e-6)
for feature in grouped:
    cols = mapping.loc[mapping.source_feature.eq(feature), "encoded_feature"]
    assert np.allclose(encoded[cols].sum(axis=1), grouped[feature])
importance = pd.read_csv(out / "global_importance.csv").set_index("feature")
repeats = pd.read_csv(out / "permutation_repeats.csv").set_index("repeat")
assert len(repeats) == meta["permutation_repeats"]
assert np.allclose(grouped.abs().mean(), importance.loc[grouped.columns, "mean_absolute_grouped_shap"])
assert np.allclose(repeats.mean(), importance.loc[repeats.columns, "permutation_ap_drop_mean"])
assert np.allclose(repeats.std(ddof=0), importance.loc[repeats.columns, "permutation_ap_drop_sd"])
assert np.isclose(importance.impurity_importance.sum(), 1)
assert hashlib.sha256((ROOT / "models" / (meta["model"] + ".joblib")).read_bytes()).hexdigest() == meta["model_sha256"]
local = pd.read_csv(out / "local_explanations.csv")
for case, rows in local.groupby("case"):
    cid = meta["cases"][case]
    assert set(rows.customerID) == {cid}
    assert np.allclose(rows.set_index("feature").loc[grouped.columns, "shap_value"], grouped.loc[cid])
print("PASS: customer IDs, prior predictions, local additivity, dummy aggregation, global ranks, permutation summaries, local cases, unchanged model.")
