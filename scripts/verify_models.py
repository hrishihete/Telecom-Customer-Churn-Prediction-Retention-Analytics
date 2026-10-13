"""Recompute reported metrics and verify split and prediction integrity."""
import json
import numpy as np
import pandas as pd
import joblib
from clean_data import normalize_records
from train_models import ROOT, metrics

out = ROOT / "reports/models"
pred = pd.read_csv(out / "test_predictions.csv")
reported = pd.read_csv(out / "test_metrics.csv").set_index("model")
manifest = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
raw = pd.read_csv(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv").set_index("customerID")
assert pred.customerID.is_unique
assert set(pred.customerID) == set(manifest.loc[manifest.split.eq("test"), "customerID"])
assert np.array_equal(pred.actual_churn, raw.loc[pred.customerID, "Churn"].map({"No": 0, "Yes": 1}))
for name in reported.index:
    score = pred[name + "_probability"]
    assert score.between(0, 1).all()
    assert np.array_equal(pred[name + "_prediction"], (score >= .5).astype(int))
    for key, value in metrics(pred.actual_churn, score).items():
        assert np.isclose(reported.loc[name, key], value), (name, key)
meta = json.loads((out / "metadata.json").read_text())
assert meta["selected_model"] == max(meta["candidates"], key=lambda k: meta["candidates"][k]["cv_average_precision"])
normalized, _ = normalize_records(raw.reset_index())
normalized = normalized.set_index("customerID")
train_ids = manifest.loc[manifest.split.eq("train"), "customerID"]
expected_median = normalized.loc[train_ids, "TotalCharges"].median()
Xtest = normalized.loc[pred.customerID].drop(columns="Churn")
Xtest["SeniorCitizen"] = Xtest.SeniorCitizen.astype(int).astype(str)
for name in meta["candidates"]:
    model = joblib.load(ROOT / "models" / f"{name}.joblib")
    assert "customerID" not in model.feature_names_in_ and "Churn" not in model.feature_names_in_
    fitted_median = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_[2]
    assert np.isclose(fitted_median, expected_median)
    assert np.allclose(model.predict_proba(Xtest)[:, 1], pred[name + "_probability"])
print("PASS: evaluation IDs, labels, probabilities, threshold predictions, all reported metrics, CV-only selection, training-only imputation, excluded identifier/target, and saved-model inference.")
