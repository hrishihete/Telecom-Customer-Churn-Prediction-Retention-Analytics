"""Independently reconcile calibration, outreach policies, artifacts and SQL outputs."""
import hashlib
import json
import sqlite3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from clean_data import normalize_records, NUMERIC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/calibration"


def check_close(actual, expected):
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12), (actual, expected)


def independent_metrics(y, p, t):
    pred = np.asarray(p) >= t
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
        f1=f1_score(y, pred, zero_division=0), tp=tp, fp=fp, fn=fn, tn=tn,
        contacts=pred.sum(), contact_rate=pred.mean())


def check_probability_metrics(row, y, p):
    funcs = {"brier": brier_score_loss, "log_loss": log_loss, "roc_auc": roc_auc_score, "average_precision": average_precision_score}
    for name, fn in funcs.items():
        check_close(row[name], fn(y, p))
    # Independently aggregate fixed-width reliability with explicit endpoint handling.
    ece = 0.
    counts = []
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        counts.append(int(mask.sum()))
        if mask.any():
            ece += mask.mean() * abs(np.asarray(y)[mask].mean() - p[mask].mean())
    assert sum(counts) == len(p)
    check_close(row.ece_10, ece)


def main():
    checks = []
    meta = json.loads((OUT / "metadata.json").read_text())
    model_dir = ROOT / "models/calibration"
    policy = json.loads((model_dir / "policy.json").read_text())
    for path, key in [(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", "source_sha256"),
                      (ROOT / "data/processed/split_manifest.csv", "original_manifest_sha256"),
                      (ROOT / "models" / f"{meta['base_family']}.joblib", "legacy_model_sha256"),
                      (model_dir / "outreach_model.joblib", "model_sha256"),
                      (model_dir / "base_model.joblib", "base_model_sha256")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta[key]
    assert policy["model_sha256"] == meta["model_sha256"]
    checks.append("Source, original split, legacy model, refitted base and outreach artifact SHA-256 values match metadata and policy.")
    raw, _ = normalize_records(pd.read_csv(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    raw = raw.set_index("customerID")
    X = raw.drop(columns="Churn")
    X["SeniorCitizen"] = X.SeniorCitizen.astype(int).astype(str)
    y = raw.Churn.map({"No": 0, "Yes": 1})
    manifest = pd.read_csv(OUT / "split_manifest.csv")
    original = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
    assert manifest.customerID.is_unique and set(manifest.customerID) == set(raw.index)
    assert set(manifest.role) == {"base_fit", "calibration", "threshold_tuning", "historical_test"}
    parts = {role: manifest.loc[manifest.role.eq(role), "customerID"].to_numpy() for role in manifest.role.unique()}
    assert set(parts["historical_test"]) == set(original.loc[original.split.eq("test"), "customerID"])
    train = original.loc[original.split.eq("train")].sort_values("export_row").customerID.to_numpy()
    b, reserve = train_test_split(train, test_size=.4, stratify=y.loc[train], random_state=meta["seed"])
    c, t = train_test_split(reserve, test_size=.5, stratify=y.loc[reserve], random_state=meta["seed"] + 1)
    for role, ids in [("base_fit", b), ("calibration", c), ("threshold_tuning", t)]:
        assert np.array_equal(parts[role], ids)
    for role, ids in parts.items():
        assert len(ids) == meta["split_counts"][role]["rows"]
        assert y.loc[ids].sum() == meta["split_counts"][role]["churners"]
    checks.append("All 7,043 IDs have exactly one role; stratified splits reproduce and original test membership is unchanged.")
    base = joblib.load(model_dir / "base_model.joblib")
    model = joblib.load(model_dir / "outreach_model.joblib")
    assert list(base.feature_names_in_) == X.columns.tolist()
    assert "customerID" not in base.feature_names_in_ and "Churn" not in base.feature_names_in_
    stats = base.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_
    assert np.allclose(stats, X.loc[b, NUMERIC].median().to_numpy())
    checks.append("Saved pipeline excludes target/ID and numeric imputation matches base-fit records only.")

    oof = pd.read_csv(OUT / "calibration_oof_predictions.csv", float_precision="round_trip")
    assert np.array_equal(oof.customerID, c)
    assert np.array_equal(oof.actual_churn, y.loc[c])
    assert np.allclose(oof.uncalibrated_probability, base.predict_proba(X.loc[c])[:, 1])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=meta["seed"] + 2)
    for fold, (fit_idx, val_idx) in enumerate(folds.split(c, y.loc[c])):
        assert oof.iloc[val_idx].fold.eq(fold).all()
        assert manifest.set_index("customerID").loc[c[val_idx], "calibration_validation_fold"].eq(fold).all()
        for method in ["sigmoid", "isotonic"]:
            cal = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X.loc[c[fit_idx]], y.loc[c[fit_idx]])
            assert np.allclose(cal.predict_proba(X.loc[c[val_idx]])[:, 1], oof.iloc[val_idx][method + "_probability"])
    cv = pd.read_csv(OUT / "calibration_cv_metrics.csv")
    for row in cv.itertuples(index=False):
        check_probability_metrics(pd.Series(row._asdict()), oof.actual_churn, oof[row.method + "_probability"].to_numpy())
    selected = cv.sort_values(["log_loss", "brier", "method"]).iloc[0].method
    assert selected == meta["selected_method"] == policy["method"]
    assert cv.loc[cv.selected, "method"].tolist() == [selected]
    checks.append("All calibration OOF probabilities reproduce from disjoint fold fits; method selection uses only OOF log loss.")

    pred = pd.read_csv(OUT / "evaluation_predictions.csv", float_precision="round_trip")
    assert pred.customerID.is_unique
    assert set(pred.customerID) == set(t) | set(parts["historical_test"])
    assert np.array_equal(pred.actual_churn, y.loc[pred.customerID])
    assert pred.selected_method.eq(selected).all()
    assert np.array_equal(pred.selected_probability, pred[selected + "_probability"])
    assert np.allclose(model.predict_proba(X.loc[pred.customerID])[:, 1], pred.selected_probability)
    candidates = {"uncalibrated": base}
    for method in ["sigmoid", "isotonic"]:
        candidates[method] = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X.loc[c], y.loc[c])
    for method, candidate in candidates.items():
        assert np.allclose(candidate.predict_proba(X.loc[pred.customerID])[:, 1], pred[method + "_probability"])
        assert pred[method + "_probability"].between(0, 1).all()
    for name, threshold in policy["thresholds"].items():
        assert np.array_equal(pred[name + "_flag"], pred.selected_probability.ge(threshold))
    checks.append("All evaluation IDs, labels, methods, probabilities and exact >= policy flags match saved-artifact inference.")
    tune = pred[pred.role.eq("threshold_tuning")]
    assert set(tune.customerID) == set(t)
    sweep = pd.read_csv(OUT / "threshold_sweep.csv", float_precision="round_trip")
    assert np.array_equal(sweep.threshold, np.unique(np.r_[0., tune.selected_probability, 1.]))
    for row in sweep.itertuples(index=False):
        actual = independent_metrics(tune.actual_churn, tune.selected_probability, row.threshold)
        for key, value in actual.items():
            check_close(getattr(row, key), value)
    balanced = sweep.sort_values(["f1", "precision", "threshold"], ascending=False).iloc[0]
    assert balanced.threshold == policy["thresholds"]["balanced_f1"]
    for name, constraint, bound, objective in [("precision_first", "precision", .65, "recall"), ("recall_first", "recall", .8, "precision")]:
        eligible = sweep[(sweep[constraint] >= bound) & sweep.contacts.gt(0)]
        if eligible.empty:
            assert name not in policy["thresholds"]
        else:
            best = eligible.sort_values([objective, "f1", "threshold"], ascending=False).iloc[0]
            assert best.threshold == policy["thresholds"][name]
    checks.append("Every tuning boundary and all precision/recall/F1/confusion counts independently reconcile; default and alternative thresholds are optimal under declared tuning rules.")
    metrics = pd.read_csv(OUT / "policy_metrics.csv")
    tune_summary = pd.read_csv(OUT / "tuning_policy_metrics.csv", float_precision="round_trip")
    for row in tune_summary.itertuples(index=False):
        assert row.threshold == policy["thresholds"][row.policy]
        for key, value in independent_metrics(tune.actual_churn, tune.selected_probability, row.threshold).items():
            check_close(getattr(row, key), value)
    for row in metrics.itertuples(index=False):
        part = pred[pred.role.eq(row.role)]
        expected_threshold = .5 if row.policy.endswith("050") else policy["thresholds"][row.policy]
        check_close(row.threshold, expected_threshold)
        for key, value in independent_metrics(part.actual_churn, part[row.method + "_probability"], expected_threshold).items():
            check_close(getattr(row, key), value)
    probability_metrics = pd.read_csv(OUT / "probability_metrics.csv")
    bins = pd.read_csv(OUT / "reliability_bins.csv")
    for _, row in probability_metrics.iterrows():
        part = pred[pred.role.eq(row.role)]
        p = part[row.method + "_probability"].to_numpy()
        check_probability_metrics(row, part.actual_churn, p)
        r = bins[(bins.role == row.role) & (bins.method == row.method)]
        assert r["count"].sum() == len(part) and len(r) == 10
        for binrow in r.itertuples(index=False):
            mask = (p >= binrow.lower) & ((p < binrow.upper) if binrow.bin < 9 else (p <= 1))
            assert mask.sum() == binrow.count
            if mask.any():
                check_close(binrow.mean_probability, p[mask].mean())
                check_close(binrow.observed_churn_rate, part.actual_churn.to_numpy()[mask].mean())
    checks.append("Tuning summaries, all evaluation policy/probability metrics and every reliability-bin count/rate reconcile independently.")
    with sqlite3.connect(":memory:") as db:
        pred.to_sql("evaluation_predictions", db, index=False)
        result = pd.read_sql_query((ROOT / "sql/calibration_policy_metrics.sql").read_text(), db)
    for row in result.itertuples(index=False):
        expected = metrics[(metrics.role == row.role) & (metrics.policy == "balanced_f1")].iloc[0]
        for key in ["tp", "fp", "fn", "tn", "contacts", "contact_rate", "precision", "recall", "f1"]:
            check_close(getattr(row, key), expected[key])
    checks.append("Executed SQL reconciles recommended-policy counts, precision, recall, F1 and workload for both roles.")
    # Explicit equality/tie edge cases guard the outreach decision semantics.
    from calibrate_thresholds import threshold_table, choose_policies
    edge = threshold_table(np.array([0, 1, 1, 0]), np.array([0., .5, .5, 1.]))
    assert int(edge.loc[edge.threshold.eq(.5), "contacts"].iloc[0]) == 3
    assert int(edge.loc[edge.threshold.eq(1.), "contacts"].iloc[0]) == 1
    assert int(edge.loc[edge.threshold.eq(0.), "contacts"].iloc[0]) == 4
    assert "balanced_f1" in choose_policies(edge)
    checks.append("Threshold equality, tied scores and 0/1 endpoints pass targeted decision-rule checks.")
    report = "# Calibration verification\n\nPASS — all checks below completed.\n\n" + "\n".join(f"- {c}" for c in checks) + "\n\nReproduce: `python scripts/verify_calibration.py`. Charts are visually reviewed separately. This verifies computation, not independent production generalization.\n"
    (OUT / "verification.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
