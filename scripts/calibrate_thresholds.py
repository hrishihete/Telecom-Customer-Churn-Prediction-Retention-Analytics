"""Compare probability calibration and freeze outreach thresholds without test tuning."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from clean_data import normalize_records

OUT = ROOT / "reports/calibration"
MODEL_DIR = ROOT / "models/calibration"
SEED = 20261120
METHODS = ("uncalibrated", "sigmoid", "isotonic")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    raw, _ = normalize_records(pd.read_csv(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    raw = raw.set_index("customerID")
    X = raw.drop(columns="Churn")
    X["SeniorCitizen"] = X.SeniorCitizen.astype(int).astype(str)
    y = raw.Churn.map({"No": 0, "Yes": 1})
    original = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
    assert original.customerID.is_unique and set(original.customerID) == set(X.index)
    assert set(original.split) == {"train", "test"}
    train = original.loc[original.split.eq("train")].sort_values("export_row").customerID.to_numpy()
    test = original.loc[original.split.eq("test")].sort_values("export_row").customerID.to_numpy()
    return X, y, train, test


def reliability(y, p, bins=10):
    frame = pd.DataFrame({"actual": np.asarray(y), "score": p})
    frame["bin"] = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    grouped = frame.groupby("bin").agg(count=("actual", "size"),
        mean_probability=("score", "mean"), observed_churn_rate=("actual", "mean"))
    grouped = grouped.reindex(range(bins))
    grouped["count"] = grouped["count"].fillna(0).astype(int)
    grouped["lower"] = np.arange(bins) / bins
    grouped["upper"] = (np.arange(bins) + 1) / bins
    return grouped.reset_index()


def probability_metrics(y, p):
    r = reliability(y, p)
    ece = (r["count"] * abs(r.mean_probability - r.observed_churn_rate)).sum() / len(y)
    return dict(brier=brier_score_loss(y, p), log_loss=log_loss(y, p, labels=[0, 1]),
        ece_10=float(ece), roc_auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p))


def threshold_table(y, p):
    """Every distinct >= decision boundary, including endpoints; all scores stay unrounded."""
    y, p = np.asarray(y), np.asarray(p)
    thresholds = np.unique(np.r_[0., p, 1.])
    flagged = p[None, :] >= thresholds[:, None]
    tp = flagged[:, y == 1].sum(axis=1)
    fp = flagged[:, y == 0].sum(axis=1)
    fn, tn = (y == 1).sum() - tp, (y == 0).sum() - fp
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=(tp + fp) != 0)
    recall = tp / (tp + fn)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp), dtype=float), where=(2 * tp + fp + fn) != 0)
    return pd.DataFrame(dict(threshold=thresholds, precision=precision, recall=recall, f1=f1,
        tp=tp, fp=fp, fn=fn, tn=tn, contacts=tp + fp, contact_rate=(tp + fp) / len(y)))


def choose_policies(table):
    balanced = table.sort_values(["f1", "precision", "threshold"], ascending=False).iloc[0]
    choices = {"balanced_f1": balanced}
    for name, constraint, minimum, objective in [
        ("precision_first", "precision", .65, "recall"),
        ("recall_first", "recall", .80, "precision"),
    ]:
        eligible = table[(table[constraint] >= minimum) & table.contacts.gt(0)]
        if not eligible.empty:
            choices[name] = eligible.sort_values([objective, "f1", "threshold"], ascending=False).iloc[0]
    return choices


def at_threshold(y, p, threshold):
    y, pred = np.asarray(y), np.asarray(p) >= threshold
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    fn, tn = int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred).sum())
    return dict(threshold=float(threshold), precision=tp / (tp + fp) if tp + fp else 0.,
        recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn), tp=tp, fp=fp, fn=fn, tn=tn,
        contacts=tp + fp, contact_rate=(tp + fp) / len(y))


def wilson(successes, total):
    if not total:
        return None, None
    z = 1.959963984540054
    rate = successes / total
    center = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return float(center - half), float(center + half)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y, train, test = load_inputs()
    # 60/20/20 of the original training set; preserve the existing test membership.
    base_ids, reserve = train_test_split(train, test_size=.4, stratify=y.loc[train], random_state=SEED)
    cal_ids, tune_ids = train_test_split(reserve, test_size=.5, stratify=y.loc[reserve], random_state=SEED + 1)
    parts = {"base_fit": base_ids, "calibration": cal_ids, "threshold_tuning": tune_ids, "historical_test": test}
    manifest = pd.concat([pd.DataFrame({"customerID": ids, "role": role}) for role, ids in parts.items()], ignore_index=True)
    manifest["calibration_validation_fold"] = -1
    assert manifest.customerID.is_unique and set(manifest.customerID) == set(X.index)

    prior = json.loads((ROOT / "reports/models/metadata.json").read_text())
    family = prior["selected_model"]
    # Clone discards all learned preprocessing and tree state from the legacy artifact.
    legacy_file = ROOT / "models" / f"{family}.joblib"
    base = clone(joblib.load(legacy_file))
    print(f"Refitting historical {family} configuration on {len(base_ids)} base-fit records", flush=True)
    base.fit(X.loc[base_ids], y.loc[base_ids])
    joblib.dump(base, MODEL_DIR / "base_model.joblib", compress=3)
    oof = pd.DataFrame({"customerID": cal_ids, "actual_churn": y.loc[cal_ids].to_numpy()})
    oof["uncalibrated_probability"] = base.predict_proba(X.loc[cal_ids])[:, 1]
    for method in METHODS[1:]:
        oof[method + "_probability"] = np.nan
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 2)
    for fold, (fit_idx, val_idx) in enumerate(folds.split(cal_ids, y.loc[cal_ids])):
        fit_ids, val_ids = cal_ids[fit_idx], cal_ids[val_idx]
        assert set(fit_ids).isdisjoint(val_ids) and set(fit_ids).isdisjoint(base_ids)
        oof.loc[val_idx, "fold"] = fold
        manifest.loc[manifest.customerID.isin(val_ids), "calibration_validation_fold"] = fold
        for method in METHODS[1:]:
            calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
            calibrated.fit(X.loc[fit_ids], y.loc[fit_ids])
            oof.loc[val_idx, method + "_probability"] = calibrated.predict_proba(X.loc[val_ids])[:, 1]
    assert not oof.isna().any().any()
    oof["fold"] = oof.fold.astype(int)
    comparison = pd.DataFrame([dict(method=m, **probability_metrics(y.loc[cal_ids], oof[m + "_probability"])) for m in METHODS])
    selected = comparison.sort_values(["log_loss", "brier", "method"]).iloc[0]["method"]
    comparison["selected"] = comparison.method.eq(selected)
    comparison.to_csv(OUT / "calibration_cv_metrics.csv", index=False)
    oof.to_csv(OUT / "calibration_oof_predictions.csv", index=False)
    manifest.to_csv(OUT / "split_manifest.csv", index=False)
    candidates = {"uncalibrated": base}
    for method in METHODS[1:]:
        candidates[method] = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X.loc[cal_ids], y.loc[cal_ids])
    selected_model = candidates[selected]
    artifact = MODEL_DIR / "outreach_model.joblib"
    joblib.dump(selected_model, artifact, compress=3)
    # Lock the method before reading tuning outcomes; lock thresholds before test scoring.
    tune_p = selected_model.predict_proba(X.loc[tune_ids])[:, 1]
    curve = threshold_table(y.loc[tune_ids], tune_p)
    policies = choose_policies(curve)
    curve.to_csv(OUT / "threshold_sweep.csv", index=False)
    policy = dict(method=selected, comparison_operator=">=", default_policy="balanced_f1",
        thresholds={name: float(row.threshold) for name, row in policies.items()},
        rule="Maximize tuning F1; ties prefer precision then higher threshold",
        assumptions="Equal precision/recall emphasis; alternative tuning constraints: precision >= 0.65, recall >= 0.80",
        model_sha256=sha(artifact), scope="Historical exploratory evaluation; no production deployment")
    (MODEL_DIR / "policy.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([dict(policy=k, **r.to_dict()) for k, r in policies.items()]).to_csv(OUT / "tuning_policy_metrics.csv", index=False)
    print(f"Locked method: {selected}; thresholds: {policy['thresholds']}", flush=True)

    predictions, prob_rows, bins, policy_rows = [], [], [], []
    for role, ids in [("threshold_tuning", tune_ids), ("historical_test", test)]:
        pred = pd.DataFrame({"customerID": ids, "role": role, "actual_churn": y.loc[ids].to_numpy()})
        for method, model in candidates.items():
            p = model.predict_proba(X.loc[ids])[:, 1]
            pred[method + "_probability"] = p
            prob_rows.append(dict(role=role, method=method, **probability_metrics(y.loc[ids], p)))
            bins.append(reliability(y.loc[ids], p).assign(role=role, method=method))
        pred["selected_probability"] = pred[selected + "_probability"]
        pred["selected_method"] = selected
        for name, threshold in policy["thresholds"].items():
            pred[name + "_flag"] = pred.selected_probability.ge(threshold).astype(int)
        predictions.append(pred)
        specs = [("uncalibrated_050", "uncalibrated", .5), ("selected_050", selected, .5)]
        specs += [(name, selected, t) for name, t in policy["thresholds"].items()]
        for name, method, t in specs:
            r = at_threshold(y.loc[ids], pred[method + "_probability"], t)
            lo_p, hi_p = wilson(r["tp"], r["contacts"])
            lo_r, hi_r = wilson(r["tp"], r["tp"] + r["fn"])
            policy_rows.append(dict(role=role, policy=name, method=method, **r,
                precision_ci_low=lo_p, precision_ci_high=hi_p, recall_ci_low=lo_r, recall_ci_high=hi_r))
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "evaluation_predictions.csv", index=False)
    pd.DataFrame(prob_rows).to_csv(OUT / "probability_metrics.csv", index=False)
    pd.concat(bins, ignore_index=True).to_csv(OUT / "reliability_bins.csv", index=False)
    metrics = pd.DataFrame(policy_rows)
    metrics.to_csv(OUT / "policy_metrics.csv", index=False)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, base_family=family,
        base_params=prior["candidates"][family]["params"], selected_method=selected,
        selection="Five-fold out-of-fold calibration-subset log loss; ties: Brier then method name",
        split_counts={role: dict(rows=len(ids), churners=int(y.loc[ids].sum())) for role, ids in parts.items()},
        calibration_folds=5, source_sha256=sha(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        original_manifest_sha256=sha(ROOT / "data/processed/split_manifest.csv"), legacy_model_sha256=sha(legacy_file),
        model_sha256=sha(artifact), base_model_sha256=sha(MODEL_DIR / "base_model.joblib"),
        python_packages={"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        limitations=["Historical model family and hyperparameters were selected using all original training records; the new subdivisions are not independent of that historical selection.",
            "Original test records were explored in prior EDA and evaluation; no pristine independent confirmation is available.",
            "No timestamps, direct complaints, detailed usage, treatment outcomes, outreach budget or intervention costs are available."])
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    make_charts(predictions[-1], metrics, policy)
    write_report(metadata, comparison, metrics, policy)
    print(metrics[metrics.role.eq("historical_test")].to_string(index=False), flush=True)


def make_charts(test, metrics, policy):
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in METHODS:
        p = test[method + "_probability"]
        r = reliability(test.actual_churn, p)
        axes[0].plot(r.mean_probability, r.observed_churn_rate, "o-", label=method)
        axes[1].hist(p, bins=np.linspace(0, 1, 16), histtype="step", linewidth=2, label=method)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.5)
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Reliability: historical test (10 fixed bins)", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Predicted churn probability", ylabel="Customers", title="Historical test probability distribution")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibration_curves.png", dpi=170)
    plt.close(fig)
    sweep = pd.read_csv(OUT / "threshold_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for column in ["precision", "recall", "f1", "contact_rate"]:
        axes[0].plot(sweep.threshold, sweep[column], label=column.replace("_", " "))
    for name, t in policy["thresholds"].items():
        axes[0].axvline(t, linestyle="--", alpha=.5, label=f"{name}: {t:.3f}")
    axes[0].set(xlabel="Outreach threshold (>=)", ylabel="Metric", title="Threshold choice: tuning subset only", xlim=(0, 1), ylim=(0, 1.03))
    precision, recall, _ = precision_recall_curve(test.actual_churn, test.selected_probability)
    axes[1].plot(recall, precision, label=policy["method"])
    for name in policy["thresholds"]:
        row = metrics[(metrics.role == "historical_test") & (metrics.policy == name)].iloc[0]
        axes[1].scatter(row.recall, row.precision, s=65, label=name)
    axes[1].axhline(test.actual_churn.mean(), color="gray", linestyle="--", label="Churn prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Frozen policies: historical test", xlim=(0, 1), ylim=(0, 1.03))
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_tradeoffs.png", dpi=170)
    plt.close(fig)


def write_report(meta, comparison, metrics, policy):
    test = metrics[metrics.role.eq("historical_test")].set_index("policy")
    balanced, baseline = test.loc["balanced_f1"], test.loc["uncalibrated_050"]
    selection_note = ("Neither sigmoid nor isotonic improved calibration-subset OOF log loss; retain the uncalibrated model and tune its outreach threshold. Calibration was evaluated rather than assumed beneficial."
        if policy["method"] == "uncalibrated" else f"{policy['method'].title()} had the lowest calibration-subset OOF log loss and is selected.")
    constraint_notes = []
    if "precision_first" in test.index:
        constraint_notes.append(f"Precision-first reaches {test.loc['precision_first', 'precision']:.1%} test precision against its 65% tuning target.")
    if "recall_first" in test.index:
        constraint_notes.append(f"Recall-first reaches {test.loc['recall_first', 'recall']:.1%} test recall against its 80% tuning target.")
    rows = ["| Policy | Threshold | Precision | Recall | F1 | Contacts | TP | FP | FN | TN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in test.iterrows():
        rows.append(f"| {name} | {r.threshold:.6f} | {r.precision:.1%} | {r.recall:.1%} | {r.f1:.3f} | {int(r.contacts)} | {int(r.tp)} | {int(r.fp)} | {int(r.fn)} | {int(r.tn)} |")
    cal_table = ["| Method | Calibration OOF log loss | Brier | ECE (10 bins) | Selected |", "|---|---:|---:|---:|---|"]
    for r in comparison.itertuples():
        cal_table.append(f"| {r.method} | {r.log_loss:.5f} | {r.brier:.5f} | {r.ece_10:.5f} | {r.selected} |")
    probability = pd.read_csv(OUT / "probability_metrics.csv")
    ptable = ["| Method | Historical test log loss | Brier | ECE | ROC-AUC | AP |", "|---|---:|---:|---:|---:|---:|"]
    for r in probability[probability.role.eq("historical_test")].itertuples():
        ptable.append(f"| {r.method} | {r.log_loss:.5f} | {r.brier:.5f} | {r.ece_10:.5f} | {r.roc_auc:.4f} | {r.average_precision:.4f} |")
    split_text = "; ".join(f"{role}: {counts['rows']:,} ({counts['churners']:,} churners)" for role, counts in meta["split_counts"].items())
    report = f"""# Weekly assignment: model calibration and threshold tuning

Prepared for Hrishikesh Hete, week ending November 20, 2026.

Recommend **{policy['method']} probabilities with the balanced-F1 threshold {balanced.threshold:.6f}** for a historical outreach pilot design. On the existing {meta['split_counts']['historical_test']['rows']:,}-customer test subset this flags {int(balanced.contacts)} customers, finds {int(balanced.tp)} of {int(balanced.tp + balanced.fn)} churners, and yields **{balanced.precision:.1%} precision / {balanced.recall:.1%} recall / {balanced.f1:.3f} F1**. This is a candidate policy for independent validation, not a deployed campaign.

## Precision and recall tradeoff

{chr(10).join(rows)}

{' '.join(constraint_notes)} These targets are tuning constraints, not guaranteed minimum performance on new data.

Compared with the same refitted uncalibrated model at 0.50, the balanced policy changes contacts by {int(balanced.contacts - baseline.contacts):+d}, true positives by {int(balanced.tp - baseline.tp):+d}, false positives by {int(balanced.fp - baseline.fp):+d}, and missed churners by {int(balanced.fn - baseline.fn):+d}. Both rows use identical base-fit data; they are not comparisons against the legacy model trained on all 5,634 training records. `selected_050` isolates calibration at the default threshold; `balanced_f1` then shows the threshold change.

A contact is a flagged historical record. Precision measures the share of flagged customers who churned; recall measures the share of observed churners flagged. Neither is an intervention success or save rate. Approximate 95% Wilson intervals for the balanced policy are precision {balanced.precision_ci_low:.1%}–{balanced.precision_ci_high:.1%} and recall {balanced.recall_ci_low:.1%}–{balanced.recall_ci_high:.1%}; they describe binomial sampling uncertainty only, not model selection, distribution drift, or prior data exposure.

## Calibration comparison

{chr(10).join(cal_table)}

{selection_note}

Choose the method using pooled five-fold out-of-fold **log loss on calibration records only**; lower is better. Brier score is the first tie-breaker. The uncalibrated option remains eligible so calibration is not assumed to improve results. Calibration CV estimates become selection statistics, not an independent final evaluation. Isotonic is flexible and may overfit with only about 900 calibration-fit observations per fold; extreme probabilities are penalized by log loss. ECE uses ten fixed equal-width bins, includes empty bins with zero weight, and is a descriptive, bin-sensitive measure. Brier and log loss reflect discrimination as well as reliability.

{chr(10).join(ptable)}

Historical test metrics above are reported after the policy was frozen. Differences here do not alter the chosen method or thresholds. Calibration need not improve every metric; isotonic can introduce ranking ties. Threshold tuning changes binary decisions, not the underlying probabilities or probability metrics.

![Calibration and score distributions](calibration_curves.png)

![Precision-recall and threshold tradeoffs](threshold_tradeoffs.png)

## Reproducible design and leakage controls

Split counts: {split_text}.

Preserve the original test membership. Stratify the original training set 60/20/20 into base fitting, calibration, and threshold tuning with seeds {SEED} and {SEED + 1}. Clone the existing {meta['base_family']} configuration, discarding every fitted tree and preprocessing statistic, then refit on base-fit records only. Fit imputation and encoding inside that pipeline; exclude customerID and Churn. No globally processed feature exports are used.

Wrap the fitted base pipeline in FrozenEstimator. For each of five stratified calibration folds (seed {SEED + 2}), fit sigmoid and isotonic calibrators on the other four folds and predict the omitted fold. Base fitting and calibration never overlap. After method selection, fit each calibrator on all calibration records for diagnostic comparison; export the selected method with its fitted preprocessing and base model. Threshold-tuning records enter neither the base fit nor any calibrator fit.

On the threshold subset, enumerate every distinct score plus 0 and 1 using the exact rule **probability >= threshold**. Maximize F1, the harmonic mean of precision and recall; ties prefer higher precision then higher threshold. This assumes equal precision/recall emphasis, not equal financial costs. Precision-first maximizes recall subject to tuning precision >= 65%; recall-first maximizes precision subject to tuning recall >= 80%. Ties for alternatives prefer F1 then higher threshold. Infeasible alternatives are omitted. Targets are illustrative and can fail on the test set or future customers. Keep full-precision thresholds in policy.json; displayed rounded values must not replace them. No final refit follows threshold selection because it would change the score scale.

## Operational handoff and limitations

Use `models/calibration/outreach_model.joblib` and `models/calibration/policy.json` together. The saved method may be uncalibrated if it wins the selection. Require matching artifact SHA-256 and compatible dependencies. Input is the original 19 predictor columns; cast SeniorCitizen to string. Scores above the threshold identify candidates for review, not automatic offers. Current eligibility, opt-outs, contact capacity and recent service history must come from a current customer feed. Rank eligible candidates by probability within a capacity limit; any capacity-based policy needs separate validation.

Use a randomized pilot to measure incremental retention, offer costs and net value. Track precision, recall, contacts, missed churners, calibration and cohort performance when mature outcomes arrive. No cost-optimal threshold or claimed revenue savings is supported by this dataset. There is no validated future prediction horizon. Service subscriptions are available; actual usage and complaints are absent and must not be inferred from subscription flags.

Historical family/hyperparameters were selected on all original training records, including records now reserved for calibration/tuning. The new fits and decisions avoid direct role leakage, but this does not undo earlier selection. The test subset has also appeared in prior EDA and evaluation. These are exploratory results; a new random split cannot make previously examined data pristine. Obtain fresh independent or later-period outcomes before production claims. The existing Power BI historical dashboard and retention queue still use their documented legacy scores and illustrative thresholds; this analysis supplies a separate validated handoff in `power_bi_handoff.md`.

## Deliverables and rerun

Run from the repository root with the pinned model environment:

```powershell
.\\.venv\\Scripts\\python.exe scripts/calibrate_thresholds.py
.\\.venv\\Scripts\\python.exe scripts/verify_calibration.py
```

Install `requirements-models-lock.txt` to reproduce the existing environment. Outputs: method-selection metrics and fold predictions; role/fold manifest; full tuning sweep; tuning and test policy metrics; probability metrics; reliability bins; customer evaluation predictions; two figures; metadata with source and model hashes; paired model/policy artifacts; Power BI handoff; and executable SQL reconciliation. All evaluation rows retain actual labels for analysis and are not a live outreach list. Verification is recorded in `verification.md`.

Method references: [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) and [CalibratedClassifierCV / FrozenEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html). The latter requires the caller to keep fitting and calibration data disjoint.
"""
    (OUT / "findings.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
