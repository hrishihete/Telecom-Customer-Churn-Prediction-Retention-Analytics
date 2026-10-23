"""Explain the saved selected forest without refitting or feature selection."""
import os
import json
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from clean_data import normalize_records, NUMERIC


def main():
    out = ROOT / "reports/explainability"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "reports/models/metadata.json").read_text())
    model_path = ROOT / "models" / (meta["selected_model"] + ".joblib")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    pipeline = joblib.load(model_path)
    source = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["source_sha256"]
    df, _ = normalize_records(pd.read_csv(source))
    df = df.set_index("customerID")
    manifest = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
    ids = manifest[manifest.split.eq("test")].sort_values("export_row").customerID
    X = df.loc[ids, meta["input_columns"]].copy()
    X.SeniorCitizen = X.SeniorCitizen.astype(int).astype(str)
    y = df.loc[ids, "Churn"].eq("Yes").astype(int)
    prep, forest = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    assert np.array_equal(forest.classes_, [0, 1])
    transformed = prep.transform(X)
    names = prep.get_feature_names_out()
    # Derive dummy-to-source mapping from fitted encoder order, never string splitting.
    mapping = list(NUMERIC)
    for col, levels in zip(meta["categorical_columns"], prep.named_transformers_["categorical"].categories_):
        mapping.extend([col] * len(levels))
    assert len(mapping) == transformed.shape[1] == len(names)
    probability = pipeline.predict_proba(X)[:, 1]
    prior = pd.read_csv(ROOT / "reports/models/test_predictions.csv").set_index("customerID")
    assert np.allclose(probability, prior.loc[ids, meta["selected_model"] + "_probability"])
    print("Computing Tree SHAP for every evaluation customer", flush=True)
    explainer = shap.TreeExplainer(forest, feature_perturbation="tree_path_dependent", model_output="raw")
    values = explainer.shap_values(transformed, check_additivity=True)
    values = values[1] if isinstance(values, list) else np.asarray(values)[:, :, 1]
    base = float(np.asarray(explainer.expected_value)[1])
    error = np.abs(base + values.sum(axis=1) - probability)
    assert error.max() < 1e-6
    encoded = pd.DataFrame(values, index=ids.to_numpy(), columns=names)
    encoded.index.name = "customerID"
    encoded.to_csv(out / "shap_encoded.csv")
    grouped = pd.DataFrame({col: values[:, np.array(mapping) == col].sum(axis=1) for col in X}, index=ids.to_numpy())
    grouped.index.name = "customerID"
    grouped.to_csv(out / "shap_by_customer.csv")
    pd.DataFrame({"encoded_feature": names, "source_feature": mapping}).to_csv(out / "feature_mapping.csv", index=False)
    explanation = pd.DataFrame({"customerID": ids.to_numpy(), "actual_churn": y.to_numpy(),
                                "base_probability": base, "predicted_probability": probability,
                                "reconstruction_error": error})
    explanation.to_csv(out / "explanation_predictions.csv", index=False)
    importance = grouped.abs().mean().rename("mean_absolute_grouped_shap").to_frame()
    importance["impurity_importance"] = pd.Series(forest.feature_importances_).groupby(mapping).sum()
    print("Computing raw-column permutation importance: 10 repeats", flush=True)
    perm = permutation_importance(pipeline, X, y, scoring="average_precision", n_repeats=10, random_state=42, n_jobs=1)
    importance["permutation_ap_drop_mean"] = pd.Series(perm.importances_mean, index=X.columns)
    importance["permutation_ap_drop_sd"] = pd.Series(perm.importances_std, index=X.columns)
    importance = importance.sort_values("mean_absolute_grouped_shap", ascending=False)
    importance.index.name = "feature"
    importance.to_csv(out / "global_importance.csv")
    pd.DataFrame(perm.importances.T, columns=X.columns).rename_axis("repeat").to_csv(out / "permutation_repeats.csv")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, col, title in zip(axes, ["mean_absolute_grouped_shap", "permutation_ap_drop_mean", "impurity_importance"],
                              ["Mean absolute grouped SHAP", "Permutation importance: AP decrease", "Forest impurity importance"]):
        t = importance.sort_values(col).tail(10)
        ax.barh(t.index, t[col], color="#247c93", xerr=t.permutation_ap_drop_sd if col.startswith("permutation") else None)
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "importance_comparison.png", dpi=160)
    plt.close(fig)
    shap.summary_plot(values, transformed, feature_names=names, max_display=15, show=False)
    plt.title("Encoded-feature SHAP: contribution toward churn")
    plt.tight_layout()
    plt.savefig(out / "shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()
    # Select cases by score alone, not correctness: highest, lowest, and nearest threshold.
    cases = {"highest_score": int(np.argmax(probability)), "lowest_score": int(np.argmin(probability)),
             "near_threshold": int(np.argmin(np.abs(probability - .5)))}
    local_rows = []
    for label, i in cases.items():
        row = grouped.iloc[i]
        for feature in row.index:
            local_rows.append(dict(case=label, customerID=ids.iloc[i], feature=feature,
                                   observed_value=X.iloc[i][feature], shap_value=row[feature]))
        local = shap.Explanation(values=row.to_numpy(), base_values=base,
                                 data=X.iloc[i].astype(str).to_numpy(), feature_names=X.columns.tolist())
        shap.plots.waterfall(local, max_display=12, show=False)
        plt.title(f"{label.replace('_', ' ')} | {ids.iloc[i]} | score={probability[i]:.3f}")
        plt.savefig(out / f"waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
    pd.DataFrame(local_rows).to_csv(out / "local_explanations.csv", index=False)
    # Group-level signed contributions supplement the magnitude-only global ranking.
    directions = []
    for feature in importance.index[:6]:
        levels = pd.qcut(X[feature], q=4, duplicates="drop").astype(str) if feature in NUMERIC else X[feature].astype(str)
        temp = pd.DataFrame({"level": levels.to_numpy(), "contribution": grouped[feature].to_numpy()})
        for level, group in temp.groupby("level"):
            directions.append(dict(feature=feature, value_group=level, customers=len(group), mean_shap=group.contribution.mean()))
    pd.DataFrame(directions).to_csv(out / "driver_direction.csv", index=False)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
    audit = dict(model=meta["selected_model"], model_sha256=model_hash, source_sha256=meta["source_sha256"],
                 shap_version=shap.__version__, customers=len(X), encoded_features=len(names), source_features=len(X.columns),
                 base_probability=base, max_additivity_error=float(error.max()), baseline_ap=average_precision_score(y, probability),
                 explainer="TreeExplainer", perturbation="tree_path_dependent", model_output="raw",
                 explained_output="RandomForestClassifier class 1 probability", permutation_repeats=10, seed=42,
                 cases={label: str(ids.iloc[i]) for label, i in cases.items()})
    (out / "metadata.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    table = ["| Feature | Mean absolute grouped SHAP | Mean AP decrease | Permutation SD |", "|---|---:|---:|---:|"]
    for feature, r in importance.head(10).iterrows():
        table.append(f"| {feature} | {r.mean_absolute_grouped_shap:.4f} | {r.permutation_ap_drop_mean:.4f} | {r.permutation_ap_drop_sd:.4f} |")
    direction_lines = []
    for feature in importance.index[:5]:
        rows = [r for r in directions if r["feature"] == feature]
        high, low = max(rows, key=lambda r:r["mean_shap"]), min(rows, key=lambda r:r["mean_shap"])
        direction_lines.append(f"- **{feature}**: `{high['value_group']}` has mean contribution {high['mean_shap']:+.4f}; `{low['value_group']}` has {low['mean_shap']:+.4f}. These are group averages, not rules applying to every customer.")
    report = f"""# Feature importance and SHAP: churn model drivers

Explained the previously selected **Random Forest**, unchanged, for all **{len(X):,} evaluation customers**. Highest mean absolute grouped SHAP drivers: **{', '.join(importance.index[:5])}**.

## Global results

{chr(10).join(table)}

![Importance comparison](importance_comparison.png)

SHAP magnitudes are in probability units: 0.01 equals one percentage point of model score. Positive signed SHAP pushes toward churn; negative pushes away. The global table reports mean absolute contributions and therefore does not itself indicate direction.

## Direction of contributions

{chr(10).join(direction_lines)}

Numeric groups are evaluation-set quartiles, used only to summarize explanations. Exact group sizes and signed averages are in `driver_direction.csv`.

![SHAP beeswarm](shap_beeswarm.png)

The beeswarm shows encoded columns. Red/blue denotes high/low transformed feature values, not actual churn labels. Numeric values were standardized by the saved pipeline; one-hot values are 0/1. A missing one-hot category may itself contribute to a prediction.

## Individual predictions

The training-path reference probability is **{base:.4f}**. Base plus all contributions reconstructs each churn score; maximum absolute error was **{error.max():.2e}**. Three examples were selected deterministically by highest score, lowest score, and closest to 0.50, without using correctness.

![Highest score](waterfall_highest_score.png)

![Lowest score](waterfall_lowest_score.png)

![Near threshold](waterfall_near_threshold.png)

Waterfalls aggregate dummy contributions into their original features. `local_explanations.csv` contains every contribution and observed value for these cases. Predictions are uncalibrated model estimates, not proven individual probabilities of future churn.

## Method and validation

Loaded the existing pipeline, raw records and split manifest; no model training, selection, or threshold changes. Confirmed predictions match the prior export. Tree SHAP uses `tree_path_dependent` and `model_output='raw'`; for this scikit-learn forest raw output is class probability, verified against `predict_proba`. The reference uses the fitted trees' training path counts, not a new test-derived background. See [official TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

For categorical features, signed encoded SHAP contributions are summed within each customer before taking the mean absolute value. This preserves local additivity; it is an aggregation of encoded-feature attributions, not a recomputed coalition/group Shapley game. Mapping is exported explicitly. Impurity importance sums encoded columns. Permutation importance shuffles each original column through the complete pipeline, preserving valid one-hot combinations, and measures mean decrease in average precision over 10 seeded repeats. Repeat SD reflects permutation variability, not a population confidence interval. Negative importance is retained.

## Interpretation and limits

Importance measures answer different questions. Impurity importance can favor features with many split opportunities. Correlated tenure, charges, contracts and service indicators may share or mask importance. Shuffling can create unrealistic combinations across correlated columns; SHAP dependence assumptions also affect attribution. Do not interpret a ranking disagreement as a bug or a zero score as proof a variable is irrelevant.

These explain **model decisions, not causal drivers of actual churn**. The prior EDA included the evaluation partition; this remains exploratory analysis, not independent confirmation. Demographic associations must not be used to justify differential treatment. No direct complaint records or detailed consumption telemetry exist in this dataset.

Use the service, contract, tenure and billing patterns as hypotheses for customer research and controlled retention experiments. Inspect individual explanations before assuming a global driver applies to a particular customer; changing a feature does not guarantee a lower real-world churn rate. No outreach or deployment was performed.

## Reproduce and deliverables

Install `requirements-explainability-lock.txt` in a compatible Python environment, then run `python scripts/explain_model.py` and `python scripts/verify_explanations.py`. Outputs include global tables, encoded and grouped customer-level SHAP values, 10-repeat permutation results, driver directions, five charts, local examples, and provenance metadata. `shap_by_customer.csv` joins to `explanation_predictions.csv` on customerID for Power BI exploration; signed contributions must not be displayed as causal effects.
"""
    (out / "findings.md").write_text(report, encoding="utf-8")
    print(importance.head(10).to_string(), flush=True)
    print(f"PASS: prediction identity, SHAP additivity ({error.max():.2e}), unchanged model hash", flush=True)


if __name__ == "__main__":
    main()
