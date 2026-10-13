"""Train three churn classifiers with fold-local preprocessing and CV selection."""
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from clean_data import normalize_records, NUMERIC
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
import numpy as np
import pandas as pd
import sklearn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve)


def metrics(y, probability):
    predicted = (probability >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, predicted), balanced_accuracy=balanced_accuracy_score(y, predicted),
                precision=precision_score(y, predicted, zero_division=0), recall=recall_score(y, predicted),
                f1=f1_score(y, predicted, zero_division=0), roc_auc=roc_auc_score(y, probability),
                average_precision=average_precision_score(y, probability), brier=brier_score_loss(y, probability),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def main():
    out = ROOT / "reports/models"
    artifacts = ROOT / "models"
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    source = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df, _ = normalize_records(pd.read_csv(source))
    df = df.set_index("customerID")
    manifest = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
    assert manifest.customerID.is_unique and set(manifest.customerID) == set(df.index)
    assert set(manifest.split) == {"train", "test"}
    train_ids = manifest.loc[manifest.split.eq("train")].sort_values("export_row").customerID
    test_ids = manifest.loc[manifest.split.eq("test")].sort_values("export_row").customerID
    assert set(train_ids).isdisjoint(test_ids)
    X = df.drop(columns="Churn")
    X["SeniorCitizen"] = X.SeniorCitizen.astype(int).astype(str)
    y = df.Churn.map({"No": 0, "Yes": 1})
    Xtrain, Xtest, ytrain, ytest = X.loc[train_ids], X.loc[test_ids], y.loc[train_ids], y.loc[test_ids]
    cats = [c for c in X if c not in NUMERIC]
    def pipeline(model):
        prep = ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)])
        return Pipeline([("preprocess", prep), ("model", model)])
    specs = {
        "logistic_regression": (LogisticRegression(max_iter=3000, random_state=42),
                                {"model__C": [.1, 1, 10], "model__class_weight": [None, "balanced"]}),
        "decision_tree": (DecisionTreeClassifier(random_state=42),
                          {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [10, 30]}),
        "random_forest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
                          {"model__max_depth": [8, None], "model__min_samples_leaf": [2, 10]})}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fitted, choices = {}, {}
    for name, (model, grid) in specs.items():
        print(f"Training {name}: five-fold stratified CV", flush=True)
        search = GridSearchCV(pipeline(model), grid, cv=cv, scoring={"ap": "average_precision", "roc_auc": "roc_auc", "f1": "f1"},
                              refit="ap", n_jobs=1, error_score="raise")
        search.fit(Xtrain, ytrain)
        pd.DataFrame(search.cv_results_).to_csv(out / f"{name}_cv.csv", index=False)
        choices[name] = dict(params=search.best_params_, cv_average_precision=search.best_score_,
                             cv_ap_std=float(search.cv_results_["std_test_ap"][search.best_index_]))
        fitted[name] = search.best_estimator_
        joblib.dump(search.best_estimator_, artifacts / f"{name}.joblib", compress=3)
    # Select exclusively using training-fold scores, before inspecting held-out scores.
    selected = max(choices, key=lambda name: choices[name]["cv_average_precision"])
    baseline = pipeline(DummyClassifier(strategy="prior"))
    baseline.fit(Xtrain, ytrain)
    fitted["majority_baseline"] = baseline
    predictions = pd.DataFrame({"customerID": test_ids.to_numpy(), "actual_churn": ytest.to_numpy()})
    rows = []
    for name, model in fitted.items():
        probability = model.predict_proba(Xtest)[:, 1]
        predictions[name + "_probability"] = probability
        predictions[name + "_prediction"] = (probability >= .5).astype(int)
        rows.append(dict(model=name, **metrics(ytest, probability)))
        if name != "majority_baseline":
            restored = joblib.load(artifacts / f"{name}.joblib")
            assert np.allclose(restored.predict_proba(Xtest)[:, 1], probability)
    results = pd.DataFrame(rows).set_index("model")
    results.to_csv(out / "test_metrics.csv")
    predictions.to_csv(out / "test_predictions.csv", index=False)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in specs:
        probability = predictions[name + "_probability"]
        fpr, tpr, _ = roc_curve(ytest, probability)
        precision, recall, _ = precision_recall_curve(ytest, probability)
        axes[0].plot(fpr, tpr, label=f"{name.replace('_', ' ')} ({results.loc[name, 'roc_auc']:.3f})")
        axes[1].plot(recall, precision, label=f"{name.replace('_', ' ')} ({results.loc[name, 'average_precision']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.5)
    axes[1].axhline(ytest.mean(), color="gray", linestyle="--", label="Test churn prevalence")
    axes[0].set(title="ROC curves (AUC)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision–recall curves (average precision)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out / "evaluation_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name in zip(axes, specs):
        cm = confusion_matrix(ytest, predictions[name + "_prediction"], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set(title=name.replace("_", " ").title(), xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    metadata = dict(selected_model=selected, selection_metric="Training five-fold mean average precision", candidates=choices,
                    threshold=.5, train_rows=len(ytrain), test_rows=len(ytest), train_churn=int(ytrain.sum()), test_churn=int(ytest.sum()),
                    input_columns=X.columns.tolist(), categorical_columns=cats, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    python_packages={"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
                    created_utc=datetime.now(timezone.utc).isoformat())
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    lines = ["| Model | CV average precision ± fold SD | Test ROC-AUC | Test AP | Precision | Recall | F1 | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.iterrows():
        cv_text = f"{choices[name]['cv_average_precision']:.3f} ± {choices[name]['cv_ap_std']:.3f}" if name in choices else "N/A"
        lines.append(f"| {name} | {cv_text} | {r.roc_auc:.3f} | {r.average_precision:.3f} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.accuracy:.3f} |")
    selected_result = results.loc[selected]
    report = f"""# Churn model training and evaluation

Trained Logistic Regression, Decision Tree, and Random Forest with a majority-class baseline. Selected **{selected.replace('_', ' ')}** using training-only five-fold mean average precision. The evaluation split contains {len(ytest):,} records ({ytest.sum():,} churners); training contains {len(ytrain):,}.

## Comparison

{chr(10).join(lines)}

AP means average precision, a summary of the precision–recall relationship, not trapezoidal PR-AUC. Fold SD describes variation across folds and is not a confidence interval. Classification metrics use the predeclared 0.50 threshold. Full metrics include balanced accuracy, Brier score, and confusion counts in `test_metrics.csv`.

![ROC and precision recall](evaluation_curves.png)

![Confusion matrices](confusion_matrices.png)

## Method

Used the original raw CSV and existing split manifest. customerID is excluded. Missing TotalCharges is median-imputed within each training fold; numeric predictors are standardized and categorical predictors one-hot encoded within the same pipeline. SeniorCitizen is categorical. Unknown categories are ignored at transformation. No globally imputed or encoded exports are used to train models.

Each candidate uses five stratified folds, shuffled with seed 42. Grid searches compare logistic C and class weighting, tree depth and leaf size, and forest depth and leaf size (200 trees). Exact grids and all fold scores are in the script and CV CSVs. Each winning pipeline is refitted on the training partition only. Model-family selection uses CV AP, never test ranking. No threshold tuning or oversampling is performed.

## Interpretation

At threshold 0.50 the selected model identifies {int(selected_result.tp)} churners, misses {int(selected_result.fn)}, and flags {int(selected_result.fp)} retained customers. These tradeoffs matter more for retention outreach than accuracy alone. Before deployment, choose a threshold using validation data and actual contact costs, retention success, and customer value; this assignment does not assume those quantities. Scores are uncalibrated estimates and should not be presented as validated individual risk probabilities.

## Evaluation limitation

The existing test partition was included in prior full-dataset descriptive EDA. It is excluded from fitting and grid selection here, but is **not a pristine independent confirmation set**. These results are an exploratory benchmark. Changing the random split would not undo prior exposure; use fresh independent or later-period data before claiming production generalization. No timestamps support temporal validation, and no direct complaint or detailed usage features exist in the source.

## Deliverables and verification

- `scripts/train_models.py`: reproducible training and evaluation.
- `models/*.joblib`: three fitted preprocessing-and-model pipelines. Load only trusted artifacts, using compatible versions recorded in metadata.
- `test_predictions.csv`: evaluation customer IDs, actual labels, scores, and 0.50-threshold predictions for each model and baseline. These are evaluation outputs, not a live outreach list.
- `metadata.json`: selected model, parameters, versions, source hash, and split counts.
- Separate CV tables, metrics, and two evaluation figures.

Verified disjoint customer IDs and complete split coverage, and reloaded each saved pipeline to confirm identical probabilities. Run `python scripts/verify_models.py` to independently recalculate every reported metric from the prediction export. Power BI can import `test_predictions.csv` linked on customerID, but must label its scope as the evaluation subset; it is not the full customer base.
"""
    (out / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(results.to_string(), flush=True)
    print(f"Selected by training CV: {selected}", flush=True)


if __name__ == "__main__":
    main()
