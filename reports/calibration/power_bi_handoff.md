# Calibration and threshold tuning: Power BI handoff

This is an import specification for an additional evaluation page. The existing PBIP has not been changed or refreshed. Its legacy scores, 0.50 risk cutoff, retention queue and KPI checks remain tied to the earlier model.

Import `reports/calibration/evaluation_predictions.csv` as **Evaluation** and `policy_metrics.csv` as **PolicyMetrics**. These are historical evaluation records containing both outcomes; do not filter to Churn=No because that removes every true positive and invalidates recall. Use the `role` slicer with **historical_test** selected by default, and keep the tuning view clearly labeled. Do not join by customerID alone to the active-only dashboard table. If relationships are needed, use a unique all-customer dimension and single-direction filters; keep PolicyMetrics disconnected from customer-level tables.

Recommended visuals:

- PolicyMetrics table: policy, threshold, precision, recall, F1, contacts, TP, FP and FN; filter to historical_test. Mark balanced_f1 as the recommendation; constraints for alternatives were applied only to tuning records.
- Line chart from `threshold_sweep.csv`: threshold on X; precision, recall, F1 and contact_rate on Y. Title: “Tuning subset — policy selection”.
- Calibration plot from `reliability_bins.csv`: mean_probability vs observed_churn_rate, method legend; role=historical_test; exclude count=0; reference y=x. Show bin counts in tooltips.
- Calibration scorecard from `probability_metrics.csv`: Brier and log loss (lower is better), AP and ROC-AUC (higher is better); role=historical_test. Selection itself uses `calibration_cv_metrics.csv`, not this test table.

Use whole-number counts, percentages with one decimal, F1/Brier/log loss with three decimals, and display thresholds with six decimals. Classification must use the exported flags or the full-precision policy, never the formatted display value.

Optional customer-filter-aware DAX for the recommended policy:

```dax
Evaluation Customers = COUNTROWS(Evaluation)
Flagged = SUM(Evaluation[balanced_f1_flag])
True Positives = COUNTROWS(FILTER(Evaluation, Evaluation[balanced_f1_flag] = 1 && Evaluation[actual_churn] = 1))
False Positives = COUNTROWS(FILTER(Evaluation, Evaluation[balanced_f1_flag] = 1 && Evaluation[actual_churn] = 0))
False Negatives = COUNTROWS(FILTER(Evaluation, Evaluation[balanced_f1_flag] = 0 && Evaluation[actual_churn] = 1))
Precision = DIVIDE([True Positives], [True Positives] + [False Positives], 0)
Recall = DIVIDE([True Positives], [True Positives] + [False Negatives], 0)
F1 = DIVIDE(2 * [True Positives], 2 * [True Positives] + [False Positives] + [False Negatives], 0)
Contact Rate = DIVIDE([Flagged], [Evaluation Customers], 0)
```

Confirm unfiltered totals against `policy_metrics.csv` and the executed `sql/calibration_policy_metrics.sql`. Scope label: **Historical, previously explored evaluation data; no production campaign or causal retention estimates.** The saved outreach model and threshold policy must be validated on fresh outcomes before replacing operational scoring.
