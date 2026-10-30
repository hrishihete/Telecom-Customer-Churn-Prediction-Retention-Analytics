# Churn risk dashboard

Open `ChurnRisk.pbip` in Power BI Desktop, with report definitions and semantic model in the adjacent folders. The standalone `Telecom_Churn_Risk.pbix` export is pending recovery of Desktop's unresponsive Save As dialog.

## Pages

1. Risk overview: active customers, high-risk customers, score-weighted monthly exposure, high-risk monthly charges, contract/risk charts, and a customer priority table. Slicers cover contract, risk band, internet service, and model cohort.
2. Customer explorer: sortable customer scores and charges with contract, payment method, tenure, and risk filters.
3. Definitions & freshness: scope, formulas, model limitations, and scoring timestamp.

## Refresh

From the repository root, using the model environment:

```powershell
.\.venv\Scripts\python.exe scripts/score_dashboard.py
```

Then use **Home > Refresh** in Desktop and save. On another computer, set the `DataPath` Power Query parameter to the absolute path of `powerbi/data/customer_risk.csv` before refreshing. The PBIP requires refresh on first opening because the local data cache is intentionally excluded from Git.

The optional `scripts/build_powerbi.py` regenerates the project layout and local data path; it overwrites report definitions, so preserve any manual edits first. The checked-in project was opened and saved by Desktop.

## Metric definitions and limits

- Active means `Churn = No` at the historical Kaggle snapshot (5,174 records); it does not establish present-day status.
- Risk score is the saved Random Forest output. High: score >= 0.50; medium: 0.30 <= score < 0.50; low: score < 0.30. Thresholds are illustrative.
- Revenue-at-risk proxy = SUM(MonthlyCharges * RiskScore), in source currency units (CU). Scores are uncalibrated and have no validated future horizon, so this is not expected financial loss.
- High-risk monthly charges = SUM(MonthlyCharges) among high-risk records.
- Model cohort distinguishes training records from held-out test records. Training scores are in-sample demonstrations.
- Scoring timestamp describes the script run, not source observation freshness.

**No live production feed is connected.** This deliverable is a refreshable historical demonstration. Real-time operation requires a current customer source, deployed scoring, and an appropriate Power BI refresh/DirectQuery setup.

Validation totals and contract comparisons are in `reports/dashboard/`. The imported dataset excludes the 1,869 historical churners from exposure totals.
