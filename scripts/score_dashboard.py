"""Refresh model scores for the local Power BI demonstration snapshot."""
from pathlib import Path
import argparse
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib
from clean_data import normalize_records

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    args = parser.parse_args()
    # This demonstration accepts the documented historical schema, including Churn.
    raw, _ = normalize_records(pd.read_csv(args.input))
    meta = json.loads((ROOT / "reports/models/metadata.json").read_text())
    model_file = ROOT / "models/random_forest.joblib"
    model = joblib.load(model_file)
    active = raw[raw.Churn.eq("No")].copy()
    X = active[meta["input_columns"]].copy()
    X.SeniorCitizen = X.SeniorCitizen.astype(int).astype(str)
    scores = model.predict_proba(X)[:, 1]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    out = active[["customerID", "Contract", "PaymentMethod", "InternetService", "tenure", "MonthlyCharges"]].copy()
    out["RiskScore"] = scores
    out["RiskBand"] = np.select([scores >= .5, scores >= .3], ["High", "Medium"], default="Low")
    out["RiskWeightedMonthlyCharges"] = out.MonthlyCharges * out.RiskScore
    out["TenureBand"] = pd.cut(out.tenure, [-1, 12, 24, 36, 48, 60, 72], labels=["00-12 months", "13-24 months", "25-36 months", "37-48 months", "49-60 months", "61-72 months"])
    manifest = pd.read_csv(ROOT / "data/processed/split_manifest.csv").set_index("customerID")
    out["ModelCohort"] = out.customerID.map(manifest.split).fillna("New record")
    out["ScoredAtUTC"] = timestamp
    out["SourceMode"] = "Historical snapshot - not a live feed"
    out["ModelVersion"] = hashlib.sha256(model_file.read_bytes()).hexdigest()[:12]
    assert out.customerID.is_unique and not out.isna().any().any()
    assert out.RiskScore.between(0, 1).all()
    folder = ROOT / "powerbi/data"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "customer_risk.csv"
    temp = folder / "customer_risk.tmp"
    out.to_csv(temp, index=False)
    temp.replace(target)
    summary = dict(active_customers=len(out), excluded_historical_churners=int(raw.Churn.eq("Yes").sum()),
                   high_risk_customers=int(out.RiskScore.ge(.5).sum()), mean_risk=float(out.RiskScore.mean()),
                   monthly_charges=float(out.MonthlyCharges.sum()),
                   high_risk_monthly_charges=float(out.loc[out.RiskScore.ge(.5), "MonthlyCharges"].sum()),
                   risk_weighted_monthly_charges=float(out.RiskWeightedMonthlyCharges.sum()),
                   scored_at_utc=timestamp, source_mode="historical snapshot", currency="source currency units (CU)",
                   model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest())
    dest = ROOT / "reports/dashboard"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "expected_kpis.json").write_text(json.dumps(summary, indent=2)+"\n")
    out.groupby("Contract").agg(Customers=("customerID", "size"), MonthlyCharges=("MonthlyCharges", "sum"),
                                WeightedCharges=("RiskWeightedMonthlyCharges", "sum")).to_csv(dest / "contract_checks.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
