"""Reproducible historical retention planning; does not contact customers."""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mpl-cache"))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/retention"

def cohort(r):
    band = "00-03" if r.tenure <= 3 else "04-12" if r.tenure <= 12 else "13-24" if r.tenure <= 24 else "25+"
    return f"{r.Contract} | {band} months | {r.InternetService}"

def play(r):
    if r.tenure <= 3:
        return "R1 Onboarding and first-bill review"
    if r.tenure <= 12:
        return "R2 Service and plan-fit review"
    if r.Contract == "Month-to-month":
        return "R3 Loyalty and optional contract review"
    return "R4 Relationship and renewal review"

def actions(r):
    result = []
    if r.InternetService == "Fiber optic":
        result.append("Ask about reliability; check outage/speed evidence before remediation")
    if r.InternetService != "No" and r.TechSupport == "No":
        result.append("Offer a guided diagnostic; support not subscribed does not imply a complaint")
    if r.StreamingTV == "Yes" or r.StreamingMovies == "Yes":
        result.append("Review bundle value and actual consumption before changing plan")
    else:
        result.append("Ask about service needs; validate actual consumption before right-sizing")
    if r.PaymentMethod == "Electronic check":
        result.append("Review bill clarity and offer optional payment-method assistance")
    return "; ".join(result)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    scorefile = ROOT / "powerbi/data/customer_risk.csv"
    raw = pd.read_csv(source)
    scores = pd.read_csv(scorefile)
    assert raw.customerID.is_unique and scores.customerID.is_unique
    extra = ["customerID", "TechSupport", "OnlineSecurity", "StreamingTV", "StreamingMovies", "PhoneService", "MultipleLines"]
    active = scores.merge(raw[extra], on="customerID", validate="one_to_one")
    assert set(active.customerID) == set(raw.loc[raw.Churn.eq("No"), "customerID"])
    active["Cohort"] = active.apply(cohort, axis=1)
    raw["Cohort"] = raw.apply(cohort, axis=1)
    high = active[active.RiskScore.ge(.5)].copy()
    high["PrimaryPlay"] = high.apply(play, axis=1)
    high["ServiceReview"] = high.apply(actions, axis=1)
    high["PlanningStatus"] = "Historical planning only - revalidate before outreach"
    high = high.sort_values(["RiskWeightedMonthlyCharges", "RiskScore", "customerID"], ascending=[False, False, True])
    high.insert(0, "ReviewRank", range(1, len(high)+1))
    high.to_csv(OUT / "customer_review_queue.csv", index=False)
    agg = high.groupby("Cohort").agg(HighRiskCustomers=("customerID", "size"), MeanScore=("RiskScore", "mean"),
        MonthlyChargesCU=("MonthlyCharges", "sum"), WeightedExposureCU=("RiskWeightedMonthlyCharges", "sum"),
        TrainingRecords=("ModelCohort", lambda s: s.eq("train").sum()), TestRecords=("ModelCohort", lambda s: s.eq("test").sum()))
    agg["ActiveCustomersInCohort"] = active.groupby("Cohort").size()
    agg["HighRiskShareOfActive"] = agg.HighRiskCustomers / agg.ActiveCustomersInCohort
    historic = raw.assign(Churned=raw.Churn.eq("Yes")).groupby("Cohort").agg(HistoricalCustomers=("customerID","size"), HistoricalChurned=("Churned","sum"))
    agg = agg.join(historic)
    agg["HistoricalChurnRate"] = agg.HistoricalChurned / agg.HistoricalCustomers
    agg = agg.sort_values("WeightedExposureCU", ascending=False)
    agg.to_csv(OUT / "cohort_priorities.csv")
    plays = high.groupby("PrimaryPlay").agg(Customers=("customerID","size"), MonthlyChargesCU=("MonthlyCharges","sum"), WeightedExposureCU=("RiskWeightedMonthlyCharges","sum"))
    plays.to_csv(OUT / "playbook_summary.csv")
    service = []
    for column in ["InternetService", "TechSupport", "OnlineSecurity", "StreamingTV", "StreamingMovies", "PaymentMethod"]:
        for value, part in high.groupby(column):
            service.append(dict(Indicator=column, Value=value, HighRiskCustomers=len(part), ShareOfHighRisk=len(part)/len(high)))
    pd.DataFrame(service).to_csv(OUT / "service_indicators.csv", index=False)
    sensitivity = []
    for cutoff in [.4,.5,.6]:
        subset = active[active.RiskScore.ge(cutoff)]
        sensitivity.append(dict(Threshold=cutoff, Customers=len(subset), MonthlyChargesCU=subset.MonthlyCharges.sum()))
    pd.DataFrame(sensitivity).to_csv(OUT / "threshold_sensitivity.csv", index=False)
    # Execute the SQL handoff against the same inputs and reconcile its totals.
    with sqlite3.connect(":memory:") as db:
        active.to_sql("customer_risk", db, index=False)
        actual = pd.read_sql_query((ROOT / "sql/retention_cohorts.sql").read_text(), db)
    expected = agg.reset_index()[["Cohort", "HighRiskCustomers", "MonthlyChargesCU", "WeightedExposureCU"]].sort_values("Cohort").reset_index(drop=True)
    actual = actual.sort_values("Cohort").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-7, rtol=1e-10)
    assert len(high) == agg.HighRiskCustomers.sum() == plays.Customers.sum()
    assert high.customerID.is_unique and high.PrimaryPlay.notna().all()
    assert high.RiskScore.ge(.5).all()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(5, len(agg)*.45)))
    sns.barplot(data=agg.reset_index(), x="HighRiskCustomers", y="Cohort", color="#247c91", ax=ax)
    ax.set(title="High-risk active-at-snapshot customers by cohort", xlabel="Customers with model score ≥ 0.50", ylabel="Contract | tenure | subscribed internet service")
    for container in ax.containers: ax.bar_label(container, padding=3)
    fig.tight_layout()
    fig.savefig(OUT / "cohort_customers.png", dpi=160)
    plt.close(fig)
    metadata = dict(high_risk_customers=len(high), active_customers=len(active), monthly_charges_cu=float(high.MonthlyCharges.sum()),
        weighted_exposure_cu=float(high.RiskWeightedMonthlyCharges.sum()), training_records=int(high.ModelCohort.eq("train").sum()),
        test_records=int(high.ModelCohort.eq("test").sum()), source_scored_at_utc=str(high.ScoredAtUTC.iloc[0]),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), scores_sha256=hashlib.sha256(scorefile.read_bytes()).hexdigest(),
        validation="PASS: one-to-one joins, population coverage, exclusive plays, cohort totals and executed SQL reconciliation")
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))
    print(plays.to_string())
    print(agg.to_string())

if __name__ == "__main__":
    main()
