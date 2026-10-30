"""Check the exported dashboard population, score arithmetic, and KPI totals."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

data = rows(ROOT / "powerbi/data/customer_risk.csv")
raw = rows(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
expected = json.loads((ROOT / "reports/dashboard/expected_kpis.json").read_text())
assert len({r["customerID"] for r in data}) == len(data)
assert {r["customerID"] for r in data} == {r["customerID"] for r in raw if r["Churn"] == "No"}
for r in data:
    score, charges = float(r["RiskScore"]), float(r["MonthlyCharges"])
    assert 0 <= score <= 1
    assert math.isclose(float(r["RiskWeightedMonthlyCharges"]), score * charges, abs_tol=1e-8)
    assert r["RiskBand"] == ("High" if score >= .5 else "Medium" if score >= .3 else "Low")
assert len(data) == expected["active_customers"]
assert sum(float(r["RiskScore"]) >= .5 for r in data) == expected["high_risk_customers"]
assert math.isclose(sum(float(r["RiskWeightedMonthlyCharges"]) for r in data), expected["risk_weighted_monthly_charges"], abs_tol=1e-6)
for contract in rows(ROOT / "reports/dashboard/contract_checks.csv"):
    subset = [r for r in data if r["Contract"] == contract["Contract"]]
    assert len(subset) == int(contract["Customers"])
    assert math.isclose(sum(float(r["RiskWeightedMonthlyCharges"]) for r in subset), float(contract["WeightedCharges"]), abs_tol=1e-6)
print(f"PASS: {len(data)} customer records, score bands, exposure arithmetic, and contract totals")
