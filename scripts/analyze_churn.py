"""Reproduce descriptive segment EDA. No predictive modeling or causal claims."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3


def main():
    df = pd.read_csv(ROOT / "data/processed/telco_churn_cleaned.csv")
    assert df.customerID.is_unique and df.Churn.isin(["Yes", "No"]).all()
    assert df.tenure.between(0, 72).all()
    df["churn_flag"] = df.Churn.eq("Yes").astype(int)
    bands = ["0–12 months", "13–24 months", "25–36 months", "37–48 months", "49–60 months", "61–72 months"]
    df["TenureBand"] = pd.cut(df.tenure, [-1, 12, 24, 36, 48, 60, 72], labels=bands)
    assert df.TenureBand.notna().all()
    out = ROOT / "reports/eda"
    out.mkdir(parents=True, exist_ok=True)
    base = df.churn_flag.mean()
    orders = {"Contract": ["Month-to-month", "One year", "Two year"],
              "PaymentMethod": sorted(df.PaymentMethod.unique()), "TenureBand": bands}
    tables = {}
    for field, order in orders.items():
        t = df.groupby(field, observed=True).churn_flag.agg(customers="size", churned="sum").reindex(order)
        t["retained"] = t.customers - t.churned
        t["churn_rate_pct"] = 100 * t.churned / t.customers
        t["customer_share_pct"] = 100 * t.customers / len(df)
        t["difference_vs_overall_pp"] = t.churn_rate_pct - 100 * base
        t["lift_vs_overall"] = t.churn_rate_pct / (100 * base)
        assert t.customers.sum() == len(df) and t.churned.sum() == df.churn_flag.sum()
        t.to_csv(out / f"churn_by_{field}.csv", float_format="%.6f")
        tables[field] = t
    cross = df.groupby(["Contract", "TenureBand"], observed=True).churn_flag.agg(customers="size", churned="sum")
    cross["churn_rate_pct"] = cross.churned / cross.customers * 100
    cross.to_csv(out / "contract_tenure_breakdown.csv", float_format="%.6f")
    sns.set_theme(style="whitegrid", context="notebook")
    for field, table in tables.items():
        fig, ax = plt.subplots(figsize=(10, 5.8))
        y = np.arange(len(table))
        ax.barh(y, table.churn_rate_pct, color="#247c93")
        ax.set_yticks(y, table.index)
        ax.invert_yaxis()
        for i, (_, row) in enumerate(table.iterrows()):
            ax.text(row.churn_rate_pct + 1, i, f"{row.churn_rate_pct:.1f}%  (n={int(row.customers):,})", va="center", fontsize=10)
        ax.axvline(base*100, color="#b54c36", linestyle="--", label=f"Overall: {base:.2%}")
        ax.set(xlim=(0, 65), xlabel="Customers who churned (%)", title=f"Churn rate by {field}")
        ax.legend(loc="lower right")
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"churn_by_{field}.png", dpi=180)
        plt.close(fig)
    rate = cross.churn_rate_pct.unstack().reindex(index=orders["Contract"], columns=bands)
    sizes = cross.customers.unstack().reindex(index=orders["Contract"], columns=bands)
    labels = rate.copy().astype(object)
    for i in range(len(rate)):
        for j in range(len(rate.columns)):
            labels.iloc[i, j] = "" if pd.isna(rate.iloc[i,j]) else f"{rate.iloc[i,j]:.1f}%\nn={sizes.iloc[i,j]:.0f}"
    fig, ax = plt.subplots(figsize=(12, 4.8))
    sns.heatmap(rate, annot=labels, fmt="", cmap="YlOrRd", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "Churn rate (%)"})
    ax.set(title="Contract and tenure: observed churn rate and segment size", xlabel="Tenure band", ylabel="Contract")
    fig.tight_layout()
    fig.savefig(out / "contract_tenure_heatmap.png", dpi=180)
    plt.close(fig)
    # Independent SQL aggregation validates every segment's count and churn total.
    with sqlite3.connect(":memory:") as conn:
        df.drop(columns="TenureBand").to_sql("telco_customers", conn, index=False)
        sql = (ROOT / "sql/churn_eda.sql").read_text()
        sql = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
        results = [pd.read_sql_query(q, conn) for q in sql.split(";") if q.strip()]
        for field, result in zip(orders, results):
            expected = tables[field].sort_index()
            result = result.set_index("segment").sort_index()
            assert result.customers.tolist() == expected.customers.tolist()
            assert result.churned.tolist() == expected.churned.tolist()
            assert np.allclose(result.churn_rate_pct, expected.churn_rate_pct)
    def md(t):
        lines = ["| Segment | Customers | Churned | Churn rate | Difference vs overall |", "|---|---:|---:|---:|---:|"]
        for name, r in t.iterrows():
            lines.append(f"| {name} | {int(r.customers):,} | {int(r.churned):,} | {r.churn_rate_pct:.2f}% | {r.difference_vs_overall_pp:+.2f} pp |")
        return "\n".join(lines)
    sections = "\n\n".join(f"## {field}\n\n{md(table)}\n\n![Churn by {field}](churn_by_{field}.png)" for field, table in tables.items())
    top_contract = tables["Contract"].churn_rate_pct.idxmax()
    top_payment = tables["PaymentMethod"].churn_rate_pct.idxmax()
    top_band = tables["TenureBand"].churn_rate_pct.idxmax()
    report = f"""# Weekly EDA: churn across customer segments

Analyzed all {len(df):,} cleaned customer records. Overall churn: **{df.churn_flag.sum():,} / {len(df):,} = {base:.2%}**. Each rate uses that segment's customer count as its denominator; percentages are not shares of all churned customers.

## Findings

- Highest contract churn: **{top_contract} ({tables['Contract'].loc[top_contract, 'churn_rate_pct']:.2f}%)**.
- Highest payment-method churn: **{top_payment} ({tables['PaymentMethod'].loc[top_payment, 'churn_rate_pct']:.2f}%)**.
- Highest tenure-band churn: **{top_band} ({tables['TenureBand'].loc[top_band, 'churn_rate_pct']:.2f}%)**.

{sections}

## Contract and tenure together

![Contract and tenure heatmap](contract_tenure_heatmap.png)

The joint view helps distinguish tenure composition from contract associations. Always consider the displayed sample counts: rates in small cells are unstable. This is a descriptive cross-tab, not an adjusted causal estimate.

## Interpretation and retention opportunities

Prioritize investigation of onboarding experiences among short-tenure customers and service/value concerns among month-to-month customers. Review the electronic-check payment journey for friction. These are hypotheses for research and controlled retention experiments; the data does not establish that changing a contract or payment method will prevent churn. Contract, payment method, tenure, and service mix can be related.

## Method and limitations

- Tenure bands include both endpoints: 0–12, 13–24, 25–36, 37–48, 49–60, and 61–72 months. Zero-tenure customers remain in the first band. Values outside 0–72 or missing tenure stop the script rather than silently dropping records.
- Full-dataset EDA uses the cleaned readable export. The 11 imputed TotalCharges values do not enter these groupings or churn-rate calculations.
- This analysis includes the previously reserved test records for descriptive reporting. Do not use these findings to tune models and then describe that test set as untouched; reserve a fresh independent evaluation dataset if making model decisions from these findings.
- These are observed proportions in the supplied dataset, not future individual risk scores or a time series. No observation dates, direct complaint records, or detailed usage measurements are available. Generalization to a live customer population is unverified.
- Validation passed: all three breakdowns sum to {len(df):,} customers and {df.churn_flag.sum():,} churners; independent SQLite queries reproduce every group count and rate. No records were filtered out.

## Reproduce and handoff

Run `python scripts/analyze_churn.py`. Tables and four PNG charts are written to `reports/eda/`. Run `sql/churn_eda.sql` after importing the cleaned CSV as `telco_customers`. CSV rates use percentage units (26.54 means 26.54%), so divide by 100 before using Power BI percentage formatting. See `power_bi_eda.md` for dashboard setup.
"""
    (out / "findings.md").write_text(report, encoding="utf-8")
    print("PASS: all segment totals and independent SQL results match.")
    for field, t in tables.items():
        print(field)
        print(t[["customers", "churned", "churn_rate_pct"]].to_string())


if __name__ == "__main__":
    main()
