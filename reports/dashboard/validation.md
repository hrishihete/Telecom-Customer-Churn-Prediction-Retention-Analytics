# Desktop validation

Checked on October 30, 2026 in Power BI Desktop.

- All three pages render without visual errors: overview, customer explorer, and definitions/freshness.
- Overview totals: 5,174 active-at-snapshot customers; 402 high-risk customers; 67,388.05 CU weighted monthly exposure; 31,054.15 CU high-risk monthly charges.
- Contract slicer tested with Month-to-month: 2,220 customers and 50,718.72 CU weighted exposure, matching `contract_checks.csv`. Charts and customer table respond to the filter.
- Contract slicer reset to All; full overview totals restored before saving.
- Customer explorer displays descending scores and the cohort column.
- Definitions page displays the scoring timestamp and explicitly distinguishes historical data from a live feed.

Source-derived expected totals are retained in `expected_kpis.json`; checks can be reproduced with `scripts/verify_dashboard.py`.
