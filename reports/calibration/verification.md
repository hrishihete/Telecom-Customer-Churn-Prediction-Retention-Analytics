# Calibration verification

PASS — all checks below completed.

- Source, original split, legacy model, refitted base and outreach artifact SHA-256 values match metadata and policy.
- All 7,043 IDs have exactly one role; stratified splits reproduce and original test membership is unchanged.
- Saved pipeline excludes target/ID and numeric imputation matches base-fit records only.
- All calibration OOF probabilities reproduce from disjoint fold fits; method selection uses only OOF log loss.
- All evaluation IDs, labels, methods, probabilities and exact >= policy flags match saved-artifact inference.
- Every tuning boundary and all precision/recall/F1/confusion counts independently reconcile; default and alternative thresholds are optimal under declared tuning rules.
- Tuning summaries, all evaluation policy/probability metrics and every reliability-bin count/rate reconcile independently.
- Executed SQL reconciles recommended-policy counts, precision, recall, F1 and workload for both roles.
- Threshold equality, tied scores and 0/1 endpoints pass targeted decision-rule checks.

Reproduce: `python scripts/verify_calibration.py`. Charts are visually reviewed separately. This verifies computation, not independent production generalization.
