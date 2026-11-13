# Incoming customer data validation

The repository runs a Python validation job on pushes, pull requests, and manual workflow runs. The job checks the historical snapshot, exercises the proposed event format with synthetic records, and validates all CSV event files under `data/incoming/` together. Use that repository path for shareable test batches. Run the same command in the ingestion environment for private customer data.

| Input | Blocking checks |
|---|---|
| Every CSV | Readable UTF-8, required columns, unique headers, nonempty batch, correct row width, required values and unique record keys across the files in the invocation |
| Kaggle snapshot | Unique customer ID, recognized contract/payment/service categories, nonnegative finite charges, whole nonnegative tenure, consistent phone/internet add-on flags |
| Telemetry events | Unique event ID, customer present in reference roster, offset-aware observation time at or before validation time, supported metric, matching unit, finite nonnegative value, whole complaint counts |

The historical snapshot contains 7,043 customers. It passed with 11 warnings for blank TotalCharges where tenure is zero. A blank charge at positive tenure fails. Validation preserves the source value for the existing cleaning step.

## Proposed event contract

Each row represents one measured increment. Supply eventID, customerID, observedAt, metric, value and unit. The example supports data_usage in MB, voice_usage in minutes and complaint_count in count. Additional columns are accepted. Whitespace surrounding values is trimmed during checking. Timestamps must have an explicit UTC offset. Historical backfills are allowed. The default customer roster is the Kaggle snapshot. Supply a current authoritative roster for operational use.

The three [example events](../../data/examples/telemetry_sample.csv) are synthetic. No live telemetry source was supplied. These rules verify structure, plausibility, and reference consistency. Confirming actual consumption or complaint totals requires reconciliation with independent network counters or ticket records.

## Run and interpret

```sh
python scripts/validate_incoming.py snapshot data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --report snapshot-result.json
python scripts/validate_incoming.py events data/examples/telemetry_sample.csv --as-of 2026-11-13T12:00:00Z --report event-result.json
python -m unittest discover -s tests -p test_incoming_validation.py -v
```

For a new batch, substitute its path and pass `--customers roster.csv` with a customerID column. Omit `--as-of` to use the current UTC time. Pass multiple event files in one invocation to detect replayed IDs across them. A successful check exits 0. Blocking violations exit 1. Diagnostics include counts by rule and up to 50 examples with file, CSV line and field. A failed batch should be corrected at source and rerun before scoring or refresh. Downstream ingestion can invoke the CLI and proceed only on exit 0.

The [snapshot result](snapshot.json) and [event result](example.json) record the local run. The [automated workflow](../../.github/workflows/validate-incoming.yml) uploads run reports. The [regression tests](../../tests/test_incoming_validation.py) exercise malformed batches, duplicates, unknown customers, timestamp and measurement errors, and contradictory service subscriptions.
