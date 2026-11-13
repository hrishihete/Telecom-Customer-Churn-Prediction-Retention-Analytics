"""Validate customer snapshots and interval telemetry before downstream use."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
ADDONS = "OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies".split()
DOMAINS = {
    "gender": {"Female", "Male"}, "SeniorCitizen": {"0", "1"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
    **{key: {"Yes", "No"} for key in ("Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn")},
    **{key: {"Yes", "No", "No internet service"} for key in ADDONS},
}
SNAPSHOT = set(DOMAINS) | {"customerID", "tenure", "MonthlyCharges", "TotalCharges"}
EVENT = {"eventID", "customerID", "observedAt", "metric", "value", "unit"}
UNITS = {"data_usage": "MB", "voice_usage": "minutes", "complaint_count": "count"}


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("timestamp requires offset")
    return result.astimezone(timezone.utc)


def nonnegative(value, integral=False):
    try:
        n = Decimal(value)
        return n.is_finite() and n >= 0 and (not integral or n == n.to_integral_value())
    except InvalidOperation:
        return False


def validate(paths, kind, as_of, customers=None):
    errors, warnings, examples = Counter(), Counter(), []
    seen = set()
    rows = 0
    required = SNAPSHOT if kind == "snapshot" else EVENT

    def issue(rule, path, line, field="", warning=False):
        (warnings if warning else errors)[rule] += 1
        if len(examples) < 50:
            examples.append(dict(file=path.name, line=line, rule=rule, field=field,
                                 severity="warning" if warning else "error"))

    for path in paths:
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader, [])
                if len(set(header)) != len(header) or required - set(header):
                    issue("invalid_header", path, 1)
                    continue
                count = 0
                for values in reader:
                    count += 1
                    rows += 1
                    line = reader.line_num
                    if len(values) != len(header):
                        issue("row_width", path, line)
                        continue
                    row = dict(zip(header, (v.strip() for v in values)))
                    key = "customerID" if kind == "snapshot" else "eventID"
                    if row[key] in seen:
                        issue("duplicate_key", path, line, key)
                    seen.add(row[key])
                    for field in sorted(required):
                        if row[field]:
                            continue
                        if field == "TotalCharges" and nonnegative(row["tenure"], True) and Decimal(row["tenure"]) == 0:
                            issue("zero_tenure_unbilled", path, line, field, warning=True)
                        else:
                            issue("missing_value", path, line, field)
                    if kind == "snapshot":
                        for field, allowed in DOMAINS.items():
                            if row[field] and row[field] not in allowed:
                                issue("unknown_category", path, line, field)
                        for field in ("tenure", "MonthlyCharges", "TotalCharges"):
                            if row[field] and not nonnegative(row[field], field == "tenure"):
                                issue("invalid_number", path, line, field)
                        for field in ADDONS:
                            if (row["InternetService"] == "No") != (row[field] == "No internet service"):
                                issue("internet_service_conflict", path, line, field)
                        if (row["PhoneService"] == "No") != (row["MultipleLines"] == "No phone service"):
                            issue("phone_service_conflict", path, line, "MultipleLines")
                    else:
                        if row["customerID"] not in customers:
                            issue("unknown_customer", path, line, "customerID")
                        metric = row["metric"]
                        if metric not in UNITS:
                            issue("unknown_metric", path, line, "metric")
                        elif row["unit"] != UNITS[metric]:
                            issue("metric_unit_mismatch", path, line, "unit")
                        if not nonnegative(row["value"], metric == "complaint_count"):
                            issue("invalid_measurement", path, line, "value")
                        try:
                            if timestamp(row["observedAt"]) > as_of:
                                issue("future_observation", path, line, "observedAt")
                        except (ValueError, OverflowError):
                            issue("invalid_timestamp", path, line, "observedAt")
                if not count:
                    issue("empty_batch", path, 1)
        except (OSError, UnicodeError, csv.Error):
            issue("unreadable_csv", path, 0)
    return dict(status="FAIL" if errors else "PASS", profile=kind, files=len(paths), rows=rows,
                as_of_utc=as_of.isoformat(), errors=dict(errors), warnings=dict(warnings), examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["snapshot", "events"])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--customers", type=Path, default=SOURCE)
    parser.add_argument("--as-of", type=timestamp, default=datetime.now(timezone.utc))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    customers = None
    if args.profile == "events":
        with args.customers.open(encoding="utf-8-sig", newline="") as stream:
            customers = {r["customerID"].strip() for r in csv.DictReader(stream)}
    report = validate(args.files, args.profile, args.as_of, customers)
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    return int(report["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
