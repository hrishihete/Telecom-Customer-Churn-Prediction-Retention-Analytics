import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_incoming import SOURCE, timestamp, validate

NOW = timestamp("2026-11-13T12:00:00Z")
HEADER = ["eventID", "customerID", "observedAt", "metric", "value", "unit"]
GOOD = ["e1", "c1", "2026-11-13T11:00:00Z", "data_usage", "12.5", "MB"]


class IncomingValidation(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)

    def write(self, name, header, *rows):
        path = self.folder / name
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([header, *rows])
        return path

    def test_event_contract(self):
        path = self.write("events.csv", HEADER, GOOD)
        self.assertEqual(validate([path], "events", NOW, {"c1"})["status"], "PASS")
        for index, value, rule in [(1, "missing", "unknown_customer"), (2, "2027-01-01T00:00:00Z", "future_observation"),
                (2, "2026-11-13", "invalid_timestamp"), (3, "other", "unknown_metric"),
                (4, "-1", "invalid_measurement"), (4, "NaN", "invalid_measurement"), (5, "GB", "metric_unit_mismatch")]:
            with self.subTest(value=value):
                row = GOOD.copy()
                row[index] = value
                self.write(path.name, HEADER, row)
                self.assertIn(rule, validate([path], "events", NOW, {"c1"})["errors"])
        self.write(path.name, HEADER, GOOD[:3] + ["complaint_count", "0.5", "count"])
        self.assertIn("invalid_measurement", validate([path], "events", NOW, {"c1"})["errors"])

    def test_batch_integrity(self):
        first = self.write("first.csv", HEADER, GOOD)
        second = self.write("second.csv", HEADER, GOOD)
        self.assertIn("duplicate_key", validate([first, second], "events", NOW, {"c1"})["errors"])
        for header, rows, rule in [(HEADER, [], "empty_batch"), (HEADER[:-1], [GOOD], "invalid_header"),
                                  (HEADER, [GOOD[:-1]], "row_width")]:
            self.write(first.name, header, *rows)
            self.assertIn(rule, validate([first], "events", NOW, {"c1"})["errors"])

    def test_snapshot_defects(self):
        with SOURCE.open() as stream:
            reader = csv.DictReader(stream)
            row = next(reader)
        row.update(tenure="1.5", TotalCharges="", InternetService="No", Contract="weekly")
        path = self.write("broken.csv", row.keys(), row.values())
        self.assertTrue({"invalid_number", "missing_value", "internet_service_conflict", "unknown_category"}
                        <= validate([path], "snapshot", NOW)["errors"].keys())
