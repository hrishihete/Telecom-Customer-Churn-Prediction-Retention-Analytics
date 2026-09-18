"""Download the public Kaggle archive and extract only the expected dataset."""
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
from io import BytesIO

root = Path(__file__).resolve().parents[1] / "data/raw"
root.mkdir(parents=True, exist_ok=True)
url = "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
with urlopen(Request(url, headers={"User-Agent": "telco-data-inspection/1.0"}), timeout=120) as response:
    payload = response.read()
with ZipFile(BytesIO(payload)) as archive:
    csv = archive.read(name)
(root / "telco-customer-churn.zip").write_bytes(payload)
(root / name).write_bytes(csv)
print(f"Downloaded {name}: {len(csv):,} bytes")
