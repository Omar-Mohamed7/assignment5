import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

RUN_ID_FILE = "model_info.txt"


def parse_accuracy(metric_file: Path) -> Optional[float]:
    if not metric_file.exists():
        return None

    lines = [l.strip() for l in metric_file.read_text().splitlines() if l.strip()]
    if not lines:
        return None

    try:
        return float(lines[-1].split()[1])
    except Exception:
        return None


def read_local_accuracy(run_id: str) -> Optional[float]:
    mlruns = Path("mlruns")

    if not mlruns.exists():
        return None

    # Direct search
    for run_dir in mlruns.rglob(run_id):
        metric_file = run_dir / "metrics" / "accuracy"
        acc = parse_accuracy(metric_file)
        if acc is not None:
            return acc

    # Fallback: latest metric
    metrics = list(mlruns.rglob("metrics/accuracy"))
    if metrics:
        latest = max(metrics, key=lambda p: p.stat().st_mtime)
        return parse_accuracy(latest)

    return None


def main():
    threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))

    if not os.path.exists(RUN_ID_FILE):
        print("ERROR: model_info.txt not found")
        return 1

    run_id = open(RUN_ID_FILE).read().strip()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", os.path.abspath("mlruns"))
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    accuracy = None

    try:
        run = client.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy")
    except Exception:
        pass

    if accuracy is None:
        print("Falling back to local mlruns parsing...")
        accuracy = read_local_accuracy(run_id)

    if accuracy is None:
        print(f"ERROR: Could not fetch accuracy for run {run_id}")
        return 1

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {threshold:.2f}")

    if accuracy < threshold:
        print("FAIL: Below threshold")
        return 1

    print("PASS: Deployment allowed")
    return 0


if __name__ == "__main__":
    sys.exit(main())