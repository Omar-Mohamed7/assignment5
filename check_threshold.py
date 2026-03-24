import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

RUN_ID_FILE = "model_info.txt"


def fetch_run(client: MlflowClient, run_id: str):
    try:
        return client.get_run(run_id)
    except Exception:
        return None


def read_accuracy_from_mlruns(run_id: str) -> Optional[float]:
    """Find a local MLflow run directory by run_id and read its accuracy metric."""
    mlruns_root = Path("mlruns")
    if not mlruns_root.exists():
        return None

    for candidate in mlruns_root.rglob(run_id):
        if candidate.is_dir():
            metric_file = candidate / "metrics" / "accuracy"
            if not metric_file.exists():
                continue

            lines = [line.strip() for line in metric_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                continue

            # MLflow file metric format is generally: "timestamp value step".
            parts = lines[-1].split()
            if len(parts) < 2:
                continue

            try:
                return float(parts[1])
            except ValueError:
                continue

    return None


def main() -> int:
    threshold = float(os.environ.get("ACCURACY_THRESHOLD", "0.85"))

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        if os.path.exists("mlruns"):
            tracking_uri = os.path.abspath("mlruns")
            print("MLFLOW_TRACKING_URI is not set. Falling back to local file tracking.")
        else:
            print("ERROR: MLFLOW_TRACKING_URI is not set and no local mlruns directory was found.")
            return 1

    if not os.path.exists(RUN_ID_FILE):
        print(f"ERROR: {RUN_ID_FILE} not found.")
        return 1

    with open(RUN_ID_FILE, "r", encoding="utf-8") as f:
        run_id = f.read().strip()

    if not run_id:
        print("ERROR: model_info.txt is empty.")
        return 1

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = fetch_run(client, run_id)

    # GitHub artifact downloads can place data under ./mlruns/mlruns.
    if run is None and os.path.exists("mlruns/mlruns"):
        nested_tracking_uri = os.path.abspath("mlruns/mlruns")
        mlflow.set_tracking_uri(nested_tracking_uri)
        client = MlflowClient()
        run = fetch_run(client, run_id)
        if run is not None:
            tracking_uri = nested_tracking_uri
            print("Run not found in primary local tracking path. Retried nested mlruns path.")

    accuracy = None
    if run is not None:
        accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        accuracy = read_accuracy_from_mlruns(run_id)
        if accuracy is not None:
            print("MLflow API lookup failed; using accuracy from local mlruns files.")

    if accuracy is None:
        print(f"ERROR: Could not fetch accuracy for run {run_id} from tracking URI {tracking_uri} or local mlruns files.")
        return 1

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {threshold:.2f}")

    if accuracy < threshold:
        print("FAIL: Accuracy below threshold. Blocking deployment.")
        return 1

    print("PASS: Accuracy meets threshold. Proceeding to deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
