import os
import sys

RUN_ID_FILE = "model_info.txt"
ACCURACY_FILE = "accuracy.txt"


def main():
    threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))

    if not os.path.exists(RUN_ID_FILE):
        print("ERROR: model_info.txt not found")
        return 1

    if not os.path.exists(ACCURACY_FILE):
        print("ERROR: accuracy.txt not found")
        return 1

    run_id = open(RUN_ID_FILE).read().strip()
    accuracy = float(open(ACCURACY_FILE).read().strip())

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {threshold:.2f}")

    if accuracy < threshold:
        print("FAIL: Accuracy below threshold")
        return 1

    print("PASS: Deployment allowed")
    return 0


if __name__ == "__main__":
    sys.exit(main())