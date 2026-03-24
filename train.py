import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("mlops-lab5")

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()