"""Train IDS-Agent classifiers for EV charging sessions."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = [
    "connectionTime",
    "disconnectTime",
    "RequestedDemand",
    "kWhDelivered",
]
LABEL_COLUMN = "label"

MODELS = {
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "lr": LogisticRegression(max_iter=200),
    "dt": DecisionTreeClassifier(random_state=42),
    "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    "svc": SVC(kernel="rbf", probability=True),
}


def build_pipeline(model):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, FEATURE_COLUMNS)]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_models(dataset_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(dataset_path)
    missing = [col for col in FEATURE_COLUMNS + [LABEL_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in MODELS.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        report = classification_report(y_test, preds)
        print(f"Model: {name}\n{report}\n")
        joblib.dump(pipeline, output_dir / f"{name}.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IDS-Agent EV charging classifiers")
    parser.add_argument("dataset", type=Path, help="Path to CSV dataset")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"), help="Directory to save models"
    )
    args = parser.parse_args()
    train_models(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
