"""Train IDS-Agent classifiers for EV charging sessions."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.ids_agent import (
    build_preprocessing_pipeline,
    drop_irrelevant_fields,
    split_feature_columns,
)

LABEL_COLUMN = "label"

MODELS = {
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "lr": LogisticRegression(max_iter=200),
    "dt": DecisionTreeClassifier(random_state=42),
    "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    "svc": SVC(kernel="rbf", probability=True),
}


def build_pipeline(model, *, numeric_columns, categorical_columns):
    preprocessing = build_preprocessing_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


def train_models(dataset_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(dataset_path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {LABEL_COLUMN}")

    y = df[LABEL_COLUMN]
    feature_frame = drop_irrelevant_fields(df, label_column=LABEL_COLUMN)
    if feature_frame.empty:
        raise ValueError("No usable feature columns remain after preprocessing.")

    numeric_columns, categorical_columns = split_feature_columns(feature_frame)
    X = feature_frame

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in MODELS.items():
        pipeline = build_pipeline(
            model, numeric_columns=numeric_columns, categorical_columns=categorical_columns
        )
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
