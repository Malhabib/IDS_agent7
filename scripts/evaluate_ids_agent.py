"""Run IDS-Agent sample-by-sample evaluation on a CSV dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.ids_agent import (
    CoreLLM,
    aggregate_with_core_llm,
    classify_sample,
    generate_model_reasoning,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    correct = (y_true == y_pred).sum()
    accuracy = float(correct / len(y_true)) if len(y_true) else 0.0
    benign_mask = y_true == 0
    false_positives = int(((y_pred == 1) & benign_mask).sum())
    total_benign = int(benign_mask.sum())
    far = float(false_positives / total_benign) if total_benign else 0.0
    return {"accuracy": accuracy, "far": far}


def load_models(models_dir: Path) -> dict[str, object]:
    models = {}
    for model_path in models_dir.glob("*.joblib"):
        models[model_path.stem] = joblib.load(model_path)
    if not models:
        raise ValueError(f"No models found in {models_dir}")
    return models


def evaluate_dataset(
    dataset_path: Path,
    models_dir: Path,
    *,
    core_llm: CoreLLM,
) -> None:
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must include a 'label' column.")

    models = load_models(models_dir)
    y_true = df["label"].to_numpy()
    y_pred = []

    for _, row in df.iterrows():
        sample = row.drop(labels=["label"]).to_dict()
        model_reasoning = []
        for model_name, model_pipeline in models.items():
            output = classify_sample(model_name, model_pipeline, sample, k=3)
            top_prediction = output.top_predictions[0]
            model_reasoning.append(
                generate_model_reasoning(model_name, top_prediction)
            )
        aggregated = aggregate_with_core_llm(core_llm, model_reasoning)
        y_pred.append(aggregated.label)

    y_pred = np.array(y_pred)

    binary_metrics = compute_binary_metrics(y_true, y_pred)
    print("Binary classification metrics:")
    print(f"Accuracy: {binary_metrics['accuracy']:.4f}")
    print(f"False Alarm Rate (FAR): {binary_metrics['far']:.4f}")
    print()

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print("Multi-class classification report:")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IDS-Agent sample-by-sample.")
    parser.add_argument("dataset", type=Path, help="Path to CSV dataset")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Directory with .joblib models"
    )
    parser.add_argument(
        "--core-llm",
        type=str,
        default=CoreLLM.GPT_4O_MINI.value,
        choices=[model.value for model in CoreLLM],
        help="Core LLM used to aggregate model reasoning.",
    )
    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.models_dir, core_llm=CoreLLM(args.core_llm))


if __name__ == "__main__":
    main()
