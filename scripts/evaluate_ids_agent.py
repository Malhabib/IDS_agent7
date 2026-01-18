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
    ModelReasoning,
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


def majority_vote(predictions: list[str]) -> str:
    counts: dict[str, int] = {}
    for label in predictions:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def render_binary_table(results: dict[str, dict[str, float]]) -> str:
    header = [
        "Metric Types",
        "Metrics",
        "RF",
        "Majority Vote",
        "IDS-Agent (GPT-3.5)",
        "IDS-Agent (GPT-4o-mini)",
        "IDS-Agent (GPT-4o)",
    ]
    rows = [
        ["Binary-Class", "Binary-Class Accuracy ↑"],
        ["Binary-Class", "FAR ↓"],
    ]
    for model_key in header[2:]:
        metrics = results.get(model_key, {"accuracy": 0.0, "far": 0.0})
        rows[0].append(f"{metrics['accuracy']:.3f}")
        rows[1].append(f"{metrics['far']:.3f}")

    table_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


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
) -> None:
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must include a 'label' column.")

    models = load_models(models_dir)
    y_true = df["label"].to_numpy()
    predictions_by_llm: dict[CoreLLM, list[str]] = {
        CoreLLM.GPT_3_5_TURBO: [],
        CoreLLM.GPT_4O_MINI: [],
        CoreLLM.GPT_4O: [],
    }
    rf_predictions: list[str] = []
    majority_predictions: list[str] = []

    for _, row in df.iterrows():
        sample = row.drop(labels=["label"]).to_dict()
        per_model_predictions: list[str] = []
        model_reasoning: list[ModelReasoning] = []
        for model_name, model_pipeline in models.items():
            output = classify_sample(model_name, model_pipeline, sample, k=3)
            top_prediction = output.top_predictions[0]
            per_model_predictions.append(top_prediction.label)
            model_reasoning.append(
                generate_model_reasoning(model_name, top_prediction)
            )
            if model_name == "rf":
                rf_predictions.append(top_prediction.label)
        majority_predictions.append(majority_vote(per_model_predictions))
        for core_llm in predictions_by_llm:
            aggregated = aggregate_with_core_llm(core_llm, model_reasoning)
            predictions_by_llm[core_llm].append(aggregated.label)

    results = {
        "RF": compute_binary_metrics(y_true, np.array(rf_predictions)),
        "Majority Vote": compute_binary_metrics(y_true, np.array(majority_predictions)),
        "IDS-Agent (GPT-3.5)": compute_binary_metrics(
            y_true, np.array(predictions_by_llm[CoreLLM.GPT_3_5_TURBO])
        ),
        "IDS-Agent (GPT-4o-mini)": compute_binary_metrics(
            y_true, np.array(predictions_by_llm[CoreLLM.GPT_4O_MINI])
        ),
        "IDS-Agent (GPT-4o)": compute_binary_metrics(
            y_true, np.array(predictions_by_llm[CoreLLM.GPT_4O])
        ),
    }

    print("Binary classification table:")
    print(render_binary_table(results))
    print()

    report = classification_report(
        y_true, predictions_by_llm[CoreLLM.GPT_4O_MINI], digits=4, zero_division=0
    )
    print("Multi-class classification report (IDS-Agent GPT-4o-mini):")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IDS-Agent sample-by-sample.")
    parser.add_argument("dataset", type=Path, help="Path to CSV dataset")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Directory with .joblib models"
    )
    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.models_dir)


if __name__ == "__main__":
    main()
