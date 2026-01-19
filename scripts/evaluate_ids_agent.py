"""Run IDS-Agent sample-by-sample evaluation on a CSV dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ids_agent import (
    CoreLLM,
    DEFAULT_CORE_LLM,
    ModelReasoning,
    aggregate_with_core_llm,
    build_lime_explainer,
    build_iterative_trace,
    classify_sample,
    generate_iterative_trace_with_llm,
    generate_model_reasoning,
    lime_explain_prediction,
    retrieve_knowledge,
    split_feature_columns,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    benign_label: str = "0",
    malicious_label: str = "1",
) -> dict[str, float]:
    correct = (y_true == y_pred).sum()
    accuracy = float(correct / len(y_true)) if len(y_true) else 0.0
    benign_mask = y_true == benign_label
    false_positives = int(((y_pred == malicious_label) & benign_mask).sum())
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
        "LR",
        "KNN",
        "MLP",
        "DT",
        "SVC",
        "Majority Vote",
        "IDS-Agent",
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
    *,
    trace_line: int | None = None,
) -> None:
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must include a 'label' column.")

    models = load_models(models_dir)
    y_true = df["label"].astype(str).to_numpy()
    feature_frame = df.drop(columns=["label"])
    numeric_columns, categorical_columns = split_feature_columns(feature_frame)
    categorical_indices = [
        feature_frame.columns.get_loc(col) for col in categorical_columns
    ]
    categorical_names = {
        feature_frame.columns.get_loc(col): sorted(
            feature_frame[col].dropna().astype(str).unique().tolist()
        )
        for col in categorical_columns
    }
    predictions_by_llm: dict[CoreLLM, list[str]] = {
        DEFAULT_CORE_LLM: [],
    }
    model_predictions: dict[str, list[str]] = {
        "rf": [],
        "lr": [],
        "knn": [],
        "mlp": [],
        "dt": [],
        "svc": [],
    }
    majority_predictions: list[str] = []

    explainers = {
        model_name: build_lime_explainer(
            feature_frame,
            class_names=[str(cls) for cls in model_pipeline.classes_],
            categorical_features=categorical_indices,
            categorical_names=categorical_names,
        )
        for model_name, model_pipeline in models.items()
    }

    for row_index, row in df.iterrows():
        sample = row.drop(labels=["label"]).to_dict()
        per_model_predictions: list[str] = []
        model_reasoning: list[ModelReasoning] = []
        model_outputs = []
        for model_name, model_pipeline in models.items():
            output = classify_sample(model_name, model_pipeline, sample, k=3)
            model_outputs.append(output)
            top_prediction = output.top_predictions[0]
            per_model_predictions.append(top_prediction.label)
            lime_explanation = lime_explain_prediction(
                explainers[model_name],
                model_pipeline,
                sample,
            )
            model_reasoning.append(
                generate_model_reasoning(
                    model_name,
                    top_prediction,
                    lime_explanation=lime_explanation,
                )
            )
            if model_name in model_predictions:
                model_predictions[model_name].append(top_prediction.label)
        query = " ".join(sorted(set(per_model_predictions)))
        knowledge = retrieve_knowledge(query)
        majority_predictions.append(majority_vote(per_model_predictions))
        aggregated = aggregate_with_core_llm(
            model_reasoning,
            core_llm=DEFAULT_CORE_LLM,
            knowledge=knowledge,
            memory_context=[],
        )
        predictions_by_llm[DEFAULT_CORE_LLM].append(aggregated.label)
        if trace_line is not None and row_index + 1 == trace_line:
            sample_frame = pd.DataFrame([sample])
            preprocessed = models["rf"].named_steps["preprocessing"].transform(sample_frame)[
                0
            ].tolist()
            trace = build_iterative_trace(
                line_number=trace_line,
                raw_features=sample,
                preprocessed_features=preprocessed,
                model_outputs=model_outputs,
            )
            print("Iterative LLM trace:")
            print("\n".join(trace))
            llm_trace = generate_iterative_trace_with_llm(
                line_number=trace_line,
                raw_features=sample,
                preprocessed_features=preprocessed,
                model_outputs=model_outputs,
                core_llm=DEFAULT_CORE_LLM,
            )
            print("\nIterative LLM trace (GPT-4o):")
            print(llm_trace)

    results = {
        "RF": compute_binary_metrics(y_true, np.array(model_predictions["rf"])),
        "LR": compute_binary_metrics(y_true, np.array(model_predictions["lr"])),
        "KNN": compute_binary_metrics(y_true, np.array(model_predictions["knn"])),
        "MLP": compute_binary_metrics(y_true, np.array(model_predictions["mlp"])),
        "DT": compute_binary_metrics(y_true, np.array(model_predictions["dt"])),
        "SVC": compute_binary_metrics(y_true, np.array(model_predictions["svc"])),
        "Majority Vote": compute_binary_metrics(y_true, np.array(majority_predictions)),
        "IDS-Agent": compute_binary_metrics(
            y_true, np.array(predictions_by_llm[DEFAULT_CORE_LLM])
        ),
    }

    print("Binary classification table:")
    print(render_binary_table(results))
    print()

    report = classification_report(
        y_true, predictions_by_llm[DEFAULT_CORE_LLM], digits=4, zero_division=0
    )
    print("Multi-class classification report (IDS-Agent):")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IDS-Agent sample-by-sample.")
    parser.add_argument("dataset", type=Path, help="Path to CSV dataset")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Directory with .joblib models"
    )
    parser.add_argument(
        "--trace-line",
        type=int,
        default=None,
        help="Optional line number to print the iterative LLM trace for.",
    )
    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.models_dir, trace_line=args.trace_line)


if __name__ == "__main__":
    main()
