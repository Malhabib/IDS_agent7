"""Run the IDS-Agent framework with the core ChatGPT/GPT-4o LLM."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ids_agent import (
    DEFAULT_CORE_LLM,
    aggregate_with_core_llm,
    classify_sample,
    generate_model_reasoning,
    generate_stepwise_llm_response,
    majority_vote_predictions,
    shap_explain_prediction,
)


def load_models(models_dir: Path) -> dict[str, object]:
    models = {}
    for model_path in models_dir.glob("*.joblib"):
        models[model_path.stem] = joblib.load(model_path)
    if not models:
        raise ValueError(f"No models found in {models_dir}")
    return models


def iter_line_numbers(
    total_rows: int,
    *,
    line_number: int,
    num_samples: int | None,
    all_samples: bool,
) -> list[int]:
    if line_number < 1 or line_number > total_rows:
        raise ValueError(f"line_number must be between 1 and {total_rows}")
    if all_samples:
        return list(range(line_number, total_rows + 1))
    if num_samples is None:
        return [line_number]
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    end = min(total_rows, line_number + num_samples - 1)
    return list(range(line_number, end + 1))


def run_framework(
    dataset_path: Path,
    models_dir: Path,
    *,
    line_number: int,
    num_samples: int | None,
    all_samples: bool,
) -> None:
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must include a 'label' column.")
    line_numbers = iter_line_numbers(
        len(df),
        line_number=line_number,
        num_samples=num_samples,
        all_samples=all_samples,
    )

    models = load_models(models_dir)
    rf_preprocessor = models["rf"].named_steps["preprocessing"]
    background = rf_preprocessor.transform(df.drop(columns=["label"]))
    model_order = ["rf", "lr", "knn", "mlp", "dt", "svc"]
    ordered_models = {name: models[name] for name in model_order if name in models}
    for model_name, model_pipeline in models.items():
        if model_name not in ordered_models:
            ordered_models[model_name] = model_pipeline

    labels_true: list[str] = []
    labels_majority: list[str] = []
    labels_ids: list[str] = []
    labels_by_model: dict[str, list[str]] = {name: [] for name in ordered_models}

    for index, line_number in enumerate(line_numbers, start=1):
        row = df.iloc[line_number - 1]
        labels_true.append(str(row["label"]))
        sample = row.drop(labels=["label"]).to_dict()
        sample_frame = pd.DataFrame([sample])
        preprocessed = rf_preprocessor.transform(sample_frame)[0]
        preprocessed_dense = (
            preprocessed.toarray().ravel() if hasattr(preprocessed, "toarray") else preprocessed
        )
        feature_count = (
            preprocessed_dense.shape[0]
            if hasattr(preprocessed_dense, "shape")
            else len(preprocessed_dense)
        )
        feature_names = [f"f{idx}" for idx in range(feature_count)]

        model_outputs = []
        model_reasoning = []
        for model_name, model_pipeline in ordered_models.items():
            output = classify_sample(model_name, model_pipeline, sample, k=3)
            model_outputs.append(output)
            top_prediction = output.top_predictions[0]
            labels_by_model[model_name].append(str(top_prediction.label))
            shap_explanation = shap_explain_prediction(
                model_pipeline.named_steps["model"],
                preprocessed_dense,
                background,
                feature_names,
            )
            model_reasoning.append(
                generate_model_reasoning(
                    model_name,
                    top_prediction,
                    explanation=shap_explanation,
                )
            )

        system_prompt, user_prompt, stepwise_response = generate_stepwise_llm_response(
            model_names=list(models.keys()),
            line_number=line_number,
            raw_features=sample,
            preprocessed_features=preprocessed_dense.tolist(),
            model_outputs=model_outputs,
            model_reasoning=model_reasoning,
            core_llm=DEFAULT_CORE_LLM,
        )
        if len(line_numbers) > 1:
            print(f"\n=== Sample {index}/{len(line_numbers)} (line {line_number}) ===")
        print("Stepwise LLM prompt (system):")
        print(system_prompt)
        print("\nStepwise LLM prompt (user):")
        print(user_prompt)
        print(stepwise_response)

        aggregated = aggregate_with_core_llm(
            model_reasoning,
            core_llm=DEFAULT_CORE_LLM,
        )
        labels_majority.append(majority_vote_predictions([item.prediction for item in model_reasoning]))
        labels_ids.append(str(aggregated.label))
        print("\nFinal aggregation:")
        print(aggregated.reasoning)
        if aggregated.raw_response:
            print("\nFinal aggregation (raw LLM response):")
            print(aggregated.raw_response)

    report_columns = []
    report_scores = []
    for model_name in model_order:
        if model_name in labels_by_model:
            report_columns.append(model_name.upper())
            report_scores.append(
                f1_score(labels_true, labels_by_model[model_name], average="macro", zero_division=0)
            )
    for model_name in labels_by_model:
        if model_name in model_order:
            continue
        report_columns.append(model_name.upper())
        report_scores.append(
            f1_score(labels_true, labels_by_model[model_name], average="macro", zero_division=0)
        )
    report_columns.extend(["Majority Vote", "IDS-Agent"])
    report_scores.extend(
        [
            f1_score(labels_true, labels_majority, average="macro", zero_division=0),
            f1_score(labels_true, labels_ids, average="macro", zero_division=0),
        ]
    )

    print("\nF1-score report (macro average):")
    header = "| Model | " + " | ".join(report_columns) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(report_columns)) + " |"
    values = "| F1-score | " + " | ".join(f"{score:.3f}" for score in report_scores) + " |"
    print(header)
    print(separator)
    print(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDS-Agent with ChatGPT core LLM.")
    parser.add_argument("dataset", type=Path, help="Path to CSV dataset")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Directory with .joblib models"
    )
    parser.add_argument(
        "--line-number",
        type=int,
        default=1,
        help="CSV line number to start from (1-indexed).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to run starting from --line-number.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all samples from --line-number to the end of the dataset.",
    )
    args = parser.parse_args()
    run_framework(
        args.dataset,
        args.models_dir,
        line_number=args.line_number,
        num_samples=args.num_samples,
        all_samples=args.all,
    )


if __name__ == "__main__":
    main()
