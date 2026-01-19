"""Run the IDS-Agent framework with the core ChatGPT/GPT-4o LLM."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ids_agent import (
    DEFAULT_CORE_LLM,
    aggregate_with_core_llm,
    classify_sample,
    generate_model_reasoning,
    generate_stepwise_llm_response,
    shap_explain_prediction,
)


def load_models(models_dir: Path) -> dict[str, object]:
    models = {}
    for model_path in models_dir.glob("*.joblib"):
        models[model_path.stem] = joblib.load(model_path)
    if not models:
        raise ValueError(f"No models found in {models_dir}")
    return models


def run_framework(dataset_path: Path, models_dir: Path, *, line_number: int) -> None:
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns:
        raise ValueError("Dataset must include a 'label' column.")
    if line_number < 1 or line_number > len(df):
        raise ValueError(f"line_number must be between 1 and {len(df)}")

    models = load_models(models_dir)
    row = df.iloc[line_number - 1]
    sample = row.drop(labels=["label"]).to_dict()
    rf_preprocessor = models["rf"].named_steps["preprocessing"]
    sample_frame = pd.DataFrame([sample])
    preprocessed = rf_preprocessor.transform(sample_frame)[0]
    feature_names = [f"f{idx}" for idx in range(len(preprocessed))]

    model_outputs = []
    model_reasoning = []
    for model_name, model_pipeline in models.items():
        output = classify_sample(model_name, model_pipeline, sample, k=3)
        model_outputs.append(output)
        top_prediction = output.top_predictions[0]
        shap_explanation = shap_explain_prediction(
            model_pipeline.named_steps["model"],
            preprocessed,
            feature_names,
        )
        model_reasoning.append(
            generate_model_reasoning(
                model_name,
                top_prediction,
                explanation=shap_explanation,
            )
        )

    stepwise_response = generate_stepwise_llm_response(
        model_names=list(models.keys()),
        line_number=line_number,
        raw_features=sample,
        preprocessed_features=preprocessed.tolist(),
        model_outputs=model_outputs,
        model_reasoning=model_reasoning,
        core_llm=DEFAULT_CORE_LLM,
    )
    print(stepwise_response)

    aggregated = aggregate_with_core_llm(
        model_reasoning,
        core_llm=DEFAULT_CORE_LLM,
    )
    print("\nFinal aggregation:")
    print(aggregated.reasoning)


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
        help="CSV line number to classify (1-indexed).",
    )
    args = parser.parse_args()
    run_framework(args.dataset, args.models_dir, line_number=args.line_number)


if __name__ == "__main__":
    main()
