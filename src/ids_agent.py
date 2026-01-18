"""IDS-Agent preprocessing and classification tools."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_LABEL_COLUMN = "label"
DEFAULT_TIMESTAMP_COLUMNS = ("connectionTime", "disconnectTime", "timestamp")
DEFAULT_FLOW_ID_COLUMNS = ("flow_id", "flowId", "flowID", "FlowID")

class CoreLLM(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


@dataclass(frozen=True)
class ModelReasoning:
    model_name: str
    prediction: str
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class AggregatedDecision:
    label: str
    reasoning: str
    model_reasoning: List[ModelReasoning]


@dataclass(frozen=True)
class KnowledgeSnippet:
    source: str
    content: str


@dataclass(frozen=True)
class KnowledgeRetrievalResult:
    query: str
    snippets: List[KnowledgeSnippet]


@dataclass(frozen=True)
class ActionStep:
    name: str
    tool: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class Observation:
    headline: str
    content: str


def drop_irrelevant_fields(
    frame: pd.DataFrame,
    *,
    label_column: str = DEFAULT_LABEL_COLUMN,
    timestamp_columns: Sequence[str] = DEFAULT_TIMESTAMP_COLUMNS,
    flow_id_columns: Sequence[str] = DEFAULT_FLOW_ID_COLUMNS,
) -> pd.DataFrame:
    """Remove label, timestamps, and flow identifiers before preprocessing."""
    drop_columns = {
        col
        for col in frame.columns
        if col == label_column
        or col in timestamp_columns
        or col in flow_id_columns
    }
    return frame.drop(columns=sorted(drop_columns), errors="ignore")


def split_feature_columns(frame: pd.DataFrame) -> tuple[List[str], List[str]]:
    """Split columns into numeric and categorical lists."""
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in frame.columns if col not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessing_pipeline(
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    k_best: int | str = "all",
) -> Pipeline:
    """Build preprocessing pipeline with encoding, F-test feature selection, and scaling."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif, k=k_best)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )


@dataclass(frozen=True)
class TopPrediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class ClassificationOutput:
    model_name: str
    top_predictions: List[TopPrediction]


def _ensure_frame(samples: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(samples)


def _top_k_from_proba(
    proba: np.ndarray,
    classes: Sequence[object],
    *,
    k: int = 3,
) -> List[TopPrediction]:
    top_count = min(k, proba.shape[1])
    indices = np.argsort(proba, axis=1)[:, ::-1][:, :top_count]
    row = indices[0]
    return [
        TopPrediction(label=str(classes[idx]), confidence=float(proba[0, idx]))
        for idx in row
    ]


def classify_sample(
    model_name: str,
    model_pipeline,
    sample: Mapping[str, object],
    *,
    k: int = 3,
) -> ClassificationOutput:
    """Run the full preprocessing + classification pipeline and return top-k labels."""
    frame = _ensure_frame([sample])
    if hasattr(model_pipeline, "predict_proba"):
        proba = model_pipeline.predict_proba(frame)
        top_predictions = _top_k_from_proba(proba, model_pipeline.classes_, k=k)
    else:
        scores = model_pipeline.decision_function(frame)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        top_predictions = _top_k_from_proba(proba, model_pipeline.classes_, k=k)
    return ClassificationOutput(model_name=model_name, top_predictions=top_predictions)


def generate_model_reasoning(
    model_name: str,
    top_prediction: TopPrediction,
) -> ModelReasoning:
    """Generate per-model reasoning text for feeding the core LLM."""
    reasoning = (
        f"{model_name} predicts {top_prediction.label} with confidence "
        f"{top_prediction.confidence:.4f} based on the preprocessed features."
    )
    return ModelReasoning(
        model_name=model_name,
        prediction=top_prediction.label,
        confidence=top_prediction.confidence,
        reasoning=reasoning,
    )


def retrieve_knowledge(query: str) -> KnowledgeRetrievalResult:
    """Stub for knowledge retrieval to support multi-level classification."""
    return KnowledgeRetrievalResult(query=query, snippets=[])


def assemble_context(
    model_reasoning: Sequence[ModelReasoning],
    knowledge: KnowledgeRetrievalResult,
    memory_context: Sequence[str],
) -> List[str]:
    context_lines = ["Model reasoning:"]
    context_lines.extend(f"- {item.model_name}: {item.reasoning}" for item in model_reasoning)
    context_lines.append(f"Knowledge retrieval query: {knowledge.query}")
    if knowledge.snippets:
        context_lines.append("Knowledge snippets:")
        context_lines.extend(
            f"- ({snippet.source}) {snippet.content}" for snippet in knowledge.snippets
        )
    if memory_context:
        context_lines.append("Long-term memory context:")
        context_lines.extend(f"- {item}" for item in memory_context)
    return context_lines


def aggregate_with_core_llm(
    core_llm: CoreLLM,
    model_reasoning: Sequence[ModelReasoning],
    *,
    knowledge: KnowledgeRetrievalResult | None = None,
    memory_context: Sequence[str] | None = None,
) -> AggregatedDecision:
    """Aggregate model reasoning into a single decision (LLM-compatible stub)."""
    if not model_reasoning:
        return AggregatedDecision(
            label="Unknown",
            reasoning="No model outputs were provided for aggregation.",
            model_reasoning=[],
        )

    votes = {}
    for item in model_reasoning:
        votes[item.prediction] = votes.get(item.prediction, 0) + 1

    sorted_votes = sorted(votes.items(), key=lambda x: (-x[1], x[0]))
    top_label, _ = sorted_votes[0]
    knowledge = knowledge or retrieve_knowledge("no-query")
    memory_context = memory_context or []
    context_lines = assemble_context(model_reasoning, knowledge, memory_context)
    reasoning_lines = [
        f"Core LLM ({core_llm.value}) aggregated {len(model_reasoning)} model outputs.",
        "Multi-level context:",
    ]
    reasoning_lines.extend(context_lines)
    reasoning_lines.append(f"Final decision (majority): {top_label}.")
    return AggregatedDecision(
        label=top_label,
        reasoning="\n".join(reasoning_lines),
        model_reasoning=list(model_reasoning),
    )


def build_initial_observation(
    user_request: str,
    tool_descriptions: Sequence[str],
) -> Observation:
    content = "\n".join([user_request, *tool_descriptions])
    return Observation(headline="initial observation", content=content)


def llm_reasoning(core_llm: CoreLLM, short_term_memory: Sequence[str]) -> str:
    """Stub for LLM reasoning over short-term memory."""
    memory_preview = " | ".join(short_term_memory[-3:])
    return f"{core_llm.value} reasoning over memory: {memory_preview}"


def llm_action_generation(reasoning: str, short_term_memory: Sequence[str]) -> ActionStep:
    """Stub for structured JSON action generation."""
    return ActionStep(
        name="Classification",
        tool="classification_tool",
        parameters={"reasoning": reasoning, "memory_size": len(short_term_memory)},
    )


def update_observation(action: ActionStep, tool_output: str) -> Observation:
    return Observation(
        headline=f"observation after {action.name}",
        content=tool_output,
    )


def run_react_loop(
    core_llm: CoreLLM,
    user_request: str,
    tool_descriptions: Sequence[str],
    tool_output: str,
    *,
    max_steps: int = 1,
) -> Observation:
    """Run a minimal ReAct-style loop and return the final observation."""
    observation = build_initial_observation(user_request, tool_descriptions)
    short_term_memory = [observation.content]
    for _ in range(max_steps):
        reasoning = llm_reasoning(core_llm, short_term_memory)
        action = llm_action_generation(reasoning, short_term_memory)
        observation = update_observation(action, tool_output)
        short_term_memory.append(f"{reasoning} -> {action.name} -> {observation.content}")
    return observation
