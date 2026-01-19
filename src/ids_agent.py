"""IDS-Agent preprocessing and classification tools."""
from __future__ import annotations

from dataclasses import dataclass
import json
from enum import Enum
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from openai import OpenAI
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


DEFAULT_CORE_LLM = CoreLLM.GPT_4O


@dataclass(frozen=True)
class ModelReasoning:
    model_name: str
    prediction: str
    confidence: float
    reasoning: str
    lime_explanation: str | None = None


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
    lime_explanation: str | None = None,
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
        lime_explanation=lime_explanation,
    )


def build_lime_explainer(
    training_frame: pd.DataFrame,
    *,
    class_names: Sequence[str],
    categorical_features: Sequence[int],
    categorical_names: dict[int, List[str]],
) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=training_frame.values,
        feature_names=training_frame.columns.tolist(),
        class_names=list(class_names),
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        discretize_continuous=True,
    )


def lime_explain_prediction(
    explainer: LimeTabularExplainer,
    model_pipeline,
    sample: Mapping[str, object],
    *,
    num_features: int = 5,
) -> str:
    sample_frame = pd.DataFrame([sample])
    explanation = explainer.explain_instance(
        data_row=sample_frame.values[0],
        predict_fn=model_pipeline.predict_proba,
        num_features=num_features,
    )
    return "; ".join(f"{feature}={weight:.3f}" for feature, weight in explanation.as_list())


def retrieve_knowledge(query: str) -> KnowledgeRetrievalResult:
    """Stub for knowledge retrieval to support multi-level classification."""
    return KnowledgeRetrievalResult(query=query, snippets=[])


def assemble_context(
    model_reasoning: Sequence[ModelReasoning],
    knowledge: KnowledgeRetrievalResult,
    memory_context: Sequence[str],
) -> List[str]:
    context_lines = ["Model reasoning:"]
    for item in model_reasoning:
        line = f"- {item.model_name}: {item.reasoning}"
        if item.lime_explanation:
            line += f" | LIME: {item.lime_explanation}"
        context_lines.append(line)
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
    model_reasoning: Sequence[ModelReasoning],
    core_llm: CoreLLM = DEFAULT_CORE_LLM,
    *,
    knowledge: KnowledgeRetrievalResult | None = None,
    memory_context: Sequence[str] | None = None,
) -> AggregatedDecision:
    """Aggregate model reasoning into a single decision using the core LLM."""
    if not model_reasoning:
        return AggregatedDecision(
            label="Unknown",
            reasoning="No model outputs were provided for aggregation.",
            model_reasoning=[],
        )

    model_labels = [item.prediction for item in model_reasoning]
    top_label = majority_vote_predictions(model_labels)
    knowledge = knowledge or retrieve_knowledge("no-query")
    memory_context = memory_context or []
    context_lines = assemble_context(model_reasoning, knowledge, memory_context)
    context_lines.append(f"Majority vote label: {top_label}")
    system_prompt = (
        "You are IDS-Agent. Use majority voting across the six ML models as the ensemble baseline, "
        "then write a short reasoning summary and final label. Respond with JSON containing keys "
        "`label` and `explanation`."
    )
    user_prompt = "\n".join(context_lines)
    client = OpenAI()
    response = client.chat.completions.create(
        model=core_llm.value,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    label = top_label
    explanation = f"Final decision (majority): {top_label}."
    try:
        payload = json.loads(content)
        label = str(payload.get("label", label))
        explanation = str(payload.get("explanation", explanation))
    except json.JSONDecodeError:
        if content.strip():
            explanation = content.strip()
    return AggregatedDecision(
        label=label,
        reasoning=explanation,
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
    user_request: str,
    tool_descriptions: Sequence[str],
    tool_output: str,
    *,
    max_steps: int = 1,
    core_llm: CoreLLM = DEFAULT_CORE_LLM,
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


def build_iterative_trace(
    line_number: int,
    raw_features: Mapping[str, object],
    preprocessed_features: Sequence[float],
    model_outputs: Sequence[ClassificationOutput],
) -> List[str]:
    """Construct an iterative LLM-style trace for classification."""
    trace = [
        "Thought: I need to follow the plan to classify the traffic features from line number "
        f"{line_number:03d}. I will start by loading the traffic features from the CSV file.",
        "Action: load_data_line",
        f'Action Input: {{"line_number": {line_number}}}',
        "Observation: request “sessionID” ?????",
        "Thought: I have successfully loaded the traffic features from line number "
        f"{line_number:03d}. Now, I will proceed to preprocess the loaded traffic features to "
        "prepare them for classification.",
        "Action: data_preprocessing",
        f'Action Input: {{"traffic_features": " connectionTime:{raw_features.get("connectionTime")}, '
        f'disconnectTime:{raw_features.get("disconnectTime")}, '
        f'RequestedDemand:{raw_features.get("RequestedDemand")}, '
        f'kWhDelivered:{raw_features.get("kWhDelivered")}"}}',
        f"Observation: {list(preprocessed_features)}",
        "Thought: I have successfully preprocessed the traffic features. Now, I will proceed to "
        "classify the preprocessed features using multiple classifiers to determine if the "
        "traffic record is an attack.",
    ]
    for output in model_outputs:
        top_labels = [
            f"{pred.label} ({pred.confidence:.3f})" for pred in output.top_predictions
        ]
        trace.extend(
            [
                "Action: classifier",
                f'Action Input: {{"modelname": "{output.model_name}", ?????}}',
                f"Observation: Top predictions: {top_labels}",
                "Thought: I have obtained the classification results from the "
                f"{output.model_name} model. Now, I will classify the same preprocessed features "
                "using additional classifiers to gather more predictions.",
            ]
        )
    return trace


def generate_iterative_trace_with_llm(
    line_number: int,
    raw_features: Mapping[str, object],
    preprocessed_features: Sequence[float],
    model_outputs: Sequence[ClassificationOutput],
    *,
    core_llm: CoreLLM = DEFAULT_CORE_LLM,
) -> str:
    """Use the core LLM to render the iterative Thought/Action/Observation trace."""
    template_lines = build_iterative_trace(
        line_number=line_number,
        raw_features=raw_features,
        preprocessed_features=preprocessed_features,
        model_outputs=model_outputs,
    )
    system_prompt = (
        "You are IDS-Agent. Render the iterative Thought/Action/Observation trace exactly in the "
        "same style as the provided template, filling in values consistently."
    )
    user_prompt = "\n".join(template_lines)
    client = OpenAI()
    response = client.chat.completions.create(
        model=core_llm.value,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def gmm_cluster_examples(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster in-context examples using a Gaussian Mixture Model."""
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gmm.fit_predict(embeddings)


def select_diverse_demos(
    examples: Sequence[Mapping[str, object]],
    cluster_labels: Sequence[int],
    *,
    max_per_cluster: int = 1,
) -> List[Mapping[str, object]]:
    """Select in-context demonstrations from different clusters."""
    selected: List[Mapping[str, object]] = []
    seen_counts: dict[int, int] = {}
    for example, label in zip(examples, cluster_labels):
        count = seen_counts.get(label, 0)
        if count >= max_per_cluster:
            continue
        selected.append(example)
        seen_counts[label] = count + 1
    return selected


def retrieve_ltm_demos(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    *,
    top_k: int = 5,
) -> List[int]:
    """Retrieve top-k LTM examples based on cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    scores = embeddings @ query_embedding / np.maximum(norms, 1e-8)
    return scores.argsort()[::-1][:top_k].tolist()


def majority_vote_predictions(predictions: Sequence[str]) -> str:
    """Baseline ensemble method: majority vote across ML model outputs."""
    counts: dict[str, int] = {}
    for label in predictions:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
