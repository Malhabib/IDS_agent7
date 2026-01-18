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
