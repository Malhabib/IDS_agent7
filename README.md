# IDS-Agent EV Charging Classifier

This repository provides IDS-Agent preprocessing and classification tools for EV charging sessions
along with a training script for six ML models.

## Preprocessing Tool
The preprocessing tool follows the implementation details:
- Remove irrelevant fields (label, timestamps, flow ID).
- Encode non-numerical fields.
- Apply F-test feature selection.
- Standardize extracted features.

## Model Training
Train six classifiers (RF, KNN, LR, DT, MLP, SVC) using a labeled CSV dataset that includes a
`label` column (0 benign, 1 malicious). The training pipeline automatically removes label/timestamp
fields, encodes categorical features, applies F-test feature selection, and standardizes the final
feature set.

Example:
```bash
python scripts/train_models.py path/to/dataset.csv --output-dir models
```

Models are persisted as `.joblib` files for inference use, and the classification tool returns
top-3 label predictions with confidence scores per model.

## Core LLM Options
The core IDS-Agent LLM choices are:
- GPT-3.5-Turbo
- GPT-4o-mini
- GPT-4o

## Evaluation (Sample-by-Sample)
Run evaluation across the CSV dataset while generating per-model reasoning and aggregating via the
core LLM:

```bash
python scripts/evaluate_ids_agent.py path/to/dataset.csv --models-dir models --core-llm gpt-4o-mini
```

The evaluation prints binary metrics (accuracy, FAR) and multi-class precision/recall/F1 along with
macro-averaged values.
