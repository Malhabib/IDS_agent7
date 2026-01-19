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

Default core LLM: GPT-4o (used by evaluation and aggregation).

## GPT-4o API اتصال
Set your API key before running evaluation so the core LLM aggregation uses GPT-4o:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Evaluation (Sample-by-Sample)
Run evaluation across the CSV dataset while generating per-model reasoning and aggregating via the
core LLM:

```bash
python scripts/evaluate_ids_agent.py path/to/dataset.csv --models-dir models --trace-line 1
```

The evaluation prints a binary-metrics table (RF, LR, KNN, MLP, DT, SVC, Majority Vote, IDS-Agent)
plus multi-class precision/recall/F1 with macro-averaged values. Use `--trace-line` to print the
iterative LLM-style reasoning/action/observation trace for a specific CSV line. The script will
also call GPT-4o to render the trace using the core LLM.

Each CSV row is evaluated as a separate request; the core LLM receives per-model predictions with
confidence scores, applies the majority-vote ensemble baseline, and returns a final label plus
explanation.

The IDS-Agent classification flow is multi-level: per-model reasoning is generated (with LIME
explanations), knowledge retrieval can be incorporated, and the core LLM aggregates across these
levels.

Baseline support includes GMM-clustered in-context demonstrations with dynamic LTM retrieval and an
ensemble majority-vote helper.

## ReAct-Style LLM Loop
The IDS-Agent uses a core LLM to iterate over reasoning, action generation, and observation update.
The loop starts with an initial observation built from the user request and tool descriptions, then
executes structured JSON-like actions and updates observations until a final answer is produced.
