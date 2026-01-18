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
