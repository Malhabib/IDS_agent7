# IDS-Agent EV Charging Classifier

This repository provides a baseline IDS-Agent classifier for EV charging sessions along with a
training script for six ML models.

## Heuristic Classification (Runtime)
Use the `src/ids_agent.py` module for quick, rules-based classification when models are not yet
trained or when a lightweight decision is required.

Indicators:
- Very low charging efficiency
- Large idle time after charging
- Large mismatch between requested demand and delivered energy

## Model Training
Train six classifiers (RF, KNN, LR, DT, MLP, SVC) using a labeled CSV dataset that includes:
`connectionTime`, `disconnectTime`, `RequestedDemand`, `kWhDelivered`, and `label`.

Example:
```bash
python scripts/train_models.py path/to/dataset.csv --output-dir models
```

Models are persisted as `.joblib` files for inference use.
