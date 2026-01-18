# IDS Agent Understanding

## Objective
Classify EV charging sessions as Normal or Attack using the EV session features and the provided label (0 = benign, 1 = malicious). The output should be a short explanation followed by `Prediction: Normal` or `Prediction: Attack`.

## Data Inputs
Key features for the classifier:
- `connectionTime`
- `disconnectTime`
- `RequestedDemand`
- `kWhDelivered`
- `label` (ground truth: 0 benign, 1 malicious)

## Classifier Suite
Train and persist six models so they are ready for inference:
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)
- Decision Tree (DT)
- Multi-Layer Perceptron (MLP)
- Support Vector Classifier (SVC)

## Training/Inference Expectations
- Preprocess numeric features (e.g., handle missing values, scale where appropriate).
- Train all six models against the labeled dataset.
- Save trained models for the framework to load later.
- At inference, compute indicators such as charging efficiency, idle time, and requested-vs-delivered mismatch to justify the prediction in the explanation.

## IDS-Agent Pipeline (ReAct-Inspired)
The IDS-Agent pipeline follows a ReAct-style loop. The core LLM generates a sequence of actions `{a1, a2, ...}` based on prior reasoning and observations. For intrusion detection requests, the agent builds an initial observation `o0` by concatenating the user request with a system prompt that includes a description of each available tool. This initial observation provides the context the agent uses for subsequent reasoning and action generation.
