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

Specifically, IDS-Agent iterates over the following three steps:
1) Reasoning: The core LLM generates a thought (in plain text) about the next action by ri = LLM(si), where si = {o0, {r1, a1, o1}, · · · , {ri−1, ai−1, oi−1}} is the short-term memory of the current session up to the (i − 1)-th iteration (with s1 = {o0}). Reasoning can optionally adopt long-term memory from previous sessions for in-context demonstration.
2) Action generation: The action is generated based on the reasoning/thought by ai = LLM(ri, si). Notably, each action we generate is a structured JSON file containing an action name and an action input. The action input consists of the name of the tool(s) to be used and the associated settings or parameters of each tool. Such a structured generation of the action allows its efficient and accurate execution using the specified tools. For example, if the action is to adopt a classifier, the action name will be “Classification” and the action input will be the model name and the associated settings.
3) Observation update: After executing the generated action ai, we obtain a new observation oi by converting the outputs of the tool(s) into plain text. For example, the the observation after applying a classifier will be the top-k labels and their corresponding prediction confidences.

The iterations terminate when the observation is updated by a ‘final answer’ headline followed by a JSON file. This JSON file, which encapsulates the final prediction on the traffic data and related analysis and explanation, will be the output of IDS-Agent.

### Action Space and Tool Design
Our IDS-Agent is designed with a comprehensive action space, allowing it to handle various tasks in the pipeline of data processing and classification through iterative reasoning and execution. The action space includes the key actions described below.

Data Extraction: The goal is to accurately extract network traffic records x specified in the user request from the data flow (a dataset in our experiments) for further analysis. We design the data extraction tool as a function that takes the given flow ID (or line number for stored traffic dataset) as the input and outputs a structured traffic sample.

Preprocessing: The goal is to clean, normalize, and transform the extracted data into a standard format for subsequence processing, especially for classification. Our preprocessing tools are designed as functions for diverse data analysis operations, including feature scaling, data encoding, handling missing values, and selecting important features for classification.

Classification: This action applies machine learning models to the preprocessed data to obtain classification results, i.e., to predict whether the traffic is benign or falling into any malicious category. The inputs to the classification tool include the preprocessed traffic features and a classifier, while the outputs include the top-k labels and their corresponding confidence scores. Note that our classification tool is not merely an ML model; it advances ML-based IDS by adaptively considering diverse model types and incorporating more information from the classification results into the inference procedure. The types of classifiers used by IDS-Agent include Random Forest, SVM, MLP, Decision Tree, KNN, etc. It is important to note that our classification toolbox is extensible. In real-world deployments, users can easily add new classifiers to the toolbox without the need for any LLM fine-tuning. This flexibility allows for easy updates and adaptation to new attack patterns or changes in the network environment. Moreover, the classifiers can also be open-sourced models trained by third parties (callable through APIs), or models locally trained based on the data collected by the user.

Knowledge Retrieval: This action aims to obtain knowledge regarding the particular types of attacks predicted by the classifiers. The knowledge can be external that is retrieved by calling search engines such as Google and Wikipedia API, or stored locally in the knowledge base. The retrieval from the local knowledge follows the RAG approach [20]. The retrieved knowledge will be used to guide the aggregation of classification results from the individual ML models.

Long-Term Memory Retrieval: Long-Term Memory carries information from previous sessions (will be discussed in Sec. 3.3) that can inform the decision making of IDS-Agent in the current session. The retrieval of long-term memory can be activated optionally by adding a specific instruction in the system prompt in o0.

Aggregation: This action aims to comprehensively integrate the results from multiple steps of classification action (based on different classifiers) to generate a structured final inference decision. The core of the aggregation tool is an LLM where the prompt is designed to include 1) the detailed results from the classifiers, 2) the short-term memory, and 3) demonstrative inputs and outputs aggregated by the LLM from previous sessions. Note that the short-term memory also includes the external knowledge previously extracted and the system prompts in o0. This system prompt, as shown in Fig. 3, includes a specification for the detection sensitivity; it can also include an instruction for revealing new attack categories. Compared to naive aggregation, such as majority voting, our method incorporates more information to resolve any discrepancies between different model outputs in a more comprehensive way.

### Memory and Knowledge Base
The memory and knowledge base of IDS-Agent stores 1) the short-term memory, 2) the long-term memory, and 3) supportive documents for IDS from the external. Short-term Memory (STR). The STR, as described in Sec. 3.1, includes the historical reasoning trace, actions, and observations of the current session in a structured format, and is renewed after each observation update. The major goal of the STM here is to track the agent’s iterative reasoning process and ensure consistency between steps in real-time.

Long-term Memory (LTM). The LTM consists of agent decisions and contextual information from previous use cases [33, 41]. Here, a structured LTM example is defined by ϕ = {t, x, R, A, O, yˆ}, where t is a timestamp, x is the feature vector after preprocessing, R = [r1, · · · , rn] is the reasoning trace, A = [a1, · · · , an] contains all the action steps, O = [o1, · · · , on] are the observations, and yˆ is the final label prediction. The long-term memory base can be initialized by running the agent on a validation dataset. During the inference, only sessions with correct agent decisions will be stored in the long-term memory base. In a real-world intrusion detection scenario, such correctness can be validated by human experts.

In our framework, LTM retrieval provides the agent with additional information while aggregating the results for individual classifiers. Here, we set the LTM retriever input as the current timestamp t and observations from previous data processing and classification actions, denoted by O˜ = [o1, . . . , om−1], where m is an arbitrary iteration where the LTM retrieval kicks in. The retriever obtains the top-k relevant final reasoning based on the weighted sum of timestamp distance and the cosine similarity between the embedding of O˜ and the observation embeddings O(j) of previous LTM examples {ϕ(1), · · · , ϕ(L)}. Specifically, we obtain the top-k solutions to
argmaxj [λ1r(t, t(j)) + λ2cosim(E (O˜), E (O(j)))],            (1)
where E (·) is the encoder. r(t, t(j)) = 1−(t−t(j))/maxk(t−t(k)) is the recency of the memory. The equation ensures that both recent observations and content-wise similar observations are considered to address the evolving nature of intrusion data. Then, the observation om for the iteration m contains the input-prediction pairs (x(j), yˆ(j)) of each retrieved ϕ(j). In our experiments, we set k = 5 to retrieve the top-5 relevant structured LTM examples.

Supportive Documents from the External. In addition to the external knowledge obtained by calling search engines, such as Google and Wikipedia, IDS-Agent is also equipped with a vector database {ψ(1), · · · , ψ(K)} containing related research papers and intrusion detection blogs (both parsed into chunks with fixed token length). The retrieval from this knowledge base is similar to the retrieval of LTM. We obtain the top-k solutions to argmaxj cosim(E (q), E (ψ(j))), where q is the query generated by the core LLM (based on the reasoning) as the action input to action step am for an arbitrary iteration m for knowledge retrieval. The retrieved document chunks are summarized (for compression) using an LLM (may be the same as the core LLM or an independent LLM) and are used to update the observation om. The definition, characteristics, and detection methods for various attacks recorded in the retrieved chunks will facilitate IDS-Agent to better understand the potential risks while aggregating the classification results for the ML models. For example, if an attack type can potentially lead to catastrophic results, IDS-Agent will be more sensitive to it when any classifier makes such a prediction.
