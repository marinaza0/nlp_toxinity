Project Overview

This project is part of a TUM Bachelor Informatics NLP Practicum.

Topic:

Multilingual Hate Speech and Toxicity Classification: 2026 Reality Check

The project is a benchmarking and comparison study. We do not develop a new model from scratch. Instead, we evaluate and compare different categories of modern NLP models on multilingual toxicity classification.

⸻

Research Question

What approaches work best for multilingual toxic/hate speech classification?

We compare:

1. Encoder-Based Transformers
2. Instruction-Tuned LLMs
3. Modern Reasoning LLMs

We evaluate both language-specific and multilingual models.

⸻

Languages

The project focuses on:

* English
* Russian
* Hindi

⸻

Team Structure

Encoder-Based Transformers

Owner: Maryna

Examples:

* BERT
* RoBERTa
* DeBERTa
* XLM-R

Responsibilities:

* One Russian-focused model
* One multilingual model

⸻

Instruction-Tuned LLMs

Owner: Dhruv

Examples:

* Llama Instruct
* Qwen Instruct
* Mistral Instruct
* Gemma Instruct

Responsibilities:

* One Hindi-focused model
* One multilingual model

⸻

Modern Reasoning LLMs

Owner: Yermukhamed

Responsibilities:

English-Focused Reasoning Model

* DeepSeek-R1:7B (Ollama)

Multilingual Reasoning Model

* Qwen3 Reasoning

⸻

Datasets

Current English Dataset

Hugging Face:

textdetox/multilingual_toxicity_dataset

English split:

* 5000 samples
* Columns:
    * text
    * toxic

Labels:

* 0 = non-toxic
* 1 = toxic

Multilingual Evaluation

The current plan is to use the same dataset source for:

* English
* Russian
* Hindi

This allows fair comparison across languages and model categories.

⸻

Evaluation Metrics

Primary metric:

* Macro-F1

Secondary metrics:

* Accuracy
* Precision
* Recall
* Confusion Matrix

For multilingual experiments:

* Overall Macro-F1
* English Macro-F1
* Russian Macro-F1
* Hindi Macro-F1

⸻

Current Baseline Result (to be updated)

Model:

* DeepSeek-R1:7B

Dataset:

* TextDetox English

Sample 100:

* 100 balanced samples
* 50 toxic
* 50 non-toxic

Confusion Matrix:

[[50, 0],
[ 2,48]]

Sample 500:

* 100 balanced samples
* 50 toxic
* 50 non-toxic

Confusion matrix:
[[230  20]
 [ 23 226]]
⸻

Repository Structure (may be not updated)

.
├── data
│   ├── raw
│   └── processed
├── experiments
├── notebooks
├── src
│   ├── evaluation
│   ├── models
│   └── preprocessing
└── README.md

data/raw

Original datasets.

data/processed

Generated evaluation samples.

experiments

Standalone experiment scripts.

src

Reusable project code.

⸻

Development Guidelines

* Keep experiments reproducible.
* Do not modify raw datasets.
* Save intermediate datasets into data/processed.
* Store model outputs separately from source datasets.
* Prefer deterministic runs (temperature=0) when benchmarking.
* Document prompts used for LLM evaluation.
* Report exact model versions.
* Keep comparisons fair across model categories.

⸻

Current Reasoning Model Workflow

1. Load dataset.
2. Create balanced evaluation sample.
3. Run model inference.
4. Save predictions.
5. Compute:
    * Accuracy
    * Precision
    * Recall
    * Macro-F1
6. Perform error analysis.
7. Scale evaluation from:
    * 100 samples
    * 500 samples
    * full dataset

⸻

Reproducibility

Always record:

* Model name
* Model version
* Prompt
* Dataset
* Dataset split
* Sample size
* Metrics

Every reported result must be reproducible from repository code.