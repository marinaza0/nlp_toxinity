# Multilingual Toxicity Classification

Benchmarking encoder transformers, instruction-tuned LLMs, and reasoning models for toxicity/hate speech classification across English, Russian, and Hindi.

See [AGENTS.md](AGENTS.md) for research questions, team structure, and evaluation guidelines.

---

## Project Structure

### 📊 Data
- **`data/raw/`** — Original datasets
  - `en/` — English (TextDetox)
  - `ru/` — Russian datasets + bad words dictionaries
  - `hi/` — Hindi datasets
- **`data/processed/`** — Prepared evaluation samples (subsets, balanced splits)

### 🧪 Experiments
- **`experiments/encoder_transformers/`** — Maryna's models (BERT, RoBERTa, DeBERTa, XLM-R)
- **`experiments/instruction_tuned_llms/`** — Dhruv's models (Llama, Qwen, Mistral, Gemma)
- **`experiments/reasoning_llms/`** — Yermukhamed's models (DeepSeek-R1, Qwen Reasoning)

Each folder contains standalone experiment scripts. Shared utilities can go in a `shared.py` file.

### 📓 Notebooks
- `compare_models.ipynb` — Model comparisons
- `finetune_monolingual_bert_model.ipynb` — Fine-tuning experiments
- `prepare_hindi_dataset.ipynb` — Dataset preparation

### 💻 Source Code
- **`src/preprocessing/`** — Data loading, cleaning
- **`src/models/`** — Model wrappers, inference
- **`src/evaluation/`** — Metrics, error analysis

### 📈 Results
- **`results/{team}/{model_name}/`** — Model predictions and metrics
  - Predictions (CSV with scores)
  - Evaluation metrics (accuracy, F1, confusion matrix, etc.)

---

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your code to your team's folder:
   - Create scripts in `experiments/{your_team}/`
   - Save results to `results/{your_team}/{model_name}/`
   - Reuse shared code from `src/`

3. Run experiments with deterministic settings (temperature=0).

4. Document your model, version, dataset, and exact metrics in results.
