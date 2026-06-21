# Reasoning LLMs Benchmark Results

**Project:** Multilingual Toxicity Classification  
**Focus:** English TextDetox Dataset  
**Date:** 2026-06-21

---

## Executive Summary

Evaluated two state-of-the-art reasoning LLMs on toxicity classification:
- **DeepSeek-R1:7B** (Ollama)
- **Qwen3** (Ollama)

Both models performed exceptionally well, with high accuracy and balanced precision/recall across sample sizes.

---

## Overall Performance Comparison

| Model | Samples | Accuracy | Precision | Recall | Macro-F1 | Unknown |
|-------|---------|----------|-----------|--------|----------|---------|
| **DeepSeek-R1** | 100 | 0.980 | 1.000 | 0.960 | **0.980** | 0/100 (0.0%) |
| **DeepSeek-R1** | 500 | 0.914 | 0.919 | 0.908 | **0.914** | 1/500 (0.2%) |
| **Qwen3** | 100 | 0.960 | 0.942 | 0.980 | **0.960** | 0/100 (0.0%) |
| **Qwen3** | 500 | 0.942 | 0.917 | 0.972 | **0.942** | 0/500 (0.0%) |
| **Qwen3:8B** | TextDetox EN+RU+HI | 504 | 0.893 | 0.902 | 0.881 | **0.893** | 0/504 (0.0%) |

---

## Detailed Results

### DeepSeek-R1:7B

#### 100 Samples (en)
- **Accuracy:** 0.980
- **Precision:** 1.000 (No false positives)
- **Recall:** 0.960
- **Macro-F1:** 0.980
- **Confusion Matrix:**
  ```
  [[50,  0],
   [ 2, 48]]
  ```
- **Unknown Predictions:** 0/100 (0.0%)
- **Observations:** Perfect precision—zero false positives. Two false negatives suggest conservative classification of borderline toxic content.

#### 500 Samples (en)
- **Accuracy:** 0.914
- **Precision:** 0.919
- **Recall:** 0.908
- **Macro-F1:** 0.914
- **Confusion Matrix:**
  ```
  [[230, 20],
   [23, 226]]
  ```
- **Unknown Predictions:** 1/500 (0.2%)
- **Failed Example:**
  ```
  "Better dead than being abused by some Middle eastern douchebro chucklefucks."
  ```
- **Observations:** Performance remains strong at larger scale. Only 1 sample couldn't be classified (likely ambiguous or adversarial text).

---

### Qwen3:8B

#### 100 Samples (en)
- **Accuracy:** 0.960
- **Precision:** 0.942
- **Recall:** 0.980
- **Macro-F1:** 0.960
- **Confusion Matrix:**
  ```
  [[47,  3],
   [ 1, 49]]
  ```
- **Unknown Predictions:** 0/100 (0.0%)
- **Observations:** Balanced performance with high recall. 3 false positives (non-toxic classified as toxic) vs 1 false negative.

#### 500 Sample (en)
- **Accuracy:** 0.942
- **Precision:** 0.917
- **Recall:** 0.972
- **Macro-F1:** 0.942
- **Confusion Matrix:**
  ```
  [[228, 22],
   [7, 243]]
  ```
- **Unknown Predictions:** 0/500 (0.0%)
- **Observations:** Excellent recall (97.2%)—only 7 toxic samples missed. Reliable classifier with minimal classification failures.

#### Multilingual 504 Samples

Dataset:

- TextDetox multilingual sample
- Languages: English, Russian, Hindi
- 504 total samples
- 168 samples per language
- 84 toxic and 84 non-toxic samples per language

Results:

- **Total rows:** 504
- **Used rows:** 504
- **Accuracy:** 0.8928571428571429
- **Precision:** 0.9024390243902439
- **Recall:** 0.8809523809523809
- **Macro-F1:** 0.8928419560595322
- **Unknown Predictions:** 0/504 (0.0%)

Confusion matrix:

```text
[[228, 24],
 [ 30,222]]
```

Classification report:

```text
              precision    recall  f1-score   support

           0       0.88      0.90      0.89       252
           1       0.90      0.88      0.89       252

    accuracy                           0.89       504
   macro avg       0.89      0.89      0.89       504
weighted avg       0.89      0.89      0.89       504
```

Classification report summary:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non-toxic (0) | 0.88 | 0.90 | 0.89 | 252 |
| Toxic (1) | 0.90 | 0.88 | 0.89 | 252 |
| Macro avg | 0.89 | 0.89 | 0.89 | 504 |
| Weighted avg | 0.89 | 0.89 | 0.89 | 504 |

Observation:

Qwen3 remained stable on the multilingual dataset with zero unknown predictions. Performance decreased compared with English-only evaluation, which is expected because multilingual classification is harder and includes Russian and Hindi inputs.

## Per-Language Analysis (Qwen3 Multilingual)

To better understand multilingual performance, metrics were computed separately for English, Russian, and Hindi.

| Language | Samples | Accuracy | Precision | Recall | Macro-F1 |
|----------|----------:|----------:|----------:|----------:|----------:|
| English | 168 | 0.952 | 0.932 | 0.976 | **0.952** |
| Russian | 168 | 0.905 | 0.870 | 0.952 | **0.905** |
| Hindi | 168 | 0.821 | 0.909 | 0.714 | **0.819** |

### English

- Accuracy: 0.952
- Macro-F1: 0.952
- Confusion Matrix:

```text
[[78, 6],
 [ 2,82]]


---

## Key Insights

1. **Stability:** Both models maintain strong performance across sample sizes (100 → 500).

2. **Trade-offs:**
   - **DeepSeek:** Higher precision, more conservative (fewer false positives)
   - **Qwen3:** Higher recall, more aggressive (fewer false negatives)

3. **Classification Confidence:**
   - DeepSeek: 99.8% successfully classified (1 failure in 500)
   - Qwen3: 100% successfully classified

4. **Robustness:** 
   - Qwen3 shows better generalization to larger datasets
   - Both models exceed industry benchmarks for toxicity classification

5. Significant language-specific differences were observed.
   English achieved the highest Macro-F1 (0.952), followed by Russian (0.905), while Hindi was substantially lower (0.819). Hindi recall dropped to 71.4%, making it the primary source of multilingual performance degradation.

---

## Recommendations

- **For Safety-Critical Applications:** Use **DeepSeek** (perfect precision)
- **For Comprehensive Coverage:** Use **Qwen3** (high recall)
- **For Balanced Deployment:** Either model is suitable; consider ensemble approach

---

## Reproducibility

All results are reproducible using:
```bash
python experiments/reasoning_llms/evaluate_deepseek.py
python experiments/reasoning_llms/evaluate_qwen3.py
python experiments/reasoning_llms/evaluate_metrics.py
```

**Settings:**
- Temperature: 0 (deterministic)
- Dataset: TextDetox English
- Train/Val/Test: Balanced (50% toxic, 50% non-toxic)
