import pandas as pd
import ollama
from pathlib import Path

MODEL = "qwen3:8b"

INPUT_PATH = "data/processed/textdetox_multilingual_en_ru_hi_sample_504.csv"

OUTPUT_PATH = Path(
    "results/reasoning_llms/qwen3/textdetox_multilingual_en_ru_hi_sample_504_predictions.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

if OUTPUT_PATH.exists():
    existing = pd.read_csv(OUTPUT_PATH)
    start_idx = len(existing)
    predictions = existing["prediction"].tolist()
    print(f"Resuming from row {start_idx + 1}")
else:
    start_idx = 0
    predictions = []

for idx in range(start_idx, len(df)):
    row = df.iloc[idx]
    text = row["text"]

    prompt = f"""
Classify the text as exactly one of:
toxic
non-toxic

Text:
{text}

Return only the final label.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False
    )

    output = response["message"]["content"].strip().lower()

    if output == "non-toxic":
        pred = 0
    elif output == "toxic":
        pred = 1
    elif "non-toxic" in output:
        pred = 0
    elif "toxic" in output:
        pred = 1
    else:
        pred = -1

    predictions.append(pred)

    partial = df.iloc[:idx + 1].copy()
    partial["prediction"] = predictions
    partial.to_csv(OUTPUT_PATH, index=False)

    print(f"{idx + 1}/{len(df)} | pred={pred}")

print("Done")