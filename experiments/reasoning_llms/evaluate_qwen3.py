import pandas as pd
import ollama

MODEL = "qwen3:8b"

# convention: textdetox_en_sample_100.csv, textdetox_en_sample_500.csv ...
df = pd.read_csv("data/processed/textdetox_en_sample_100.csv") 

predictions = []

for idx, row in df.iterrows():
    text = row["text"]

    prompt = f"""
You are a toxicity classifier.

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
        options={"temperature": 0}
    )

    output = response["message"]["content"].lower()

    if "non-toxic" in output:
        pred = 0
    elif "toxic" in output:
        pred = 1
    else:
        pred = -1

    predictions.append(pred)

    print(f"{idx+1}/{len(df)}")

df["prediction"] = predictions

# convention: textdetox_en_sample_100_predictions.csv, textdetox_en_sample_500_predictions.csv...
df.to_csv(
    "results/reasoning_llms/qwen3/textdetox_en_sample_100_predictions.csv",
    index=False
) 

print("Done")