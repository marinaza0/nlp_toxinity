import ollama

MODEL = "deepseek-r1:7b"

def classify(text: str) -> str:
    prompt = f"""
You are a toxicity classification system.

Classify the following text as exactly one of:
- toxic
- non-toxic

Text:
"{text}"

Return only the final label: toxic or non-toxic.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    output = response["message"]["content"].strip().lower()

    if "non-toxic" in output:
        return "non-toxic"
    if "toxic" in output:
        return "toxic"

    return "unknown"


examples = [
    ("I hope you have a great day.", "non-toxic"),
    ("You are an idiot and nobody likes you.", "toxic"),
]

for text, gold in examples:
    pred = classify(text)
    print("TEXT:", text)
    print("GOLD:", gold)
    print("PRED:", pred)
    print()