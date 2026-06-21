import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

FILE = "results/reasoning_llms/qwen3/textdetox_multilingual_en_ru_hi_sample_504_predictions.csv"

df = pd.read_csv(FILE)

# remove unknown predictions if any
df = df[df["prediction"] != -1]

print("=" * 60)
print("OVERALL")
print("=" * 60)

print("Accuracy:", accuracy_score(df["toxic"], df["prediction"]))
print("Precision:", precision_score(df["toxic"], df["prediction"]))
print("Recall:", recall_score(df["toxic"], df["prediction"]))
print("Macro-F1:", f1_score(df["toxic"], df["prediction"], average="macro"))

print("\n")

for language in sorted(df["language"].unique()):
    lang_df = df[df["language"] == language]

    y_true = lang_df["toxic"]
    y_pred = lang_df["prediction"]

    print("=" * 60)
    print(f"LANGUAGE: {language}")
    print("=" * 60)

    print("Samples:", len(lang_df))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print()