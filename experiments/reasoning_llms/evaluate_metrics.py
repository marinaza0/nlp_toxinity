import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# df = pd.read_csv("results/reasoning_llms/deepseek/textdetox_en_sample_500_predictions.csv")
df = pd.read_csv("results/reasoning_llms/qwen3/textdetox_multilingual_en_ru_hi_sample_504_predictions.csv")

unknown_count = (df["prediction"] == -1).sum()

print("Total rows:", len(df))
print("Unknown predictions:", unknown_count)
print("Unknown rate:", unknown_count / len(df))

df = df[df["prediction"] != -1]

y_true = df["toxic"]
y_pred = df["prediction"]

print("\nUsed rows:", len(df))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred))