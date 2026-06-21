import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("results/reasoning_llms/deepseek/textdetox_en_sample_500_predictions.csv")

df = df[df["prediction"] != -1]

y_true = df["toxic"]
y_pred = df["prediction"]

print("Used rows:", len(df))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))