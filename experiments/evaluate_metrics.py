import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("data/processed/textdetox_en_sample_100_predictions.csv")

y_true = df["toxic"]
y_pred = df["prediction"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
print()
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
print()
print(classification_report(y_true, y_pred))