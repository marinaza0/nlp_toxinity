import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def calculate_metrics(y_true, y_pred, languages=None, model_name="model"):
    print(f"\n{'=' * 40}")
    print(f"Metrics for: {model_name}")
    print(f"{'=' * 40}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))

    if languages is not None:
        print("\nPer-language Macro-F1:")

        temp_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'lang': languages})


        for lang in temp_df['lang'].unique():
            subset = temp_df[temp_df['lang'] == lang]
            lang_f1 = f1_score(subset['y_true'], subset['y_pred'], average="macro")
            print(f"  - {lang}: {lang_f1:.4f} (based on {len(subset)} samples)")


    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))



if __name__ == "__main__":
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
    calculate_metrics(y_true, y_pred, "Qwen3")
