import pandas as pd
from datasets import load_dataset
from pathlib import Path

DATASET = "textdetox/multilingual_toxicity_dataset"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LANGS = {
    "en": "textdetox_en",
    "ru": "textdetox_ru",
    "hi": "textdetox_hi",
}

N_PER_LABEL = 84
RANDOM_STATE = 42


def load_and_save_raw(lang: str, name: str) -> pd.DataFrame:
    raw_path = RAW_DIR / lang / f"{name}.csv" if lang != "en" else RAW_DIR / f"{name}.csv"

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        ds = load_dataset(DATASET, split=lang)
        df = ds.to_pandas()

        if lang != "en":
            (RAW_DIR / lang).mkdir(parents=True, exist_ok=True)

        df.to_csv(raw_path, index=False)

    df = df[["text", "toxic"]].copy()
    df["language"] = lang
    return df


samples = []

for lang, name in LANGS.items():
    df = load_and_save_raw(lang, name)

    toxic = df[df["toxic"] == 1].sample(n=N_PER_LABEL, random_state=RANDOM_STATE)
    non_toxic = df[df["toxic"] == 0].sample(n=N_PER_LABEL, random_state=RANDOM_STATE)

    sample = pd.concat([toxic, non_toxic]).sample(frac=1, random_state=RANDOM_STATE)

    sample_path = PROCESSED_DIR / f"{name}_sample_168.csv"
    sample.to_csv(sample_path, index=False)

    samples.append(sample)

    print(f"{lang}:")
    print(sample["toxic"].value_counts())
    print()

merged = pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE)
merged.to_csv(PROCESSED_DIR / "textdetox_multilingual_en_ru_hi_sample_504.csv", index=False)

print("All samples created successfully!")
print("\nFinal multilingual sample:")
print(merged["toxic"].value_counts())
print(f"Total samples: {len(merged)}")
print(f"Languages: {merged['language'].value_counts().to_dict()}")