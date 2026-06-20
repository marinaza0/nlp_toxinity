from datasets import load_dataset

DATASET = "textdetox/multilingual_toxicity_dataset"

ds = load_dataset(DATASET, split="en")
df = ds.to_pandas()

print(df.head())
print(df.columns)
print(df.shape)

df.to_csv("data/raw/textdetox_en.csv", index=False)