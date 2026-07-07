import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
fake["label"] = "FAKE"

real = pd.read_csv("dataset/True.csv")
real["label"] = "REAL"

df = pd.concat([fake, real])
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("dataset/news.csv", index=False)
print(f"Saved {len(df)} rows to dataset/news.csv")
print(df['label'].value_counts())