# prepare_data.py
import pandas as pd
from pathlib import Path

def load_file(p):
    p = Path(p)
    if p.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(p)
    return pd.read_csv(p, encoding='utf-8', engine='python')

# Paths (adjust if your files are named differently)
fake = load_file("data/Fake.csv")
true = load_file("data/True.csv")

# Ensure columns exist: try to standardize column names
def ensure_columns(df):
    # if text column called something else, try common alternatives
    if "text" not in df.columns:
        for c in ["content", "article", "body", "Body", "Text", "CONTENT"]:
            if c in df.columns:
                df.rename(columns={c: "text"}, inplace=True)
                break
    if "title" not in df.columns:
        for c in ["Title", "headline"]:
            if c in df.columns:
                df.rename(columns={c: "title"}, inplace=True)
                break
    # if still missing, add empty title
    if "title" not in df.columns:
        df["title"] = ""
    if "text" not in df.columns:
        df["text"] = df["title"].copy()  # fallback

ensure_columns(fake)
ensure_columns(true)

# add labels: 1=fake, 0=true
fake["label"] = 1
true["label"] = 0

# keep only title, text, label
fake = fake[["title", "text", "label"]]
true = true[["title", "text", "label"]]

# concat and shuffle
df = pd.concat([fake, true], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

Path("data").mkdir(parents=True, exist_ok=True)
df.to_csv("data/news_dataset.csv", index=False)
print("Saved merged dataset to data/news_dataset.csv — rows:", len(df))
