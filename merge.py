import pandas as pd

files = [
    "raw_google_news.csv",
    "raw_gdelt_news.csv",
    "raw_wiki_context.csv"
]

dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)

merged.to_csv("raw_public_signals.csv", index=False)
print("Saved raw_public_signals.csv")
