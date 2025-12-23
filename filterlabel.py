import pandas as pd
import re

df = pd.read_csv("raw_public_signals.csv")
print("Initial rows:", len(df))
def extract_text(row):
    for col in ["text", "title", "snippet", "description"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return ""

df["raw_text"] = df.apply(extract_text, axis=1)

df["clean_text"] = (
    df["raw_text"]
    .str.lower()
    .str.replace(r"http\S+", "", regex=True)
    .str.replace(r"[^a-z\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
df = df[df["clean_text"].str.len() > 30]

print("After text cleaning:", len(df))

HIGH_RISK = [
    "layoff", "mass layoff", "retrenchment",
    "strike", "walkout", "union busting",
    "workplace grievance", "labor dispute",
    "termination", "job cuts", "downsizing",
    "class action", "lawsuit"
]

ELEVATED_RISK = [
    "cost cutting", "hiring freeze",
    "restructuring", "union",
    "worker protest", "employee complaint",
    "working conditions", "labor issue",
    "wage dispute"
]
high_pattern = re.compile("|".join(HIGH_RISK))
elevated_pattern = re.compile("|".join(ELEVATED_RISK))

def assign_risk_label(text):
    if high_pattern.search(text):
        return 2
    if elevated_pattern.search(text):
        return 1
    return 0

def extract_triggers(text):
    triggers = []
    for kw in HIGH_RISK + ELEVATED_RISK:
        if kw in text:
            triggers.append(kw)
    return ",".join(sorted(set(triggers)))

df["risk_label"] = df["clean_text"].apply(assign_risk_label)
df["trigger_terms"] = df["clean_text"].apply(extract_triggers)
df = df[df["risk_label"] > 0]
print("After risk filtering:", len(df))

final_columns = [
    "company",
    "source_type",
    "source_name",
    "publish_date",
    "risk_label",
    "trigger_terms",
    "clean_text"
]
final_columns = [c for c in final_columns if c in df.columns]
final_df = df[final_columns]
final_df.to_csv("labeled_public_signals.csv", index=False)
print("Saved labeled_public_signals.csv")
print(final_df["risk_label"].value_counts())