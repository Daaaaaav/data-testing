import feedparser
import pandas as pd
from datetime import datetime
from tqdm import tqdm

QUERIES = {
    "Meta": [
        "Meta layoffs",
        "Meta hiring freeze",
        "Meta cost cutting"
    ],
    "Starbucks": [
        "Starbucks worker protest",
        "Starbucks union",
        "Starbucks employee complaints"
    ]
}

rows = []

for company, queries in QUERIES.items():
    for query in tqdm(queries, desc=f"Fetching {company}"):
        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
        feed = feedparser.parse(url)

        for entry in feed.entries:
            rows.append({
                "doc_id": entry.id,
                "company": company,
                "source_type": "news",
                "source_name": entry.source.title if "source" in entry else "Google News",
                "publish_date": entry.published,
                "text": entry.title,
                "engagement": 0,
                "url": entry.link
            })

df = pd.DataFrame(rows)
df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
df.to_csv("raw_google_news.csv", index=False)

print("Saved raw_google_news.csv")
