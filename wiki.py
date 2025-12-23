import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": "AcademicResearchBot/1.0 (student project, non-commercial)"
}

URLS = {
    "Meta": "https://en.wikipedia.org/wiki/Meta_Platforms",
    "Starbucks": "https://en.wikipedia.org/wiki/Starbucks"
}

rows = []

for company, url in URLS.items():
    print(f"Scraping {company} Wikipedia")

    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        print("No content container found")
        continue

    paragraphs = content.find_all("p")

    print(f"Found {len(paragraphs)} paragraphs")

    for p in paragraphs:
        text = p.get_text(strip=True)

        if len(text) < 80:
            continue

        rows.append({
            "company": company,
            "source_type": "wikipedia",
            "text": text
        })

if not rows:
    print("No Wikipedia data collected.")
else:
    df = pd.DataFrame(rows)
    df.to_csv("raw_wiki_context.csv", index=False)
    print(f"Saved raw_wiki_context.csv ({len(df)} rows)")
