import pandas as pd
import requests
import time
from io import StringIO
from datetime import datetime

HEADERS = {
    "User-Agent": "AcademicResearchBot/1.0 (contact: student-project)"
}

QUERIES = {
    "Meta": [
        "Meta layoffs",
        "Meta cost cutting"
    ],
    "Starbucks": [
        "Starbucks worker protest",
        "Starbucks union"
    ]
}

OUTPUT_FILE = "raw_gdelt_news.csv"

def fetch_gdelt_safe(keyword, retries=3, sleep_time=10):
    """
    Safely fetch GDELT DOC API data with retries and throttling.
    Returns a DataFrame (possibly empty).
    """
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={keyword}"
        "&mode=ArtList"
        "&format=CSV"
        "&maxrecords=100"
        "&timespan=3m"
    )

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)

            if response.status_code == 429:
                print(f"429 Too Many Requests, sleeping {sleep_time}s")
                time.sleep(sleep_time)
                continue

            response.raise_for_status()

            if not response.text.strip():
                print("Empty response body")
                return pd.DataFrame()

            return pd.read_csv(StringIO(response.text))

        except Exception as e:
            print(f"Attempt {attempt} failed:", e)
            time.sleep(sleep_time)

    print("Failed after maximum retries")
    return pd.DataFrame()

def normalize_schema(df):
    """
    Ensure output DataFrame always conforms to required schema.
    Missing columns are created with safe defaults.
    """
    rename_map = {
        "SOURCECOMMONNAME": "source_name",
        "URL": "url",
        "TONE": "sentiment_score",
        "DATE": "publish_date"
    }

    df = df.rename(columns=rename_map)

    required_columns = {
        "company": None,
        "source_type": "news",
        "source_name": "GDELT",
        "publish_date": None,
        "sentiment_score": 0.0,
        "url": None,
        "engagement": 0
    }

    for col, default in required_columns.items():
        if col not in df.columns:
            df[col] = default

    # Parse publish_date safely
    df["publish_date"] = pd.to_datetime(
        df["publish_date"], errors="coerce"
    )

    return df[list(required_columns.keys())]

def main():
    all_frames = []

    for company, query_list in QUERIES.items():
        for query in query_list:
            print(f"Fetching GDELT for {company}: {query}")

            df = fetch_gdelt_safe(query)

            if df.empty:
                print("No data returned")
            else:
                df["company"] = company
                df = normalize_schema(df)
                all_frames.append(df)

            # Throttle between queries
            time.sleep(15)

    if not all_frames:
        print("No data collected from GDELT. Exiting.")
        return

    final_df = pd.concat(all_frames, ignore_index=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nGDELT scraping completed")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows collected: {len(final_df)}")
    print(f"Columns: {final_df.columns.tolist()}")

if __name__ == "__main__":
    main()
