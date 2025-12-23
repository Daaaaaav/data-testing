# Overview
Test project which implements a predictive AI prototype model trained using Logistic Regression and Random Forest that detects early warning signals of retrenchment and workplace grievance risks using publicly available data sources, including:
= Online news articles
- Global event databases (GDELT)
- Wikipedia corporate context
The system applies weak supervision, time-series feature engineering, and interpretable machine learning models to predict elevated labor risk periods before known real-world incidents.

Two real-world case studies are used:
- Meta (Facebook) — mass layoffs and restructuring
- Starbucks — labor unionization and worker protests


# Data Collection
- Scraped public news and corporate context
- Signal Filtering & Weak Labeling
- Rule-based risk labeling using domain keywords
- Temporal Aggregation (weekly aggregation of public risk signals)
- Model Traaining of Logistic Regression (primary) vs Random Forest (comparison)

# Evaluation

