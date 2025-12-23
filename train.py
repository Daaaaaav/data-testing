import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score
)

df = pd.read_csv("labeled_public_signals.csv")
print("Loaded rows:", len(df))

df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
df = df.dropna(subset=["publish_date"])

df["week"] = df["publish_date"].dt.to_period("W").apply(lambda r: r.start_time)

weekly = df.groupby(["company", "week"]).agg(
    signal_count=("risk_label", "count"),
    high_risk_count=("risk_label", lambda x: (x == 2).sum()),
    avg_risk=("risk_label", "mean"),
).reset_index()

weekly = weekly.sort_values(["company", "week"])

for lag in [1, 2]:
    weekly[f"signal_count_lag{lag}"] = weekly.groupby("company")["signal_count"].shift(lag)
    weekly[f"avg_risk_lag{lag}"] = weekly.groupby("company")["avg_risk"].shift(lag)

weekly["signal_trend"] = weekly["signal_count_lag1"] - weekly["signal_count_lag2"]
weekly["avg_risk_trend"] = weekly["avg_risk_lag1"] - weekly["avg_risk_lag2"]

weekly["risk_volatility"] = weekly[
    ["avg_risk_lag1", "avg_risk_lag2"]
].std(axis=1)

weekly["target"] = (weekly["high_risk_count"] > 0).astype(int)

weekly = weekly.dropna()
print("Weekly samples:", len(weekly))

features = [
    "signal_count_lag1",
    "signal_count_lag2",
    "avg_risk_lag1",
    "avg_risk_lag2",
    "signal_trend",
    "avg_risk_trend",
    "risk_volatility"
]

X = weekly[features]
y = weekly["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
split_idx = int(len(weekly) * 0.7)

X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

log_model = LogisticRegression(
    class_weight="balanced",
    random_state=42
)

log_model.fit(X_train, y_train)

log_prob = log_model.predict_proba(X_test)[:, 1]
threshold = 0.4
log_pred = (log_prob >= threshold).astype(int)

print("\n=== Logistic Regression (Improved) ===")
print(classification_report(y_test, log_pred))
print("ROC-AUC:", roc_auc_score(y_test, log_prob))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, log_pred))

log_importance = pd.Series(
    log_model.coef_[0],
    index=features
).sort_values(ascending=False)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob >= threshold).astype(int)

print("\n=== Random Forest ===")
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, rf_pred))

rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_comparison.png")
plt.show()

plt.figure(figsize=(6, 4))
log_importance.plot(kind="barh")
plt.title("Logistic Regression Feature Importance")
plt.tight_layout()
plt.savefig("logistic_feature_importance.png")
plt.show()

plt.figure(figsize=(6, 4))
rf_importance.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.show()

weekly["risk_probability_logistic"] = log_model.predict_proba(X_scaled)[:, 1]
weekly["risk_probability_rf"] = rf_model.predict_proba(X_scaled)[:, 1]

weekly.to_csv("weekly_risk_predictions.csv", index=False)

print("\n Saved weekly_risk_predictions.csv")
print("Saved plots:")
print("- roc_comparison.png")
print("- logistic_feature_importance.png")
print("- rf_feature_importance.png")
