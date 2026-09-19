import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# 1. FILE PATHS
# ==========================================

MODEL_FILE = Path(
    "models/random_forest.joblib"
)

OUTPUT_FILE = Path(
    "results/feature_importance.csv"
)


# ==========================================
# 2. FEATURES
# ==========================================

features = [
    "spf_pass",
    "dkim_pass",
    "dmarc_pass",
    "auth_fail_count",
    "subject_length",
    "body_length",
    "sending_hour",
    "has_html",
    "contains_tracking_token",
    "attachment_count"
]


# ==========================================
# 3. LOAD MODEL
# ==========================================

model = joblib.load(
    MODEL_FILE
)

print("\n======================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("======================================")


# ==========================================
# 4. GET FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_


# ==========================================
# 5. CREATE TABLE
# ==========================================

importance_df = pd.DataFrame(
    {
        "Feature": features,
        "Importance": importance
    }
)


importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


importance_df[
    "Importance Percentage"
] = (
    importance_df["Importance"]
    * 100
)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print()

print(
    importance_df.to_string(
        index=False,
        formatters={
            "Importance": lambda x: f"{x:.4f}",
            "Importance Percentage":
                lambda x: f"{x:.2f}%"
        }
    )
)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

importance_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n======================================")
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("======================================")

print("\nSaved to:")
print(OUTPUT_FILE)