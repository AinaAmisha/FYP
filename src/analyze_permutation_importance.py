import pandas as pd
import joblib
from pathlib import Path

from sklearn.inspection import permutation_importance


# ==========================================
# 1. FILE PATHS
# ==========================================

TEST_FILE = Path(
    "data/split/test_data.csv"
)

MODEL_FILE = Path(
    "models/random_forest.joblib"
)

OUTPUT_FILE = Path(
    "results/permutation_importance.csv"
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
# 3. LOAD TEST DATA
# ==========================================

test_df = pd.read_csv(
    TEST_FILE
)

X_test = test_df[
    features
]

y_test = test_df[
    "label"
]


# ==========================================
# 4. LOAD MODEL
# ==========================================

model = joblib.load(
    MODEL_FILE
)

print("\n======================================")
print("PERMUTATION FEATURE IMPORTANCE")
print("======================================")


# ==========================================
# 5. CALCULATE PERMUTATION IMPORTANCE
# ==========================================

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="f1",
    n_jobs=-1
)


# ==========================================
# 6. CREATE RESULT TABLE
# ==========================================

importance_df = pd.DataFrame(
    {
        "Feature": features,
        "Importance Mean":
            result.importances_mean,
        "Importance Standard Deviation":
            result.importances_std
    }
)


importance_df = (
    importance_df
    .sort_values(
        by="Importance Mean",
        ascending=False
    )
    .reset_index(drop=True)
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print()

print(
    importance_df.to_string(
        index=False,
        formatters={
            "Importance Mean":
                lambda x: f"{x:.4f}",
            "Importance Standard Deviation":
                lambda x: f"{x:.4f}"
        }
    )
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

importance_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n======================================")
print("PERMUTATION IMPORTANCE COMPLETED")
print("======================================")

print("\nSaved to:")
print(OUTPUT_FILE)