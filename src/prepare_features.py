import pandas as pd
from pathlib import Path


# ==========================================
# 1. FILE PATHS
# ==========================================

INPUT_FILE = Path(
    "data/processed/email_dataset_100k_cleaned.csv"
)

OUTPUT_FILE = Path(
    "data/processed/email_dataset_100k_ml_ready.csv"
)


# ==========================================
# 2. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("\n======================================")
print("FEATURE SELECTION")
print("======================================")

print("\nOriginal cleaned dataset shape:")
print(df.shape)


# ==========================================
# 3. CHECK CLASS DISTRIBUTION
# ==========================================

print("\nOriginal class distribution:")

print(
    df["class_label"]
    .value_counts()
)


# ==========================================
# 4. REMOVE DUPLICATE EMAIL CONTENT
# ==========================================

print("\nDuplicate subject + body emails:")

duplicates = df.duplicated(
    subset=[
        "subject",
        "body_plain"
    ]
).sum()

print(duplicates)


df = (
    df
    .drop_duplicates(
        subset=[
            "subject",
            "body_plain"
        ]
    )
    .reset_index(drop=True)
)


print("\nDataset shape after removing repeated emails:")
print(df.shape)


# ==========================================
# 5. SELECT FEATURES
# ==========================================

selected_features = [

    # Authentication
    "spf_pass",
    "dkim_pass",
    "dmarc_pass",
    "auth_fail_count",

    # Email structure
    "subject_length",
    "body_length",

    # Time
    "sending_hour",

    # Content characteristics
    "has_html",
    "contains_tracking_token",

    # Attachment characteristic
    "attachment_count"
]


print("\nSelected features:")

for feature in selected_features:
    print("-", feature)


# ==========================================
# 6. CHECK FEATURES EXIST
# ==========================================

missing_features = [

    feature

    for feature in selected_features

    if feature not in df.columns
]


if missing_features:

    print(
        "\nERROR: These features are missing:"
    )

    print(missing_features)

    raise SystemExit


# ==========================================
# 7. CREATE ML DATAFRAME
# ==========================================

ml_df = df[
    selected_features
    +
    ["label"]
].copy()


# ==========================================
# 8. CONVERT FEATURES TO NUMERIC
# ==========================================

for column in selected_features:

    ml_df[column] = pd.to_numeric(
        ml_df[column],
        errors="coerce"
    )


ml_df["label"] = pd.to_numeric(
    ml_df["label"],
    errors="coerce"
)


# ==========================================
# 9. CHECK MISSING VALUES
# ==========================================

print("\nMissing values before final cleanup:")

print(
    ml_df
    .isnull()
    .sum()
)


# ==========================================
# 10. REMOVE ROWS WITH INVALID VALUES
# ==========================================

before_drop = len(ml_df)

ml_df = (
    ml_df
    .dropna()
    .reset_index(drop=True)
)

after_drop = len(ml_df)


print("\nRows removed because of missing/invalid values:")

print(
    before_drop - after_drop
)


# ==========================================
# 11. MAKE LABEL INTEGER
# ==========================================

ml_df["label"] = (
    ml_df["label"]
    .astype(int)
)


# ==========================================
# 12. VALIDATE LABEL
# ==========================================

valid_labels = [
    0,
    1
]

ml_df = ml_df[
    ml_df["label"]
    .isin(valid_labels)
].copy()


# ==========================================
# 13. CHECK CONSTANT FEATURES
# ==========================================

print("\nUnique values for each feature:")

for feature in selected_features:

    print(
        feature,
        "=",
        ml_df[feature].nunique()
    )


# ==========================================
# 14. FINAL CLASS DISTRIBUTION
# ==========================================

print("\nFinal class distribution:")

print(
    ml_df["label"]
    .value_counts()
)


print("\nFinal class percentage:")

print(
    ml_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nLabel meaning:")

print(
    "0 = legitimate"
)

print(
    "1 = spam"
)


# ==========================================
# 15. FINAL DATASET INFORMATION
# ==========================================

print("\nFinal ML dataset shape:")

print(
    ml_df.shape
)


print("\nFinal ML columns:")

print(
    ml_df.columns.tolist()
)


print("\nFirst 5 rows:")

print(
    ml_df.head()
)


# ==========================================
# 16. SAVE ML-READY DATASET
# ==========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n======================================")
print("FEATURE SELECTION COMPLETED")
print("======================================")

print("\nML-ready dataset saved to:")

print(
    OUTPUT_FILE
)
