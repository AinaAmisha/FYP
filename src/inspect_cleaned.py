import pandas as pd
from pathlib import Path


# ==========================================
# 1. FILE PATH
# ==========================================

FILE = Path(
    "data/processed/email_dataset_100k_cleaned.csv"
)


# ==========================================
# 2. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv(
    FILE,
    low_memory=False
)


print("\n======================================")
print("CLEANED DATASET INSPECTION")
print("======================================")


# ==========================================
# 3. DATASET SIZE
# ==========================================

print("\nDataset shape:")
print(df.shape)


# ==========================================
# 4. COLUMNS
# ==========================================

print("\nNumber of columns:")
print(len(df.columns))

print("\nColumn names:")

for column in df.columns:
    print("-", column)


# ==========================================
# 5. DUPLICATES
# ==========================================

print("\nExact duplicate rows:")
print(df.duplicated().sum())


# ==========================================
# 6. MISSING VALUES
# ==========================================

print("\nMissing values:")

missing = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing[
        missing > 0
    ]
)


# ==========================================
# 7. CLASS DISTRIBUTION
# ==========================================

print("\nClass distribution:")

print(
    df["class_label"]
    .value_counts()
)


print("\nClass percentage:")

print(
    df["class_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==========================================
# 8. SPF DISTRIBUTION
# ==========================================

print("\nSPF by class:")

print(
    pd.crosstab(
        df["spf_result"],
        df["class_label"]
    )
)


# ==========================================
# 9. DKIM DISTRIBUTION
# ==========================================

print("\nDKIM by class:")

print(
    pd.crosstab(
        df["dkim_result"],
        df["class_label"]
    )
)


# ==========================================
# 10. DMARC DISTRIBUTION
# ==========================================

print("\nDMARC by class:")

print(
    pd.crosstab(
        df["dmarc_result"],
        df["class_label"]
    )
)


# ==========================================
# 11. AUTH FAILURE COUNT
# ==========================================

print("\nAuthentication failures by class:")

print(
    pd.crosstab(
        df["auth_fail_count"],
        df["class_label"]
    )
)


# ==========================================
# 12. URL DISTRIBUTION
# ==========================================

print("\nHas URL by class:")

print(
    pd.crosstab(
        df["has_url"],
        df["class_label"]
    )
)


print("\nNumber of URLs by class:")

print(
    pd.crosstab(
        df["num_urls"],
        df["class_label"]
    )
)


# ==========================================
# 13. ATTACHMENT DISTRIBUTION
# ==========================================

print("\nAttachments by class:")

print(
    pd.crosstab(
        df["has_attachments"],
        df["class_label"]
    )
)


print("\nDangerous attachment by class:")

print(
    pd.crosstab(
        df["dangerous_attachment"],
        df["class_label"]
    )
)


# ==========================================
# 14. HTML DISTRIBUTION
# ==========================================

print("\nHTML emails by class:")

print(
    pd.crosstab(
        df["has_html"],
        df["class_label"]
    )
)


# ==========================================
# 15. TRACKING TOKEN
# ==========================================

print("\nTracking token by class:")

print(
    pd.crosstab(
        df["contains_tracking_token"],
        df["class_label"]
    )
)


# ==========================================
# 16. CONTENT DUPLICATES
# ==========================================

print("\nDuplicate raw email content:")

print(
    df.duplicated(
        subset=["raw_text"]
    ).sum()
)


print("\nDuplicate subject + body:")

print(
    df.duplicated(
        subset=[
            "subject",
            "body_plain"
        ]
    ).sum()
)


# ==========================================
# 17. TEXT LENGTH STATISTICS
# ==========================================

print("\nSubject length statistics:")

print(
    df.groupby("class_label")
    ["subject_length"]
    .describe()
)


print("\nBody length statistics:")

print(
    df.groupby("class_label")
    ["body_length"]
    .describe()
)


# ==========================================
# 18. NUMERICAL FEATURE SUMMARY
# ==========================================

features = [
    "subject_length",
    "body_length",
    "num_urls",
    "attachment_count",
    "auth_fail_count",
    "auth_nonpass_count",
    "sending_hour"
]


print("\nNumerical feature summary:")

print(
    df[
        features
    ].describe()
)


# ==========================================
# 19. FINAL MESSAGE
# ==========================================

print("\n======================================")
print("DATASET INSPECTION COMPLETED")
print("======================================")
