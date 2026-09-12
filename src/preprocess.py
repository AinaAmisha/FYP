import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================
# 1. FILE PATHS
# ==========================================

RAW_FILE = Path("data/raw/email_dataset_100k.csv")

OUTPUT_FILE = Path(
    "data/processed/email_dataset_100k_cleaned.csv"
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(RAW_FILE)

print("\n========== ORIGINAL DATASET ==========")

print("\nFirst 5 rows:")
print(df.head())

print("\nOriginal dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nOriginal label distribution:")
print(df["label"].value_counts(dropna=False))


# ==========================================
# 3. CLEAN COLUMN NAMES
# ==========================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())


# ==========================================
# 4. REMOVE DUPLICATES
# ==========================================

print("\nDuplicates before:")
print(df.duplicated().sum())

df = (
    df
    .drop_duplicates()
    .reset_index(drop=True)
)

print("Duplicates after:")
print(df.duplicated().sum())


# ==========================================
# 5. CLEAN TEXT COLUMNS
# ==========================================

text_columns = [
    "raw_text",
    "subject",
    "body_plain",
    "body_html",
    "from_address",
    "from_domain",
    "reply_to",
    "to_addresses",
    "cc_addresses",
    "message_id",
    "in_reply_to",
    "received_origin_ip",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "attachment_types",
    "user_agent",
    "list_unsubscribe",
    "language"
]

for column in text_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )


# ==========================================
# 6. HANDLE MISSING TEXT VALUES
# ==========================================

fill_empty_columns = [
    "subject",
    "body_plain",
    "body_html",
    "reply_to",
    "cc_addresses",
    "in_reply_to",
    "attachment_types",
    "list_unsubscribe"
]

for column in fill_empty_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
        )


# ==========================================
# 7. CLEAN EMAIL / DOMAIN VALUES
# ==========================================

df["from_address"] = (
    df["from_address"]
    .str.lower()
)

df["from_domain"] = (
    df["from_domain"]
    .str.lower()
)


# ==========================================
# 8. VALIDATE SENDER EMAIL
# ==========================================

email_pattern = (
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
)

df["sender_email_valid"] = (
    df["from_address"]
    .str.match(
        email_pattern,
        na=False
    )
    .astype(int)
)

print("\nInvalid sender emails:")

print(
    df.loc[
        df["sender_email_valid"] == 0,
        [
            "from_address",
            "from_domain"
        ]
    ]
)


# ==========================================
# 9. EXTRACT DOMAIN FROM EMAIL
# ==========================================

df["sender_domain_extracted"] = (
    df["from_address"]
    .str.rsplit("@", n=1)
    .str[-1]
)


# ==========================================
# 10. CHECK DOMAIN CONSISTENCY
# ==========================================

df["sender_domain_match"] = (
    df["sender_domain_extracted"]
    ==
    df["from_domain"]
).astype(int)


# ==========================================
# 11. CLEAN DATE
# ==========================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
    utc=True
)

print("\nInvalid dates:")
print(df["date"].isnull().sum())


# ==========================================
# 12. NUMERIC COLUMNS
# ==========================================

numeric_columns = [
    "hour_of_day",
    "num_received_headers",
    "num_urls",
    "num_emails_in_body",
    "num_phone_numbers"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ==========================================
# 13. REMOVE NEGATIVE NUMERIC VALUES
# ==========================================

for column in numeric_columns:

    if column in df.columns:

        df.loc[
            df[column] < 0,
            column
        ] = np.nan


# ==========================================
# 14. FILL MISSING NUMERIC VALUES
# ==========================================

for column in numeric_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna(0)
        )


# ==========================================
# 15. CLEAN BOOLEAN COLUMNS
# ==========================================

boolean_columns = [
    "has_attachments",
    "has_html",
    "contains_tracking_token"
]

boolean_mapping = {
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
    "yes": 1,
    "no": 0
}

for column in boolean_columns:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(boolean_mapping)
        .fillna(0)
        .astype(int)
    )


# ==========================================
# 16. CLEAN AUTHENTICATION VALUES
# ==========================================

authentication_columns = [
    "spf_result",
    "dkim_result",
    "dmarc_result"
]

for column in authentication_columns:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("unknown")
    )


print("\nSPF values:")
print(df["spf_result"].value_counts())

print("\nDKIM values:")
print(df["dkim_result"].value_counts())

print("\nDMARC values:")
print(df["dmarc_result"].value_counts())


# ==========================================
# 17. CREATE AUTHENTICATION FEATURES
# ==========================================

df["spf_pass"] = (
    df["spf_result"] == "pass"
).astype(int)

df["dkim_pass"] = (
    df["dkim_result"] == "pass"
).astype(int)

df["dmarc_pass"] = (
    df["dmarc_result"] == "pass"
).astype(int)


# ==========================================
# 18. AUTHENTICATION FAILURE COUNT
# ==========================================

df["auth_fail_count"] = (
    (df["spf_result"] == "fail").astype(int)
    +
    (df["dkim_result"] == "fail").astype(int)
    +
    (df["dmarc_result"] == "fail").astype(int)
)


# ==========================================
# 19. AUTHENTICATION NON-PASS COUNT
# ==========================================

df["auth_nonpass_count"] = (
    (df["spf_result"] != "pass").astype(int)
    +
    (df["dkim_result"] != "pass").astype(int)
    +
    (df["dmarc_result"] != "pass").astype(int)
)


# ==========================================
# 20. SUBJECT LENGTH
# ==========================================

df["subject_length"] = (
    df["subject"]
    .astype(str)
    .str.len()
)


# ==========================================
# 21. BODY LENGTH
# ==========================================

df["body_length"] = (
    df["body_plain"]
    .astype(str)
    .str.len()
)


# ==========================================
# 22. SENDING HOUR
# ==========================================

df["sending_hour"] = (
    df["date"]
    .dt.hour
)


# ==========================================
# 23. HAS URL FEATURE
# ==========================================

df["has_url"] = (
    df["num_urls"] > 0
).astype(int)


# ==========================================
# 24. ATTACHMENT COUNT
# ==========================================

df["attachment_count"] = np.where(
    df["attachment_types"].fillna("") == "",
    0,
    df["attachment_types"]
    .str.count(";")
    .fillna(0)
    + 1
)

df["attachment_count"] = (
    df["attachment_count"]
    .astype(int)
)


# ==========================================
# 25. SUSPICIOUS ATTACHMENT FEATURE
# ==========================================

dangerous_extension_pattern = (
    r"\.(?:exe|js|vbs|scr|bat|cmd|ps1|"
    r"docm|xlsm|pptm)(?:;|$)"
)

df["dangerous_attachment"] = (
    df["attachment_types"]
    .str.contains(
        dangerous_extension_pattern,
        case=False,
        regex=True,
        na=False
    )
    .astype(int)
)


# ==========================================
# 26. CLEAN LABEL
# ==========================================

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

valid_labels = [0, 1]

df = df[
    df["label"].isin(valid_labels)
].copy()

df["label"] = (
    df["label"]
    .astype(int)
)


# ==========================================
# 27. HUMAN-READABLE LABEL
# ==========================================

# IMPORTANT:
# This dataset is binary spam classification.
# Do NOT call label 1 "malicious" yet.

label_mapping = {
    0: "legitimate",
    1: "spam"
}

df["class_label"] = (
    df["label"]
    .map(label_mapping)
)


# ==========================================
# 28. REMOVE DATA LEAKAGE
# ==========================================

# x_spam_score perfectly reveals the label
# in this dataset.
#
# Therefore it MUST NOT be used as an ML
# input feature.

if "x_spam_score" in df.columns:

    df = df.drop(
        columns=["x_spam_score"]
    )

    print(
        "\nx_spam_score removed "
        "to prevent data leakage."
    )


# ==========================================
# 29. ADD DATASET SOURCE
# ==========================================

df["source_dataset"] = (
    "email_dataset_100k"
)


# ==========================================
# 30. FINAL CHECK
# ==========================================

print(
    "\n========== CLEANED DATASET =========="
)

print("\nCleaned shape:")
print(df.shape)

print("\nRemaining duplicates:")
print(df.duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["class_label"].value_counts())

print("\nAuthentication failure counts:")
print(
    df["auth_fail_count"]
    .value_counts()
    .sort_index()
)

print("\nURL feature:")
print(df["has_url"].value_counts())

print("\nAttachment feature:")
print(
    df["has_attachments"]
    .value_counts()
)

print("\nDangerous attachments:")
print(
    df["dangerous_attachment"]
    .value_counts()
)

print("\nFirst 5 cleaned rows:")

print(
    df[
        [
            "subject",
            "from_address",
            "from_domain",
            "spf_result",
            "dkim_result",
            "dmarc_result",
            "num_urls",
            "attachment_count",
            "auth_fail_count",
            "class_label"
        ]
    ].head()
)


# ==========================================
# 31. SAVE CLEANED DATASET
# ==========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n======================================")
print("REAL DATA PREPROCESSING COMPLETED")
print("======================================")

print("\nSaved to:")
print(OUTPUT_FILE)