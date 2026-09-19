import pandas as pd
from pathlib import Path


# ==========================================
# 1. FILE PATHS
# ==========================================

PREDICTION_FILE = Path(
    "results/test_predictions.csv"
)

OUTPUT_FILE = Path(
    "results/false_detection_pattern_summary.csv"
)


# ==========================================
# 2. LOAD PREDICTIONS
# ==========================================

df = pd.read_csv(
    PREDICTION_FILE,
    low_memory=False
)

print("\n======================================")
print("FALSE DETECTION PATTERN ANALYSIS")
print("======================================")

print("\nDataset shape:")
print(df.shape)


# ==========================================
# 3. FEATURES TO ANALYSE
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
# 4. RESULT GROUP COUNTS
# ==========================================

print("\nClassification result counts:")

print(
    df["result_type"]
    .value_counts()
)


# ==========================================
# 5. GROUP FEATURE SUMMARY
# ==========================================

summary = (
    df.groupby(
        "result_type"
    )[features]
    .mean()
    .round(3)
)


print("\n======================================")
print("AVERAGE FEATURE VALUES")
print("======================================")

print(summary)


# ==========================================
# 6. ADD NUMBER OF EMAILS
# ==========================================

counts = (
    df["result_type"]
    .value_counts()
    .rename("email_count")
)

summary = summary.join(
    counts
)


# ==========================================
# 7. FALSE POSITIVE ANALYSIS
# ==========================================

false_positive = df[
    df["result_type"]
    ==
    "False Positive"
]

print("\n======================================")
print("FALSE POSITIVE PATTERNS")
print("======================================")

print("\nNumber of False Positives:")
print(len(false_positive))

print("\nSPF pass percentage:")

print(
    round(
        false_positive["spf_pass"].mean()
        * 100,
        2
    ),
    "%"
)

print("\nDKIM pass percentage:")

print(
    round(
        false_positive["dkim_pass"].mean()
        * 100,
        2
    ),
    "%"
)

print("\nDMARC pass percentage:")

print(
    round(
        false_positive["dmarc_pass"].mean()
        * 100,
        2
    ),
    "%"
)

print("\nAverage authentication failures:")

print(
    round(
        false_positive[
            "auth_fail_count"
        ].mean(),
        2
    )
)

print("\nAverage subject length:")

print(
    round(
        false_positive[
            "subject_length"
        ].mean(),
        2
    )
)

print("\nAverage body length:")

print(
    round(
        false_positive[
            "body_length"
        ].mean(),
        2
    )
)


# ==========================================
# 8. FALSE NEGATIVE ANALYSIS
# ==========================================

false_negative = df[
    df["result_type"]
    ==
    "False Negative"
]

print("\n======================================")
print("FALSE NEGATIVE PATTERNS")
print("======================================")

print("\nNumber of False Negatives:")
print(len(false_negative))

print("\nSPF pass percentage:")

print(
    round(
        false_negative["spf_pass"].mean()
        * 100,
        2
    ),
    "%"
)

print("\nDKIM pass percentage:")

print(
    round(
        false_negative["dkim_pass"].mean()
        * 100,
        2
    ),
    "%"
)

print("\nDMARC pass percentage:")

print(
    round(
        false_negative["dmarc_pass"].mean()
        * 100,
        2
    ),
    "%"
)

print("\nAverage authentication failures:")

print(
    round(
        false_negative[
            "auth_fail_count"
        ].mean(),
        2
    )
)

print("\nAverage subject length:")

print(
    round(
        false_negative[
            "subject_length"
        ].mean(),
        2
    )
)

print("\nAverage body length:")

print(
    round(
        false_negative[
            "body_length"
        ].mean(),
        2
    )
)


# ==========================================
# 9. TOP FALSE-POSITIVE DOMAINS
# ==========================================

if "from_domain" in df.columns:

    print("\n======================================")
    print("TOP FALSE POSITIVE DOMAINS")
    print("======================================")

    print(
        false_positive[
            "from_domain"
        ]
        .value_counts()
        .head(10)
    )


# ==========================================
# 10. TOP FALSE-NEGATIVE DOMAINS
# ==========================================

if "from_domain" in df.columns:

    print("\n======================================")
    print("TOP FALSE NEGATIVE DOMAINS")
    print("======================================")

    print(
        false_negative[
            "from_domain"
        ]
        .value_counts()
        .head(10)
    )


# ==========================================
# 11. SAVE SUMMARY
# ==========================================

summary.to_csv(
    OUTPUT_FILE
)


print("\n======================================")
print("PATTERN ANALYSIS COMPLETED")
print("======================================")

print("\nSummary saved to:")
print(OUTPUT_FILE)
