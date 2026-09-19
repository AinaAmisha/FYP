import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# 1. FILE PATHS
# ==========================================

CLEANED_FILE = Path(
    "data/processed/email_dataset_100k_cleaned.csv"
)

TEST_FILE = Path(
    "data/split/test_data.csv"
)

MODEL_FILE = Path(
    "models/random_forest.joblib"
)

RESULT_FOLDER = Path(
    "results"
)

PREDICTION_FILE = RESULT_FOLDER / "test_predictions.csv"

FALSE_DETECTION_FILE = RESULT_FOLDER / "false_detections.csv"

SUMMARY_FILE = RESULT_FOLDER / "false_detection_summary.csv"


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

print("\n======================================")
print("FALSE DETECTION ANALYSIS")
print("======================================")

print("\nTesting dataset shape:")
print(test_df.shape)


# ==========================================
# 4. LOAD RANDOM FOREST MODEL
# ==========================================

model = joblib.load(
    MODEL_FILE
)

print("\nRandom Forest model loaded:")
print(MODEL_FILE)


# ==========================================
# 5. PREPARE TEST FEATURES
# ==========================================

X_test = test_df[
    features
]

y_test = test_df[
    "label"
]


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test
)

results_df = test_df.copy()

results_df[
    "predicted_label"
] = predictions


# ==========================================
# 7. HUMAN-READABLE LABELS
# ==========================================

label_mapping = {
    0: "legitimate",
    1: "spam"
}

results_df[
    "actual_class"
] = results_df[
    "label"
].map(label_mapping)

results_df[
    "predicted_class"
] = results_df[
    "predicted_label"
].map(label_mapping)


# ==========================================
# 8. CLASSIFY RESULT TYPE
# ==========================================

def classify_result(row):

    actual = row["label"]
    predicted = row["predicted_label"]

    if actual == 0 and predicted == 0:
        return "True Negative"

    elif actual == 0 and predicted == 1:
        return "False Positive"

    elif actual == 1 and predicted == 0:
        return "False Negative"

    elif actual == 1 and predicted == 1:
        return "True Positive"

    else:
        return "Unknown"


results_df[
    "result_type"
] = results_df.apply(
    classify_result,
    axis=1
)

# ==========================================
# LOAD ORIGINAL EMAIL INFORMATION
# ==========================================

cleaned_df = pd.read_csv(
    CLEANED_FILE,
    low_memory=False
)

cleaned_df["source_row_id"] = cleaned_df.index


detail_columns = [
    "source_row_id",
    "subject",
    "from_address",
    "from_domain",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "num_urls",
    "attachment_types",
    "has_attachments",
    "dangerous_attachment",
    "body_plain",
    "class_label"
]


results_df = results_df.merge(
    cleaned_df[detail_columns],
    on="source_row_id",
    how="left"
)


# ==========================================
# 9. COUNT RESULT TYPES
# ==========================================

result_counts = (
    results_df[
        "result_type"
    ]
    .value_counts()
)

print("\n======================================")
print("CLASSIFICATION RESULTS")
print("======================================")

print(
    result_counts
)


# ==========================================
# 10. EXTRACT FALSE DETECTIONS
# ==========================================

false_detections = results_df[
    results_df[
        "result_type"
    ].isin(
        [
            "False Positive",
            "False Negative"
        ]
    )
].copy()


false_positive = results_df[
    results_df[
        "result_type"
    ]
    ==
    "False Positive"
]

false_negative = results_df[
    results_df[
        "result_type"
    ]
    ==
    "False Negative"
]


# ==========================================
# 11. DISPLAY FALSE DETECTION COUNTS
# ==========================================

print("\nFalse Positive:")
print(len(false_positive))

print("\nFalse Negative:")
print(len(false_negative))

print("\nTotal false detections:")
print(len(false_detections))


# ==========================================
# 12. FALSE DETECTION PERCENTAGE
# ==========================================

total_test = len(results_df)

total_false = len(
    false_detections
)

false_detection_rate = (
    total_false
    /
    total_test
)

print("\nFalse Detection Rate:")

print(
    round(
        false_detection_rate * 100,
        2
    ),
    "%"
)


# ==========================================
# 13. FALSE POSITIVE RATE
# ==========================================

actual_legitimate = len(
    results_df[
        results_df["label"] == 0
    ]
)

false_positive_rate = (
    len(false_positive)
    /
    actual_legitimate
)

print("\nFalse Positive Rate:")

print(
    round(
        false_positive_rate * 100,
        2
    ),
    "%"
)


# ==========================================
# 14. FALSE NEGATIVE RATE
# ==========================================

actual_spam = len(
    results_df[
        results_df["label"] == 1
    ]
)

false_negative_rate = (
    len(false_negative)
    /
    actual_spam
)

print("\nFalse Negative Rate:")

print(
    round(
        false_negative_rate * 100,
        2
    ),
    "%"
)


# ==========================================
# 15. SUMMARY TABLE
# ==========================================

summary_df = pd.DataFrame(
    {
        "Metric": [
            "Total Test Emails",
            "True Negative",
            "True Positive",
            "False Positive",
            "False Negative",
            "Total False Detection",
            "False Detection Rate",
            "False Positive Rate",
            "False Negative Rate"
        ],

        "Value": [
            total_test,

            len(
                results_df[
                    results_df[
                        "result_type"
                    ]
                    ==
                    "True Negative"
                ]
            ),

            len(
                results_df[
                    results_df[
                        "result_type"
                    ]
                    ==
                    "True Positive"
                ]
            ),

            len(
                false_positive
            ),

            len(
                false_negative
            ),

            total_false,

            false_detection_rate,

            false_positive_rate,

            false_negative_rate
        ]
    }
)


# ==========================================
# 16. CREATE RESULT FOLDER
# ==========================================

RESULT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 17. SAVE ALL PREDICTIONS
# ==========================================

results_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# ==========================================
# 18. SAVE FALSE DETECTIONS
# ==========================================

false_detections.to_csv(
    FALSE_DETECTION_FILE,
    index=False
)


# ==========================================
# 19. SAVE SUMMARY
# ==========================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ==========================================
# 20. DISPLAY EXAMPLES
# ==========================================

print("\n======================================")
print("FALSE POSITIVE EXAMPLES")
print("======================================")

print(
    false_positive.head()
)


print("\n======================================")
print("FALSE NEGATIVE EXAMPLES")
print("======================================")

print(
    false_negative.head()
)


# ==========================================
# 21. SAVED FILES
# ==========================================

print("\n======================================")
print("FILES SAVED")
print("======================================")

print("\nAll predictions:")
print(PREDICTION_FILE)

print("\nFalse detections:")
print(FALSE_DETECTION_FILE)

print("\nFalse detection summary:")
print(SUMMARY_FILE)


# ==========================================
# 22. FINAL MESSAGE
# ==========================================

print("\n======================================")
print("FALSE DETECTION ANALYSIS COMPLETED")
print("======================================")
