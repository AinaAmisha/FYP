import pandas as pd
import joblib
from pathlib import Path
import argparse


# ==========================================
# 1. FILE PATHS
# ==========================================

MODEL_FILE = Path(
    "models/random_forest.joblib"
)

DEFAULT_INPUT_FILE = Path(
    "data/split/test_data.csv"
)

OUTPUT_FILE = Path(
    "results/predicted_email_logs.csv"
)


# ==========================================
# 2. MODEL FEATURES
# ==========================================

FEATURES = [
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

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    return model


# ==========================================
# 4. BOOLEAN CONVERSION
# ==========================================

def convert_boolean(series):

    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    }

    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0)
        .astype(int)
    )


# ==========================================
# 5. CREATE FEATURES FROM RAW EMAIL LOG
# ==========================================

def prepare_raw_features(df):

    print(
        "\nRaw email log detected."
    )

    print(
        "Creating ML features..."
    )


    # --------------------------------------
    # Authentication
    # --------------------------------------

    authentication_columns = [
        "spf_result",
        "dkim_result",
        "dmarc_result"
    ]

    for column in authentication_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )


    df["spf_pass"] = (
        df["spf_result"] == "pass"
    ).astype(int)

    df["dkim_pass"] = (
        df["dkim_result"] == "pass"
    ).astype(int)

    df["dmarc_pass"] = (
        df["dmarc_result"] == "pass"
    ).astype(int)


    df["auth_fail_count"] = (
        (df["spf_result"] == "fail").astype(int)
        +
        (df["dkim_result"] == "fail").astype(int)
        +
        (df["dmarc_result"] == "fail").astype(int)
    )


    # --------------------------------------
    # Subject
    # --------------------------------------

    if "subject" not in df.columns:

        raise ValueError(
            "Missing required column: subject"
        )

    df["subject"] = (
        df["subject"]
        .fillna("")
        .astype(str)
    )

    df["subject_length"] = (
        df["subject"]
        .str.len()
    )


    # --------------------------------------
    # Body
    # --------------------------------------

    if "body_plain" not in df.columns:

        raise ValueError(
            "Missing required column: body_plain"
        )

    df["body_plain"] = (
        df["body_plain"]
        .fillna("")
        .astype(str)
    )

    df["body_length"] = (
        df["body_plain"]
        .str.len()
    )


    # --------------------------------------
    # Sending hour
    # --------------------------------------

    if "date" not in df.columns:

        raise ValueError(
            "Missing required column: date"
        )

    parsed_date = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True
    )

    df["sending_hour"] = (
        parsed_date
        .dt.hour
        .fillna(0)
        .astype(int)
    )


    # --------------------------------------
    # HTML
    # --------------------------------------

    if "has_html" not in df.columns:

        raise ValueError(
            "Missing required column: has_html"
        )

    df["has_html"] = convert_boolean(
        df["has_html"]
    )


    # --------------------------------------
    # Tracking token
    # --------------------------------------

    if "contains_tracking_token" not in df.columns:

        raise ValueError(
            "Missing required column: "
            "contains_tracking_token"
        )

    df[
        "contains_tracking_token"
    ] = convert_boolean(
        df["contains_tracking_token"]
    )


    # --------------------------------------
    # Attachment count
    # --------------------------------------

    if "attachment_types" not in df.columns:

        raise ValueError(
            "Missing required column: "
            "attachment_types"
        )

    attachment_text = (
        df["attachment_types"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["attachment_count"] = (
        attachment_text
        .apply(
            lambda value:
            0
            if value == ""
            else value.count(";") + 1
        )
    )


    return df


# ==========================================
# 6. PREPARE INPUT DATA
# ==========================================

def prepare_input(df):

    # Check whether uploaded file already
    # contains all 10 ML-ready features.

    if all(
        feature in df.columns
        for feature in FEATURES
    ):

        print(
            "\nML-ready dataset detected."
        )

        return df


    # Otherwise attempt to create the
    # required features from raw email logs.

    return prepare_raw_features(
        df
    )


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

def predict(input_file):

    print("\n======================================")
    print("EMAIL LOG PREDICTION")
    print("======================================")


    # --------------------------------------
    # Check file
    # --------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )


    # --------------------------------------
    # Load input
    # --------------------------------------

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print("\nInput file:")
    print(input_file)

    print("\nInput shape:")
    print(df.shape)


    # --------------------------------------
    # Prepare features
    # --------------------------------------

    df = prepare_input(
        df
    )


    # --------------------------------------
    # Validate features
    # --------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing ML features: "
            + ", ".join(missing_features)
        )


    X = df[
        FEATURES
    ].copy()


    # Make sure everything is numeric.

    for column in FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )


    if X.isnull().any().any():

        print(
            "\nWARNING:"
            " Missing/invalid feature values detected."
        )

        print(
            "They will be replaced with 0."
        )

        X = X.fillna(0)


    # --------------------------------------
    # Load model
    # --------------------------------------

    model = load_model()


    # --------------------------------------
    # Prediction
    # --------------------------------------

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )


    # --------------------------------------
    # Save predictions
    # --------------------------------------

    result_df = df.copy()

    result_df[
        "predicted_label"
    ] = predictions

    result_df[
        "predicted_class"
    ] = result_df[
        "predicted_label"
    ].map(
        {
            0: "legitimate",
            1: "spam"
        }
    )


    # Probability of each class

    result_df[
        "legitimate_probability"
    ] = probabilities[:, 0]

    result_df[
        "spam_probability"
    ] = probabilities[:, 1]


    # Prediction confidence

    result_df[
        "prediction_confidence"
    ] = probabilities.max(
        axis=1
    )


    # ======================================
    # OPTIONAL FALSE-DETECTION CHECK
    # ======================================

    if "label" in result_df.columns:

        actual_label = pd.to_numeric(
            result_df["label"],
            errors="coerce"
        )

        result_df[
            "possible_false_detection"
        ] = (
            actual_label
            !=
            result_df["predicted_label"]
        )

        result_df[
            "actual_class"
        ] = actual_label.map(
            {
                0: "legitimate",
                1: "spam"
            }
        )

        result_df[
            "result_status"
        ] = "Correct Classification"

        result_df.loc[
            (
                actual_label == 0
            )
            &
            (
                result_df[
                    "predicted_label"
                ] == 1
            ),
            "result_status"
        ] = "False Positive"

        result_df.loc[
            (
                actual_label == 1
            )
            &
            (
                result_df[
                    "predicted_label"
                ] == 0
            ),
            "result_status"
        ] = "False Negative"


    # --------------------------------------
    # Save file
    # --------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ======================================
    # DISPLAY SUMMARY
    # ======================================

    print("\n======================================")
    print("PREDICTION SUMMARY")
    print("======================================")

    print(
        result_df[
            "predicted_class"
        ].value_counts()
    )


    print("\nPrediction percentage:")

    print(
        result_df[
            "predicted_class"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )


    print("\nAverage prediction confidence:")

    print(
        round(
            result_df[
                "prediction_confidence"
            ].mean()
            * 100,
            2
        ),
        "%"
    )


    # --------------------------------------
    # False detection summary
    # --------------------------------------

    if "result_status" in result_df.columns:

        print("\nResult status:")

        print(
            result_df[
                "result_status"
            ].value_counts()
        )


    print("\n======================================")
    print("PREDICTION COMPLETED")
    print("======================================")

    print("\nResults saved to:")
    print(OUTPUT_FILE)


# ==========================================
# 8. COMMAND LINE INPUT
# ==========================================

# ==========================================
# COMMAND LINE MODE
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Predict legitimate/spam "
            "email logs using Random Forest."
        )
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(DEFAULT_INPUT_FILE),
        help="CSV file to classify"
    )

    args = parser.parse_args()

    predict(
        Path(args.input_file)
    )