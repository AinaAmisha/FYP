import pandas as pd
import streamlit as st

from src.predict_email_logs import (
    FEATURES,
    prepare_input,
    load_model
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Email Log Classification",
    page_icon="📧",
    layout="wide"
)

# ==========================================
# CUSTOM INTERFACE STYLE
# ==========================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 20px;
        border-radius: 12px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    .system-subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# TITLE
# ==========================================

st.title("🛡️ Email Security Analysis System")

st.markdown(
    """
    <div class="system-subtitle">
    Machine Learning-Based False Detection in Email Log Classification
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    """
    Upload an email log CSV file to classify
    each email as Legitimate or Spam using
    the trained Random Forest model.
    """
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def get_model():

    return load_model()


try:

    model = get_model()

except Exception as error:

    st.error(
        f"Unable to load Random Forest model: {error}"
    )

    st.stop()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🛡️ Email Security")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload & Classification",
        "False Detection",
        "Model Performance",
        "About System"
    ]
)

st.sidebar.divider()

with st.sidebar.expander("System Information"):

    st.write("**Selected Model:** Random Forest")
    st.write("**Classification:** Binary")

    st.write(
        """
        **Classes**
        - 0 = Legitimate
        - 1 = Spam
        """
    )

    st.write("**ML Features:** 10")


# ==========================================
# PAGE NAVIGATION
# ==========================================

if page == "Dashboard":

    st.header("📊 Dashboard")

    st.write(
        "Overview of the trained email classification system."
    )


    # ======================================
    # LOAD SAVED RESULTS
    # ======================================

    try:

        model_results = pd.read_csv(
            "results/model_results.csv"
        )

        false_summary = pd.read_csv(
            "results/false_detection_summary.csv"
        )


        # ==================================
        # RANDOM FOREST RESULT
        # ==================================

        rf_result = model_results[
            model_results["Model"]
            ==
            "Random Forest"
        ].iloc[0]


        accuracy = (
            rf_result["Accuracy"]
            * 100
        )

        f1_score_value = (
            rf_result["F1 Score"]
            * 100
        )


        # ==================================
        # FALSE DETECTION VALUES
        # ==================================

        summary_dict = dict(
            zip(
                false_summary["Metric"],
                false_summary["Value"]
            )
        )


        total_test = int(
            summary_dict[
                "Total Test Emails"
            ]
        )

        false_positive = int(
            summary_dict[
                "False Positive"
            ]
        )

        false_negative = int(
            summary_dict[
                "False Negative"
            ]
        )

        false_detection_rate = (
            summary_dict[
                "False Detection Rate"
            ]
            * 100
        )


        # ==================================
        # OVERVIEW METRICS
        # ==================================

        st.subheader(
            "System Overview"
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )


        col1.metric(
            "Best Model",
            "Random Forest"
        )

        col2.metric(
            "Accuracy",
            f"{accuracy:.2f}%"
        )

        col3.metric(
            "F1 Score",
            f"{f1_score_value:.2f}%"
        )

        col4.metric(
            "False Detection Rate",
            f"{false_detection_rate:.2f}%"
        )


        # ==================================
        # TESTING SUMMARY
        # ==================================

        st.subheader(
            "Testing Summary"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Test Emails",
            f"{total_test:,}"
        )

        col2.metric(
            "False Positive",
            f"{false_positive:,}"
        )

        col3.metric(
            "False Negative",
            f"{false_negative:,}"
        )


        # ==================================
        # MODEL COMPARISON
        # ==================================

        st.subheader(
            "Model Performance Comparison"
        )


        comparison = (
            model_results[
                [
                    "Model",
                    "Accuracy",
                    "F1 Score"
                ]
            ]
            .copy()
        )


        comparison[
            "Accuracy"
        ] *= 100

        comparison[
            "F1 Score"
        ] *= 100


        comparison = (
            comparison
            .set_index("Model")
        )


        st.bar_chart(
            comparison
        )


        # ==================================
        # MODEL RESULT TABLE
        # ==================================

        with st.expander(
            "View Full Model Results"
        ):

            display_results = (
                model_results.copy()
            )

            percentage_columns = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "False Positive Rate",
                "False Negative Rate"
            ]

            for column in percentage_columns:

                display_results[
                    column
                ] = (
                    display_results[
                        column
                    ]
                    * 100
                ).round(2)


            st.dataframe(
                display_results,
                width="stretch"
            )


    except FileNotFoundError:

        st.warning(
            "Result files were not found. "
            "Run the ML analysis scripts first."
        )


    st.stop()


elif page == "False Detection":

    st.header("⚠️ False Detection Analysis")

    try:

        false_df = pd.read_csv(
            "results/false_detections.csv",
            low_memory=False
        )

        summary_df = pd.read_csv(
            "results/false_detection_summary.csv"
        )


        # ==================================
        # SUMMARY VALUES
        # ==================================

        summary = dict(
            zip(
                summary_df["Metric"],
                summary_df["Value"]
            )
        )

        total_test = int(
            summary["Total Test Emails"]
        )

        false_positive = int(
            summary["False Positive"]
        )

        false_negative = int(
            summary["False Negative"]
        )

        total_false = int(
            summary["Total False Detection"]
        )

        false_rate = (
            summary["False Detection Rate"]
            * 100
        )


        # ==================================
        # METRICS
        # ==================================

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Test Emails",
            f"{total_test:,}"
        )

        col2.metric(
            "False Positive",
            f"{false_positive:,}"
        )

        col3.metric(
            "False Negative",
            f"{false_negative:,}"
        )

        col4.metric(
            "False Detection Rate",
            f"{false_rate:.2f}%"
        )


        # ==================================
        # DESCRIPTION
        # ==================================

        st.info(
            """
            False Positive:
            A legitimate email is incorrectly
            classified as spam.

            False Negative:
            A spam email is incorrectly
            classified as legitimate.
            """
        )


        # ==================================
        # TABS
        # ==================================

        tab1, tab2, tab3 = st.tabs(
            [
                "All False Detections",
                "False Positives",
                "False Negatives"
            ]
        )


        with tab1:

            st.subheader(
                f"All False Detections ({total_false})"
            )

            st.dataframe(
                false_df,
                width="stretch"
            )


        with tab2:

            fp_df = false_df[
                false_df["result_type"]
                ==
                "False Positive"
            ]

            st.subheader(
                f"False Positives ({len(fp_df)})"
            )

            st.dataframe(
                fp_df,
                width="stretch"
            )


        with tab3:

            fn_df = false_df[
                false_df["result_type"]
                ==
                "False Negative"
            ]

            st.subheader(
                f"False Negatives ({len(fn_df)})"
            )

            st.dataframe(
                fn_df,
                width="stretch"
            )


        # ==================================
        # DOWNLOAD
        # ==================================

        st.subheader(
            "Download False Detection Results"
        )

        csv_data = (
            false_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download False Detection CSV",
            data=csv_data,
            file_name="false_detections.csv",
            mime="text/csv",
            width="stretch"
        )


    except FileNotFoundError:

        st.warning(
            "False detection result files were not found."
        )


    st.stop()


elif page == "Model Performance":

    st.header("📈 Model Performance")

    try:

        model_results = pd.read_csv(
            "results/model_results.csv"
        )

        feature_importance = pd.read_csv(
            "results/feature_importance.csv"
        )

        permutation_importance = pd.read_csv(
            "results/permutation_importance.csv"
        )


        # ==================================
        # MODEL COMPARISON
        # ==================================

        st.subheader(
            "Model Comparison"
        )


        display_results = (
            model_results.copy()
        )


        percentage_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "False Positive Rate",
            "False Negative Rate"
        ]


        for column in percentage_columns:

            display_results[
                column
            ] = (
                display_results[
                    column
                ]
                * 100
            ).round(2)


        st.dataframe(
            display_results,
            width="stretch"
        )


        # ==================================
        # ACCURACY / F1 CHART
        # ==================================

        chart_df = (
            model_results[
                [
                    "Model",
                    "Accuracy",
                    "F1 Score"
                ]
            ]
            .copy()
        )


        chart_df[
            "Accuracy"
        ] *= 100

        chart_df[
            "F1 Score"
        ] *= 100


        chart_df = (
            chart_df
            .set_index("Model")
        )


        st.subheader(
            "Accuracy and F1 Score"
        )

        st.bar_chart(
            chart_df
        )


        # ==================================
        # BEST MODEL
        # ==================================

        rf = model_results[
            model_results["Model"]
            ==
            "Random Forest"
        ].iloc[0]


        st.subheader(
            "Selected Model"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        col1.metric(
            "Model",
            "Random Forest"
        )

        col2.metric(
            "Accuracy",
            f"{rf['Accuracy'] * 100:.2f}%"
        )

        col3.metric(
            "F1 Score",
            f"{rf['F1 Score'] * 100:.2f}%"
        )


        # ==================================
        # FEATURE IMPORTANCE
        # ==================================

        st.subheader(
            "Random Forest Feature Importance"
        )


        importance_chart = (
            feature_importance[
                [
                    "Feature",
                    "Importance Percentage"
                ]
            ]
            .copy()
        )


        importance_chart = (
            importance_chart
            .set_index("Feature")
        )


        st.bar_chart(
            importance_chart
        )


        with st.expander(
            "View Feature Importance Table"
        ):

            st.dataframe(
                feature_importance,
                width="stretch"
            )


        # ==================================
        # PERMUTATION IMPORTANCE
        # ==================================

        st.subheader(
            "Permutation Importance"
        )


        permutation_chart = (
            permutation_importance[
                [
                    "Feature",
                    "Importance Mean"
                ]
            ]
            .copy()
        )


        permutation_chart = (
            permutation_chart
            .set_index("Feature")
        )


        st.bar_chart(
            permutation_chart
        )


        with st.expander(
            "View Permutation Importance Table"
        ):

            st.dataframe(
                permutation_importance,
                width="stretch"
            )


    except FileNotFoundError:

        st.warning(
            "Model analysis files were not found."
        )


    st.stop()


elif page == "About System":

    st.header("ℹ️ About System")

    st.write(
        """
        **Project Title:**  
        Machine Learning-Based False Detection
        in Email Log Classification

        **Selected Model:** Random Forest

        **Classification:** Legitimate / Spam

        **Machine Learning Models Evaluated:**
        - Support Vector Machine
        - Naive Bayes
        - Random Forest
        """
    )

    st.stop()


# If page == "Upload & Classification",
# the existing upload/classification code below will run.


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Email Log CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # ==================================
        # LOAD CSV
        # ==================================

        df = pd.read_csv(
            uploaded_file,
            low_memory=False
        )

        st.success(
            "CSV file uploaded successfully."
        )


        # ==================================
        # DATASET INFORMATION
        # ==================================

        st.subheader(
            "Uploaded Dataset"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Number of Emails",
                len(df)
            )

        with col2:

            st.metric(
                "Number of Columns",
                len(df.columns)
            )


        with st.expander(
            "Preview Uploaded Dataset"
        ):

            st.dataframe(
                df.head(20),
                use_container_width=True
            )


        # ==================================
        # ANALYZE BUTTON
        # ==================================

        if st.button(
            "🔍 Analyze Email Logs",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing email logs..."
            ):

                # ==========================
                # PREPARE FEATURES
                # ==========================

                prepared_df = prepare_input(
                    df.copy()
                )


                missing_features = [

                    feature

                    for feature in FEATURES

                    if feature not in prepared_df.columns
                ]


                if missing_features:

                    st.error(
                        "Missing required features: "
                        +
                        ", ".join(
                            missing_features
                        )
                    )

                    st.stop()


                # ==========================
                # ML INPUT
                # ==========================

                X = prepared_df[
                    FEATURES
                ].copy()


                for column in FEATURES:

                    X[column] = pd.to_numeric(
                        X[column],
                        errors="coerce"
                    )


                if X.isnull().any().any():

                    st.warning(
                        """
                        Missing or invalid feature
                        values were detected.
                        They have been replaced
                        with 0.
                        """
                    )

                    X = X.fillna(0)


                # ==========================
                # PREDICTION
                # ==========================

                predictions = model.predict(
                    X
                )

                probabilities = (
                    model.predict_proba(
                        X
                    )
                )


                # ==========================
                # RESULTS
                # ==========================

                result_df = (
                    prepared_df.copy()
                )


                result_df[
                    "predicted_label"
                ] = predictions


                result_df[
                    "predicted_class"
                ] = result_df[
                    "predicted_label"
                ].map(
                    {
                        0: "Legitimate",
                        1: "Spam"
                    }
                )


                result_df[
                    "legitimate_probability"
                ] = probabilities[:, 0]


                result_df[
                    "spam_probability"
                ] = probabilities[:, 1]


                result_df[
                    "prediction_confidence"
                ] = probabilities.max(
                    axis=1
                )


                # ==========================
                # FALSE DETECTION
                # ==========================

                has_actual_label = (
                    "label"
                    in result_df.columns
                )


                if has_actual_label:

                    actual_label = (
                        pd.to_numeric(
                            result_df[
                                "label"
                            ],
                            errors="coerce"
                        )
                    )


                    result_df[
                        "actual_class"
                    ] = actual_label.map(
                        {
                            0: "Legitimate",
                            1: "Spam"
                        }
                    )


                    result_df[
                        "result_status"
                    ] = (
                        "Correct Classification"
                    )


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


                # ==========================
                # SUMMARY
                # ==========================

                st.success(
                    "Analysis completed successfully."
                )

                st.header(
                    "Classification Summary"
                )


                total_email = len(
                    result_df
                )

                legitimate_count = (

                    result_df[
                        "predicted_label"
                    ]
                    ==
                    0

                ).sum()


                spam_count = (

                    result_df[
                        "predicted_label"
                    ]
                    ==
                    1

                ).sum()


                average_confidence = (

                    result_df[
                        "prediction_confidence"
                    ]
                    .mean()
                    *
                    100
                )


                col1, col2, col3, col4 = (
                    st.columns(4)
                )


                col1.metric(
                    "Total Emails",
                    total_email
                )


                col2.metric(
                    "Legitimate",
                    legitimate_count
                )


                col3.metric(
                    "Spam",
                    spam_count
                )


                col4.metric(
                    "Avg. Confidence",
                    f"{average_confidence:.2f}%"
                )


                # ==========================
                # CLASS DISTRIBUTION
                # ==========================

                st.subheader(
                    "Prediction Distribution"
                )


                distribution = (

                    result_df[
                        "predicted_class"
                    ]
                    .value_counts()
                    .rename_axis(
                        "Classification"
                    )
                    .reset_index(
                        name="Count"
                    )

                )


                st.bar_chart(
                    distribution,
                    x="Classification",
                    y="Count"
                )


                # ==========================
                # FALSE DETECTION SUMMARY
                # ==========================

                if has_actual_label:

                    st.header(
                        "False Detection Analysis"
                    )


                    false_positive = (

                        result_df[
                            "result_status"
                        ]
                        ==
                        "False Positive"

                    ).sum()


                    false_negative = (

                        result_df[
                            "result_status"
                        ]
                        ==
                        "False Negative"

                    ).sum()


                    correct = (

                        result_df[
                            "result_status"
                        ]
                        ==
                        "Correct Classification"

                    ).sum()


                    total_false = (

                        false_positive
                        +
                        false_negative

                    )


                    false_rate = (

                        total_false
                        /
                        total_email
                        *
                        100

                    )


                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )


                    col1.metric(
                        "Correct",
                        correct
                    )


                    col2.metric(
                        "False Positive",
                        false_positive
                    )


                    col3.metric(
                        "False Negative",
                        false_negative
                    )


                    col4.metric(
                        "False Detection Rate",
                        f"{false_rate:.2f}%"
                    )


                # ==========================
                # RESULT TABS
                # ==========================

                st.header(
                    "Detailed Results"
                )


                if has_actual_label:

                    tab1, tab2, tab3 = (
                        st.tabs(
                            [
                                "All Predictions",
                                "False Positives",
                                "False Negatives"
                            ]
                        )
                    )


                    with tab1:

                        st.dataframe(
                            result_df,
                            width="stretch"
                        )


                    with tab2:

                        fp_df = result_df[

                            result_df[
                                "result_status"
                            ]
                            ==
                            "False Positive"

                        ]

                        st.write(
                            f"{len(fp_df)} "
                            "false positive emails"
                        )

                        st.dataframe(
                            fp_df,
                            use_container_width=True
                        )


                    with tab3:

                        fn_df = result_df[

                            result_df[
                                "result_status"
                            ]
                            ==
                            "False Negative"

                        ]

                        st.write(
                            f"{len(fn_df)} "
                            "false negative emails"
                        )

                        st.dataframe(
                            fn_df,
                            use_container_width=True
                        )


                else:

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )


                # ==========================
                # DOWNLOAD
                # ==========================

                st.header(
                    "Download Results"
                )


                csv_data = (
                    result_df
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )


                st.download_button(
                    label=(
                        "⬇️ Download Classification Results"
                    ),
                    data=csv_data,
                    file_name=(
                        "email_classification_results.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )


    except Exception as error:

        st.error(
            f"Error processing file: {error}"
        )


else:

    st.write(
        "👆 Upload a CSV file to begin classification."
    )