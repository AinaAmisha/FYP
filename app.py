from flask import send_file, session
from uuid import uuid4, UUID
import secrets

from flask import (
    Flask,
    render_template,
    request,
    abort,
    redirect,
    url_for
)

from datetime import datetime
import pandas as pd
from pathlib import Path

from src.predict_email_logs import (
    FEATURES,
    prepare_input,
    load_model
)

# ==========================================
# FLASK CONFIGURATION
# ==========================================

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)


# ==========================================
# FILE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_FOLDER = PROJECT_ROOT / "results"
EXPORT_FOLDER = RESULT_FOLDER / "exports"


# ==========================================
# PERSISTENT FLASK SESSION
# ==========================================

RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

SECRET_FILE = RESULT_FOLDER / ".flask_secret_key"

if not SECRET_FILE.exists():
    SECRET_FILE.write_text(
        secrets.token_hex(32),
        encoding="utf-8"
    )

app.secret_key = SECRET_FILE.read_text(
    encoding="utf-8"
).strip()


MODEL_RESULTS_FILE = (
    RESULT_FOLDER / "model_results.csv"
)

FALSE_SUMMARY_FILE = (
    RESULT_FOLDER / "false_detection_summary.csv"
)

PREDICTION_FILE = (
    RESULT_FOLDER / "test_predictions.csv"
)

FALSE_DETECTION_FILE = (
    RESULT_FOLDER / "false_detections.csv"
)

REVIEW_FILE = (
    RESULT_FOLDER / "review_decisions.csv"
)

# ==========================================
# DASHBOARD DATA
# ==========================================

def load_dashboard_data():

    dashboard = {
        "best_model": "Not Available",
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "fpr": 0,
        "fnr": 0,
        "total_test": 0,
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "false_detection_rate": 0
    }


    # ======================================
    # MODEL RESULTS
    # ======================================

    if MODEL_RESULTS_FILE.exists():

        model_results = pd.read_csv(
            MODEL_RESULTS_FILE
        )

        if not model_results.empty:

            # Results are already sorted by F1
            # in train_models.py.
            best = model_results.iloc[0]

            dashboard["best_model"] = (
                best["Model"]
            )

            dashboard["accuracy"] = (
                float(best["Accuracy"]) * 100
            )

            dashboard["precision"] = (
                float(best["Precision"]) * 100
            )

            dashboard["recall"] = (
                float(best["Recall"]) * 100
            )

            dashboard["f1_score"] = (
                float(best["F1 Score"]) * 100
            )

            dashboard["fpr"] = (
                float(
                    best["False Positive Rate"]
                ) * 100
            )

            dashboard["fnr"] = (
                float(
                    best["False Negative Rate"]
                ) * 100
            )


    # ======================================
    # FALSE DETECTION RESULTS
    # ======================================

    if FALSE_SUMMARY_FILE.exists():

        false_summary = pd.read_csv(
            FALSE_SUMMARY_FILE
        )

        summary = dict(
            zip(
                false_summary["Metric"],
                false_summary["Value"]
            )
        )

        dashboard["total_test"] = int(
            summary.get(
                "Total Test Emails",
                0
            )
        )

        dashboard["true_positive"] = int(
            summary.get(
                "True Positive",
                0
            )
        )

        dashboard["true_negative"] = int(
            summary.get(
                "True Negative",
                0
            )
        )

        dashboard["false_positive"] = int(
            summary.get(
                "False Positive",
                0
            )
        )

        dashboard["false_negative"] = int(
            summary.get(
                "False Negative",
                0
            )
        )

        dashboard["false_detection_rate"] = (
            float(
                summary.get(
                    "False Detection Rate",
                    0
                )
            ) * 100
        )


    return dashboard


# ==========================================
# REVIEW DECISIONS
# ==========================================

def load_review_decisions():

    if not REVIEW_FILE.exists():

        return pd.DataFrame(
            columns=[
                "source_row_id",
                "decision",
                "reviewed_at"
            ]
        )

    reviews = pd.read_csv(
        REVIEW_FILE
    )

    if "source_row_id" in reviews.columns:

        reviews["source_row_id"] = pd.to_numeric(
            reviews["source_row_id"],
            errors="coerce"
        )

    return reviews


def save_review_decision(
    source_row_id,
    decision
):

    reviews = load_review_decisions()

    reviews = reviews[
        reviews["source_row_id"]
        != source_row_id
    ]

    new_review = pd.DataFrame(
        [
            {
                "source_row_id":
                    source_row_id,

                "decision":
                    decision,

                "reviewed_at":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
            }
        ]
    )

    reviews = pd.concat(
        [
            reviews,
            new_review
        ],
        ignore_index=True
    )

    REVIEW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    reviews.to_csv(
        REVIEW_FILE,
        index=False
    )


# ==========================================
# DASHBOARD ROUTE
# ==========================================

@app.route("/")
def dashboard():

    data = load_dashboard_data()

    return render_template(
        "dashboard.html",
        data=data
    )

# ==========================================
# MESSAGE AUDIT LOGS
# ==========================================

@app.route("/audit-logs")
def audit_logs():

    if not PREDICTION_FILE.exists():

        return render_template(
            "audit_logs.html",
            messages=[],
            total=0,
            page=1,
            pages=1,
            query="",
            result_filter="All"
        )


    df = pd.read_csv(
        PREDICTION_FILE,
        low_memory=False
    )

    df = df.fillna("")


    # ======================================
    # SEARCH
    # ======================================

    query = request.args.get(
        "q",
        ""
    ).strip()


    if query:

        search_columns = [
            "from_address",
            "from_domain",
            "subject",
            "actual_class",
            "predicted_class",
            "result_type"
        ]

        mask = pd.Series(
            False,
            index=df.index
        )

        for column in search_columns:

            if column in df.columns:

                mask = (
                    mask
                    |
                    df[column]
                    .astype(str)
                    .str.contains(
                        query,
                        case=False,
                        regex=False,
                        na=False
                    )
                )

        df = df[mask]


    # ======================================
    # RESULT FILTER
    # ======================================

    result_filter = request.args.get(
        "result",
        "All"
    )


    if (
        result_filter != "All"
        and
        "result_type" in df.columns
    ):

        df = df[
            df["result_type"]
            ==
            result_filter
        ]


    # ======================================
    # PAGINATION
    # ======================================

    page = request.args.get(
        "page",
        1,
        type=int
    )

    if page < 1:
        page = 1


    per_page = 25

    total = len(df)

    pages = max(
        (total + per_page - 1)
        // per_page,
        1
    )


    if page > pages:
        page = pages


    start = (
        (page - 1)
        * per_page
    )

    end = (
        start
        + per_page
    )


    page_df = df.iloc[
        start:end
    ].copy()


    messages = page_df.to_dict(
        orient="records"
    )


    return render_template(
        "audit_logs.html",
        messages=messages,
        total=total,
        page=page,
        pages=pages,
        query=query,
        result_filter=result_filter
    )

# ==========================================
# MESSAGE DETAILS
# ==========================================

@app.route("/audit-logs/<int:source_row_id>")
def message_detail(source_row_id):

    if not PREDICTION_FILE.exists():
        abort(404)

    df = pd.read_csv(
        PREDICTION_FILE,
        low_memory=False
    )

    if "source_row_id" not in df.columns:
        abort(404)

    df["source_row_id"] = pd.to_numeric(
        df["source_row_id"],
        errors="coerce"
    )

    matched = df[
        df["source_row_id"] == source_row_id
    ]

    if matched.empty:
        abort(404)

    message = (
        matched
        .fillna("")
        .iloc[0]
        .to_dict()
    )

    return render_template(
        "message_detail.html",
        message=message
    )

# ==========================================
# FALSE DETECTION PAGE
# ==========================================

@app.route("/false-detection")
def false_detection():

    if not FALSE_DETECTION_FILE.exists():

        return render_template(
            "false_detection.html",
            messages=[],
            total_false=0,
            false_positive=0,
            false_negative=0,
            false_detection_rate=0,
            detection_filter="All"
        )


    df = pd.read_csv(
        FALSE_DETECTION_FILE,
        low_memory=False
    )

    df = df.fillna("")


    # ======================================
    # COUNTS
    # ======================================

    total_false = len(df)

    false_positive = len(
        df[
            df["result_type"]
            ==
            "False Positive"
        ]
    )

    false_negative = len(
        df[
            df["result_type"]
            ==
            "False Negative"
        ]
    )


    # ======================================
    # FALSE DETECTION RATE
    # ======================================

    total_test = 0

    if PREDICTION_FILE.exists():

        prediction_df = pd.read_csv(
            PREDICTION_FILE,
            low_memory=False
        )

        total_test = len(
            prediction_df
        )


    if total_test > 0:

        false_detection_rate = (
            total_false
            /
            total_test
            *
            100
        )

    else:

        false_detection_rate = 0


    # ======================================
    # FILTER
    # ======================================

    detection_filter = request.args.get(
        "type",
        "All"
    )


    filtered_df = df.copy()


    if detection_filter in [
        "False Positive",
        "False Negative"
    ]:

        filtered_df = filtered_df[
            filtered_df["result_type"]
            ==
            detection_filter
        ]


    messages = filtered_df.to_dict(
        orient="records"
    )


    return render_template(
        "false_detection.html",
        messages=messages,
        total_false=total_false,
        false_positive=false_positive,
        false_negative=false_negative,
        false_detection_rate=false_detection_rate,
        detection_filter=detection_filter
    )

# ==========================================
# REVIEW CENTER
# ==========================================

@app.route("/review-center")
def review_center():

    if not FALSE_DETECTION_FILE.exists():

        return render_template(
            "review_center.html",
            messages=[],
            total_candidates=0,
            pending_count=0,
            reviewed_count=0
        )


    df = pd.read_csv(
        FALSE_DETECTION_FILE,
        low_memory=False
    )

    df = df.fillna("")


    # Only false positives are reviewed here
    candidates = df[
        df["result_type"]
        ==
        "False Positive"
    ].copy()


    candidates["source_row_id"] = pd.to_numeric(
        candidates["source_row_id"],
        errors="coerce"
    )


    reviews = load_review_decisions()


    reviewed_ids = set(
        reviews[
            "source_row_id"
        ].dropna().astype(int)
    )


    candidates["reviewed"] = (
        candidates[
            "source_row_id"
        ]
        .astype(int)
        .isin(reviewed_ids)
    )


    pending_df = candidates[
        candidates["reviewed"]
        ==
        False
    ]


    total_candidates = len(
        candidates
    )

    pending_count = len(
        pending_df
    )

    reviewed_count = (
        total_candidates
        -
        pending_count
    )


    messages = pending_df.to_dict(
        orient="records"
    )


    return render_template(
        "review_center.html",
        messages=messages,
        total_candidates=total_candidates,
        pending_count=pending_count,
        reviewed_count=reviewed_count
    )


# ==========================================
# REVIEW MESSAGE
# ==========================================

@app.route(
    "/review/<int:source_row_id>"
)
def review_message(source_row_id):

    if not FALSE_DETECTION_FILE.exists():
        abort(404)


    df = pd.read_csv(
        FALSE_DETECTION_FILE,
        low_memory=False
    )


    df["source_row_id"] = pd.to_numeric(
        df["source_row_id"],
        errors="coerce"
    )


    matched = df[
        (
            df["source_row_id"]
            ==
            source_row_id
        )
        &
        (
            df["result_type"]
            ==
            "False Positive"
        )
    ]


    if matched.empty:
        abort(404)


    message = (
        matched
        .fillna("")
        .iloc[0]
        .to_dict()
    )


    return render_template(
        "review_message.html",
        message=message
    )


# ==========================================
# SAVE REVIEW DECISION
# ==========================================

@app.route(
    "/review/<int:source_row_id>/decision",
    methods=["POST"]
)
def review_decision(source_row_id):

    decision = request.form.get(
        "decision"
    )

    allowed_decisions = {
        "release": "Release as Legitimate",
        "keep": "Keep as Spam"
    }

    if decision not in allowed_decisions:
        abort(400)

    save_review_decision(
        source_row_id,
        allowed_decisions[decision]
    )

    return redirect(
        url_for("review_center")
    )

# ==========================================
# QUARANTINE
# ==========================================

@app.route("/quarantine")
def quarantine():

    if not FALSE_DETECTION_FILE.exists():

        return render_template(
            "quarantine.html",
            messages=[],
            total_candidates=0,
            quarantined_count=0,
            pending_count=0,
            released_count=0
        )


    df = pd.read_csv(
        FALSE_DETECTION_FILE,
        low_memory=False
    )

    df = df.fillna("")


    # False-positive records are used as the
    # current simulated quarantine candidates.
    candidates = df[
        df["result_type"]
        ==
        "False Positive"
    ].copy()


    candidates["source_row_id"] = pd.to_numeric(
        candidates["source_row_id"],
        errors="coerce"
    )


    # ======================================
    # LOAD REVIEW DECISIONS
    # ======================================

    reviews = load_review_decisions()


    candidates = candidates.merge(
        reviews,
        on="source_row_id",
        how="left"
    )


    candidates["decision"] = (
        candidates["decision"]
        .fillna("Pending Review")
    )


    candidates["reviewed_at"] = (
        candidates["reviewed_at"]
        .fillna("")
    )


    # ======================================
    # COUNTS
    # ======================================

    total_candidates = len(
        candidates
    )


    released_count = len(
        candidates[
            candidates["decision"]
            ==
            "Release as Legitimate"
        ]
    )


    pending_count = len(
        candidates[
            candidates["decision"]
            ==
            "Pending Review"
        ]
    )


    # Anything not released remains 
    # in the simulated quarantine.
    active_quarantine = candidates[
        candidates["decision"]
        !=
        "Release as Legitimate"
    ].copy()


    quarantined_count = len(
        active_quarantine
    )


    messages = active_quarantine.to_dict(
        orient="records"
    )


    return render_template(
        "quarantine.html",
        messages=messages,
        total_candidates=total_candidates,
        quarantined_count=quarantined_count,
        pending_count=pending_count,
        released_count=released_count
    )

# ==========================================
# MODEL PERFORMANCE
# ==========================================

@app.route("/model-performance")
def model_performance():

    if not MODEL_RESULTS_FILE.exists():

        return render_template(
            "model_performance.html",
            models=[],
            selected_model=None
        )


    df = pd.read_csv(
        MODEL_RESULTS_FILE
    )


    if df.empty:

        return render_template(
            "model_performance.html",
            models=[],
            selected_model=None
        )


    # Convert decimal metrics to percentages
    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "False Positive Rate",
        "False Negative Rate"
    ]


    for column in percentage_columns:

        if column in df.columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
                *
                100
            )


    # Make confusion matrix counts integers
    count_columns = [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive"
    ]


    for column in count_columns:

        if column in df.columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )


    models = df.to_dict(
        orient="records"
    )


    # train_models.py already sorts
    # model_results.csv by F1 Score.
    selected_model = models[0]

    # train_models.py already sorts
    # model_results.csv by F1 Score.
    selected_model = models[0]


    # ==========================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ==========================================

    FEATURE_IMPORTANCE_FILE = (
        RESULT_FOLDER / "feature_importance.csv"
    )

    feature_importance = []

    if FEATURE_IMPORTANCE_FILE.exists():

        importance_df = pd.read_csv(
            FEATURE_IMPORTANCE_FILE
        )

        if {"Feature", "Importance"}.issubset(
            importance_df.columns
        ):

            importance_df["Importance"] = (
                pd.to_numeric(
                    importance_df["Importance"],
                    errors="coerce"
                )
                .fillna(0)
                .clip(lower=0)
            )

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            max_importance = importance_df["Importance"].max()

            importance_df["Percentage"] = (
                importance_df["Importance"] * 100
            )

            importance_df["Bar Width"] = (
                importance_df["Importance"]
                / max_importance * 100
                if max_importance > 0
                else 0
            )

            feature_importance = importance_df.to_dict(
                orient="records"
            )




    # ==========================================
    # PERMUTATION IMPORTANCE
    # ==========================================

    permutation_importance = []

    permutation_file = (
        RESULT_FOLDER / "permutation_importance.csv"
    )

    if permutation_file.exists():

        perm_df = pd.read_csv(permutation_file)

        required_columns = {
            "Feature",
            "Importance Mean",
            "Importance Standard Deviation"
        }

        if required_columns.issubset(perm_df.columns):

            for column in [
                "Importance Mean",
                "Importance Standard Deviation"
            ]:
                perm_df[column] = (
                    pd.to_numeric(
                        perm_df[column],
                        errors="coerce"
                    ).fillna(0)
                )

            perm_df = perm_df.sort_values(
                by="Importance Mean",
                ascending=False
            )

            perm_df["F1 Drop"] = (
                perm_df["Importance Mean"] * 100
            )

            perm_df["Std Dev"] = (
                perm_df["Importance Standard Deviation"]
                * 100
            )

            max_value = max(
                perm_df["Importance Mean"].max(),
                0
            ) if not perm_df.empty else 0

            if max_value > 0:
                perm_df["Bar Width"] = (
                    perm_df["Importance Mean"]
                    .clip(lower=0)
                    / max_value * 100
                )
            else:
                perm_df["Bar Width"] = 0

            permutation_importance = (
                perm_df.to_dict(orient="records")
            )

    # ==========================================
    # RENDER PAGE
    # ==========================================

    return render_template(
        "model_performance.html",
        models=models,
        selected_model=selected_model,
        feature_importance=feature_importance,
        permutation_importance=permutation_importance
    )


# ==========================================
# MESSAGE ANALYSIS
# ==========================================

@app.route("/message-analysis", methods=["GET", "POST"])
def message_analysis():

    messages = []
    total = 0
    legitimate_count = 0
    spam_count = 0
    error = None
    export_id = None

    if request.method == "POST":

        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a CSV file."

        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only CSV files are supported."

        else:
            try:
                df = pd.read_csv(
                    uploaded_file,
                    low_memory=False
                )

                if df.empty:
                    raise ValueError(
                        "The uploaded CSV is empty."
                    )

                # Reuse the existing feature pipeline.
                df = prepare_input(df.copy())

                X = df[FEATURES].copy()

                for column in FEATURES:
                    X[column] = pd.to_numeric(
                        X[column],
                        errors="coerce"
                    )

                if X.isnull().any().any():
                    raise ValueError(
                        "Some ML features contain "
                        "missing or invalid values."
                    )

                model = load_model()

                predictions = model.predict(X)
                probabilities = model.predict_proba(X)

                class_indexes = {
                    int(label): index
                    for index, label
                    in enumerate(model.classes_)
                }

                df["predicted_label"] = predictions

                df["predicted_class"] = (
                    df["predicted_label"].map({
                        0: "legitimate",
                        1: "spam"
                    })
                )

                df["spam_probability"] = (
                    probabilities[:, class_indexes[1]] * 100
                )

                df["prediction_confidence"] = (
                    probabilities.max(axis=1) * 100
                )

                
                # ======================================
                # EXPORT CLASSIFICATION RESULTS
                # ======================================

                EXPORT_FOLDER.mkdir(
                    parents=True,
                    exist_ok=True
                )

                export_id = uuid4()

                export_columns = [
                    column
                    for column in [
                        "from_address",
                        "subject",
                        "predicted_class",
                        "prediction_confidence",
                        "spam_probability"
                    ]
                    if column in df.columns
                ]

                export_df = df[export_columns].copy()

                export_path = (
                    EXPORT_FOLDER / f"{export_id.hex}.csv"
                )

                export_df.to_csv(
                    export_path,
                    index=False
                )

                # Remember this classification for future visits
                session["last_export_id"] = export_id.hex

                total = len(df)

                legitimate_count = int(
                    (df["predicted_label"] == 0).sum()
                )

                spam_count = int(
                    (df["predicted_label"] == 1).sum()
                )

                # Display the first 25 records.
                preview = df.head(25).copy()

                if "from_address" not in preview.columns:
                    preview["from_address"] = "-"

                if "subject" not in preview.columns:
                    preview["subject"] = "-"

                messages = (
                    preview.fillna("")
                    .to_dict(orient="records")
                )

            except (
                ValueError,
                OSError,
                ImportError,
                pd.errors.ParserError
            ) as exc:
                error = str(exc)




    # ==========================================
    # RESTORE PREVIOUS CLASSIFICATION RESULTS
    # ==========================================

    if total == 0 and session.get("last_export_id"):

        try:
            saved_id = UUID(session["last_export_id"])

            saved_path = (
                EXPORT_FOLDER / f"{saved_id.hex}.csv"
            )

            if saved_path.is_file():

                saved_df = pd.read_csv(
                    saved_path,
                    low_memory=False
                )

                total = len(saved_df)

                legitimate_count = int(
                    saved_df["predicted_class"]
                    .eq("legitimate")
                    .sum()
                )

                spam_count = int(
                    saved_df["predicted_class"]
                    .eq("spam")
                    .sum()
                )

                if "from_address" not in saved_df.columns:
                    saved_df["from_address"] = "-"

                if "subject" not in saved_df.columns:
                    saved_df["subject"] = "-"

                messages = (
                    saved_df.head(25)
                    .fillna("")
                    .to_dict(orient="records")
                )

                export_id = saved_id

        except (
            ValueError,
            OSError,
            pd.errors.ParserError
        ):
            session.pop("last_export_id", None)




    return render_template(
            "message_analysis.html",
            messages=messages,
            total=total,
            legitimate_count=legitimate_count,
            spam_count=spam_count,
            error=error,
            export_id=export_id
    )


# ==========================================
# DOWNLOAD CLASSIFICATION CSV
# ==========================================

@app.route("/download-classification/<uuid:export_id>")
def download_classification(export_id):

    export_path = (
        EXPORT_FOLDER / f"{export_id.hex}.csv"
    )

    if not export_path.is_file():
        abort(404)

    return send_file(
        export_path,
        as_attachment=True,
        download_name="classified_email_logs.csv",
        mimetype="text/csv"
    )

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

    