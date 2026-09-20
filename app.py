from flask import Flask, render_template, request, abort
import pandas as pd
from pathlib import Path


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

MODEL_RESULTS_FILE = (
    RESULT_FOLDER / "model_results.csv"
)

FALSE_SUMMARY_FILE = (
    RESULT_FOLDER / "false_detection_summary.csv"
)

PREDICTION_FILE = (
    RESULT_FOLDER / "test_predictions.csv"
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
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )