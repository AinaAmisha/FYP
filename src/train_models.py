import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. FILE PATHS
# ==========================================

TRAIN_FILE = Path(
    "data/split/train_data.csv"
)

TEST_FILE = Path(
    "data/split/test_data.csv"
)

MODEL_FOLDER = Path(
    "models"
)

RESULT_FOLDER = Path(
    "results"
)


# ==========================================
# 2. LOAD TRAINING AND TEST DATA
# ==========================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\n======================================")
print("MODEL TRAINING")
print("======================================")

print("\nTraining dataset shape:")
print(train_df.shape)

print("\nTesting dataset shape:")
print(test_df.shape)


# ==========================================
# 3. DEFINE FEATURES
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
# 4. PREPARE X AND Y
# ==========================================

X_train = train_df[features]
y_train = train_df["label"]

X_test = test_df[features]
y_test = test_df["label"]

print("\nFeatures used:")

for feature in features:
    print("-", feature)


# ==========================================
# 5. CREATE MODELS
# ==========================================

models = {

    # SVM requires feature scaling
    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                random_state=42
            )
        )
    ]),

    # Gaussian Naive Bayes
    "Naive Bayes": GaussianNB(),

    # Random Forest
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 6. CREATE OUTPUT FOLDERS
# ==========================================

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 7. RESULT STORAGE
# ==========================================

results = []


# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================

for model_name, model in models.items():

    print("\n======================================")
    print(model_name)
    print("======================================")

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    predictions = model.predict(
        X_test
    )


    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    # ======================================
    # CONFUSION MATRIX
    # ======================================

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()


    # ======================================
    # FALSE POSITIVE RATE
    # ======================================

    if (fp + tn) > 0:

        fpr = fp / (fp + tn)

    else:

        fpr = 0


    # ======================================
    # FALSE NEGATIVE RATE
    # ======================================

    if (fn + tp) > 0:

        fnr = fn / (fn + tp)

    else:

        fnr = 0


    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\nAccuracy:")
    print(round(accuracy, 4))

    print("\nPrecision:")
    print(round(precision, 4))

    print("\nRecall:")
    print(round(recall, 4))

    print("\nF1 Score:")
    print(round(f1, 4))

    print("\nFalse Positive Rate:")
    print(round(fpr, 4))

    print("\nFalse Negative Rate:")
    print(round(fnr, 4))


    print("\nConfusion Matrix:")

    print(
        f"""
        Predicted
                  Legitimate    Spam
Actual
Legitimate        {tn:<12} {fp}
Spam              {fn:<12} {tp}
        """
    )


    # ======================================
    # SAVE RESULT
    # ======================================

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "False Positive Rate": fpr,

        "False Negative Rate": fnr,

        "True Negative": tn,

        "False Positive": fp,

        "False Negative": fn,

        "True Positive": tp
    })


    # ======================================
    # SAVE MODEL
    # ======================================

    model_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = (
        MODEL_FOLDER
        /
        model_filename
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved to:")
    print(model_path)


# ==========================================
# 9. CREATE RESULT DATAFRAME
# ==========================================

results_df = pd.DataFrame(
    results
)


# ==========================================
# 10. SORT MODELS BY F1 SCORE
# ==========================================

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)


# ==========================================
# 11. DISPLAY MODEL COMPARISON
# ==========================================

print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

display_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "False Positive Rate",
    "False Negative Rate"
]

print(
    results_df[
        display_columns
    ].round(4)
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

RESULT_FILE = (
    RESULT_FOLDER
    /
    "model_results.csv"
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print("\nResults saved to:")
print(RESULT_FILE)


# ==========================================
# 13. BEST MODEL
# ==========================================

best_model = results_df.iloc[0]

print("\n======================================")
print("BEST MODEL")
print("======================================")

print("\nModel:")
print(best_model["Model"])

print("\nF1 Score:")
print(
    round(
        best_model["F1 Score"],
        4
    )
)

print("\nAccuracy:")
print(
    round(
        best_model["Accuracy"],
        4
    )
)


# ==========================================
# 14. FINAL MESSAGE
# ==========================================

print("\n======================================")
print("MODEL TRAINING COMPLETED")
print("======================================")
