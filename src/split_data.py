import pandas as pd
from pathlib import Path


# ==========================================
# 1. FILE PATHS
# ==========================================

INPUT_FILE = Path(
    "data/processed/email_dataset_100k_ml_ready.csv"
)

TRAIN_FILE = Path(
    "data/split/train_data.csv"
)

TEST_FILE = Path(
    "data/split/test_data.csv"
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("\n======================================")
print("TRAIN / TEST SPLIT")
print("======================================")

print("\nOriginal dataset shape:")
print(df.shape)

print("\nOriginal class distribution:")
print(df["label"].value_counts())


# ==========================================
# 3. SEPARATE EACH CLASS
# ==========================================

legitimate = df[
    df["label"] == 0
].copy()

spam = df[
    df["label"] == 1
].copy()


print("\nLegitimate emails:")
print(len(legitimate))

print("\nSpam emails:")
print(len(spam))


# ==========================================
# 4. SHUFFLE EACH CLASS
# ==========================================

legitimate = legitimate.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

spam = spam.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 5. CALCULATE 80% SPLIT POSITION
# ==========================================

legitimate_split = int(
    len(legitimate) * 0.80
)

spam_split = int(
    len(spam) * 0.80
)


# ==========================================
# 6. TRAINING DATA
# ==========================================

legitimate_train = legitimate.iloc[
    :legitimate_split
]

spam_train = spam.iloc[
    :spam_split
]


train_df = pd.concat(
    [
        legitimate_train,
        spam_train
    ],
    ignore_index=True
)


# ==========================================
# 7. TESTING DATA
# ==========================================

legitimate_test = legitimate.iloc[
    legitimate_split:
]

spam_test = spam.iloc[
    spam_split:
]


test_df = pd.concat(
    [
        legitimate_test,
        spam_test
    ],
    ignore_index=True
)


# ==========================================
# 8. SHUFFLE FINAL TRAIN / TEST SET
# ==========================================

train_df = train_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("SPLIT RESULTS")
print("======================================")

print("\nTraining dataset shape:")
print(train_df.shape)

print("\nTesting dataset shape:")
print(test_df.shape)


print("\nTraining class distribution:")
print(
    train_df["label"]
    .value_counts()
)

print("\nTraining percentages:")
print(
    train_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nTesting class distribution:")
print(
    test_df["label"]
    .value_counts()
)

print("\nTesting percentages:")
print(
    test_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==========================================
# 10. CREATE OUTPUT FOLDER
# ==========================================

TRAIN_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 11. SAVE TRAIN / TEST DATA
# ==========================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ==========================================
# 12. FINAL MESSAGE
# ==========================================

print("\n======================================")
print("TRAIN / TEST SPLIT COMPLETED")
print("======================================")

print("\nTraining dataset saved to:")
print(TRAIN_FILE)

print("\nTesting dataset saved to:")
print(TEST_FILE)

print("\nLabel meaning:")
print("0 = legitimate")
print("1 = spam")