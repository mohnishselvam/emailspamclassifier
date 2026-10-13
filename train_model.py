
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "spam.csv"

# Load and clean the dataset
df = pd.read_csv(DATASET_PATH, encoding="latin-1")
df = df.iloc[:, :2].copy()
df.columns = ["label", "message"]
df = df.dropna()
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["message"] = df["message"].astype(str)
df = df.drop_duplicates(subset=["label", "message"])

X = df["message"]
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Fit TF-IDF on training data only to prevent data leakage
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "MultinomialNB_alpha_1.0": MultinomialNB(alpha=1.0),
    "MultinomialNB_alpha_0.5": MultinomialNB(alpha=0.5),
    "MultinomialNB_alpha_0.25": MultinomialNB(alpha=0.25),
    "LogisticRegression_balanced": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
}

results = []

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train_vec, y_train)
    training_seconds = time.perf_counter() - start

    predictions = model.predict(X_test_vec)
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "name": name,
        "model": model,
        "accuracy": accuracy_score(y_test, predictions),
        "spam_precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "spam_recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "training_seconds": training_seconds,
    })

print("\n=== MODEL COMPARISON ===")
for result in results:
    print(f"\n{result['name']}")
    print(f"Accuracy:           {result['accuracy'] * 100:.2f}%")
    print(f"Spam precision:     {result['spam_precision'] * 100:.2f}%")
    print(f"Spam recall:        {result['spam_recall'] * 100:.2f}%")
    print(f"False positives:    {result['false_positives']}")
    print(f"False negatives:    {result['false_negatives']}")
    print(f"Training time:      {result['training_seconds']:.4f} seconds")

# Prefer models with fewer false positives, then higher spam recall.
# This prioritizes protecting legitimate messages while tracking spam detection.
best = min(
    results,
    key=lambda r: (
        r["false_positives"],
        -r["spam_recall"],
        -r["accuracy"],
    ),
)

print("\n=== SELECTED MODEL ===")
print(f"Model: {best['name']}")
print(f"Accuracy: {best['accuracy'] * 100:.2f}%")
print(f"False positives: {best['false_positives']}")
print(f"False negatives: {best['false_negatives']}")
print("\nClassification report:")
print(classification_report(
    y_test,
    best["model"].predict(X_test_vec),
    target_names=["HAM", "SPAM"],
    zero_division=0,
))

# Save the selected model and its matching vectorizer
joblib.dump(best["model"], BASE_DIR / "spam_model.pkl")
joblib.dump(vectorizer, BASE_DIR / "tfidf_vectorizer.pkl")

print("Selected model and vectorizer saved successfully.")
