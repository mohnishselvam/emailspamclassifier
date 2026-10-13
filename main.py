
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load and prepare dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ["label", "message"]
df = df.drop_duplicates().copy()

# 2. Encode labels: ham = 0, spam = 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test_tfidf)

print("\n===== SPAM CLASSIFIER RESULTS =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham (Legitimate)", "Spam"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Test sample messages
samples = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hey, are we meeting for lunch today?"
]

sample_features = vectorizer.transform(samples)
predictions = model.predict(sample_features)

print("\n===== SAMPLE PREDICTIONS =====")
for message, prediction in zip(samples, predictions):
    result = "SPAM" if prediction == 1 else "HAM (LEGITIMATE)"
    print(f"\nMessage: {message}")
    print(f"Prediction: {result}")
