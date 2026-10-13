
import joblib
from pathlib import Path

# Locate saved model files
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

print("=" * 40)
print("       EMAIL SPAM CLASSIFIER")
print("=" * 40)
print("Enter a message to check it.")
print("Type 'exit' to close the program.\n")

while True:
    message = input("Enter message: ").strip()

    if message.lower() == "exit":
        print("Friday signing off. Goodbye!")
        break

    if not message:
        print("Please enter a message.\n")
        continue

    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    confidence = probabilities[int(prediction)] * 100

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM (LEGITIMATE)"

    print(f"\nPrediction: {result}")
    print(f"Model confidence: {confidence:.2f}%")
    print("-" * 40, "\n")
