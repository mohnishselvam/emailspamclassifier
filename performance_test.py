
from pathlib import Path
import time
import statistics
import os
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

# Check that the trained files exist
if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        "Model files not found. Run train_model.py first."
    )

print("Loading SpamGuard model...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Representative test messages
messages = [
    "Congratulations! You have won a free prize. Claim it now!",
    "Hey, are we meeting at the library after class?",
    "URGENT! Your account will be blocked. Verify your password now.",
    "Your appointment is confirmed for tomorrow at 10 AM.",
    "Win cash today! Click this link to claim your reward.",
]

# Warm-up: allow initial loading/first-call overhead to settle
for message in messages:
    model.predict(vectorizer.transform([message]))

# Measure individual end-to-end predictions
latencies_ms = []
predictions = 0
test_runs = 1000

print(f"Running {test_runs} predictions...")

start_all = time.perf_counter()

for i in range(test_runs):
    message = messages[i % len(messages)]

    start = time.perf_counter()
    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]
    elapsed = time.perf_counter() - start

    latencies_ms.append(elapsed * 1000)
    predictions += 1

total_seconds = time.perf_counter() - start_all

# Model file sizes
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
vectorizer_size_mb = os.path.getsize(VECTORIZER_PATH) / (1024 * 1024)
total_size_mb = model_size_mb + vectorizer_size_mb

print("\n========== SPAMGUARD PERFORMANCE REPORT ==========")
print(f"Total predictions:       {predictions}")
print(f"Total elapsed time:      {total_seconds:.4f} seconds")
print(f"Average prediction time: {statistics.mean(latencies_ms):.4f} ms")
print(f"Median prediction time:  {statistics.median(latencies_ms):.4f} ms")
print(f"Fastest prediction:      {min(latencies_ms):.4f} ms")
print(f"Slowest prediction:      {max(latencies_ms):.4f} ms")
print(f"Throughput:              {predictions / total_seconds:.2f} predictions/sec")
print(f"Model file size:         {model_size_mb:.4f} MB")
print(f"Vectorizer file size:    {vectorizer_size_mb:.4f} MB")
print(f"Total saved file size:   {total_size_mb:.4f} MB")
print("==================================================")

print("\nSample predictions:")
for message in messages:
    features = vectorizer.transform([message])
    prediction = int(model.predict(features)[0])
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"[{label}] {message}")
