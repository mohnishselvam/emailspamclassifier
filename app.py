from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import joblib

from database import (
    initialize_database,
    save_prediction,
    get_history,
    get_analytics,
)

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

# Initialize the SQLite database when the application starts.
initialize_database()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Email Spam Classifier API is running!",
        "status": "success"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Please provide a valid message."
        }), 400

    # Convert message into TF-IDF features.
    features = vectorizer.transform([message])

    # Predict SPAM or HAM.
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction]) * 100

    prediction_label = "SPAM" if prediction == 1 else "HAM"

    # Identify words present in the trained vocabulary.
    terms = vectorizer.get_feature_names_out()
    nonzero_indices = features.nonzero()[1]
    keyword_scores = []

    if hasattr(model, "feature_log_prob_"):
        spam_index = list(model.classes_).index(1)
        ham_index = list(model.classes_).index(0)

        for index in nonzero_indices:
            contribution = (
                model.feature_log_prob_[spam_index, index]
                - model.feature_log_prob_[ham_index, index]
            ) * float(features[0, index])

            keyword_scores.append({
                "word": str(terms[index]),
                "score": round(float(contribution), 5)
            })

        keyword_scores.sort(
            key=lambda item: abs(item["score"]),
            reverse=True
        )

    else:
        for index in nonzero_indices:
            keyword_scores.append({
                "word": str(terms[index]),
                "score": round(float(features[0, index]), 5)
            })

        keyword_scores.sort(
            key=lambda item: item["score"],
            reverse=True
        )

    # Save this prediction to SQLite.
    try:
        record_id = save_prediction(
            message=message.strip(),
            prediction=prediction_label,
            confidence=round(confidence, 2)
        )
    except Exception:
        app.logger.exception("Failed to save prediction to SQLite")
        return jsonify({
            "error": "Prediction completed, but saving history failed."
        }), 500

    return jsonify({
        "id": record_id,
        "message": message,
        "prediction": prediction_label,
        "confidence": round(confidence, 2),
        "keywords": keyword_scores[:8]
    })


@app.route("/history", methods=["GET"])
def history():
    try:
        limit = request.args.get("limit", default=50, type=int)
        offset = request.args.get("offset", default=0, type=int)

        if limit is None or offset is None:
            return jsonify({
                "error": "Limit and offset must be integers."
            }), 400

        if limit < 1 or offset < 0:
            return jsonify({
                "error": "Limit must be positive and offset cannot be negative."
            }), 400

        records = get_history(limit=limit, offset=offset)

        return jsonify({
            "status": "success",
            "count": len(records),
            "history": records
        })

    except Exception:
        app.logger.exception("Failed to retrieve prediction history")
        return jsonify({
            "error": "Unable to retrieve prediction history."
        }), 500


@app.route("/analytics", methods=["GET"])
def analytics():
    try:
        statistics = get_analytics()

        return jsonify({
            "status": "success",
            "analytics": statistics
        })

    except Exception:
        app.logger.exception("Failed to retrieve analytics")
        return jsonify({
            "error": "Unable to retrieve analytics."
        }), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )