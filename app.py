import os
from pathlib import Path
import joblib
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from database import (
    get_analytics,
    get_history,
    initialize_database,
    save_prediction,
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Initialize Flask with frontend as the static directory
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# Load trained models
model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

# Initialize database
initialize_database()


# ---------------------------------------------------------------------------
# Frontend Serving Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serve the frontend single-page application."""
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        "status": "healthy",
        "message": "Email Spam Classifier API is running!"
    })


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Please provide a valid message."
        }), 400

    # Convert message into TF-IDF features
    features = vectorizer.transform([message])

    # Predict SPAM or HAM
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction]) * 100

    prediction_label = "SPAM" if prediction == 1 else "HAM"

    # Identify words present in the trained vocabulary
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

    # Save prediction to SQLite
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


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
