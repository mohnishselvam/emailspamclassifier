
import sqlite3
from pathlib import Path

# Store the database in the main project folder.
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "spam_history.db"


def get_connection():
    """Create a connection to the SQLite database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the prediction history table if it doesn't exist."""
    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                prediction TEXT NOT NULL
                    CHECK (prediction IN ('SPAM', 'HAM')),
                confidence REAL NOT NULL
                    CHECK (confidence >= 0 AND confidence <= 100),
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')
                )
            )
        """)
        connection.commit()
    finally:
        connection.close()


def save_prediction(message, prediction, confidence):
    """Save a prediction and return its database ID."""
    if prediction not in ("SPAM", "HAM"):
        raise ValueError("Prediction must be SPAM or HAM.")

    confidence = float(confidence)
    if not 0 <= confidence <= 100:
        raise ValueError("Confidence must be between 0 and 100.")

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO prediction_history
                (message, prediction, confidence)
            VALUES (?, ?, ?)
            """,
            (message, prediction, confidence),
        )
        connection.commit()
        return cursor.lastrowid
    finally:
        connection.close()


def get_history(limit=50, offset=0):
    """Return recent predictions, newest first."""
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT id, message, prediction, confidence, created_at
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_analytics():
    """Return summary statistics for the dashboard."""
    connection = get_connection()

    try:
        row = connection.execute("""
            SELECT
                COUNT(*) AS total_scans,
                SUM(CASE WHEN prediction = 'SPAM'
                    THEN 1 ELSE 0 END) AS spam_count,
                SUM(CASE WHEN prediction = 'HAM'
                    THEN 1 ELSE 0 END) AS ham_count,
                ROUND(AVG(confidence), 2) AS average_confidence
            FROM prediction_history
        """).fetchone()

        return {
            "total_scans": row["total_scans"],
            "spam_count": row["spam_count"] or 0,
            "ham_count": row["ham_count"] or 0,
            "average_confidence": row["average_confidence"] or 0,
        }
    finally:
        connection.close()


def clear_history():
    """Delete all stored predictions."""
    connection = get_connection()

    try:
        connection.execute("DELETE FROM prediction_history")
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()
    print("SpamGuard database initialized successfully!")
    print(f"Database location: {DATABASE_PATH}")

