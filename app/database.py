
import sqlite3
from pathlib import Path


# Database file ka path
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "tickets.db"


def get_connection():
    """Database connection create karta hai."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row

    return connection


def get_db_connection():
    """Alternative function name for database connection."""
    return get_connection()


def init_db():
    """Database table create karta hai aur missing columns add karta hai."""

    connection = get_connection()

    try:
        # Tickets table create karo agar pehle se nahi hai
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                sentiment_label TEXT NOT NULL DEFAULT 'UNKNOWN',
                sentiment_score REAL NOT NULL DEFAULT 0.0,
                priority TEXT NOT NULL DEFAULT 'UNKNOWN',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Existing table ke columns check karo
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(tickets)"
            ).fetchall()
        }

        # Sentiment label column missing ho toh add karo
        if "sentiment_label" not in columns:
            connection.execute(
                """
                ALTER TABLE tickets
                ADD COLUMN sentiment_label TEXT NOT NULL DEFAULT 'UNKNOWN'
                """
            )

        # Sentiment score column missing ho toh add karo
        if "sentiment_score" not in columns:
            connection.execute(
                """
                ALTER TABLE tickets
                ADD COLUMN sentiment_score REAL NOT NULL DEFAULT 0.0
                """
            )

        # Priority column missing ho toh add karo
        if "priority" not in columns:
            connection.execute(
                """
                ALTER TABLE tickets
                ADD COLUMN priority TEXT NOT NULL DEFAULT 'UNKNOWN'
                """
            )

        connection.commit()
        print("Database initialized successfully!")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()
