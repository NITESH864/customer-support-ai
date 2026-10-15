
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from .knowledge_base import search_knowledge_base

from .database import get_connection, init_db
from .schemas import TicketCreate, TicketPredictionResponse, TicketResponse
from .sentiment import analyze_sentiment
from .response_generator import (
    MIN_RELEVANCE_SCORE,
    generate_support_response,
)
# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CATEGORY_MODEL_PATH = (
    BASE_DIR / "models" / "ticket_category_pipeline.joblib"
)

PRIORITY_MODEL_PATH = (
    BASE_DIR / "models" / "priority_pipeline.joblib"
)


# --------------------------------------------------
# 2. Load trained machine learning models
# --------------------------------------------------

if not CATEGORY_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Category model not found: {CATEGORY_MODEL_PATH}"
    )

if not PRIORITY_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Priority model not found: {PRIORITY_MODEL_PATH}. "
        "Run train_priority_model.py first."
    )

ticket_model = joblib.load(CATEGORY_MODEL_PATH)
priority_model = joblib.load(PRIORITY_MODEL_PATH)


# --------------------------------------------------
# 3. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI Customer Support Ticket Intelligence System",
    description=(
        "Predict ticket category, priority, and sentiment. "
        "Save and retrieve support tickets."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# 4. Initialize the SQLite database
# --------------------------------------------------

@app.on_event("startup")
def startup_event():
    init_db()


# --------------------------------------------------
# 5. Analyze ticket using ML and sentiment analysis
# --------------------------------------------------

def analyze_ticket(title: str, description: str) -> dict:
    text = f"{title} {description}"

    # Predict ticket category
    category = str(ticket_model.predict([text])[0])

    # Predict ticket priority
    priority = str(priority_model.predict([text])[0])

    # Analyze ticket sentiment
    sentiment = analyze_sentiment(text)

    return {
        "predicted_category": category,
        "priority": priority,
        "sentiment_label": sentiment["sentiment_label"],
        "sentiment_score": sentiment["sentiment_score"]
    }


# --------------------------------------------------
# 6. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Customer Support Ticket API is running."
    }


# --------------------------------------------------
# 7. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# 8. Predict ticket category, priority, and sentiment
# --------------------------------------------------

@app.post(
    "/tickets/predict",
    response_model=TicketPredictionResponse
)
def predict_ticket(ticket: TicketCreate):

    try:
        predictions = analyze_ticket(
            ticket.title,
            ticket.description
        )

        return {
            "title": ticket.title,
            "description": ticket.description,
            **predictions,
            "saved": False
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ticket prediction failed: {str(error)}"
        ) from error


# --------------------------------------------------
# 9. Save a ticket to the database
# --------------------------------------------------

@app.post("/tickets", response_model=TicketResponse)
def create_ticket(ticket: TicketCreate):

    connection = get_connection()

    try:
        # Predict category, priority, and sentiment
        predictions = analyze_ticket(
            ticket.title,
            ticket.description
        )

        # Insert ticket into SQLite
        cursor = connection.execute(
            """
            INSERT INTO tickets (
                title,
                description,
                predicted_category,
                priority,
                sentiment_label,
                sentiment_score
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticket.title,
                ticket.description,
                predictions["predicted_category"],
                predictions["priority"],
                predictions["sentiment_label"],
                predictions["sentiment_score"]
            )
        )

        ticket_id = cursor.lastrowid
        connection.commit()

        # Retrieve the newly saved ticket
        saved_ticket = connection.execute(
            """
            SELECT
                id,
                title,
                description,
                predicted_category,
                priority,
                sentiment_label,
                sentiment_score,
                created_at
            FROM tickets
            WHERE id = ?
            """,
            (ticket_id,)
        ).fetchone()

        if saved_ticket is None:
            raise HTTPException(
                status_code=500,
                detail="Ticket was saved but could not be retrieved."
            )

        return dict(saved_ticket)

    except HTTPException:
        connection.rollback()
        raise

    except Exception as error:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Could not save ticket: {str(error)}"
        ) from error

    finally:
        connection.close()


# --------------------------------------------------
# 10. Get all saved tickets
# --------------------------------------------------

@app.get("/tickets", response_model=list[TicketResponse])
def get_tickets():

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                title,
                description,
                predicted_category,
                priority,
                sentiment_label,
                sentiment_score,
                created_at
            FROM tickets
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve tickets: {str(error)}"
        ) from error

    finally:
        connection.close()


# --------------------------------------------------
# 11. Get a single ticket by ID
# --------------------------------------------------

@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: int):

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                id,
                title,
                description,
                predicted_category,
                priority,
                sentiment_label,
                sentiment_score,
                created_at
            FROM tickets
            WHERE id = ?
            """,
            (ticket_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Ticket not found."
            )

        return dict(row)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve ticket: {str(error)}"
        ) from error

    finally:
        connection.close()

# --------------------------------------------------
# 12. Search the knowledge base
# --------------------------------------------------

@app.get("/knowledge/search")
def knowledge_search(
    q: str = Query(
        ...,
        min_length=1,
        description="Describe the user's issue"
    ),
    top_k: int = Query(
        default=3,
        ge=1,
        le=10
    )
):
    try:
        results = search_knowledge_base(
            query=q,
            top_k=top_k
        )

        return {
            "query": q,
            "results": results
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge search failed: {str(error)}"
        ) from error


# --------------------------------------------------
# 13. Generate an AI customer support response
# --------------------------------------------------

@app.get("/knowledge/response")
def draft_support_response(
    q: str = Query(
        ...,
        min_length=1,
        description="Describe the customer's issue"
    ),
    top_k: int = Query(
        default=3,
        ge=1,
        le=10
    )
):
    try:
        # Step 1: Retrieve relevant knowledge articles.
        articles = search_knowledge_base(
            query=q,
            top_k=top_k
        )

        # Step 2: Generate a response grounded in those articles.
        draft = generate_support_response(
            query=q,
            knowledge_results=articles
        )

        # Return the draft and the sources used.
        relevant_sources = [
            article
            for article in articles
            if article["similarity_score"] >= MIN_RELEVANCE_SCORE
        ]

        return {
            "query": q,
            "draft_response": draft,
            "sources": relevant_sources
        }

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Response generation failed: {str(error)}"
        ) from error
