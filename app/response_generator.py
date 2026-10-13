
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. Load environment variables from .env
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------
# 2. Generate a customer support response using Gemini
# --------------------------------------------------

def generate_support_response(
    query: str,
    knowledge_results: list[dict]
) -> str:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add your API key to the project's .env file."
        )

    if not query or not query.strip():
        raise ValueError("Customer query cannot be empty.")

    # Keep only articles that appear reasonably relevant.
    # This is an initial heuristic, not a calibrated threshold.
    relevant_articles = [
        article
        for article in knowledge_results
        if float(article.get("similarity_score", 0)) >= 0.35
    ]

    # Do not generate a misleading answer without useful context.
    if not relevant_articles:
        return (
            "Sorry you're experiencing this issue. "
            "I couldn't find a sufficiently relevant solution "
            "in our knowledge base. Please contact the IT help "
            "desk and describe the issue and any error message."
        )

    # Prepare context from retrieved knowledge articles.
    context_parts = []

    for number, article in enumerate(relevant_articles, start=1):
        context_parts.append(
            f"Article {number}\n"
            f"Category: {article['category']}\n"
            f"Question: {article['question']}\n"
            f"Approved information: {article['answer']}"
        )

    context = "\n\n".join(context_parts)

    # Create the Gemini client.
    client = genai.Client(api_key=api_key)

    # Instruct Gemini to stay grounded in the knowledge base.
    prompt = f"""
You are a professional IT customer support assistant.

Write a clear, polite, concise response to the customer.

CUSTOMER ISSUE:
{query}

RETRIEVED KNOWLEDGE BASE:
{context}

RULES:
1. Use the retrieved articles as your source of technical guidance.
2. Do not invent troubleshooting steps or company policies.
3. Ignore articles that are unrelated to the customer's issue.
4. If the available information is insufficient, say so and
   recommend contacting the IT help desk.
5. Never ask for passwords, verification codes, or secret keys.
6. Write in a friendly professional tone, in 2 to 5 sentences.
7. Return only the customer-facing response draft.
"""

    # Generate the response.
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    if not response.text or not response.text.strip():
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()
