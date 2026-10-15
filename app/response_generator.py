
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Minimum similarity score required before using a knowledge article.
MIN_RELEVANCE_SCORE = 0.65

# Use the configured model, or the model verified in local testing.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def _build_knowledge_base_fallback(
    query: str,
    relevant_articles: list[dict],
) -> str:
    """Build a safe draft using approved knowledge-base information."""

    if not relevant_articles:
        return (
            "Hello, I understand you're experiencing an issue. "
            "I couldn't find a relevant troubleshooting article "
            "in the knowledge base. Please contact the IT help desk "
            "and include a description of the issue and any error message. "
            "Please review this draft before sending."
        )

    article = relevant_articles[0]
    question = str(
        article.get("question", "the reported issue")
    ).strip()
    answer = str(article.get("answer", "")).strip()

    if not answer:
        return (
            "Hello, I understand you're experiencing an issue. "
            "Please contact the IT help desk for further assistance. "
            "Please review this draft before sending."
        )

    return (
        "Hello, I understand this issue is frustrating, "
        "and I'm happy to help. "
        f"For '{question}', our knowledge base recommends: "
        f"{answer} "
        "If the issue continues, please contact the IT help desk "
        "for further assistance.\n\n"
        "Note: Gemini is temporarily unavailable, so this draft "
        "was assembled from the retrieved knowledge-base article. "
        "Please review it before sending."
    )


def _is_transient_gemini_error(error: Exception) -> bool:
    """Identify temporary overload, rate-limit, and network errors."""

    message = str(error).lower()

    markers = (
        "503",
        "unavailable",
        "429",
        "resource_exhausted",
        "rate limit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "500 internal",
        "502",
        "504",
    )

    return any(marker in message for marker in markers)


def generate_support_response(
    query: str,
    knowledge_results: list[dict],
) -> str:
    """Generate a grounded Gemini response with a safe KB fallback."""

    if not query or not query.strip():
        raise ValueError("Customer query cannot be empty.")

    # Keep only articles that meet the configured relevance threshold.
    relevant_articles = [
        article
        for article in knowledge_results
        if float(article.get("similarity_score", 0))
        >= MIN_RELEVANCE_SCORE
    ]

    # Do not ask Gemini to invent an answer without relevant context.
    if not relevant_articles:
        return _build_knowledge_base_fallback(
            query,
            relevant_articles,
        )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Configure it in the "
            "backend environment variables."
        )

    context_parts = []

    for number, article in enumerate(
        relevant_articles,
        start=1,
    ):
        context_parts.append(
            f"Article {number}\n"
            f"Category: {article.get('category', 'General')}\n"
            f"Question: {article.get('question', '')}\n"
            f"Approved information: {article.get('answer', '')}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a professional IT customer support assistant.

Write a clear, polite, concise response to the customer using only the
retrieved knowledge-base information for technical guidance.

CUSTOMER ISSUE:
{query}

RETRIEVED KNOWLEDGE BASE:
{context}

RULES:
1. Do not invent troubleshooting steps or company policies.
2. Ignore articles that are unrelated to the customer's issue.
3. If information is insufficient, recommend contacting the IT help desk.
4. Never ask for passwords, verification codes, or secret keys.
5. Write in a friendly, professional tone in 2 to 5 sentences.
6. Return only the customer-facing response draft.
"""

    model_name = os.getenv(
        "GEMINI_MODEL",
        DEFAULT_GEMINI_MODEL,
    )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        if response.text and response.text.strip():
            return response.text.strip()

        print(
            "Gemini returned an empty response. "
            "Using the knowledge-base fallback."
        )

        return _build_knowledge_base_fallback(
            query,
            relevant_articles,
        )

    except Exception as error:
        # Redact the API key before writing errors to logs.
        error_message = str(error).replace(
            api_key,
            "[REDACTED]",
        )

        print(
            f"Gemini generation error | "
            f"model={model_name} | "
            f"type={type(error).__name__} | "
            f"message={error_message[:300]}"
        )

        if _is_transient_gemini_error(error):
            return _build_knowledge_base_fallback(
                query,
                relevant_articles,
            )

        raise RuntimeError(
            f"Gemini response generation failed: "
            f"{error_message[:300]}"
        ) from error
