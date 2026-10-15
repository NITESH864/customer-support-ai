from pathlib import Path
import json
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.csv"
EMBEDDINGS_PATH = BASE_DIR / "data" / "knowledge_embeddings.json"

MODEL_NAME = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=API_KEY) if API_KEY else None


def normalize_search_text(text: str) -> str:
    """Normalize common Wi-Fi spellings."""
    return re.sub(
        r"\bwi[\s-]?fi\b",
        "wifi",
        str(text),
        flags=re.IGNORECASE,
    )


if not KNOWLEDGE_BASE_PATH.exists():
    raise FileNotFoundError(
        f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}"
    )

knowledge_df = pd.read_csv(KNOWLEDGE_BASE_PATH)

required_columns = {"question", "answer", "category"}

if not required_columns.issubset(knowledge_df.columns):
    raise ValueError(
        "knowledge_base.csv must contain: question, answer, category"
    )

knowledge_df = knowledge_df.dropna(
    subset=["question", "answer", "category"]
).copy()

for column in ["question", "answer", "category"]:
    knowledge_df[column] = (
        knowledge_df[column].astype(str).str.strip()
    )

knowledge_df = knowledge_df[
    (knowledge_df["question"] != "")
    & (knowledge_df["answer"] != "")
    & (knowledge_df["category"] != "")
].reset_index(drop=True)

if knowledge_df.empty:
    raise ValueError("The knowledge base contains no valid records.")


# TF-IDF remains available as a fallback.
knowledge_texts = [
    normalize_search_text(
        f"{row['question']} {row['answer']} {row['category']}"
    )
    for _, row in knowledge_df.iterrows()
]

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
)

tfidf_matrix = tfidf_vectorizer.fit_transform(knowledge_texts)


# Load pre-generated semantic embeddings.
semantic_matrix = None

try:
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as file:
        embedding_data = json.load(file)

    if embedding_data.get("model") != MODEL_NAME:
        raise ValueError("Embedding model does not match.")

    if embedding_data.get("dimensions") != EMBEDDING_DIMENSIONS:
        raise ValueError("Embedding dimensions do not match.")

    saved_articles = embedding_data["articles"]

    if len(saved_articles) != len(knowledge_df):
        raise ValueError(
            "Article count changed. Regenerate knowledge embeddings."
        )

    for index, (_, row) in enumerate(knowledge_df.iterrows()):
        saved = saved_articles[index]

        # Avoid silently using stale or incorrectly ordered embeddings.
        for field in ["question", "answer", "category"]:
            if saved.get(field) != row[field]:
                raise ValueError(
                    "Knowledge base changed. Regenerate embeddings."
                )

    semantic_matrix = np.asarray(
        [article["embedding"] for article in saved_articles],
        dtype=np.float32,
    )

    if semantic_matrix.shape != (
        len(knowledge_df),
        EMBEDDING_DIMENSIONS,
    ):
        raise ValueError("Invalid embedding matrix shape.")

    # Normalize once so search uses cosine similarity.
    norms = np.linalg.norm(semantic_matrix, axis=1, keepdims=True)
    semantic_matrix = semantic_matrix / np.maximum(norms, 1e-12)

    print(
        f"Semantic search ready: {len(knowledge_df)} articles loaded."
    )

except (OSError, ValueError, KeyError, TypeError) as exc:
    semantic_matrix = None
    print(f"Semantic embeddings unavailable: {exc}")
    print("TF-IDF fallback is ready.")


def _tfidf_search(query: str, top_k: int) -> list[dict]:
    """Search with TF-IDF if semantic search is unavailable."""
    query_vector = tfidf_vectorizer.transform(
        [normalize_search_text(query)]
    )

    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
    best_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "question": str(knowledge_df.iloc[int(index)]["question"]),
            "answer": str(knowledge_df.iloc[int(index)]["answer"]),
            "category": str(knowledge_df.iloc[int(index)]["category"]),
            "similarity_score": round(float(scores[index]), 4),
        }
        for index in best_indices
    ]


def search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    normalized_query = normalize_search_text(query.strip())

    # Fall back when the key or saved embeddings are unavailable.
    if semantic_matrix is None or gemini_client is None:
        return _tfidf_search(normalized_query, top_k)

    try:
        result = gemini_client.models.embed_content(
            model=MODEL_NAME,
            contents=normalized_query,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSIONS
            ),
        )

        if not result.embeddings or not result.embeddings[0].values:
            raise ValueError("Gemini returned no query embedding.")

        query_vector = np.asarray(
            result.embeddings[0].values,
            dtype=np.float32,
        )

        if query_vector.size != EMBEDDING_DIMENSIONS:
            raise ValueError("Unexpected query embedding dimensions.")

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            raise ValueError("Query embedding has zero magnitude.")

        query_vector = query_vector / query_norm
        scores = semantic_matrix @ query_vector

        best_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "question": str(knowledge_df.iloc[int(index)]["question"]),
                "answer": str(knowledge_df.iloc[int(index)]["answer"]),
                "category": str(knowledge_df.iloc[int(index)]["category"]),
                "similarity_score": round(
                    float(scores[index]), 4
                ),
            }
            for index in best_indices
        ]

    except Exception as exc:
        print(
            "Semantic search failed; using TF-IDF fallback. "
            f"Error type: {type(exc).__name__}"
        )
        return _tfidf_search(normalized_query, top_k)