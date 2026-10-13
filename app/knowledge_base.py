
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Find the knowledge base CSV file
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.csv"


# --------------------------------------------------
# 2. Load and validate knowledge base
# --------------------------------------------------

if not KNOWLEDGE_BASE_PATH.exists():
    raise FileNotFoundError(
        f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}"
    )

knowledge_df = pd.read_csv(KNOWLEDGE_BASE_PATH)

required_columns = {"question", "answer", "category"}

if not required_columns.issubset(knowledge_df.columns):
    raise ValueError(
        "knowledge_base.csv must contain: "
        "question, answer, category"
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


# --------------------------------------------------
# 3. Load sentence embedding model
# --------------------------------------------------

print("Loading semantic search model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Convert knowledge base text into embeddings
# --------------------------------------------------

knowledge_texts = (
    knowledge_df["question"] + " " + knowledge_df["answer"]
).tolist()

knowledge_embeddings = embedding_model.encode(
    knowledge_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False
)

print(
    f"Knowledge base loaded: {len(knowledge_df)} articles."
)


# --------------------------------------------------
# 5. Search relevant knowledge base articles
# --------------------------------------------------

def search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    # Convert user's query into an embedding
    query_embedding = embedding_model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    # Calculate semantic similarity
    scores = knowledge_embeddings @ query_embedding

    # Get indices of the most similar articles
    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in best_indices:
        row = knowledge_df.iloc[int(index)]

        results.append({
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "category": str(row["category"]),
            "similarity_score": round(float(scores[index]), 4)
        })

    return results
