from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.csv"

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

print("Preparing lightweight TF-IDF search...")

knowledge_texts = (
    knowledge_df["question"] + " " + knowledge_df["answer"]
).tolist()

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

knowledge_matrix = vectorizer.fit_transform(knowledge_texts)

print(f"Knowledge base loaded: {len(knowledge_df)} articles.")


def search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    query_vector = vectorizer.transform([query.strip()])

    scores = (
        knowledge_matrix @ query_vector.T
    ).toarray().ravel()

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