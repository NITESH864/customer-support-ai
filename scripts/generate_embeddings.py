
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_PATH = DATA_DIR / "knowledge_base.csv"
OUTPUT_PATH = DATA_DIR / "knowledge_embeddings.json"

MODEL_NAME = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


def main():
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY missing. Check your .env file."
        )

    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KB_PATH}"
        )

    df = pd.read_csv(KB_PATH)
    required = {"question", "answer", "category"}

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain question, answer, category columns."
        )

    df = df.dropna(
        subset=["question", "answer", "category"]
    ).copy()

    client = genai.Client(api_key=api_key)
    articles = []

    for index, row in df.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        category = str(row["category"]).strip()

        if not question or not answer or not category:
            continue

        document_text = (
            f"title: {question} | "
            f"text: {answer} Category: {category}"
        )

        print(f"Embedding article {len(articles) + 1}: {question}")

        result = client.models.embed_content(
            model=MODEL_NAME,
            contents=document_text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSIONS
            ),
        )

        if not result.embeddings:
            raise RuntimeError(
                f"No embedding returned for row {index}."
            )

        vector = result.embeddings[0].values

        if not vector or len(vector) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Unexpected embedding dimensions for row {index}."
            )

        articles.append({
            "question": question,
            "answer": answer,
            "category": category,
            "embedding": [float(value) for value in vector],
        })

    if not articles:
        raise RuntimeError("No valid articles found.")

    payload = {
        "model": MODEL_NAME,
        "dimensions": EMBEDDING_DIMENSIONS,
        "articles": articles,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = OUTPUT_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(OUTPUT_PATH)

    print(f"\nSUCCESS: Saved {len(articles)} article embeddings.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
