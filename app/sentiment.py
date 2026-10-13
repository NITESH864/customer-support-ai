from transformers import pipeline

# Load the pretrained sentiment model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment(text: str) -> dict:
    result = sentiment_analyzer(
        text,
        truncation=True,
        max_length=512
    )[0]

    return {
        "sentiment_label": str(result["label"]),
        "sentiment_score": round(float(result["score"]), 4)
    }