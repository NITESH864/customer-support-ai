from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.0
        }

    scores = sentiment_analyzer.polarity_scores(text)
    compound = float(scores["compound"])

    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "sentiment_label": label,
        "sentiment_score": round(abs(compound), 4)
    }