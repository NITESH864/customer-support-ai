from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# 1. Load the dataset
data_path = Path("data/tickets.csv")

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()


# 2. Validate required columns
required_columns = ["title", "description", "category"]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if df[required_columns].isna().any().any():
    raise ValueError(
        "Please handle missing values in title, description, and category."
    )


# 3. Prepare text and target label
df["text"] = (
    df["title"].astype(str).str.strip()
    + " "
    + df["description"].astype(str).str.strip()
)

X = df["text"]
y = df["category"].astype(str).str.strip()


# 4. Check whether there is enough data to split
class_counts = y.value_counts()

if len(class_counts) < 2:
    raise ValueError("Classification requires at least two categories.")

if (class_counts < 2).any():
    raise ValueError(
        "Each category needs at least two examples for this split."
    )


# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)


# 6. Create the complete text-classification pipeline
model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)


# 7. Train the model
model.fit(X_train, y_train)


# 8. Make predictions on unseen test examples
y_pred = model.predict(X_test)


# 9. Evaluate the model
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

class_names = sorted(y.unique())

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)

print(
    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    )
)


# 10. Save the complete pipeline
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "ticket_category_pipeline.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved successfully: {model_path}")


# 11. Try one new example
sample_ticket = [
    "Wi-Fi connection problem. "
    "My office laptop cannot connect to the internet."
]

prediction = model.predict(sample_ticket)

print("\nSample ticket:")
print(sample_ticket[0])
print("Predicted category:", prediction[0])