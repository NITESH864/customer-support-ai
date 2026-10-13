from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# 1. Load dataset
df = pd.read_csv("data/tickets.csv")
df.columns = df.columns.str.strip()

required = ["title", "description", "priority"]

if not set(required).issubset(df.columns):
    raise ValueError(f"Dataset must contain: {required}")

df = df.dropna(subset=required).copy()

# Remove blank inputs and labels
for column in required:
    df[column] = df[column].astype(str).str.strip()

df = df[
    (df["title"] != "")
    & (df["description"] != "")
    & (df["priority"] != "")
]

# 2. Prepare input and target
df["text"] = df["title"] + " " + df["description"]

X = df["text"]
y = df["priority"].str.title()

print("Priority counts:")
print(y.value_counts())

# 3. Validate data before splitting
if y.nunique() < 2:
    raise ValueError("At least two priority classes are required.")

if (y.value_counts() < 2).any():
    raise ValueError(
        "Each priority class needs at least two examples. "
        "Add more labeled data before training."
    )

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

# 5. Create the pipeline
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

# 6. Train
model.fit(X_train, y_train)

# 7. Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# 8. Save the trained pipeline
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "priority_pipeline.joblib"
joblib.dump(model, model_path)

print(f"\nPriority model saved to: {model_path}")