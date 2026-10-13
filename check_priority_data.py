import pandas as pd

df = pd.read_csv("data/tickets.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nPriority distribution:")
print(df["priority"].value_counts(dropna=False))

print("\nExamples:")
print(df[["title", "description", "priority"]].head(10))