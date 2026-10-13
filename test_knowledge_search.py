
from app.knowledge_base import search_knowledge_base


query = "My internet keeps disconnecting and Wi-Fi is unstable"

results = search_knowledge_base(query, top_k=3)

print("\nUser query:")
print(query)

print("\nMost relevant solutions:")

for number, result in enumerate(results, start=1):
    print(f"\nResult {number}")
    print("Category:", result["category"])
    print("Question:", result["question"])
    print("Answer:", result["answer"])
    print("Similarity:", result["similarity_score"])
