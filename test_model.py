import joblib

model = joblib.load(
    "models/ticket_category_pipeline.joblib"
)

tickets = [
    "My account is locked and I cannot log in.",
    "The office Wi-Fi is not connecting.",
    "My laptop keyboard is broken."
]

predictions = model.predict(tickets)

for ticket, category in zip(tickets, predictions):
    print("Ticket:", ticket)
    print("Predicted category:", category)
    print("-" * 40) 