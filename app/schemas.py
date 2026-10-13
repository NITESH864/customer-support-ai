
from pydantic import BaseModel, Field


class TicketCreate(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class TicketPredictionResponse(TicketCreate):
    predicted_category: str
    priority: str
    sentiment_label: str
    sentiment_score: float
    saved: bool = False


class TicketResponse(TicketCreate):
    id: int
    predicted_category: str
    priority: str
    sentiment_label: str
    sentiment_score: float
    created_at: str
