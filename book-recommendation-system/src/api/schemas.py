# src/api/schemas.py

from pydantic import BaseModel

# Schema for the request body of the /recommend endpoint
class RecommendationRequest(BaseModel):
    user_id: int
    liked_book_title: str
    count: int = 10
    cf_weight: float = 0.6

# Schema for the response body of the /recommend endpoint
class RecommendationResponse(BaseModel):
    recommendations: list[dict]