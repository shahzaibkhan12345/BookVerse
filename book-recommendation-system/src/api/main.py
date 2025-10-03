# src/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.hybrid_recommender import HybridRecommender
from src.api.schemas import RecommendationRequest, RecommendationResponse

# --- Initialize FastAPI App ---
app = FastAPI(
    title="BookVerse API",
    description="API for the BookVerse hybrid book recommendation system.",
    version="1.0.0",
)

# --- Load the Hybrid Model ---
# This is done once when the application starts
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
hybrid_model = HybridRecommender(models_path=MODELS_PATH)
hybrid_model.load_models()
books_df = hybrid_model.books_df

# --- Mount Static Files and Templates ---
# This allows FastAPI to serve your frontend files
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")

# --- API Endpoints ---

@app.get("/", tags=["Root"])
async def read_root():
    """Serve the main index.html file."""
    return FileResponse(os.path.join(PROJECT_ROOT, "templates", "index.html"))

@app.get("/books/search", tags=["Books"])
@app.get("/books/search", tags=["Books"])
async def search_books(q: str, limit: int = 10):
    """
    Search for books by title to use as a seed for recommendations.
    """
    # Use regex=False to handle special characters in titles like parentheses
    results = books_df[books_df['title'].str.contains(q, case=False, na=False, regex=False)]
    if results.empty:
        raise HTTPException(status_code=404, detail="No books found matching the query.")
    
    # Return a list of books with title and id
    return results.head(limit)[['book_id', 'title']].to_dict(orient='records')

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Get hybrid book recommendations for a user.
    """
    user_id = request.user_id
    liked_book_title = request.liked_book_title

    # Validate that the liked book exists
    if liked_book_title not in books_df['title'].values:
        raise HTTPException(status_code=404, detail=f"Book '{liked_book_title}' not found.")
    
    # Validate that the user exists in our CF data
    if user_id not in hybrid_model.cf_recommender.user_to_index_map:
        raise HTTPException(status_code=404, detail=f"User ID '{user_id}' not found in training data.")
        
    # Get recommendations from our hybrid model
    recommendations_df = hybrid_model.get_hybrid_recommendations(
        user_id=user_id,
        liked_book_title=liked_book_title,
        n=request.count,
        cf_weight=request.cf_weight
    )
    
    # Convert DataFrame to a list of dictionaries for the JSON response
    recommendations = recommendations_df.to_dict(orient='records')
    
    return {"recommendations": recommendations}