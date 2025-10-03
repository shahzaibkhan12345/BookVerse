from fastapi import FastAPI

app = FastAPI(
    title="BookVerse API",
    description="API for the BookVerse recommendation system",
    version="0.1.0",
)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the BookVerse API!"}