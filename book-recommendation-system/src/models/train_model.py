# src/models/train_model.py

import os
import sys
import logging

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filter import CollaborativeFilter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train and save both the content-based and collaborative filtering models.
    """
    logger.info("🚀 Starting the full model training pipeline...")

    # --- Define Paths ---
    # Use absolute paths for robustness
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_data_path = os.path.join(project_root, 'data', 'processed')
    model_artifacts_path = os.path.join(project_root, 'models')

    # File paths for our processed data
    books_content_path = os.path.join(processed_data_path, 'books_with_content.csv')
    ratings_path = os.path.join(processed_data_path, 'ratings.csv')

    # Check if processed data exists
    if not os.path.exists(books_content_path) or not os.path.exists(ratings_path):
        logger.error("❌ Processed data files not found. Please run the feature engineering notebook first.")
        return

    # --- 1. Train the Content-Based Model ---
    logger.info("📚 Step 1: Training the Content-Based Recommender...")
    try:
        cb_recommender = ContentBasedRecommender()
        cb_recommender.train(books_content_path)
        
        # Save the model artifacts
        cb_model_path = os.path.join(model_artifacts_path, 'content_based')
        cb_recommender.save_model(cb_model_path)
        logger.info("✅ Content-Based model trained and saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to train Content-Based model: {e}")
        return

    # --- 2. Train the Collaborative Filtering Model ---
    logger.info("👥 Step 2: Training the Collaborative Filtering Recommender...")
    try:
        cf_recommender = CollaborativeFilter(n_components=100) # You can tune n_components
        cf_recommender.train(ratings_path)
        
        # Save the model artifacts
        cf_model_path = os.path.join(model_artifacts_path, 'collaborative_filter')
        cf_recommender.save_model(cf_model_path)
        logger.info("✅ Collaborative Filtering model trained and saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to train Collaborative Filtering model: {e}")
        return

    logger.info("🎉 All models have been trained and saved successfully!")
    logger.info(f"Model artifacts are located in: {model_artifacts_path}")


if __name__ == "__main__":
    main()