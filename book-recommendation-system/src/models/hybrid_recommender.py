# src/models/hybrid_recommender.py

import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filter import CollaborativeFilter

class HybridRecommender:
    """
    A hybrid recommender that combines content-based and collaborative filtering results.
    """
    def __init__(self, models_path):
        self.models_path = models_path
        self.cb_recommender = ContentBasedRecommender()
        self.cf_recommender = CollaborativeFilter()
        self.books_df = None

    def load_models(self):
        """
        Loads the pre-trained content-based and collaborative filtering models.
        """
        print("Loading Hybrid Recommender models...")
        
        # Load Content-Based Model
        cb_model_path = os.path.join(self.models_path, 'content_based')
        self.cb_recommender.load_model(cb_model_path)
        self.books_df = self.cb_recommender.books_df

        # Load Collaborative Filtering Model
        cf_model_path = os.path.join(self.models_path, 'collaborative_filter')
        self.cf_recommender.load_model(cf_model_path)
        
        print("All models loaded successfully.")

    def get_hybrid_recommendations(self, user_id, liked_book_title, n=10, cf_weight=0.6):
        """
        Generates hybrid recommendations for a user.
        """
        if self.books_df is None:
            raise RuntimeError("Models are not loaded. Please call load_models() first.")

        # --- 1. Get Collaborative Filtering Recommendations ---
        cf_recs = self.cf_recommender.get_user_recommendations(user_id, n=n*2)
        
        # --- 2. Get Content-Based Recommendations ---
        cb_recs_df = self.cb_recommender.get_recommendations(liked_book_title, n=n*2)
        cb_recs = cb_recs_df['title'].tolist()

        # --- 3. Combine the Recommendations ---
        combined_scores = {}

        # Add scores from CF
        for i, book_id in enumerate(cf_recs):
            # Use the mapping from the CF model to get the title directly
            book_title = self.cf_recommender.index_to_book_map.get(book_id)
            if book_title:
                combined_scores[book_title] = combined_scores.get(book_title, 0) + (len(cf_recs) - i) * cf_weight

        # Add scores from CB
        for i, book_title in enumerate(cb_recs):
            combined_scores[book_title] = combined_scores.get(book_title, 0) + (len(cb_recs) - i) * (1 - cf_weight)

        # --- 4. Sort and Return Final Recommendations ---
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        final_recs_titles = [rec[0] for rec in sorted_recs[:n]]
        
        # --- THIS IS THE FIX ---
        # Filter the final list to only include titles that exist in our main books_df
        valid_titles = [title for title in final_recs_titles if title in self.books_df['title'].values]
        
        if not valid_titles:
            return pd.DataFrame(columns=['title', 'authors', 'image_url']) # Return empty df if no valid titles

        # Return the full book details for the valid recommendations
        final_recs_df = self.books_df[self.books_df['title'].isin(valid_titles)].set_index('title')
        
        # Re-index to preserve the original sorted order, but only for valid titles
        final_recs_df = final_recs_df.loc[valid_titles]
        # --- END OF FIX ---

        return final_recs_df[['authors', 'image_url']].reset_index()