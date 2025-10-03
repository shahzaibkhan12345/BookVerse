# src/models/collaborative_filter.py

import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import os

class CollaborativeFilter:
    """
    A collaborative filtering recommender using Scikit-learn's TruncatedSVD.
    """
    def __init__(self, n_components=100):
        self.model = None
        self.ratings_df = None
        self.interaction_matrix = None
        self.user_to_index_map = None
        self.book_to_index_map = None
        self.index_to_book_map = None
        self.n_components = n_components

    def train(self, ratings_path):
        """
        Trains the SVD model on user ratings."""
        print("Starting Collaborative Filtering Model Training (with Scikit-learn)...")
        
        # Load processed data
        self.ratings_df = pd.read_csv(ratings_path)

        # --- THIS IS THE FIX ---
        # Group by user and book, and take the mean of their ratings to handle duplicates
        print("Aggregating duplicate ratings...")
        ratings_agg = self.ratings_df.groupby(['user_id', 'book_id'])['rating'].mean().reset_index()
        print("Duplicates handled.")
        # --- END OF FIX ---

        # Create the user-item interaction matrix from the aggregated data
        interaction_matrix_df = ratings_agg.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
        
        # Create mappings to convert between IDs and matrix indices
        self.user_to_index_map = {user_id: i for i, user_id in enumerate(interaction_matrix_df.index)}
        self.book_to_index_map = {book_id: i for i, book_id in enumerate(interaction_matrix_df.columns)}
        self.index_to_book_map = {i: book_id for book_id, i in self.book_to_index_map.items()}

        # Convert to a sparse matrix format for efficiency
        self.interaction_matrix = csr_matrix(interaction_matrix_df.values)

        # Instantiate and train the SVD algorithm
        self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.model.fit(self.interaction_matrix)
        
        print("Collaborative Filtering Model Training Complete!")
        print(f"Interaction Matrix Shape: {self.interaction_matrix.shape}")
        print(f"SVD Model Components: {self.n_components}")

    def _predict_rating(self, user_id, book_id):
        """
        Internal helper to predict a single rating.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained yet. Please call train() first.")
        
        # Check if user and book exist in our training data
        if user_id not in self.user_to_index_map or book_id not in self.book_to_index_map:
            return np.nan # Return NaN if we can't predict

        user_idx = self.user_to_index_map[user_id]
        book_idx = self.book_to_index_map[book_id]

        # Reconstruct the matrix and get the predicted rating
        # This is one way to do it, but can be slow for many predictions.
        # A faster way is to use the user and item latent features directly.
        user_vec = self.model.transform(self.interaction_matrix[user_idx])
        item_vec = self.model.components_.T[book_idx]
        
        # Predict rating is the dot product of user and item latent vectors
        # We also need to add the global mean and biases for a more accurate prediction
        # For simplicity, we'll use the dot product here.
        return np.dot(user_vec, item_vec)[0]


    def get_user_recommendations(self, user_id, n=10):
        """
        Gets top N recommendations for a given user.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained yet. Please call train() first.")

        if user_id not in self.user_to_index_map:
            print(f"User {user_id} not found in training data. Cannot provide recommendations.")
            return []

        user_idx = self.user_to_index_map[user_id]
        
        # Get the user's latent feature vector
        user_vec = self.model.transform(self.interaction_matrix[user_idx])
        
        # Get all item latent feature vectors
        all_item_vecs = self.model.components_.T
        
        # Calculate the predicted rating for every item by taking the dot product
        # of the user vector with all item vectors
        predicted_ratings = np.dot(all_item_vecs, user_vec.T)
        
        # Get the list of books the user has already rated
        rated_book_ids = self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id'].values
        
        # Create a list of (book_index, predicted_rating) for books not yet rated
        recommendations = []
        for book_idx, rating in enumerate(predicted_ratings):
            book_id = self.index_to_book_map[book_idx]
            if book_id not in rated_book_ids:
                recommendations.append((book_id, rating))
        
        # Sort the recommendations by predicted rating
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top N book IDs
        return [rec[0] for rec in recommendations[:n]]

    def save_model(self, model_path):
        """Saves the trained model and mappings."""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, 'svd_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(model_path, 'cf_mappings.pkl'), 'wb') as f:
            pickle.dump({
                'user_to_index_map': self.user_to_index_map,
                'book_to_index_map': self.book_to_index_map,
                'index_to_book_map': self.index_to_book_map
            }, f)
        print(f"Collaborative Filtering model artifacts saved to {model_path}")

    def load_model(self, model_path):
        """Loads the trained model and mappings."""
        with open(os.path.join(model_path, 'svd_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_path, 'cf_mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            self.user_to_index_map = mappings['user_to_index_map']
            self.book_to_index_map = mappings['book_to_index_map']
            self.index_to_book_map = mappings['index_to_book_map']
        print("Collaborative Filtering model artifacts loaded.")