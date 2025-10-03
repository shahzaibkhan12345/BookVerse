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
        Trains the SVD model on user ratings.
        """
        print("Starting Collaborative Filtering Model Training (with Scikit-learn)...")
        
        # Load processed data
        self.ratings_df = pd.read_csv(ratings_path)
        print(f"Loaded {len(self.ratings_df)} ratings.")

        # Group by user and book, and take the mean of their ratings to handle duplicates
        print("Aggregating duplicate ratings...")
        ratings_agg = self.ratings_df.groupby(['user_id', 'book_id'])['rating'].mean().reset_index()
        print(f"Aggregated to {len(ratings_agg)} unique user-book ratings.")

        # Create the user-item interaction matrix from the aggregated data
        print("Creating interaction matrix...")
        interaction_matrix_df = ratings_agg.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
        print("Interaction matrix created successfully.")

        # Create mappings to convert between IDs and matrix indices
        print("Creating user and book mappings...")
        self.user_to_index_map = {user_id: i for i, user_id in enumerate(interaction_matrix_df.index)}
        self.book_to_index_map = {book_id: i for i, book_id in enumerate(interaction_matrix_df.columns)}
        self.index_to_book_map = {i: book_id for book_id, i in self.book_to_index_map.items()}

        # Convert to a sparse matrix format for efficiency
        print("Converting to sparse matrix...")
        self.interaction_matrix = csr_matrix(interaction_matrix_df.values)
        print(f"Sparse matrix created with shape: {self.interaction_matrix.shape}")

        # Instantiate and train the SVD algorithm
        print("Training SVD model...")
        self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.model.fit(self.interaction_matrix)
        
        print("Collaborative Filtering Model Training Complete!")
        print(f"SVD Model Components: {self.n_components}")

    def get_user_recommendations(self, user_id, n=10):
        """
        Gets top N recommendations for a given user.
        """
        if self.model is None or self.interaction_matrix is None or self.ratings_df is None:
            raise RuntimeError("Model is not trained or loaded correctly. Please call train() or load_model() first.")

        if user_id not in self.user_to_index_map:
            print(f"User {user_id} not found in training data. Cannot provide recommendations.")
            return []

        user_idx = self.user_to_index_map[user_id]
        
        # Get the user's latent feature vector
        user_vec = self.model.transform(self.interaction_matrix[user_idx])
        
        # Get all item latent feature vectors
        all_item_vecs = self.model.components_.T
        
        # Calculate the predicted rating for every item
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
        """Saves the trained model and all necessary data."""
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
        # Save the interaction matrix and ratings_df as well
        with open(os.path.join(model_path, 'interaction_matrix.pkl'), 'wb') as f:
            pickle.dump(self.interaction_matrix, f)
        with open(os.path.join(model_path, 'ratings_df.pkl'), 'wb') as f:
            pickle.dump(self.ratings_df, f)
        print(f"Collaborative Filtering model artifacts saved to {model_path}")

    def load_model(self, model_path):
        """Loads the trained model and all necessary data."""
        print(f"Attempting to load CF model from: {model_path}")
        with open(os.path.join(model_path, 'svd_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_path, 'cf_mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            self.user_to_index_map = mappings['user_to_index_map']
            self.book_to_index_map = mappings['book_to_index_map']
            self.index_to_book_map = mappings['index_to_book_map']
        # Load the interaction matrix and ratings_df
        with open(os.path.join(model_path, 'interaction_matrix.pkl'), 'rb') as f:
            self.interaction_matrix = pickle.load(f)
        with open(os.path.join(model_path, 'ratings_df.pkl'), 'rb') as f:
            self.ratings_df = pickle.load(f)
        print("Collaborative Filtering model artifact loaded.")