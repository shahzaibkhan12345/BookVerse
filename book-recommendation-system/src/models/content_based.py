# src/models/content_based.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class ContentBasedRecommender:
    """
    A content-based recommender system using TF-IDF and Cosine Similarity.
    """
    def __init__(self):
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.books_df = None
        self.cosine_sim = None

    def train(self, books_with_content_path):
        """
        Trains the content-based model.
        - Loads book data with content.
        - Creates a TF-IDF matrix.
        - Calculates the cosine similarity matrix.
        """
        print("Starting Content-Based Model Training...")
        
        # Load processed data
        self.books_df = pd.read_csv(books_with_content_path)
        self.books_df['content'] = self.books_df['content'].fillna('')

        # Create a TF-IDF Vectorizer
        # max_features limits the vocabulary size to prevent memory issues
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        
        # Construct the TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.books_df['content'])
        
        # Compute the cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print("Content-Based Model Training Complete!")
        print(f"TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")

    def get_recommendations(self, book_title, n=10):
        """
        Gets recommendations for a book based on its content.
        """
        if self.cosine_sim is None:
            raise RuntimeError("Model is not trained yet. Please call train() first.")
            
        # Get the index of the book that matches the title
        if book_title not in self.books_df['title'].values:
            return f"Book '{book_title}' not found in the dataset."
            
        idx = self.books_df[self.books_df['title'] == book_title].index[0]

        # Get the pairwise similarity scores of all books with that book
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar books (excluding the book itself)
        sim_scores = sim_scores[1:n+1]

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar books
        return self.books_df[['title', 'authors', 'image_url']].iloc[book_indices]

    def save_model(self, model_path):
        """Saves the trained model components."""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(os.path.join(model_path, 'tfidf_matrix.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(os.path.join(model_path, 'books_df.pkl'), 'wb') as f:
            pickle.dump(self.books_df, f)
        print(f"Content-Based model artifacts saved to {model_path}")

    def load_model(self, model_path):
        """Loads the trained model components."""
        with open(os.path.join(model_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(os.path.join(model_path, 'tfidf_matrix.pkl'), 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        with open(os.path.join(model_path, 'books_df.pkl'), 'rb') as f:
            self.books_df = pickle.load(f)
        # Re-calculate cosine similarity on load
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Content-Based model artifacts loaded.")
