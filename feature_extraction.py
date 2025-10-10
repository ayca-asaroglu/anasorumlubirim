"""
Feature Extraction Module

This module handles feature extraction for Turkish text classification,
including SBERT embeddings, TF-IDF vectors, and categorical feature encoding.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from config import PREPROCESSING_CONFIG


class FeatureExtractor:
    """Feature extraction class for Turkish text classification."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or PREPROCESSING_CONFIG
        self.tfidf_vectorizer = None
        self.sbert_model = None
        self.label_encoders = {}
        
    def get_sbert_embeddings(self, texts: List[str], cache_file: str = None) -> np.ndarray:
        """
        Generate SBERT embeddings for texts.
        
        Args:
            texts: List of texts to embed
            cache_file: Optional cache file path
            
        Returns:
            SBERT embeddings as numpy array
        """
        cache_file = cache_file or self.config["embedding_cache_file"]
        
        # For API predictions (single text), always compute new embeddings
        if len(texts) == 1:
            print("📦 Computing SBERT embeddings for single text...")
            if self.sbert_model is None:
                self.sbert_model = SentenceTransformer(self.config["sbert_model"])
            embeddings = self.sbert_model.encode(texts, show_progress_bar=False)
            return embeddings
        
        # For training (multiple texts), use cache if available
        if os.path.exists(cache_file):
            print("✅ SBERT embeddings loading from cache...")
            return np.load(cache_file)
        
        print("📦 Computing SBERT embeddings...")
        
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(self.config["sbert_model"])
        
        embeddings = self.sbert_model.encode(texts, show_progress_bar=True)
        np.save(cache_file, embeddings)
        
        return embeddings
    
    def get_tfidf_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Generate TF-IDF features for texts.
        
        Args:
            texts: List of texts to process
            
        Returns:
            TF-IDF features as DataFrame
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not loaded. Please load trained models first.")
        
        # Transform texts using the pre-trained vectorizer
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.tfidf_vectorizer.get_feature_names_out()
        )
        
        return tfidf_df
    
    def get_categorical_features(self, df: pd.DataFrame, 
                               categorical_columns: List[str]) -> pd.DataFrame:
        """
        Generate one-hot encoded features for categorical columns.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            Combined one-hot encoded features as DataFrame
        """
        feature_dfs = []
        
        for col in categorical_columns:
            if col in df.columns:
                # Create one-hot encoding
                dummies = pd.get_dummies(
                    df[col].astype(str).str.strip(),
                    prefix=col
                )
                feature_dfs.append(dummies)
        
        if feature_dfs:
            return pd.concat(feature_dfs, axis=1)
        else:
            return pd.DataFrame()
    
    def encode_labels(self, labels: pd.Series, label_name: str) -> np.ndarray:
        """
        Encode categorical labels to numerical values.
        
        Args:
            labels: Series of categorical labels
            label_name: Name for the label encoder
            
        Returns:
            Encoded labels as numpy array
        """
        if label_name not in self.label_encoders:
            self.label_encoders[label_name] = LabelEncoder()
        
        return self.label_encoders[label_name].fit_transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray, label_name: str) -> np.ndarray:
        """
        Decode numerical labels back to categorical values.
        
        Args:
            encoded_labels: Encoded labels as numpy array
            label_name: Name of the label encoder
            
        Returns:
            Decoded labels as numpy array
        """
        if label_name not in self.label_encoders:
            raise ValueError(f"Label encoder '{label_name}' not found")
        
        return self.label_encoders[label_name].inverse_transform(encoded_labels)
    
    def combine_features(self, feature_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple feature matrices horizontally.
        
        Args:
            feature_matrices: List of feature matrices
            
        Returns:
            Combined feature matrix
        """
        if not feature_matrices:
            return np.array([])
        
        return np.hstack(feature_matrices)
    
    def extract_all_features(self, df: pd.DataFrame, 
                           text_columns: List[str],
                           categorical_columns: List[str],
                           target_column: str = None) -> tuple:
        """
        Extract all features from the DataFrame.
        
        Args:
            df: Input DataFrame
            text_columns: List of text column names
            categorical_columns: List of categorical column names
            target_column: Optional target column name
            
        Returns:
            Tuple of (features_matrix, target_labels, feature_names)
        """
        # Combine text columns
        combined_texts = []
        for col in text_columns:
            if col in df.columns:
                combined_texts.append(df[col])
        
        if combined_texts:
            combined_text = pd.concat(combined_texts, axis=1).apply(
                lambda row: " ".join(row.astype(str)), axis=1
            ).tolist()
        else:
            combined_text = []
        
        # Extract features
        features = []
        feature_names = []
        
        # SBERT embeddings
        if combined_text:
            sbert_embeddings = self.get_sbert_embeddings(combined_text)
            features.append(sbert_embeddings)
            feature_names.append("sbert")
        
        # TF-IDF features
        if combined_text:
            tfidf_df = self.get_tfidf_features(combined_text)
            features.append(tfidf_df.values)
            feature_names.append("tfidf")
        
        # Categorical features
        if categorical_columns:
            cat_features = self.get_categorical_features(df, categorical_columns)
            if not cat_features.empty:
                features.append(cat_features.values)
                feature_names.append("categorical")
        
        # Combine all features
        if features:
            combined_features = self.combine_features(features)
        else:
            combined_features = np.array([])
        
        # Extract target labels if provided
        target_labels = None
        if target_column and target_column in df.columns:
            target_labels = self.encode_labels(df[target_column], target_column)
        
        return combined_features, target_labels, feature_names
