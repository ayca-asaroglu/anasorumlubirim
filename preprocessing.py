"""
Turkish Hierarchical Text Classification Model

This module contains text preprocessing utilities for Turkish text classification.
Includes text cleaning, stopword removal, and normalization functions.
"""

import re
import pandas as pd
from typing import List, Set
from config import TURKISH_STOPWORDS


class TextPreprocessor:
    """Text preprocessing class for Turkish text classification."""
    
    def __init__(self, stopwords: Set[str] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            stopwords: Set of stopwords to remove. If None, uses default Turkish stopwords.
        """
        self.stopwords = stopwords or set(TURKISH_STOPWORDS)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Turkish text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        
        # Keep only Turkish characters and whitespace
        text = re.sub(r"[^\w\sğüşıöçİĞÜŞÖÇ]", " ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove stopwords
        words = text.split()
        filtered_words = [w for w in words if w not in self.stopwords]
        
        return " ".join(filtered_words)
    
    def clean_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Clean text columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names containing text to clean
            
        Returns:
            DataFrame with cleaned text columns
        """
        df_cleaned = df.copy()
        
        for col in text_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].apply(self.clean_text)
        
        return df_cleaned
    
    def filter_classes_by_min_samples(self, df: pd.DataFrame, class_column: str, 
                                    min_samples: int = 30) -> pd.DataFrame:
        """
        Filter DataFrame to keep only classes with minimum number of samples.
        
        Args:
            df: Input DataFrame
            class_column: Name of the class column
            min_samples: Minimum number of samples per class
            
        Returns:
            Filtered DataFrame
        """
        class_counts = df[class_column].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        
        return df[df[class_column].isin(valid_classes)].reset_index(drop=True)
