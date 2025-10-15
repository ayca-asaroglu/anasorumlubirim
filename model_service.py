"""
Model Service for API

This module handles model loading, caching, and prediction logic for the API.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import HierarchicalClassifier
from api_models import PredictionRequest, PredictionResponse
from config import PREPROCESSING_CONFIG, MODEL_CONFIG, RAG_CONFIG

logger = logging.getLogger(__name__)


class ModelService:
    """Service class for managing models and predictions."""
    
    def __init__(self):
        """Initialize the model service."""
        self.preprocessor = None
        self.feature_extractor = None
        self.classifier = None
        self.upper_label_encoder = None
        self.lower_label_encoder = None
        self.is_loaded = False
        self.model_info = {}
        self.rag = None
        
    async def load_models(self):
        """Load all required models and components."""
        try:
            logger.info("🔄 Loading models...")
            
            # Initialize components
            self.preprocessor = TextPreprocessor()
            self.feature_extractor = FeatureExtractor()
            self.classifier = HierarchicalClassifier()
            
            # Load label encoders (these would be saved during training)
            # For now, we'll create dummy encoders - in production, load from files
            await self._load_label_encoders()
            
            # Load pre-trained models (these would be saved during training)
            # For now, we'll create dummy models - in production, load from files
            await self._load_trained_models()
            
            self.is_loaded = True
            logger.info("✅ Models loaded successfully")

            # Initialize optional RAG retriever
            if RAG_CONFIG.get("enabled", False):
                try:
                    from rag_retriever import AppCatalogRetriever
                    self.rag = AppCatalogRetriever().load()
                    logger.info("✅ RAG retriever initialized")
                except Exception as e:
                    logger.warning(f"RAG initialization failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {str(e)}")
            raise
    
    async def _load_label_encoders(self):
        """Load label encoders from saved files."""
        import joblib
        
        try:
            # Try to load from saved files
            self.upper_label_encoder = joblib.load('models/upper_label_encoder.pkl')
            self.lower_label_encoder = joblib.load('models/lower_label_encoder.pkl')
            logger.info("✅ Label encoders loaded from saved files")
        except FileNotFoundError:
            logger.warning("⚠️ Saved label encoders not found, creating dummy encoders")
            # For demo purposes, create dummy encoders
            from sklearn.preprocessing import LabelEncoder
            
            # Dummy upper-level classes
            upper_classes = ['IT', 'HR', 'Finance', 'Operations', 'Marketing']
            self.upper_label_encoder = LabelEncoder()
            self.upper_label_encoder.fit(upper_classes)
            
            # Dummy lower-level classes
            lower_classes = [
                'IT_Support', 'IT_Development', 'IT_Infrastructure',
                'HR_Recruitment', 'HR_Payroll', 'HR_Training',
                'Finance_Accounting', 'Finance_Budgeting',
                'Operations_Logistics', 'Operations_Quality',
                'Marketing_Digital', 'Marketing_Brand'
            ]
            self.lower_label_encoder = LabelEncoder()
            self.lower_label_encoder.fit(lower_classes)
    
    async def _load_trained_models(self):
        """Load trained models from saved files."""
        import joblib
        
        try:
            # Try to load from saved files
            self.classifier.upper_model = joblib.load('models/upper_model.pkl')
            self.classifier.lower_model = joblib.load('models/lower_model.pkl')
            self.classifier.upper_selector = joblib.load('models/upper_selector.pkl')
            self.classifier.lower_selector = joblib.load('models/lower_selector.pkl')
            
            # Load feature extractor components
            self.feature_extractor.tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            self.feature_extractor.sbert_model = joblib.load('models/sbert_model.pkl')
            self.categorical_columns = joblib.load('models/categorical_columns.pkl')
            # Ensure the FeatureExtractor uses the exact categorical schema from training
            self.feature_extractor.categorical_columns = self.categorical_columns
            
            logger.info("✅ Trained models loaded from saved files")
            
        except FileNotFoundError:
            logger.warning("⚠️ Saved models not found, creating dummy models")
            # For demo purposes, create dummy models
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import LinearSVC
            from sklearn.feature_selection import SelectKBest, f_classif
            
            # Create dummy feature selectors
            self.classifier.upper_selector = SelectKBest(f_classif, k=10)
            self.classifier.lower_selector = SelectKBest(f_classif, k=10)
            
            # Create dummy models
            self.classifier.upper_model = LinearSVC()
            self.classifier.lower_model = LogisticRegression()
            
            # Create dummy training data for fitting
            np.random.seed(42)
            X_dummy = np.random.randn(100, 20)
            y_upper_dummy = np.random.randint(0, len(self.upper_label_encoder.classes_), 100)
            y_lower_dummy = np.random.randint(0, len(self.lower_label_encoder.classes_), 100)
            
            # Fit dummy models
            X_upper_selected = self.classifier.upper_selector.fit_transform(X_dummy, y_upper_dummy)
            X_lower_selected = self.classifier.lower_selector.fit_transform(X_dummy, y_lower_dummy)
            
            self.classifier.upper_model.fit(X_upper_selected, y_upper_dummy)
            self.classifier.lower_model.fit(X_lower_selected, y_lower_dummy)
    
    async def predict_single(self, request: PredictionRequest) -> Dict[str, Any]:
        """Predict organizational units for a single text."""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")
        
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                'description': request.text,
                'SUMMARY': request.summary or "",
                'EtkilenecekKanallar': request.etkilenecek_kanallar or "",
                'talep_tipi': request.talep_tipi or "",
                'talep_alt_tipi': request.talep_alt_tipi or "",
                'reporterBirim': request.reporter_birim or "",
                'reporterDirektorluk': request.reporter_direktorluk or ""
            }
            
            df_input = pd.DataFrame([input_data])
            
            # Clean text
            df_cleaned = self.preprocessor.clean_dataframe(
                df_input, 
                text_columns=["SUMMARY", "description"]
            )
            
            # Extract features
            X_base, _, _ = self.feature_extractor.extract_all_features(
                df_cleaned,
                text_columns=["SUMMARY", "description"],
                categorical_columns=[
                    "EtkilenecekKanallar", "talep_tipi", "talep_alt_tipi",
                    "reporterBirim", "reporterDirektorluk"
                ]
            )
            
            # Ensure X_base is 2D and has correct shape
            if X_base.ndim == 1:
                X_base = X_base.reshape(1, -1)
            elif X_base.shape[0] == 0:
                # Handle empty array case
                X_base = np.zeros((1, 100))  # Default feature size
            elif X_base.shape[0] > 1:
                # Take only the first row if multiple rows exist
                X_base = X_base[:1, :]
            
            # Make predictions
            predictions = await self._make_hierarchical_prediction(X_base)

            # Optional RAG retrieval from application catalog
            rag_candidates = []
            if self.rag is not None and (request.summary or request.text):
                try:
                    query_text = ((request.summary or "") + " \n " + (request.text or "")).strip()
                    rag_candidates = self.rag.retrieve(query_text)
                except Exception as e:
                    logger.warning(f"RAG retrieve failed: {e}")
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "upper_level_prediction": predictions["upper_prediction"],
                "lower_level_prediction": predictions["lower_prediction"],
                "upper_level_confidence": predictions["upper_confidence"],
                "lower_level_confidence": predictions["lower_confidence"],
                "top3_predictions": predictions["top3_predictions"],
                "rag_candidates": rag_candidates,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    async def predict_batch(self, request) -> Dict[str, Any]:
        """Predict organizational units for multiple texts."""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")
        
        start_time = time.time()
        predictions = []
        
        try:
            for item in request.texts:
                prediction = await self.predict_single(item)
                predictions.append(prediction)
            
            total_processing_time = (time.time() - start_time) * 1000
            
            return {
                "predictions": predictions,
                "total_processing_time_ms": total_processing_time,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise
    
    async def _make_hierarchical_prediction(self, X_base: np.ndarray) -> Dict[str, Any]:
        """Make hierarchical prediction using the loaded models."""
        try:
            # Predict upper level
            X_upper = self.classifier.upper_selector.transform(X_base)
            upper_pred_encoded = self.classifier.upper_model.predict(X_upper)[0]
            upper_prediction = self.upper_label_encoder.inverse_transform([upper_pred_encoded])[0]
            
            # Get upper level confidence
            if hasattr(self.classifier.upper_model, 'predict_proba'):
                upper_proba = self.classifier.upper_model.predict_proba(X_upper)[0]
                upper_confidence = float(np.max(upper_proba))
            else:
                upper_confidence = 0.8  # Default confidence for SVM
            
            # Create upper-level one-hot features
            upper_onehot = pd.get_dummies(pd.Series([upper_prediction]), prefix="ust")
            
            # Ensure all upper-level classes are represented
            all_upper_classes = self.upper_label_encoder.classes_
            for class_name in all_upper_classes:
                col_name = f"ust_{class_name}"
                if col_name not in upper_onehot.columns:
                    upper_onehot[col_name] = 0
            
            # Prepare lower-level input
            X_lower_input = np.hstack([X_base, upper_onehot.values])
            X_lower = self.classifier.lower_selector.transform(X_lower_input)
            
            # Predict lower level
            lower_pred_encoded = self.classifier.lower_model.predict(X_lower)[0]
            lower_prediction = self.lower_label_encoder.inverse_transform([lower_pred_encoded])[0]
            
            # Get lower level probabilities and confidence
            lower_proba = self.classifier.lower_model.predict_proba(X_lower)[0]
            lower_confidence = float(np.max(lower_proba))
            
            # Get top 3 predictions
            top3_indices = np.argsort(lower_proba)[-3:][::-1]
            top3_predictions = [
                self.lower_label_encoder.inverse_transform([idx])[0] 
                for idx in top3_indices
            ]
            
            return {
                "upper_prediction": upper_prediction,
                "lower_prediction": lower_prediction,
                "upper_confidence": upper_confidence,
                "lower_confidence": lower_confidence,
                "top3_predictions": top3_predictions
            }
            
        except Exception as e:
            logger.error(f"Hierarchical prediction error: {str(e)}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        if not self.is_loaded:
            return {
                "upper_level_classes": [],
                "lower_level_classes": [],
                "model_version": "1.0.0",
                "training_date": None,
                "feature_count": 0,
                "is_loaded": False
            }
        
        return {
            "upper_level_classes": self.upper_label_encoder.classes_.tolist(),
            "lower_level_classes": self.lower_label_encoder.classes_.tolist(),
            "model_version": "1.0.0",
            "training_date": "2024-01-01",  # Would be loaded from model metadata
            "feature_count": 20,  # Would be calculated from actual features
            "is_loaded": True
        }
    
    async def get_upper_classes(self) -> List[str]:
        """Get list of available upper-level classes."""
        if not self.is_loaded:
            return []
        return self.upper_label_encoder.classes_.tolist()
    
    async def get_lower_classes(self) -> List[str]:
        """Get list of available lower-level classes."""
        if not self.is_loaded:
            return []
        return self.lower_label_encoder.classes_.tolist()
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up model service...")
        self.is_loaded = False
