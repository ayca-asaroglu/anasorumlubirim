"""
Hierarchical Classification Models

This module implements hierarchical classification models for Turkish text classification,
including upper-level and lower-level classification with proper feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from config import MODEL_CONFIG


class HierarchicalClassifier:
    """Hierarchical classification system for Turkish text classification."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the hierarchical classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.upper_model = None
        self.lower_model = None
        self.upper_selector = None
        self.lower_selector = None
        self.upper_label_encoder = None
        self.lower_label_encoder = None
        
    def train_upper_level_model(self, X: np.ndarray, y: np.ndarray, 
                              label_encoder) -> dict:
        """
        Train the upper-level classification model.
        
        Args:
            X: Feature matrix
            y: Target labels (encoded)
            label_encoder: Label encoder for upper-level classes
            
        Returns:
            Dictionary containing training results
        """
        self.upper_label_encoder = label_encoder
        
        # Create stratified train/test split using indices to avoid leakage
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y,
            test_size=self.config["test_size"],
            stratify=y,
            random_state=self.config["random_state"]
        )

        # Fit selector ONLY on training data to prevent leakage
        self.upper_selector = SelectKBest(
            score_func=f_classif,
            k=self.config["feature_selection_k"]
        )
        X_train_selected = self.upper_selector.fit_transform(X[train_idx], y_train)
        X_test_selected = self.upper_selector.transform(X[test_idx])
        
        # Train model
        self.upper_model = CalibratedClassifierCV(
            LinearSVC(
                class_weight='balanced', 
                max_iter=self.config["svm_max_iter"]
            ), 
            cv=3
        )
        self.upper_model.fit(X_train_selected, y_train)
        
        # Evaluate
        y_train_pred = self.upper_model.predict(X_train_selected)
        train_acc = self._calculate_accuracy(y_train, y_train_pred)
        train_f1 = self._calculate_f1_score(y_train, y_train_pred)

        # Test metrics
        y_test_pred = self.upper_model.predict(X_test_selected)
        test_acc = self._calculate_accuracy(y_test, y_test_pred)
        test_f1 = self._calculate_f1_score(y_test, y_test_pred)
        
        return {
            "train_indices": train_idx,
            "test_indices": test_idx,
            "y_train": y_train,
            "y_test": y_test,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1
        }
    
    def train_lower_level_model(self, X: np.ndarray, y: np.ndarray, 
                              label_encoder, upper_features: np.ndarray = None) -> dict:
        """
        Train the lower-level classification model.
        
        Args:
            X: Feature matrix
            y: Target labels (encoded)
            label_encoder: Label encoder for lower-level classes
            upper_features: Optional upper-level features to include
            
        Returns:
            Dictionary containing training results
        """
        self.lower_label_encoder = label_encoder
        
        # Combine features with upper-level features if provided (aligned by rows)
        if upper_features is not None:
            X_combined = np.hstack([X, upper_features])
        else:
            X_combined = X

        # Create stratified train/test split using indices to avoid leakage
        num_samples = X_combined.shape[0]
        indices = np.arange(num_samples)
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y,
            test_size=self.config["test_size"],
            stratify=y,
            random_state=self.config["random_state"]
        )

        # Feature selection ONLY on training data to prevent leakage
        self.lower_selector = SelectKBest(
            score_func=f_classif, 
            k=self.config["feature_selection_k"]
        )
        X_train_selected = self.lower_selector.fit_transform(X_combined[train_idx], y_train)
        X_test_selected = self.lower_selector.transform(X_combined[test_idx])
        
        # Train model
        self.lower_model = LogisticRegression(
            max_iter=self.config["logistic_max_iter"],
            class_weight="balanced"
        )
        self.lower_model.fit(X_train_selected, y_train)
        
        # Evaluate
        y_train_pred = self.lower_model.predict(X_train_selected)
        train_acc = self._calculate_accuracy(y_train, y_train_pred)
        train_f1 = self._calculate_f1_score(y_train, y_train_pred)

        # Test metrics
        y_test_pred = self.lower_model.predict(X_test_selected)
        test_acc = self._calculate_accuracy(y_test, y_test_pred)
        test_f1 = self._calculate_f1_score(y_test, y_test_pred)
        
        return {
            "train_indices": train_idx,
            "test_indices": test_idx,
            "y_train": y_train,
            "y_test": y_test,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1
        }
    
    def predict_hierarchical(self, X_test: np.ndarray, 
                           df_test: pd.DataFrame,
                           upper_class_column: str,
                           lower_class_column: str) -> dict:
        """
        Make hierarchical predictions using both upper and lower level models.
        
        Args:
            X_test: Test feature matrix
            df_test: Test DataFrame
            upper_class_column: Name of upper-level class column
            lower_class_column: Name of lower-level class column
            
        Returns:
            Dictionary containing predictions and metrics
        """
        # Predict upper level
        X_test_upper = self.upper_selector.transform(X_test)
        upper_preds_encoded = self.upper_model.predict(X_test_upper)
        upper_preds = self.upper_label_encoder.inverse_transform(upper_preds_encoded)
        
        # Create upper-level one-hot features
        upper_pred_onehot = pd.get_dummies(pd.Series(upper_preds), prefix="ust")
        
        # Ensure all upper-level classes are represented
        all_upper_classes = self.upper_label_encoder.classes_
        for class_name in all_upper_classes:
            col_name = f"ust_{class_name}"
            if col_name not in upper_pred_onehot.columns:
                upper_pred_onehot[col_name] = 0
        
        # Prepare lower-level input
        X_test_lower_input = np.hstack([X_test, upper_pred_onehot.values])
        X_test_lower = self.lower_selector.transform(X_test_lower_input)
        
        # Get lower-level probabilities
        lower_probs = self.lower_model.predict_proba(X_test_lower)
        
        # Filter predictions based on upper-level constraints
        final_preds = []
        for i in range(len(df_test)):
            upper_pred = upper_preds[i]
            
            # Get valid lower classes for this upper class
            valid_lower_classes = df_test[df_test[upper_class_column] == upper_pred][lower_class_column].unique()
            
            # Find indices of valid classes
            valid_indices = []
            for class_name in valid_lower_classes:
                if class_name in self.lower_label_encoder.classes_:
                    idx = np.where(self.lower_label_encoder.classes_ == class_name)[0]
                    if len(idx) > 0:
                        valid_indices.append(idx[0])
            
            # Select best prediction from valid classes
            if valid_indices:
                valid_probs = {idx: lower_probs[i][idx] for idx in valid_indices}
                best_idx = max(valid_probs, key=valid_probs.get)
                final_pred = self.lower_label_encoder.inverse_transform([best_idx])[0]
            else:
                # Fallback to overall best prediction
                best_idx = np.argmax(lower_probs[i])
                final_pred = self.lower_label_encoder.inverse_transform([best_idx])[0]
            
            final_preds.append(final_pred)
        
        return {
            "upper_predictions": upper_preds,
            "lower_predictions": final_preds,
            "lower_probabilities": lower_probs
        }
    
    def calculate_top_n_accuracy(self, y_true: np.ndarray, 
                               y_proba: np.ndarray, n: int) -> float:
        """
        Calculate top-n accuracy.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            n: Number of top predictions to consider
            
        Returns:
            Top-n accuracy score
        """
        top_n_preds = np.argsort(y_proba, axis=1)[:, -n:]
        return np.mean([y_true[i] in top_n_preds[i] for i in range(len(y_true))])
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    
    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="weighted")
