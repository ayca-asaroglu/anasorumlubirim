"""
Evaluation and Metrics Module

This module provides comprehensive evaluation metrics and visualization
for the Turkish hierarchical text classification model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    ConfusionMatrixDisplay, classification_report
)
from config import OUTPUT_CONFIG


class ModelEvaluator:
    """Evaluation class for Turkish text classification model."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or OUTPUT_CONFIG
        
    def evaluate_predictions(self, y_true: list, y_pred: list, 
                           model_name: str = "Model") -> dict:
        """
        Evaluate model predictions and return comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": report
        }
    
    def evaluate_hierarchical_predictions(self, 
                                       upper_true: list, upper_pred: list,
                                       lower_true: list, lower_pred: list,
                                       lower_proba: np.ndarray = None) -> dict:
        """
        Evaluate hierarchical predictions for both upper and lower levels.
        
        Args:
            upper_true: True upper-level labels
            upper_pred: Predicted upper-level labels
            lower_true: True lower-level labels
            lower_pred: Predicted lower-level labels
            lower_proba: Lower-level prediction probabilities
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        # Upper-level evaluation
        upper_metrics = self.evaluate_predictions(upper_true, upper_pred, "Upper Level")
        
        # Lower-level evaluation
        lower_metrics = self.evaluate_predictions(lower_true, lower_pred, "Lower Level")
        
        # Top-n accuracy for lower level
        top_n_metrics = {}
        if lower_proba is not None:
            lower_true_encoded = self._encode_labels_for_evaluation(lower_true, lower_pred)
            top_n_metrics = {
                "top_2_accuracy": self._calculate_top_n_accuracy(lower_true_encoded, lower_proba, 2),
                "top_3_accuracy": self._calculate_top_n_accuracy(lower_true_encoded, lower_proba, 3)
            }
        
        return {
            "upper_level": upper_metrics,
            "lower_level": lower_metrics,
            "top_n_metrics": top_n_metrics
        }
    
    def plot_confusion_matrix(self, y_true: list, y_pred: list, 
                            class_names: list, title: str = "Confusion Matrix",
                            save_path: str = None) -> None:
        """
        Plot and display confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Title for the plot
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=self.config["plot_figure_size"])
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=class_names
        )
        
        ax = plt.gca()
        display.plot(
            ax=ax, 
            cmap=self.config["plot_cmap"], 
            xticks_rotation=90, 
            colorbar=True
        )
        
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_class_distribution(self, labels: list, title: str = "Class Distribution") -> None:
        """
        Print class distribution statistics.
        
        Args:
            labels: List of labels
            title: Title for the distribution
        """
        print(f"\n📊 {title}:")
        counts = pd.Series(labels).value_counts().sort_values(ascending=False)
        print(counts)
        print(f"Total samples: {len(labels)}")
        print(f"Number of classes: {len(counts)}")
    
    def print_evaluation_results(self, results: dict) -> None:
        """
        Print formatted evaluation results.
        
        Args:
            results: Dictionary containing evaluation results
        """
        print(f"\n📊 {results['model_name']} Performance:")
        print(f"✔ Accuracy: {results['accuracy']:.4f}")
        print(f"✔ F1 Score: {results['f1_score']:.4f}")
    
    def print_hierarchical_results(self, results: dict) -> None:
        """
        Print formatted hierarchical evaluation results.
        
        Args:
            results: Dictionary containing hierarchical evaluation results
        """
        print("\n" + "="*50)
        print("📊 HIERARCHICAL MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Upper level results
        print(f"\n📊 Upper Level Performance:")
        print(f"✔ Accuracy: {results['upper_level']['accuracy']:.4f}")
        print(f"✔ F1 Score: {results['upper_level']['f1_score']:.4f}")
        
        # Lower level results
        print(f"\n📊 Lower Level Performance:")
        print(f"✔ Accuracy: {results['lower_level']['accuracy']:.4f}")
        print(f"✔ F1 Score: {results['lower_level']['f1_score']:.4f}")
        
        # Top-n accuracy
        if results['top_n_metrics']:
            print(f"\n📊 Top-N Accuracy:")
            for metric, value in results['top_n_metrics'].items():
                print(f"✔ {metric.replace('_', ' ').title()}: {value:.4f}")
    
    def create_results_dataframe(self, df_test: pd.DataFrame,
                               upper_preds: list, lower_preds: list,
                               top3_preds: list = None) -> pd.DataFrame:
        """
        Create results DataFrame for export.
        
        Args:
            df_test: Test DataFrame
            upper_preds: Upper-level predictions
            lower_preds: Lower-level predictions
            top3_preds: Optional top-3 predictions
            
        Returns:
            Results DataFrame
        """
        results_df = pd.DataFrame({
            "Gerçek Üst Birim": df_test["anaSorumluBirimUstBirim"],
            "Tahmin Üst Birim": upper_preds,
            "Gerçek Alt Birim": df_test["AnaSorumluBirim_Duzenlenmis"],
            "Tahmin Alt Birim": lower_preds
        })
        
        if top3_preds:
            results_df["Alt Tahmin 1"] = [p[0] if len(p) > 0 else None for p in top3_preds]
            results_df["Alt Tahmin 2"] = [p[1] if len(p) > 1 else None for p in top3_preds]
            results_df["Alt Tahmin 3"] = [p[2] if len(p) > 2 else None for p in top3_preds]
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, filename: str = None) -> None:
        """
        Save results to Excel file.
        
        Args:
            results_df: Results DataFrame
            filename: Optional filename, uses default if None
        """
        filename = filename or self.config["results_file"]
        results_df.to_excel(filename, index=False)
        print(f"📁 Results saved to: {filename}")
    
    def _encode_labels_for_evaluation(self, y_true: list, y_pred: list) -> np.ndarray:
        """Encode labels for evaluation purposes."""
        all_labels = list(set(y_true + y_pred))
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        return np.array([label_to_idx[label] for label in y_true])
    
    def _calculate_top_n_accuracy(self, y_true: np.ndarray, 
                                y_proba: np.ndarray, n: int) -> float:
        """Calculate top-n accuracy."""
        top_n_preds = np.argsort(y_proba, axis=1)[:, -n:]
        return np.mean([y_true[i] in top_n_preds[i] for i in range(len(y_true))])
