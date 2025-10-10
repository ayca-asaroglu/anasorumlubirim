"""
Example Usage Script

This script demonstrates how to use individual components of the
Turkish Hierarchical Text Classification system.
"""

import pandas as pd
import numpy as np
from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import HierarchicalClassifier
from evaluation import ModelEvaluator


def example_text_preprocessing():
    """Example of text preprocessing."""
    print("🔤 Text Preprocessing Example")
    print("-" * 30)
    
    # Sample Turkish text
    sample_text = "Merhaba, bu bir test metnidir. Lütfen bu talebi değerlendiriniz."
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Clean text
    cleaned_text = preprocessor.clean_text(sample_text)
    
    print(f"Original: {sample_text}")
    print(f"Cleaned:  {cleaned_text}")
    print()


def example_feature_extraction():
    """Example of feature extraction."""
    print("🔧 Feature Extraction Example")
    print("-" * 30)
    
    # Sample data
    sample_data = {
        'text': ['Bu bir test metnidir', 'Başka bir örnek metin'],
        'category': ['A', 'B']
    }
    df = pd.DataFrame(sample_data)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Extract TF-IDF features
    tfidf_features = feature_extractor.get_tfidf_features(df['text'].tolist())
    print(f"TF-IDF features shape: {tfidf_features.shape}")
    
    # Extract categorical features
    cat_features = feature_extractor.get_categorical_features(df, ['category'])
    print(f"Categorical features shape: {cat_features.shape}")
    print()


def example_model_training():
    """Example of model training."""
    print("🤖 Model Training Example")
    print("-" * 30)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Initialize classifier
    classifier = HierarchicalClassifier()
    
    # Create dummy label encoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(['class_0', 'class_1', 'class_2'])
    
    # Train model
    results = classifier.train_upper_level_model(X, y, le)
    
    print(f"Training accuracy: {results['train_accuracy']:.4f}")
    print(f"Training F1 score: {results['train_f1']:.4f}")
    print()


def example_evaluation():
    """Example of model evaluation."""
    print("📊 Model Evaluation Example")
    print("-" * 30)
    
    # Sample predictions
    y_true = ['class_A', 'class_B', 'class_A', 'class_C']
    y_pred = ['class_A', 'class_B', 'class_B', 'class_C']
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate predictions
    results = evaluator.evaluate_predictions(y_true, y_pred, "Example Model")
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print()


def main():
    """Run all examples."""
    print("🚀 Turkish Hierarchical Text Classification - Examples")
    print("=" * 60)
    
    example_text_preprocessing()
    example_feature_extraction()
    example_model_training()
    example_evaluation()
    
    print("✅ All examples completed!")


if __name__ == "__main__":
    main()
