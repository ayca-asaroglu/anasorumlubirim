"""
Main Execution Script for Turkish Hierarchical Text Classification

This script orchestrates the entire pipeline from data loading to model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import HierarchicalClassifier
from evaluation import ModelEvaluator
from config import DATA_CONFIG, PREPROCESSING_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG


def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print("📂 Loading data...")
    
    # Load data
    df = pd.read_csv(
        DATA_CONFIG["csv_file"], 
        delimiter=DATA_CONFIG["delimiter"], 
        encoding=DATA_CONFIG["encoding"]
    )
    
    # Select required columns and drop missing values
    df = df[DATA_CONFIG["required_columns"]].dropna()
    
    print(f"✅ Data loaded: {len(df)} samples")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Clean text columns
    df_cleaned = preprocessor.clean_dataframe(
        df, 
        text_columns=["SUMMARY", "description"]
    )
    
    # Filter classes by minimum samples
    df_filtered = preprocessor.filter_classes_by_min_samples(
        df_cleaned, 
        "anaSorumluBirimUstBirim", 
        PREPROCESSING_CONFIG["min_samples_per_class"]
    )
    
    df_filtered = preprocessor.filter_classes_by_min_samples(
        df_filtered, 
        "AnaSorumluBirim_Duzenlenmis", 
        PREPROCESSING_CONFIG["min_samples_per_class"]
    )
    
    print(f"✅ Data filtered: {len(df_filtered)} samples")
    print("\n📌 Included sub-units:")
    print(df_filtered["AnaSorumluBirim_Duzenlenmis"].value_counts())
    
    return df_filtered


def extract_features(df):
    """Extract features from the dataset."""
    print("\n🔧 Extracting features...")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Define categorical columns for one-hot encoding
    categorical_columns = [
        "EtkilenecekKanallar", "talep_tipi", "talep_alt_tipi", 
        "reporterBirim", "reporterDirektorluk"
    ]
    
    # Extract all features
    X_base, _, _ = feature_extractor.extract_all_features(
        df,
        text_columns=["SUMMARY", "description"],
        categorical_columns=categorical_columns
    )
    
    # Create upper-level one-hot features
    upper_onehot = pd.get_dummies(df["anaSorumluBirimUstBirim"], prefix="ust")
    
    print(f"✅ Features extracted: {X_base.shape}")
    
    return X_base, upper_onehot, feature_extractor


def train_models(X_base, upper_onehot, df):
    """Train hierarchical classification models."""
    print("\n🤖 Training models...")
    
    # Initialize classifier
    classifier = HierarchicalClassifier()
    
    # Prepare labels
    le_upper = LabelEncoder()
    le_lower = LabelEncoder()
    
    y_upper = le_upper.fit_transform(df["anaSorumluBirimUstBirim"])
    y_lower = le_lower.fit_transform(df["AnaSorumluBirim_Duzenlenmis"])
    
    # Train upper-level model
    print("📚 Training upper-level model...")
    upper_results = classifier.train_upper_level_model(X_base, y_upper, le_upper)
    
    print(f"✔ Upper-level Train Accuracy: {upper_results['train_accuracy']:.4f}")
    print(f"✔ Upper-level Train F1 Score: {upper_results['train_f1']:.4f}")
    
    # Train lower-level model
    print("📚 Training lower-level model...")
    lower_results = classifier.train_lower_level_model(
        X_base, y_lower, le_lower, upper_onehot.values
    )
    
    print(f"✔ Lower-level Train Accuracy: {lower_results['train_accuracy']:.4f}")
    print(f"✔ Lower-level Train F1 Score: {lower_results['train_f1']:.4f}")
    print(f"✔ Cross-validation F1 Scores: {np.round(lower_results['cv_scores'], 4)}")
    print(f"✔ Mean CV F1: {lower_results['cv_mean']:.4f}")
    
    return classifier, le_upper, le_lower, upper_results, lower_results


def evaluate_models(classifier, le_upper, le_lower, df, X_base, upper_onehot):
    """Evaluate the trained models."""
    print("\n📊 Evaluating models...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_base, df, df, 
        test_size=MODEL_CONFIG["test_size"],
        stratify=le_lower.transform(df["AnaSorumluBirim_Duzenlenmis"]),
        random_state=MODEL_CONFIG["random_state"]
    )
    
    # Make hierarchical predictions
    predictions = classifier.predict_hierarchical(
        X_test, df_test,
        "anaSorumluBirimUstBirim",
        "AnaSorumluBirim_Duzenlenmis"
    )
    
    # Evaluate predictions
    evaluation_results = evaluator.evaluate_hierarchical_predictions(
        df_test["anaSorumluBirimUstBirim"].tolist(),
        predictions["upper_predictions"],
        df_test["AnaSorumluBirim_Duzenlenmis"].tolist(),
        predictions["lower_predictions"],
        predictions["lower_probabilities"]
    )
    
    # Print results
    evaluator.print_hierarchical_results(evaluation_results)
    
    # Print class distributions
    evaluator.print_class_distribution(
        le_upper.inverse_transform(y_train), 
        "Upper Level Training Data Class Distribution"
    )
    
    evaluator.print_class_distribution(
        le_upper.inverse_transform(y_test), 
        "Upper Level Test Data Class Distribution"
    )
    
    evaluator.print_class_distribution(
        le_lower.inverse_transform(y_train), 
        "Lower Level Training Data Class Distribution"
    )
    
    evaluator.print_class_distribution(
        le_lower.inverse_transform(y_test), 
        "Lower Level Test Data Class Distribution"
    )
    
    # Create top-3 predictions
    top3_indices = np.argsort(predictions["lower_probabilities"], axis=1)[:, -3:]
    top3_preds = [le_lower.inverse_transform(top[::-1]) for top in top3_indices]
    
    # Create results DataFrame
    results_df = evaluator.create_results_dataframe(
        df_test,
        predictions["upper_predictions"],
        predictions["lower_predictions"],
        top3_preds
    )
    
    # Save results
    evaluator.save_results(results_df)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        df_test["AnaSorumluBirim_Duzenlenmis"].tolist(),
        predictions["lower_predictions"],
        le_lower.classes_,
        "Alt Birim Confusion Matrix"
    )
    
    return evaluation_results, results_df


def main():
    """Main execution function."""
    print("🚀 Starting Turkish Hierarchical Text Classification Pipeline")
    print("="*60)
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Extract features
        X_base, upper_onehot, feature_extractor = extract_features(df)
        
        # Train models
        classifier, le_upper, le_lower, upper_results, lower_results = train_models(
            X_base, upper_onehot, df
        )
        
        # Evaluate models
        evaluation_results, results_df = evaluate_models(
            classifier, le_upper, le_lower, df, X_base, upper_onehot
        )
        
        print("\n🎉 Pipeline completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
