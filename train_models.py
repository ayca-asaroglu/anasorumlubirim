"""
Model Training and Saving Script

This script trains the models and saves them for API deployment.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import HierarchicalClassifier
from config import DATA_CONFIG, PREPROCESSING_CONFIG, MODEL_CONFIG


def train_and_save_models():
    """Train models and save them for API deployment."""
    print("🚀 Starting model training and saving...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Load and preprocess data
    print("📂 Loading data...")
    df = pd.read_csv(
        DATA_CONFIG["csv_file"], 
        delimiter=DATA_CONFIG["delimiter"], 
        encoding=DATA_CONFIG["encoding"]
    )
    
    df = df[DATA_CONFIG["required_columns"]].dropna()
    
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
    
    # Extract features
    print("🔧 Extracting features...")
    feature_extractor = FeatureExtractor()
    
    categorical_columns = [
        "EtkilenecekKanallar", "talep_tipi", "talep_alt_tipi", 
        "reporterBirim", "reporterDirektorluk"
    ]
    
    X_base, _, _ = feature_extractor.extract_all_features(
        df_filtered,
        text_columns=["SUMMARY", "description"],
        categorical_columns=categorical_columns,
        fit_vectorizer=True
    )
    
    # Create upper-level one-hot features
    upper_onehot = pd.get_dummies(df_filtered["anaSorumluBirimUstBirim"], prefix="ust")
    
    # Create categorical features to save column structure
    categorical_columns = [
        "EtkilenecekKanallar", "talep_tipi", "talep_alt_tipi", 
        "reporterBirim", "reporterDirektorluk"
    ]
    cat_features = feature_extractor.get_categorical_features(df_filtered, categorical_columns)
    
    print(f"✅ Features extracted: {X_base.shape}")
    
    # Prepare labels
    le_upper = LabelEncoder()
    le_lower = LabelEncoder()
    
    y_upper = le_upper.fit_transform(df_filtered["anaSorumluBirimUstBirim"])
    y_lower = le_lower.fit_transform(df_filtered["AnaSorumluBirim_Duzenlenmis"])
    
    # Train models
    print("🤖 Training models...")
    classifier = HierarchicalClassifier()
    
    # Train upper-level model
    print("📚 Training upper-level model...")
    upper_results = classifier.train_upper_level_model(X_base, y_upper, le_upper)
    
    # Train lower-level model
    print("📚 Training lower-level model...")
    lower_results = classifier.train_lower_level_model(
        X_base, y_lower, le_lower, upper_onehot.values
    )
    
    print("✅ Models trained successfully")

    # ================= Hierarchical evaluation on a held-out test split =================
    # Use the upper model's test split indices to define a consistent evaluation set
    test_idx = upper_results["test_indices"]
    df_test = df_filtered.iloc[test_idx]
    X_test = X_base[test_idx]

    # Ground-truth labels (strings) for the test set
    y_upper_true_str = df_test["anaSorumluBirimUstBirim"].values
    y_lower_true_str = df_test["AnaSorumluBirim_Duzenlenmis"].values

    # End-to-end (upper prediction feeds lower) evaluation
    hier_preds = classifier.predict_hierarchical(
        X_test=X_test,
        df_test=df_test,
        upper_class_column="anaSorumluBirimUstBirim",
        lower_class_column="AnaSorumluBirim_Duzenlenmis"
    )
    upper_pred_str = hier_preds["upper_predictions"]
    lower_pred_str = hier_preds["lower_predictions"]

    # Exact-match (both upper and lower must be correct)
    exact_match = (upper_pred_str == y_upper_true_str) & (lower_pred_str == y_lower_true_str)
    e2e_accuracy = float(np.mean(exact_match))
    # Joint F1 over combined labels
    from sklearn.metrics import f1_score, accuracy_score
    joint_true = [f"{u}__{l}" for u, l in zip(y_upper_true_str, y_lower_true_str)]
    joint_pred = [f"{u}__{l}" for u, l in zip(upper_pred_str, lower_pred_str)]
    e2e_f1 = f1_score(joint_true, joint_pred, average="weighted")

    # Lower-oracle (provide true upper as one-hot to lower model) on the same test split
    upper_true_onehot = pd.get_dummies(pd.Series(y_upper_true_str), prefix="ust")
    # Ensure all upper classes exist and order matches encoder
    for class_name in classifier.upper_label_encoder.classes_:
        col = f"ust_{class_name}"
        if col not in upper_true_onehot.columns:
            upper_true_onehot[col] = 0
    upper_true_onehot = upper_true_onehot[[f"ust_{c}" for c in classifier.upper_label_encoder.classes_]]

    X_lower_input_oracle = np.hstack([X_test, upper_true_onehot.values])
    X_lower_sel_oracle = classifier.lower_selector.transform(X_lower_input_oracle)
    lower_oracle_pred_enc = classifier.lower_model.predict(X_lower_sel_oracle)
    lower_oracle_pred_str = classifier.lower_label_encoder.inverse_transform(lower_oracle_pred_enc)
    lower_oracle_acc = accuracy_score(y_lower_true_str, lower_oracle_pred_str)
    lower_oracle_f1 = f1_score(y_lower_true_str, lower_oracle_pred_str, average="weighted")
    
    # Save models and components
    print("💾 Saving models...")
    
    # Save label encoders
    joblib.dump(le_upper, "models/upper_label_encoder.pkl")
    joblib.dump(le_lower, "models/lower_label_encoder.pkl")
    
    # Save feature extractor components
    joblib.dump(feature_extractor.tfidf_vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(feature_extractor.sbert_model, "models/sbert_model.pkl")
    joblib.dump(cat_features.columns.tolist(), "models/categorical_columns.pkl")
    
    # Save classifier components
    joblib.dump(classifier.upper_model, "models/upper_model.pkl")
    joblib.dump(classifier.lower_model, "models/lower_model.pkl")
    joblib.dump(classifier.upper_selector, "models/upper_selector.pkl")
    joblib.dump(classifier.lower_selector, "models/lower_selector.pkl")
    
    # Save model metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "model_version": "1.0.0",
        "upper_classes": le_upper.classes_.tolist(),
        "lower_classes": le_lower.classes_.tolist(),
        "feature_count": X_base.shape[1],
        "training_samples": len(df_filtered),
        # Split-aware metrics
        "upper_train_accuracy": upper_results["train_accuracy"],
        "upper_train_f1": upper_results["train_f1"],
        "upper_test_accuracy": upper_results.get("test_accuracy"),
        "upper_test_f1": upper_results.get("test_f1"),
        "lower_train_accuracy": lower_results["train_accuracy"],
        "lower_train_f1": lower_results["train_f1"],
        "lower_test_accuracy": lower_results.get("test_accuracy"),
        "lower_test_f1": lower_results.get("test_f1"),
        # Hierarchical metrics on test split
        "e2e_accuracy": e2e_accuracy,
        "e2e_f1": e2e_f1,
        "lower_oracle_accuracy": lower_oracle_acc,
        "lower_oracle_f1": lower_oracle_f1
    }
    
    joblib.dump(metadata, "models/model_metadata.pkl")
    
    print("✅ Models saved successfully!")
    print(f"📊 Upper (train) acc: {upper_results['train_accuracy']:.4f} | f1: {upper_results['train_f1']:.4f}")
    if upper_results.get('test_accuracy') is not None:
        print(f"📊 Upper (test)  acc: {upper_results['test_accuracy']:.4f} | f1: {upper_results['test_f1']:.4f}")
    print(f"📊 Lower (train) acc: {lower_results['train_accuracy']:.4f} | f1: {lower_results['train_f1']:.4f}")
    if lower_results.get('test_accuracy') is not None:
        print(f"📊 Lower (test)  acc: {lower_results['test_accuracy']:.4f} | f1: {lower_results['test_f1']:.4f}")
    print(f"📦 End-to-end exact-match acc: {e2e_accuracy:.4f} | joint F1: {e2e_f1:.4f}")
    print(f"🎯 Lower-oracle acc: {lower_oracle_acc:.4f} | f1: {lower_oracle_f1:.4f}")
    
    return metadata


if __name__ == "__main__":
    metadata = train_and_save_models()
    print("\n🎉 Model training completed!")
    print("📁 Models saved in 'models/' directory")
    print("🚀 Ready for API deployment!")
