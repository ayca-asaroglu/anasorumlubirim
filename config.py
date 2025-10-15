# Turkish Hierarchical Text Classification Model Configuration

# Data Configuration
DATA_CONFIG = {
    "csv_file": "0910.csv",
    "delimiter": ";",
    "encoding": "utf-8-sig",
    "required_columns": [
        "SUMMARY", "description", "anaSorumluBirimUstBirim", 
        "EtkilenecekKanallar", "talep_tipi", "talep_alt_tipi", 
        "reporterBirim", "reporterDirektorluk", 
        "AnaSorumluBirim_Duzenlenmis", "EFOR_ANASORUMLU_HARCANAN"
    ]
}

# Text Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "min_samples_per_class": 5,
    "tfidf_max_features": 100,
    "sbert_model": "paraphrase-multilingual-mpnet-base-v2",
    "embedding_cache_file": "sbert_embeddings_pipe11_min30_0910_1.npy"
}

# Model Configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "feature_selection_k": 300,
    "cv_folds": 5,
    "svm_max_iter": 1000,
    "logistic_max_iter": 1000
}

# Output Configuration
OUTPUT_CONFIG = {
    "results_file": "tahmin_ust_alt_model_2609.xlsx",
    "plot_figure_size": (12, 12),
    "plot_cmap": "Greens"
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "cors_origins": ["*"],
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "timeout": 30,  # seconds
    "max_batch_size": 100
}

# Retrieval-Augmented Generation (RAG) Configuration
RAG_CONFIG = {
    "enabled": False,                 # Optional, off by default
    "excel_path": "app_catalog.xlsx", # Application catalog Excel
    "sheet_name": 0,                  # Sheet index or name
    "app_name_col": "UygulamaAdi",
    "app_desc_col": "UygulamaAciklamasi",
    "directorate_col": "Direktorluk",
    "units_col": "Birimler",         # Delimited list or multiple columns
    "top_k": 5                        # Top-k candidates to retrieve
}

# Turkish Stopwords
TURKISH_STOPWORDS = [
    # Basic Turkish stopwords
    "bir", "bu", "ve", "veya", "ile", "de", "da", "için", "gibi", "şu", "şöyle", "o", "ki",
    "ne", "mi", "mı", "mu", "mü", "ise", "ama", "fakat", "ancak", "çünkü", "ya", "en",
    "çok", "az", "daha", "her", "hiç", "bazı", "hangi", "nasıl", "neden", "hem", "olan",
    "olur", "olacak", "olmuş", "oldu", "olduğu", "olduğunu", "olması", "olmasına",
    
    # Common email/formality words
    "merhaba", "mail", "maildedir", "bilgilerinize", "saygılarımla", "teşekkürler",
    "lütfen", "rica", "ediyorum", "ediyoruz", "edilmesi", "edilmesini",
    
    # Generic business terms
    "talep", "eklenmesi", "yeni", "yer", "ilgili", "yapılan", "yapılması", "alan",
    "hesap", "gerekmektedir", "geliştirme", "edilmektedir", "ekranında", "limit",
    "işlem", "şekilde", "tarafından", "durum", "durumda", "durumunda", "konu",
    "konuda", "konusunda", "hakkında", "ile", "ilgili", "bağlı", "bağlı",
    
    # Time/date references
    "bugün", "dün", "yarın", "hafta", "ay", "yıl", "saat", "dakika", "saniye",
    "geçen", "gelecek", "şu", "an", "anda", "zaman", "zamanında",
    
    # Common pronouns and determiners
    "ben", "sen", "o", "biz", "siz", "onlar", "kendi", "kendisi", "kendileri",
    "bunun", "bunların", "şunun", "şunların", "onun", "onların",
    
    # Modal/auxiliary verbs
    "olabilir", "olabilir", "olmalı", "olmalı", "olması", "olmasına", "olmasından",
    "edebilir", "edebilir", "etmeli", "etmeli", "etmesi", "etmesine",
    
    # Common conjunctions and prepositions
    "ancak", "fakat", "lakin", "yalnız", "sadece", "yalnızca", "sadece",
    "üzerinde", "altında", "yanında", "karşısında", "içinde", "dışında",
    "arasında", "ortasında", "sonunda", "başında", "sonrasında", "öncesinde"
]
