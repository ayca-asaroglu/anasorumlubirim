import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import RAG_CONFIG, PREPROCESSING_CONFIG


class AppCatalogRetriever:
    """Simple SBERT-based retriever over an application catalog Excel."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or RAG_CONFIG
        self.model = None
        self.df = None
        self.embeddings = None

    def load(self):
        if not self.config.get("enabled", False):
            return self
        excel_path = self.config["excel_path"]
        sheet_name = self.config.get("sheet_name", 0)
        self.df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # Prepare corpus texts (name + description)
        name_col = self.config["app_name_col"]
        desc_col = self.config["app_desc_col"]
        texts = (
            self.df[name_col].fillna("").astype(str).str.strip() + " \n " +
            self.df[desc_col].fillna("").astype(str).str.strip()
        ).tolist()

        # Build SBERT model and embeddings
        if self.model is None:
            self.model = SentenceTransformer(PREPROCESSING_CONFIG["sbert_model"])
        self.embeddings = self.model.encode(texts, show_progress_bar=False)
        return self

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if not self.config.get("enabled", False):
            return []
        if self.df is None or self.embeddings is None:
            raise RuntimeError("Retriever not loaded. Call load() first.")
        top_k = top_k or self.config.get("top_k", 5)
        query_vec = self.model.encode([query], show_progress_bar=False)
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        name_col = self.config["app_name_col"]
        desc_col = self.config["app_desc_col"]
        dir_col = self.config["directorate_col"]
        units_col = self.config["units_col"]
        results = []
        for i in top_idx:
            row = self.df.iloc[i]
            results.append({
                "app_name": row.get(name_col, ""),
                "app_description": row.get(desc_col, ""),
                "directorate": row.get(dir_col, ""),
                "units": row.get(units_col, ""),
                "score": float(sims[i])
            })
        return results


