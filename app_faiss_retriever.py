from typing import List, Dict, Any
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
from config import RAG_APP_CONFIG, PREPROCESSING_CONFIG, AZURE_OPENAI_CONFIG


class AppFaissRetriever:
    """SBERT + FAISS retriever for application catalog."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or RAG_APP_CONFIG
        self.model = None
        self.df = None
        self.index = None
        self.embeddings = None

    def load(self):
        if not self.config.get("enabled", False):
            return self

        excel_path = self.config["excel_path"]
        sheet_name = self.config.get("sheet_name", 0)
        self.df = pd.read_excel(excel_path, sheet_name=sheet_name)

        name_col = self.config["app_name_col"]
        desc_col = self.config["app_desc_col"]
        texts = (
            self.df[name_col].fillna("").astype(str).str.strip() + " \n " +
            self.df[desc_col].fillna("").astype(str).str.strip()
        ).tolist()

        provider = self.config.get("provider", "sbert")
        if provider == "azure_openai":
            self.embeddings = self._embed_with_azure_openai(texts)
        else:
            # SBERT embeddings
            if self.model is None:
                self.model = SentenceTransformer(PREPROCESSING_CONFIG["sbert_model"])
            self.embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        # Build / load FAISS index (cosine via inner product on normalized vectors)
        dim = self.embeddings.shape[1]
        index_path = self.config.get("faiss_index_path", "faiss_app.index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings.astype("float32"))
            faiss.write_index(self.index, index_path)
        return self

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if not self.config.get("enabled", False):
            return []
        if self.df is None or self.index is None:
            raise RuntimeError("FAISS retriever not loaded. Call load() first.")
        top_k = top_k or self.config.get("top_k", 5)

        provider = self.config.get("provider", "sbert")
        if provider == "azure_openai":
            q = self._embed_with_azure_openai([query]).astype("float32")
        else:
            q = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, top_k)
        scores = scores[0]
        idxs = idxs[0]

        name_col = self.config["app_name_col"]
        desc_col = self.config["app_desc_col"]
        dir_col = self.config["directorate_col"]
        units_col = self.config["units_col"]

        results: List[Dict[str, Any]] = []
        for score, i in zip(scores, idxs):
            if i < 0:
                continue
            row = self.df.iloc[int(i)]
            results.append({
                "app_name": row.get(name_col, ""),
                "app_description": row.get(desc_col, ""),
                "directorate": row.get(dir_col, ""),
                "units": row.get(units_col, ""),
                "score": float(score)
            })
        return results

    def _embed_with_azure_openai(self, texts: List[str]) -> np.ndarray:
        try:
            # Lazy import to avoid mandatory dependency when unused
            from openai import AzureOpenAI
        except Exception as exc:
            raise RuntimeError(f"openai package not installed or incompatible: {exc}")

        endpoint = os.getenv(AZURE_OPENAI_CONFIG["endpoint_env"]) or ""
        api_key = os.getenv(AZURE_OPENAI_CONFIG["api_key_env"]) or ""
        api_version = AZURE_OPENAI_CONFIG["api_version"]
        deployment = AZURE_OPENAI_CONFIG["embedding_deployment"]
        if not endpoint or not api_key:
            raise RuntimeError("Azure OpenAI endpoint/key not set in environment variables")

        client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        # Batch to respect token limits if needed; simple single-call here
        resp = client.embeddings.create(input=texts, model=deployment)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        # Normalize for cosine via inner product
        arr = np.vstack(vecs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        return arr


