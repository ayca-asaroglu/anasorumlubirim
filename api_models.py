"""
Pydantic models for API request and response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single text prediction."""
    
    text: str = Field(..., description="Text to classify", min_length=1, max_length=10000)
    summary: Optional[str] = Field(None, description="Optional summary text", max_length=5000)
    etkilenecek_kanallar: Optional[str] = Field(None, description="Affected channels")
    talep_tipi: Optional[str] = Field(None, description="Request type")
    talep_alt_tipi: Optional[str] = Field(None, description="Request sub-type")
    reporter_birim: Optional[str] = Field(None, description="Reporter unit")
    reporter_direktorluk: Optional[str] = Field(None, description="Reporter directorate")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Sistemde bir hata oluştu ve kullanıcılar giriş yapamıyor.",
                "summary": "Giriş sistemi hatası",
                "talep_tipi": "Teknik Destek",
                "reporter_birim": "IT Departmanı"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for single text prediction."""
    
    upper_level_prediction: str = Field(..., description="Predicted upper-level organizational unit")
    lower_level_prediction: str = Field(..., description="Predicted lower-level organizational unit")
    upper_level_confidence: float = Field(..., description="Confidence score for upper-level prediction", ge=0, le=1)
    lower_level_confidence: float = Field(..., description="Confidence score for lower-level prediction", ge=0, le=1)
    top3_predictions: List[str] = Field(..., description="Top 3 lower-level predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    rag_candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Top-k application/unit candidates from RAG retriever"
    )
    app_candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Top-k application candidates from FAISS-backed app retriever"
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch text prediction."""
    
    texts: List[PredictionRequest] = Field(..., description="List of texts to classify", min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {
                        "text": "Sistemde bir hata oluştu.",
                        "summary": "Sistem hatası",
                        "talep_tipi": "Teknik Destek"
                    },
                    {
                        "text": "Yeni kullanıcı kaydı gerekiyor.",
                        "summary": "Kullanıcı kaydı",
                        "talep_tipi": "İnsan Kaynakları"
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch text prediction."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    message: str = Field(..., description="Health status message")
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response model."""
    
    upper_level_classes: List[str] = Field(..., description="Available upper-level classes")
    lower_level_classes: List[str] = Field(..., description="Available lower-level classes")
    model_version: str = Field(..., description="Model version")
    training_date: Optional[str] = Field(None, description="Model training date")
    feature_count: int = Field(..., description="Number of features used")
    is_loaded: bool = Field(..., description="Whether models are loaded")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str = Field(..., description="Error detail message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
