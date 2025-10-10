"""
FastAPI Application for Turkish Hierarchical Text Classification

This module provides REST API endpoints for the Turkish text classification model.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List, Optional, Dict, Any
import asyncio
from contextlib import asynccontextmanager

from api_models import (
    PredictionRequest, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse
)
from model_service import ModelService
from config import API_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model service instance
model_service: Optional[ModelService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    global model_service
    
    # Startup
    logger.info("🚀 Starting Turkish Text Classification API...")
    try:
        model_service = ModelService()
        await model_service.load_models()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")
    if model_service:
        await model_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Turkish Hierarchical Text Classification API",
    description="API for classifying Turkish text into organizational units using hierarchical classification",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model_service() -> ModelService:
    """Dependency to get the model service instance."""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service not available")
    return model_service


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic API information."""
    return HealthResponse(
        message="Turkish Hierarchical Text Classification API",
        status="healthy",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        message="API is healthy",
        status="healthy",
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(service: ModelService = Depends(get_model_service)):
    """Get information about the loaded models."""
    try:
        info = await service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    service: ModelService = Depends(get_model_service)
):
    """
    Predict organizational units for a single text input.
    
    Args:
        request: Prediction request containing text and metadata
        service: Model service instance
        
    Returns:
        Prediction response with upper and lower level predictions
    """
    try:
        result = await service.predict_single(request)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    service: ModelService = Depends(get_model_service)
):
    """
    Predict organizational units for multiple text inputs.
    
    Args:
        request: Batch prediction request containing multiple texts
        service: Model service instance
        
    Returns:
        Batch prediction response with predictions for all inputs
    """
    try:
        result = await service.predict_batch(request)
        return BatchPredictionResponse(**result)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes/upper")
async def get_upper_classes(service: ModelService = Depends(get_model_service)):
    """Get list of available upper-level classes."""
    try:
        classes = await service.get_upper_classes()
        return {"classes": classes}
    except Exception as e:
        logger.error(f"Error getting upper classes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes/lower")
async def get_lower_classes(service: ModelService = Depends(get_model_service)):
    """Get list of available lower-level classes."""
    try:
        classes = await service.get_lower_classes()
        return {"classes": classes}
    except Exception as e:
        logger.error(f"Error getting lower classes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
        log_level="info"
    )
