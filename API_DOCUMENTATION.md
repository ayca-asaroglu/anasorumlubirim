# Turkish Hierarchical Text Classification API

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Optional)
```bash
python train_models.py
```

### 3. Start API Server
```bash
python api.py
```

### 4. Access API Documentation
Open your browser and go to: http://localhost:8000/docs

## 📡 API Endpoints

### Health Check
- **GET** `/health` - Check API health status
- **GET** `/` - Root endpoint with basic info

### Model Information
- **GET** `/model/info` - Get model information and available classes
- **GET** `/classes/upper` - Get available upper-level classes
- **GET** `/classes/lower` - Get available lower-level classes

### Predictions
- **POST** `/predict` - Predict organizational units for single text
- **POST** `/predict/batch` - Predict organizational units for multiple texts

## 🔧 API Usage Examples

### Single Text Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Sistemde bir hata oluştu ve kullanıcılar giriş yapamıyor.",
       "summary": "Giriş sistemi hatası",
       "talep_tipi": "Teknik Destek",
       "reporter_birim": "IT Departmanı"
     }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

### Get Model Info
```bash
curl -X GET "http://localhost:8000/model/info"
```

## 🐳 Docker Deployment

### Build and Run with Docker
```bash
# Build the image
docker build -t turkish-classification-api .

# Run the container
docker run -p 8000:8000 turkish-classification-api
```

### Using Docker Compose
```bash
# Start the service
docker-compose up -d

# Start with nginx (production)
docker-compose --profile production up -d
```

## 📊 Response Format

### Single Prediction Response
```json
{
  "upper_level_prediction": "IT",
  "lower_level_prediction": "IT_Support",
  "upper_level_confidence": 0.85,
  "lower_level_confidence": 0.92,
  "top3_predictions": ["IT_Support", "IT_Development", "IT_Infrastructure"],
  "processing_time_ms": 45.2,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Batch Prediction Response
```json
{
  "predictions": [
    {
      "upper_level_prediction": "IT",
      "lower_level_prediction": "IT_Support",
      "upper_level_confidence": 0.85,
      "lower_level_confidence": 0.92,
      "top3_predictions": ["IT_Support", "IT_Development", "IT_Infrastructure"],
      "processing_time_ms": 45.2,
      "timestamp": "2024-01-01T12:00:00"
    }
  ],
  "total_processing_time_ms": 89.5,
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Configuration

API settings can be modified in `config.py`:

```python
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "cors_origins": ["*"],
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "timeout": 30,  # seconds
    "max_batch_size": 100
}
```

## 📝 Request Parameters

### Required Parameters
- `text`: The main text to classify (string, 1-10000 characters)

### Optional Parameters
- `summary`: Optional summary text (string, max 5000 characters)
- `etkilenecek_kanallar`: Affected channels (string)
- `talep_tipi`: Request type (string)
- `talep_alt_tipi`: Request sub-type (string)
- `reporter_birim`: Reporter unit (string)
- `reporter_direktorluk`: Reporter directorate (string)

## 🚨 Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error
- `503`: Service Unavailable (models not loaded)

Error response format:
```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔍 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Model Status
```bash
curl http://localhost:8000/model/info
```

## 🛠️ Development

### Run in Development Mode
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
# Add test files and run
python -m pytest tests/
```

## 📈 Performance

- **Single Prediction**: ~50ms average
- **Batch Prediction**: ~30ms per item
- **Memory Usage**: ~500MB (with SBERT model)
- **Concurrent Requests**: Up to 100 (configurable)

## 🔒 Security

- CORS enabled for cross-origin requests
- Input validation and sanitization
- Request size limits
- Timeout protection
- Error message sanitization
