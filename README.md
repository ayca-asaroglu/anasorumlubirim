# Turkish Hierarchical Text Classification Model

A comprehensive machine learning pipeline for Turkish text classification using hierarchical classification approach. This model classifies Turkish text into upper-level and lower-level organizational units using a two-stage classification process.

## 🏗️ Architecture

The project follows a modular architecture with the following components:

```
birim/
├── main.py                 # Main execution script
├── api.py                 # FastAPI application
├── api_models.py          # Pydantic models for API
├── model_service.py       # Model service for API
├── train_models.py        # Model training and saving script
├── test_api.py           # API testing script
├── start_api.sh          # API startup script
├── config.py             # Configuration settings
├── preprocessing.py      # Text preprocessing utilities
├── feature_extraction.py # Feature extraction (SBERT, TF-IDF, categorical)
├── models.py            # Hierarchical classification models
├── evaluation.py        # Evaluation metrics and visualization
├── example.py           # Example usage script
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── nginx.conf          # Nginx configuration
├── API_DOCUMENTATION.md # API documentation
└── README.md           # This file
```

## 🚀 Features

- **Hierarchical Classification**: Two-stage classification (upper-level → lower-level)
- **Turkish Text Processing**: Specialized preprocessing for Turkish language
- **Multiple Feature Types**: 
  - SBERT embeddings for semantic understanding
  - TF-IDF vectors for term frequency analysis
  - One-hot encoded categorical features
- **Feature Selection**: Automatic feature selection using statistical tests
- **Comprehensive Evaluation**: Multiple metrics including top-N accuracy
- **Visualization**: Confusion matrices and performance plots

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🔧 Configuration

The model can be configured through `config.py`:

- **Data Configuration**: CSV file path, encoding, required columns
- **Preprocessing**: Minimum samples per class, TF-IDF parameters
- **Model Configuration**: Test size, feature selection parameters
- **Output Configuration**: Result file paths, visualization settings

## 📊 Usage

### Basic Usage (Command Line)

Run the complete pipeline:

```bash
python main.py
```

Run examples:

```bash
python example.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### API Usage

#### Quick Start
```bash
# Start the API server
./start_api.sh

# Or manually:
python api.py
```

#### Access API Documentation
Open your browser and go to: http://localhost:8000/docs

#### Test the API
```bash
python test_api.py
```

#### Docker Deployment
```bash
# Build and run with Docker
docker build -t turkish-classification-api .
docker run -p 8000:8000 turkish-classification-api

# Or use Docker Compose
docker-compose up -d
```

#### API Endpoints
- **GET** `/health` - Health check
- **GET** `/model/info` - Model information
- **POST** `/predict` - Single text prediction
- **POST** `/predict/batch` - Batch text prediction
- **GET** `/classes/upper` - Available upper-level classes
- **GET** `/classes/lower` - Available lower-level classes

#### Example API Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Sistemde bir hata oluştu ve kullanıcılar giriş yapamıyor.",
       "summary": "Giriş sistemi hatası",
       "talep_tipi": "Teknik Destek"
     }'
```

For detailed API documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

### Custom Configuration

Modify `config.py` to adjust:
- Minimum samples per class
- Feature selection parameters
- Model hyperparameters
- Output file names
- API settings

## 🔍 Model Pipeline

1. **Data Loading**: Load CSV data with specified columns
2. **Text Preprocessing**: 
   - Clean Turkish text
   - Remove stopwords
   - Normalize characters
3. **Feature Extraction**:
   - Generate SBERT embeddings
   - Create TF-IDF vectors
   - Encode categorical features
4. **Model Training**:
   - Train upper-level classifier (SVM with calibration)
   - Train lower-level classifier (Logistic Regression)
5. **Hierarchical Prediction**:
   - Predict upper-level class
   - Use upper-level prediction to constrain lower-level predictions
6. **Evaluation**:
   - Calculate accuracy and F1 scores
   - Generate confusion matrices
   - Export results to Excel

## 📈 Performance Metrics

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score for imbalanced classes
- **Top-N Accuracy**: Accuracy considering top-N predictions
- **Cross-Validation**: 5-fold CV for robust performance estimation

## 🎯 Key Features

### Turkish Language Support
- Comprehensive Turkish stopword list
- Proper handling of Turkish characters (ğ, ü, ş, ı, ö, ç)
- Specialized text cleaning for Turkish business documents

### Hierarchical Classification
- Two-stage prediction process
- Upper-level predictions constrain lower-level predictions
- Maintains logical consistency between levels

### Robust Feature Engineering
- SBERT embeddings for semantic understanding
- TF-IDF for term frequency analysis
- Categorical feature encoding
- Automatic feature selection

### Comprehensive Evaluation
- Multiple accuracy metrics
- Confusion matrix visualization
- Class distribution analysis
- Cross-validation for robust estimates

## 📁 Output Files

The pipeline generates:
- `tahmin_ust_alt_model_2609.xlsx`: Detailed prediction results
- Confusion matrix plots
- Performance metrics in console output

## 🔧 Customization

### Adding New Features
1. Modify `feature_extraction.py` to add new feature types
2. Update `config.py` with new parameters
3. Adjust `main.py` to include new features

### Modifying Preprocessing
1. Edit `preprocessing.py` to change text cleaning rules
2. Update stopword list in `config.py`
3. Modify filtering criteria as needed

### Changing Models
1. Update `models.py` to use different algorithms
2. Adjust hyperparameters in `config.py`
3. Modify evaluation metrics in `evaluation.py`

## 📝 Data Format

The model expects a CSV file with the following columns:
- `SUMMARY`: Text summary
- `description`: Detailed description
- `anaSorumluBirimUstBirim`: Upper-level organizational unit
- `EtkilenecekKanallar`: Affected channels
- `talep_tipi`: Request type
- `talep_alt_tipi`: Request sub-type
- `reporterBirim`: Reporter unit
- `reporterDirektorluk`: Reporter directorate
- `AnaSorumluBirim_Duzenlenmis`: Lower-level organizational unit
- `EFOR_ANASORUMLU_HARCANAN`: Effort spent

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
