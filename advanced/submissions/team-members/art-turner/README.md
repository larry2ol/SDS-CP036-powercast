# ⚡ Powercast API - Advanced Power Consumption Forecasting

A production-ready FastAPI deployment featuring an **AttentionLSTM** model for multi-zone power consumption forecasting.

## 🚀 Features

- **High-Performance Model**: AttentionLSTM achieving R² = 0.9941
- **RESTful API**: FastAPI with automatic documentation
- **Advanced UI**: Interactive dashboard with real-time visualizations  
- **Docker Support**: Containerized deployment
- **Cloud Ready**: Configured for Render.com deployment
- **Real-time Predictions**: WebSocket-like real-time prediction updates
- **Comprehensive Testing**: Built-in API testing and validation

## 📊 Model Performance

- **Architecture**: AttentionLSTM (256 hidden, 2 layers, 0.2 dropout)
- **Best R²**: 0.9941
- **RMSE**: 343.4 kW
- **MAE**: 242.9 kW
- **Input**: 36 timesteps × 11 features (environmental + cyclical)
- **Output**: 3 zones power consumption predictions

## 🔧 Local Development

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
# Clone and navigate to the project
cd powercast-deployment

# Install dependencies
pip install -r requirements.txt

# Start the development server
python app.py
# or
uvicorn app:app --reload --port 8000
```

### Access the Application
- **API Documentation**: http://localhost:8000/docs
- **Advanced Dashboard**: http://localhost:8000/dashboard
- **Simple UI**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

### Build and Run
```bash
# Build Docker image
docker build -t powercast-api .

# Run container
docker run -p 8000:8000 powercast-api

# Or using docker-compose
docker-compose up --build
```

## ☁️ Render.com Deployment

### Deploy to Render.com

1. **Connect Repository**:
   - Connect your GitHub repository to Render.com
   - Select "Web Service" deployment type

2. **Configuration**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120`
   - **Environment**: Python 3.10

3. **Environment Variables** (Optional):
   - `PYTHON_VERSION`: 3.10.12
   - `PORT`: 8000 (automatically set by Render)

4. **Health Check Path**: `/health`

### Alternative: YAML Configuration
Use the included `render.yaml` for infrastructure-as-code deployment:
```bash
render deploy
```

## 📡 API Endpoints

### Core Prediction Endpoints

- `POST /predict` - Custom feature prediction
- `POST /predict-demo` - Quick prediction with dummy data
- `GET /generate-dummy-data` - Generate sample input data

### Information Endpoints

- `GET /health` - Service health status
- `GET /model-info` - Model architecture and performance metrics
- `GET /` - Basic web UI
- `GET /dashboard` - Advanced interactive dashboard

### API Request Example

```python
import requests

# Custom prediction
response = requests.post("https://your-app.onrender.com/predict", json={
    "features": [[25.5, 60.2, 3.1, 0.8, 0.6, 0.5, 0.87, -0.71, 0.71, 0.0, 1.0]] * 36,
    "normalize": True
})

prediction = response.json()
print(f"Zone 1: {prediction['zone_predictions']['Zone 1']:.2f} kW")
print(f"Zone 2: {prediction['zone_predictions']['Zone 2']:.2f} kW") 
print(f"Zone 3: {prediction['zone_predictions']['Zone 3']:.2f} kW")
```

### JavaScript/Browser Example

```javascript
// Quick demo prediction
fetch('/predict-demo', { method: 'POST' })
  .then(response => response.json())
  .then(data => {
    console.log('Predictions:', data.zone_predictions);
    console.log('Total Power:', 
      data.predictions[0] + data.predictions[1] + data.predictions[2]
    );
  });
```

## 📊 Features Overview

### Input Features (11 total)
1. **Environmental** (5 features):
   - Temperature (°C)
   - Humidity (%)
   - Wind Speed (m/s) 
   - General Diffuse Flows (W/m²)
   - Diffuse Flows (W/m²)

2. **Cyclical Time** (6 features):
   - Hour sin/cos (24h cycle)
   - Day of week sin/cos (weekly cycle)
   - Month sin/cos (seasonal cycle)

### Output Predictions
- Zone 1 Power Consumption (kW)
- Zone 2 Power Consumption (kW) 
- Zone 3 Power Consumption (kW)

## 🎯 Advanced Dashboard Features

- **Real-time Predictions**: Live prediction generation
- **Interactive Charts**: Dynamic time series visualization using Chart.js
- **Model Metrics**: Performance statistics display
- **Custom Input**: JSON input validation and testing
- **Prediction History**: Track and analyze recent predictions
- **Batch Processing**: Generate multiple predictions
- **Export Ready**: Copy-paste predictions for external analysis

## 🔒 Production Considerations

- **Model Security**: Models loaded securely without external dependencies
- **Scalability**: Gunicorn multi-worker deployment
- **Monitoring**: Health checks and logging
- **Error Handling**: Comprehensive error responses
- **Input Validation**: Pydantic models for request validation
- **CORS**: Configured for cross-origin requests

## 🛠️ Development Commands

```bash
# Run development server with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Test API endpoints
curl -X POST "http://localhost:8000/predict-demo"

# View API documentation
open http://localhost:8000/docs

# Check health
curl http://localhost:8000/health
```

## 📁 Project Structure

```
powercast-deployment/
├── app.py                          # FastAPI application
├── advanced_models.py              # Model architectures
├── week2_feature_engineering_fixed.py  # Data preprocessing
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── render.yaml                     # Render.com deployment config
├── templates/
│   └── dashboard.html             # Advanced UI template
├── static/                        # Static assets (if needed)
├── *.pkl                         # Model scalers
├── *.json                        # Dataset metadata
└── README.md                     # This file
```

## 🚀 Live Demo

Once deployed to Render.com, your API will be available at:
- `https://your-app-name.onrender.com/dashboard` - Interactive Dashboard
- `https://your-app-name.onrender.com/docs` - API Documentation

## 📈 Performance Notes

- **Cold Start**: ~10-15 seconds (free tier)
- **Response Time**: <200ms for predictions
- **Memory Usage**: ~512MB
- **Model Size**: ~2MB (PyTorch state dict)

## 🔍 Troubleshooting

### Common Issues
1. **ImportError**: Ensure all dependencies in requirements.txt
2. **Model Loading**: Check file paths for .pkl and .json files
3. **Memory Limits**: Consider reducing model size for free tier
4. **Timeout**: Increase gunicorn timeout for slower predictions

### Logs
Check Render.com deployment logs for detailed error information.

---

Built with ❤️ using FastAPI, PyTorch, and Chart.js for the **Advanced Power Consumption Forecasting Challenge**.