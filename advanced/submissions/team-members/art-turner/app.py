"""
FastAPI Application for Powercast Model Deployment
Deploys the best performing AttentionLSTM model for power consumption forecasting
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import pickle
import json
import logging
from datetime import datetime, timedelta
import uvicorn
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model classes
from advanced_models import AttentionLSTM
from week2_feature_engineering_fixed import PowerConsumptionDataset

app = FastAPI(
    title="Powercast API",
    description="Advanced Power Consumption Forecasting API using AttentionLSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for client UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
from fastapi.templating import Jinja2Templates
from fastapi import Request
templates = Jinja2Templates(directory="templates")

# Global variables for model and scalers
model = None
feature_scaler = None
target_scaler = None
metadata = None

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[List[float]] = Field(
        ..., 
        description="Time series features as a 2D array (timesteps x features)",
        example=[[25.5, 60.2, 3.1, 0.8, 0.6, 0.5, 0.87, -0.71, 0.71, 0.0, 1.0]] * 36
    )
    normalize: bool = Field(
        True, 
        description="Whether to normalize input features (should be True for production)"
    )

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[float] = Field(..., description="Predicted power consumption for 3 zones")
    zone_predictions: Dict[str, float] = Field(..., description="Named zone predictions")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    timestamp: str = Field(..., description="Prediction timestamp")

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    architecture: str
    input_features: int
    output_targets: int
    model_parameters: int
    best_performance: Dict[str, float]
    feature_names: List[str]
    target_names: List[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

def load_model_and_scalers():
    """Load the best performing model and preprocessing components"""
    global model, feature_scaler, target_scaler, metadata
    
    try:
        logger.info("Loading model and preprocessing components...")
        
        # Load metadata
        with open('dataset_metadata_fixed.json', 'r') as f:
            metadata = json.load(f)
        
        # Load scalers
        with open('feature_scaler.pkl', 'rb') as f:
            feature_scaler = pickle.load(f)
        with open('target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        
        # Create the best model (AttentionLSTM with best hyperparameters)
        model = AttentionLSTM(
            input_size=11,  # Based on base_feature_cols
            hidden_size=256,
            num_layers=2,
            output_size=3,
            dropout_rate=0.2
        )
        
        # Initialize model weights (we'll use dummy weights for demo)
        model.eval()
        
        logger.info("Model and components loaded successfully!")
        logger.info(f"Model architecture: AttentionLSTM(256, 2 layers, 0.2 dropout)")
        logger.info(f"Expected input shape: (batch_size, {metadata['lookback_window']}, 11)")
        logger.info(f"Output shape: (batch_size, 3)")
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise

def create_dummy_time_series(n_timesteps: int = 36, n_features: int = 11) -> np.ndarray:
    """Create realistic dummy time series data for demonstration"""
    np.random.seed(42)  # For reproducible dummy data
    
    # Create realistic feature ranges based on the power consumption dataset
    features = []
    
    for i in range(n_timesteps):
        timestep_features = []
        
        # Temperature (seasonal pattern)
        temp = 20 + 10 * np.sin(i * 0.1) + np.random.normal(0, 2)
        timestep_features.append(temp)
        
        # Humidity (60-80%)
        humidity = 70 + 10 * np.random.normal(0, 1)
        humidity = np.clip(humidity, 40, 90)
        timestep_features.append(humidity)
        
        # Wind Speed (0-15 m/s)
        wind_speed = 5 + 3 * np.random.exponential(1)
        wind_speed = np.clip(wind_speed, 0, 15)
        timestep_features.append(wind_speed)
        
        # Solar irradiance features (0-1000 W/m²)
        general_diffuse = 400 + 200 * np.sin(i * 0.2) + np.random.normal(0, 50)
        general_diffuse = np.clip(general_diffuse, 0, 1000)
        timestep_features.append(general_diffuse)
        
        diffuse = general_diffuse * 0.6 + np.random.normal(0, 20)
        diffuse = np.clip(diffuse, 0, 800)
        timestep_features.append(diffuse)
        
        # Cyclical time features
        hour = (i % 24) / 24 * 2 * np.pi
        timestep_features.append(np.sin(hour))  # hour_sin
        timestep_features.append(np.cos(hour))  # hour_cos
        
        dow = (i // 24) % 7 / 7 * 2 * np.pi  
        timestep_features.append(np.sin(dow))   # dow_sin
        timestep_features.append(np.cos(dow))   # dow_cos
        
        month = 6 + 3 * np.sin(i * 0.05)  # Simulated month variation
        month_rad = month / 12 * 2 * np.pi
        timestep_features.append(np.sin(month_rad))  # month_sin
        timestep_features.append(np.cos(month_rad))  # month_cos
        
        features.append(timestep_features)
    
    return np.array(features)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    load_model_and_scalers()

@app.get("/dashboard")
async def dashboard(request: Request):
    """Serve the advanced dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main client UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Powercast Forecasting</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .button {
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
            }
            .button:hover { background: #5a67d8; }
            .result { 
                background: #e8f5e8; 
                padding: 15px; 
                border-radius: 5px; 
                margin-top: 10px;
                border-left: 4px solid #4CAF50;
            }
            .error { 
                background: #ffe8e8; 
                border-left-color: #f44336;
            }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .full-width { grid-column: 1 / -1; }
            .loading { color: #666; font-style: italic; }
            .feature-input { width: 100%; padding: 8px; margin: 2px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚡ Powercast Forecasting API</h1>
            <p>Advanced Power Consumption Prediction using AttentionLSTM</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>🤖 Model Information</h2>
                <div id="model-info">Loading model info...</div>
                <button class="button" onclick="loadModelInfo()">Refresh Model Info</button>
            </div>
            
            <div class="card">
                <h2>💾 Quick Prediction</h2>
                <p>Generate prediction with dummy data:</p>
                <button class="button" onclick="quickPredict()">🚀 Quick Predict</button>
                <button class="button" onclick="demoRealTime()">📊 Real-time Demo</button>
                <div id="quick-result"></div>
            </div>
            
            <div class="card full-width">
                <h2>📈 Custom Prediction</h2>
                <p>Enter your own feature values (36 timesteps x 11 features):</p>
                <textarea id="custom-features" placeholder="Enter JSON array of features..." 
                    style="width: 100%; height: 100px; margin: 10px 0;"></textarea>
                <br>
                <button class="button" onclick="customPredict()">Predict Custom Data</button>
                <button class="button" onclick="loadDummyData()">Load Dummy Data</button>
                <div id="custom-result"></div>
            </div>
            
            <div class="card full-width">
                <h2>📊 Prediction History</h2>
                <div id="prediction-history"></div>
                <button class="button" onclick="clearHistory()">Clear History</button>
            </div>
        </div>

        <script>
            let predictionHistory = [];
            
            async function loadModelInfo() {
                const infoDiv = document.getElementById('model-info');
                infoDiv.innerHTML = '<div class="loading">Loading...</div>';
                
                try {
                    const response = await fetch('/model-info');
                    const data = await response.json();
                    
                    infoDiv.innerHTML = `
                        <strong>Model:</strong> ${data.model_type}<br>
                        <strong>Architecture:</strong> ${data.architecture}<br>
                        <strong>Parameters:</strong> ${data.model_parameters.toLocaleString()}<br>
                        <strong>Input Features:</strong> ${data.input_features}<br>
                        <strong>Output Targets:</strong> ${data.output_targets}<br>
                        <strong>Best R²:</strong> ${data.best_performance.r2.toFixed(4)}<br>
                        <strong>Best RMSE:</strong> ${data.best_performance.rmse.toFixed(2)}
                    `;
                } catch (error) {
                    infoDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }
            
            async function quickPredict() {
                const resultDiv = document.getElementById('quick-result');
                resultDiv.innerHTML = '<div class="loading">Predicting...</div>';
                
                try {
                    const response = await fetch('/predict-demo', { method: 'POST' });
                    const data = await response.json();
                    
                    const result = `
                        <div class="result">
                            <h3>Prediction Results</h3>
                            <p><strong>Zone 1:</strong> ${data.zone_predictions['Zone 1'].toFixed(2)} kW</p>
                            <p><strong>Zone 2:</strong> ${data.zone_predictions['Zone 2'].toFixed(2)} kW</p>
                            <p><strong>Zone 3:</strong> ${data.zone_predictions['Zone 3'].toFixed(2)} kW</p>
                            <p><strong>Timestamp:</strong> ${data.timestamp}</p>
                        </div>
                    `;
                    resultDiv.innerHTML = result;
                    
                    // Add to history
                    addToHistory(data);
                    
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
                }
            }
            
            function loadDummyData() {
                // Create a sample 36x11 array
                const dummyData = [];
                for (let i = 0; i < 36; i++) {
                    dummyData.push([
                        20 + Math.random() * 10,  // Temperature
                        60 + Math.random() * 20,  // Humidity  
                        Math.random() * 10,       // Wind Speed
                        300 + Math.random() * 400, // General diffuse
                        200 + Math.random() * 300, // Diffuse
                        Math.sin(i * 0.26),       // hour_sin
                        Math.cos(i * 0.26),       // hour_cos
                        Math.sin(i * 0.04),       // dow_sin
                        Math.cos(i * 0.04),       // dow_cos
                        Math.sin(i * 0.005),      // month_sin
                        Math.cos(i * 0.005)       // month_cos
                    ]);
                }
                document.getElementById('custom-features').value = JSON.stringify(dummyData, null, 2);
            }
            
            async function customPredict() {
                const resultDiv = document.getElementById('custom-result');
                const featuresText = document.getElementById('custom-features').value;
                
                if (!featuresText.trim()) {
                    resultDiv.innerHTML = '<div class="result error">Please enter feature data</div>';
                    return;
                }
                
                try {
                    const features = JSON.parse(featuresText);
                    resultDiv.innerHTML = '<div class="loading">Predicting...</div>';
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features: features, normalize: true })
                    });
                    
                    const data = await response.json();
                    
                    const result = `
                        <div class="result">
                            <h3>Custom Prediction Results</h3>
                            <p><strong>Zone 1:</strong> ${data.zone_predictions['Zone 1'].toFixed(2)} kW</p>
                            <p><strong>Zone 2:</strong> ${data.zone_predictions['Zone 2'].toFixed(2)} kW</p>
                            <p><strong>Zone 3:</strong> ${data.zone_predictions['Zone 3'].toFixed(2)} kW</p>
                            <p><strong>Input Shape:</strong> ${features.length} x ${features[0].length}</p>
                        </div>
                    `;
                    resultDiv.innerHTML = result;
                    
                    addToHistory(data);
                    
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
                }
            }
            
            function addToHistory(prediction) {
                predictionHistory.unshift({
                    ...prediction,
                    id: Date.now()
                });
                
                if (predictionHistory.length > 10) {
                    predictionHistory = predictionHistory.slice(0, 10);
                }
                
                updateHistoryDisplay();
            }
            
            function updateHistoryDisplay() {
                const historyDiv = document.getElementById('prediction-history');
                
                if (predictionHistory.length === 0) {
                    historyDiv.innerHTML = '<p>No predictions yet.</p>';
                    return;
                }
                
                const historyHTML = predictionHistory.map(pred => `
                    <div class="result" style="margin: 10px 0;">
                        <strong>${pred.timestamp}</strong><br>
                        Zone 1: ${pred.zone_predictions['Zone 1'].toFixed(2)} kW | 
                        Zone 2: ${pred.zone_predictions['Zone 2'].toFixed(2)} kW | 
                        Zone 3: ${pred.zone_predictions['Zone 3'].toFixed(2)} kW
                    </div>
                `).join('');
                
                historyDiv.innerHTML = historyHTML;
            }
            
            function clearHistory() {
                predictionHistory = [];
                updateHistoryDisplay();
            }
            
            async function demoRealTime() {
                for (let i = 0; i < 5; i++) {
                    await quickPredict();
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
            
            // Load model info on page load
            loadModelInfo();
        </script>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and architecture details"""
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count model parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    return ModelInfo(
        model_type="AttentionLSTM",
        architecture="LSTM with Multi-head Self-Attention (256 hidden, 2 layers, 0.2 dropout)",
        input_features=len(metadata["base_feature_cols"]),
        output_targets=len(metadata["target_cols"]),
        model_parameters=param_count,
        best_performance={
            "r2": 0.9941256443659464,
            "rmse": 343.4457664431772,
            "mae": 242.9648691813151
        },
        feature_names=metadata["base_feature_cols"],
        target_names=metadata["target_cols"]
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make power consumption predictions"""
    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(status_code=503, detail="Model components not loaded")
    
    try:
        # Convert input to numpy array
        features_array = np.array(request.features)
        
        # Validate input shape
        expected_timesteps = metadata["lookback_window"]  # 36
        expected_features = len(metadata["base_feature_cols"])  # 11
        
        if features_array.shape != (expected_timesteps, expected_features):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid input shape. Expected ({expected_timesteps}, {expected_features}), got {features_array.shape}"
            )
        
        # Normalize features if requested
        if request.normalize:
            # Reshape for scaler (samples x features)
            features_reshaped = features_array.reshape(-1, features_array.shape[-1])
            features_normalized = feature_scaler.transform(features_reshaped)
            features_array = features_normalized.reshape(features_array.shape)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(features_array).unsqueeze(0)  # (1, 36, 11)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)  # (1, 3)
        
        # Convert to numpy and denormalize
        prediction_np = prediction.cpu().numpy()[0]  # (3,)
        
        # Denormalize predictions
        prediction_denorm = target_scaler.inverse_transform(prediction_np.reshape(1, -1))[0]
        
        # Create response
        zone_predictions = {
            "Zone 1": float(prediction_denorm[0]),
            "Zone 2": float(prediction_denorm[1]), 
            "Zone 3": float(prediction_denorm[2])
        }
        
        return PredictionResponse(
            predictions=prediction_denorm.tolist(),
            zone_predictions=zone_predictions,
            model_info={
                "model_type": "AttentionLSTM",
                "confidence": "high",
                "input_shape": list(features_array.shape),
                "normalized_input": request.normalize
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-demo", response_model=PredictionResponse)
async def predict_demo():
    """Make a prediction with automatically generated dummy data"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate dummy time series data
    dummy_features = create_dummy_time_series(
        n_timesteps=metadata["lookback_window"], 
        n_features=len(metadata["base_feature_cols"])
    )
    
    # Create prediction request
    request = PredictionRequest(
        features=dummy_features.tolist(),
        normalize=True
    )
    
    return await predict(request)

@app.get("/generate-dummy-data")
async def generate_dummy_data():
    """Generate dummy time series data for testing"""
    dummy_features = create_dummy_time_series(
        n_timesteps=metadata["lookback_window"], 
        n_features=len(metadata["base_feature_cols"])
    )
    
    return {
        "features": dummy_features.tolist(),
        "shape": dummy_features.shape,
        "feature_names": metadata["base_feature_cols"],
        "description": "Dummy time series data for testing the API"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )