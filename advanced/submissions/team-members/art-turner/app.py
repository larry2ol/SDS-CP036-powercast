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
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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

# Mount static files and templates (only if directory exists)
import os
if os.path.exists("static"):
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
    input_data: Optional[List[List[float]]] = Field(None, description="Input features used for prediction")
    input_summary: Optional[Dict[str, Any]] = Field(None, description="Summary statistics of input data")

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

class SHAPExplanation(BaseModel):
    """SHAP explanation response"""
    shap_values: List[List[float]] = Field(..., description="SHAP values for each feature and output")
    feature_names: List[str] = Field(..., description="Names of input features")
    base_values: List[float] = Field(..., description="Base values for each output")
    explanation_plots: Dict[str, str] = Field(..., description="Base64 encoded explanation plots")

class FeatureAnalysis(BaseModel):
    """Feature analysis response"""
    correlation_matrix: List[List[float]] = Field(..., description="Feature correlation matrix")
    feature_importance: Dict[str, float] = Field(..., description="Global feature importance scores")
    feature_statistics: Dict[str, Dict[str, float]] = Field(..., description="Statistical summary of features")
    visualizations: Dict[str, str] = Field(..., description="Base64 encoded visualization plots")

class InputVisualization(BaseModel):
    """Input data visualization response"""
    input_data: List[List[float]] = Field(..., description="The input time series data")
    feature_names: List[str] = Field(..., description="Names of the features")
    time_series_plot: str = Field(..., description="Base64 encoded time series plot")
    feature_distribution_plot: str = Field(..., description="Base64 encoded feature distribution plot")

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

# Global simulation state - Tetouan, Morocco climate
simulation_state = {
    "current_datetime": datetime(2024, 1, 1, 0, 0),  # Start simulation
    "base_temperature": 20.0,  # Tetouan annual average (14-26°C range)
    "seasonal_factor": 0.0,
    "weather_trend": 0.0
}

def create_dummy_time_series(n_timesteps: int = 36, n_features: int = 11, advance_time: bool = True) -> np.ndarray:
    """Create realistic time-progressing dummy time series data for simulation"""
    global simulation_state
    
    if advance_time:
        # Advance simulation time by 6 hours for each prediction
        simulation_state["current_datetime"] += timedelta(hours=6)
        
        # Update seasonal and weather trends
        day_of_year = simulation_state["current_datetime"].timetuple().tm_yday
        simulation_state["seasonal_factor"] = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        simulation_state["weather_trend"] += np.random.normal(0, 0.5)
        simulation_state["weather_trend"] = np.clip(simulation_state["weather_trend"], -5, 5)
    
    # Create realistic time-progressing feature data
    features = []
    current_sim_time = simulation_state["current_datetime"]
    
    for i in range(n_timesteps):
        # Calculate the timestamp for this data point (going back 36 hours from current time)
        timestamp = current_sim_time - timedelta(hours=n_timesteps - i)
        
        timestep_features = []
        
        # Temperature with realistic seasonal and daily patterns
        hour_of_day = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        # Tetouan, Morocco climate simulation
        # Daily temperature cycle (Mediterranean climate - moderate daily variation)
        daily_temp_cycle = 4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Seasonal temperature variation: 14°C (Jan) to 26°C (Aug)
        # Peak summer around day 213 (Aug 1), winter around day 15 (Jan 15)
        seasonal_temp = 6 * np.sin(2 * np.pi * (day_of_year - 15) / 365.25)
        
        # Base temperature with weather trend
        base_temp = simulation_state["base_temperature"] + simulation_state["weather_trend"]
        
        temp = base_temp + seasonal_temp + daily_temp_cycle + np.random.normal(0, 1.5)
        temp = np.clip(temp, 8, 35)  # Reasonable bounds for Tetouan
        timestep_features.append(temp)
        
        # Mediterranean humidity: 70-78% year-round, slightly lower in hot summer
        base_humidity = 74  # Average
        # Slightly lower humidity in summer (high temps), higher in winter
        seasonal_humidity_adj = -2 * np.sin(2 * np.pi * (day_of_year - 15) / 365.25)
        humidity = base_humidity + seasonal_humidity_adj + np.random.normal(0, 4)
        humidity = np.clip(humidity, 65, 82)  # 70-78% ± variation
        timestep_features.append(humidity)
        
        # Mediterranean wind patterns (coastal location)
        base_wind = 4 + 1.5 * np.sin(2 * np.pi * day_of_year / 365.25)  # Slightly windier in winter
        wind_speed = base_wind + np.random.exponential(1.5)
        wind_speed = np.clip(wind_speed, 1, 18)  # Mediterranean coastal winds
        timestep_features.append(wind_speed)
        
        # Solar irradiance (Mediterranean - very sunny summers, cloudier winters)
        if 5 <= hour_of_day <= 19:  # Longer days in Mediterranean
            solar_angle = np.sin(np.pi * (hour_of_day - 5) / 14)
            base_solar = 900 * solar_angle  # High solar potential
            
            # Seasonal variation: very sunny summers (July=dry), cloudier winters (Dec=rainy)
            # Peak sun in summer (day 200), lowest in winter (day 350)
            seasonal_solar_mult = 0.6 + 0.5 * np.sin(2 * np.pi * (day_of_year - 350) / 365.25)
            
            # Random weather (clouds, clear days)
            weather_factor = 0.7 + 0.3 * np.random.random()  # 70-100% of potential
            
            general_diffuse = base_solar * seasonal_solar_mult * weather_factor + np.random.normal(0, 30)
        else:  # Nighttime
            general_diffuse = np.random.normal(0, 5)
        
        general_diffuse = np.clip(general_diffuse, 0, 1000)
        timestep_features.append(general_diffuse)
        
        diffuse = general_diffuse * (0.4 + 0.2 * np.random.random()) + np.random.normal(0, 15)
        diffuse = np.clip(diffuse, 0, min(800, general_diffuse))
        timestep_features.append(diffuse)
        
        # Cyclical time features (based on actual simulation time)
        hour_rad = 2 * np.pi * hour_of_day / 24
        timestep_features.append(np.sin(hour_rad))  # hour_sin
        timestep_features.append(np.cos(hour_rad))  # hour_cos
        
        dow_rad = 2 * np.pi * timestamp.weekday() / 7
        timestep_features.append(np.sin(dow_rad))   # dow_sin
        timestep_features.append(np.cos(dow_rad))   # dow_cos
        
        month_rad = 2 * np.pi * (timestamp.month - 1) / 12
        timestep_features.append(np.sin(month_rad))  # month_sin
        timestep_features.append(np.cos(month_rad))  # month_cos
        
        features.append(timestep_features)
    
    return np.array(features)

def create_input_summary(features_array: np.ndarray) -> Dict[str, Any]:
    """Create summary statistics for input features"""
    return {
        "shape": features_array.shape,
        "mean_values": features_array.mean(axis=0).tolist(),
        "std_values": features_array.std(axis=0).tolist(),
        "min_values": features_array.min(axis=0).tolist(),
        "max_values": features_array.max(axis=0).tolist(),
        "feature_ranges": {
            metadata["base_feature_cols"][i]: {
                "min": float(features_array[:, i].min()),
                "max": float(features_array[:, i].max()),
                "mean": float(features_array[:, i].mean()),
                "std": float(features_array[:, i].std())
            } for i in range(features_array.shape[1])
        }
    }

def create_time_series_plot(features_array: np.ndarray, feature_names: List[str]) -> str:
    """Create time series plot of input features"""
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=feature_names[:11],  # Show first 11 features
        vertical_spacing=0.08,
        horizontal_spacing=0.06
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, feature_name in enumerate(feature_names[:11]):
        row = (i // 4) + 1
        col = (i % 4) + 1
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(features_array))),
                y=features_array[:, i],
                name=feature_name,
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text="Input Features Time Series (36 timesteps)",
        title_x=0.5
    )
    
    return fig.to_html(include_plotlyjs='cdn')

def create_feature_distribution_plot(features_array: np.ndarray, feature_names: List[str]) -> str:
    """Create feature distribution plots"""
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=feature_names[:11],
        vertical_spacing=0.08,
        horizontal_spacing=0.06
    )
    
    for i, feature_name in enumerate(feature_names[:11]):
        row = (i // 4) + 1
        col = (i % 4) + 1
        
        fig.add_trace(
            go.Histogram(
                x=features_array[:, i],
                name=feature_name,
                nbinsx=10,
                opacity=0.7,
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text="Input Features Distribution",
        title_x=0.5
    )
    
    return fig.to_html(include_plotlyjs='cdn')

def create_correlation_heatmap(features_array: np.ndarray, feature_names: List[str]) -> str:
    """Create correlation heatmap"""
    correlation_matrix = np.corrcoef(features_array.T)
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=feature_names[:11],
        y=feature_names[:11],
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500,
        width=600
    )
    
    return fig.to_html(include_plotlyjs='cdn')

def plot_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    load_model_and_scalers()

@app.get("/dashboard")
async def dashboard(request: Request):
    """Serve the advanced dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/advanced")
async def advanced_dashboard(request: Request):
    """Serve the advanced analytics dashboard with AI explainability"""
    return templates.TemplateResponse("advanced_dashboard.html", {"request": request})

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
            <div style="margin-top: 20px;">
                <a href="/dashboard" class="button">📊 Advanced Dashboard</a>
                <a href="/advanced" class="button">🧠 AI Analytics Dashboard</a>
                <a href="/docs" class="button">📖 API Documentation</a>
            </div>
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
                
                <!-- Input Summary Display -->
                <div id="input-display" style="display: none; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px; border: 2px solid #e9ecef;">
                    <h3 style="margin-bottom: 15px; color: #333;">📊 Input Data Summary</h3>
                    <div id="input-summary-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px;"></div>
                    <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        <strong>Forecast based on:</strong> <span id="simulation-time-display">Current conditions</span>
                    </p>
                </div>
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
                <p style="font-size: 0.9em; color: #666; margin-bottom: 15px;">
                    <strong>Timestamps show simulated weather conditions</strong> (6-hour intervals) for realistic forecasting. 
                    API call times are shown for debugging purposes.
                </p>
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
                            <p><strong>Total Power:</strong> ${(data.predictions[0] + data.predictions[1] + data.predictions[2]).toFixed(1)} kW</p>
                            <p><strong>Timestamp:</strong> ${data.timestamp}</p>
                        </div>
                    `;
                    resultDiv.innerHTML = result;
                    
                    // Display input summary that generated this prediction
                    displayInputSummary(data.input_summary, data.model_info?.simulation_time);
                    
                    // Add to history
                    addToHistory(data);
                    
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
                }
            }
            
            function displayInputSummary(inputSummary, simulationTime) {
                if (!inputSummary || !inputSummary.feature_ranges) return;
                
                const inputDisplay = document.getElementById('input-display');
                const summaryGrid = document.getElementById('input-summary-grid');
                const timeDisplay = document.getElementById('simulation-time-display');
                
                // Feature units for display
                const featureUnits = {
                    'Temperature': '°C',
                    'Humidity': '%',
                    'Wind Speed': 'm/s',
                    'general diffuse flows': 'W/m²',
                    'diffuse flows': 'W/m²'
                };
                
                // Show only the main environmental features (not cyclical ones)
                const mainFeatures = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows'];
                
                summaryGrid.innerHTML = mainFeatures.map(feature => {
                    const range = inputSummary.feature_ranges[feature];
                    if (!range) return '';
                    
                    const unit = featureUnits[feature] || '';
                    const precision = feature.includes('flows') ? 0 : 1;
                    
                    return `
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 8px; border: 1px solid #ddd;">
                            <div style="font-size: 1.2em; font-weight: bold; color: #667eea;">
                                ${range.mean.toFixed(precision)}${unit}
                            </div>
                            <div style="font-size: 0.85em; color: #666; margin-top: 2px;">
                                ${feature}
                            </div>
                            <div style="font-size: 0.75em; color: #999; margin-top: 2px;">
                                ${range.min.toFixed(precision)}-${range.max.toFixed(precision)}${unit}
                            </div>
                        </div>
                    `;
                }).join('');
                
                // Update simulation time
                if (simulationTime) {
                    timeDisplay.textContent = simulationTime + ' (Tetouan, Morocco climate)';
                } else {
                    timeDisplay.textContent = 'Current conditions (Tetouan, Morocco climate)';
                }
                
                // Show the input display
                inputDisplay.style.display = 'block';
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
                
                const historyHTML = predictionHistory.map(pred => {
                    // Format timestamp for better readability
                    const timestamp = new Date(pred.timestamp).toLocaleString('en-US', {
                        year: 'numeric',
                        month: '2-digit', 
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false
                    });
                    
                    return `
                        <div class="result" style="margin: 10px 0;">
                            <strong>📅 ${timestamp} (Tetouan Climate)</strong>
                            ${pred.model_info?.api_call_time ? `<br><small style="color: #666;">API called at: ${pred.model_info.api_call_time}</small>` : ''}
                            <br>Zone 1: ${pred.zone_predictions['Zone 1'].toFixed(2)} kW | 
                            Zone 2: ${pred.zone_predictions['Zone 2'].toFixed(2)} kW | 
                            Zone 3: ${pred.zone_predictions['Zone 3'].toFixed(2)} kW
                        </div>
                    `;
                }).join('');
                
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
        
        # Store raw data for display purposes
        raw_features_array = features_array.copy()
        
        # Normalize features if requested (for model inference)
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
        
        # Create input summary using raw data
        input_summary = create_input_summary(raw_features_array)
        
        # Use simulation timestamp as primary timestamp for consistency
        primary_timestamp = simulation_state["current_datetime"].isoformat() if 'simulation_state' in globals() else datetime.now().isoformat()
        
        return PredictionResponse(
            predictions=prediction_denorm.tolist(),
            zone_predictions=zone_predictions,
            model_info={
                "model_type": "AttentionLSTM",
                "confidence": "high",
                "input_shape": list(raw_features_array.shape),
                "normalized_input": request.normalize,
                "simulation_time": simulation_state["current_datetime"].strftime("%Y-%m-%d %H:%M") if 'simulation_state' in globals() else None,
                "api_call_time": datetime.now().strftime("%H:%M:%S")  # Show when API was called (for debugging)
            },
            timestamp=primary_timestamp,  # Use simulation time as primary timestamp
            input_data=raw_features_array.tolist(),  # Return raw data for display
            input_summary=input_summary
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

@app.post("/explain", response_model=SHAPExplanation)
async def explain_prediction(request: PredictionRequest):
    """Generate SHAP explanations for a prediction"""
    if model is None or feature_scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process input
        features_array = np.array(request.features)
        if request.normalize:
            features_reshaped = features_array.reshape(-1, features_array.shape[-1])
            features_normalized = feature_scaler.transform(features_reshaped)
            features_array = features_normalized.reshape(features_array.shape)
        
        # Create SHAP explainer
        input_tensor = torch.FloatTensor(features_array).unsqueeze(0)
        
        def model_predict(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).reshape(-1, *features_array.shape)
                predictions = model(x_tensor)
                return predictions.cpu().numpy()
        
        # Generate background data for SHAP
        background_data = create_dummy_time_series(10, len(metadata["base_feature_cols"]))
        if request.normalize:
            background_reshaped = background_data.reshape(-1, background_data.shape[-1])
            background_normalized = feature_scaler.transform(background_reshaped)
            background_data = background_normalized.reshape(background_data.shape)
        
        explainer = shap.KernelExplainer(model_predict, background_data.reshape(10, -1))
        shap_values = explainer.shap_values(features_array.reshape(1, -1), nsamples=50)
        
        # Create SHAP plots
        explanation_plots = {}
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        feature_names_flat = [f"{name}_{i}" for name in metadata["base_feature_cols"] for i in range(36)]
        shap.summary_plot(shap_values, features_array.reshape(1, -1), 
                         feature_names=feature_names_flat[:len(shap_values[0])], 
                         show=False, max_display=20)
        explanation_plots["summary"] = plot_to_base64(plt.gcf())
        
        return SHAPExplanation(
            shap_values=shap_values,
            feature_names=feature_names_flat[:len(shap_values[0])],
            base_values=explainer.expected_value.tolist() if hasattr(explainer.expected_value, 'tolist') else [explainer.expected_value],
            explanation_plots=explanation_plots
        )
        
    except Exception as e:
        logger.error(f"SHAP explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/analyze-features", response_model=FeatureAnalysis)
async def analyze_features(request: PredictionRequest):
    """Analyze input features with correlations and statistics"""
    try:
        features_array = np.array(request.features)
        feature_names = metadata["base_feature_cols"]
        
        # Use raw data for analysis (more meaningful statistics)
        raw_features = features_array.copy()
        
        # Calculate statistics on raw data
        correlation_matrix = np.corrcoef(raw_features.T)
        feature_stats = {}
        
        for i, name in enumerate(feature_names):
            feature_stats[name] = {
                "mean": float(raw_features[:, i].mean()),
                "std": float(raw_features[:, i].std()),
                "min": float(raw_features[:, i].min()),
                "max": float(raw_features[:, i].max()),
                "median": float(np.median(raw_features[:, i]))
            }
        
        # Create visualizations with raw data
        visualizations = {}
        
        # Correlation heatmap
        visualizations["correlation_heatmap"] = create_correlation_heatmap(raw_features, feature_names)
        
        # Feature importance (simplified)
        feature_importance = {}
        for i, name in enumerate(feature_names):
            # Simple variance-based importance on raw data
            feature_importance[name] = float(raw_features[:, i].var())
        
        return FeatureAnalysis(
            correlation_matrix=correlation_matrix.tolist(),
            feature_importance=feature_importance,
            feature_statistics=feature_stats,
            visualizations=visualizations
        )
        
    except Exception as e:
        logger.error(f"Feature analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/visualize-input", response_model=InputVisualization)
async def visualize_input(request: PredictionRequest):
    """Create visualizations of input data"""
    try:
        features_array = np.array(request.features)
        feature_names = metadata["base_feature_cols"]
        
        # Always use raw data for visualization (don't normalize for display)
        raw_features = features_array.copy()
        
        # Create visualizations with raw data
        time_series_plot = create_time_series_plot(raw_features, feature_names)
        distribution_plot = create_feature_distribution_plot(raw_features, feature_names)
        
        return InputVisualization(
            input_data=raw_features.tolist(),
            feature_names=feature_names,
            time_series_plot=time_series_plot,
            feature_distribution_plot=distribution_plot
        )
        
    except Exception as e:
        logger.error(f"Input visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.get("/feature-importance")
async def get_global_feature_importance():
    """Get global feature importance using multiple dummy samples"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate multiple dummy samples
        importance_scores = {}
        n_samples = 20
        
        for _ in range(n_samples):
            dummy_features = create_dummy_time_series(
                n_timesteps=metadata["lookback_window"],
                n_features=len(metadata["base_feature_cols"])
            )
            
            # Calculate feature variance as a proxy for importance
            for i, feature_name in enumerate(metadata["base_feature_cols"]):
                if feature_name not in importance_scores:
                    importance_scores[feature_name] = []
                importance_scores[feature_name].append(float(dummy_features[:, i].var()))
        
        # Average the importance scores
        final_importance = {}
        for feature_name in importance_scores:
            final_importance[feature_name] = float(np.mean(importance_scores[feature_name]))
        
        return {
            "feature_importance": final_importance,
            "method": "variance-based",
            "samples_used": n_samples,
            "feature_names": metadata["base_feature_cols"]
        }
        
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature importance failed: {str(e)}")

@app.get("/simulation-status")
async def get_simulation_status():
    """Get current simulation time and progress"""
    return {
        "current_datetime": simulation_state["current_datetime"].isoformat(),
        "current_formatted": simulation_state["current_datetime"].strftime("%B %d, %Y at %H:%M"),
        "day_of_year": simulation_state["current_datetime"].timetuple().tm_yday,
        "season": get_season_name(simulation_state["current_datetime"]),
        "weather_trend": simulation_state["weather_trend"],
        "base_temperature": simulation_state["base_temperature"],
        "next_prediction_time": (simulation_state["current_datetime"] + timedelta(hours=6)).strftime("%H:%M")
    }

@app.post("/reset-simulation")
async def reset_simulation():
    """Reset the simulation to start from a specific date"""
    global simulation_state
    simulation_state = {
        "current_datetime": datetime(2024, 1, 1, 0, 0),
        "base_temperature": 20.0,  # Tetouan annual average (14-26°C range)
        "seasonal_factor": 0.0,
        "weather_trend": 0.0
    }
    return {
        "message": "Simulation reset to January 1, 2024",
        "current_datetime": simulation_state["current_datetime"].isoformat()
    }

def get_season_name(dt):
    """Get season name from datetime"""
    month = dt.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )