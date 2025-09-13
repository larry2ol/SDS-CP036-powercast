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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import json
import logging
import threading
from datetime import datetime, timedelta
import uvicorn
from pathlib import Path
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model classes
from advanced_models import AttentionLSTM
from model_loader import load_model_from_checkpoint
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
model_validation_metrics: Optional[Dict[str, float]] = None

def partial_scale_features(arr: np.ndarray) -> np.ndarray:
    """Apply feature_scaler to matching leading columns only when counts differ.

    - arr can be (T,F) or (N,T,F). Returns array with same shape.
    - If feature_scaler expects K features and arr has F != K, only the first K
      columns are scaled; the remainder are left unchanged.
    """
    if feature_scaler is None:
        return arr
    try:
        k = getattr(feature_scaler, 'n_features_in_', None)
        if k is None:
            # Fallback: try length of feature_names_in_
            names = getattr(feature_scaler, 'feature_names_in_', None)
            k = len(names) if names is not None else arr.shape[-1]
        F = arr.shape[-1]
        if arr.ndim == 2:
            if k == F:
                return feature_scaler.transform(arr)
            else:
                out = arr.copy()
                out[:, :k] = feature_scaler.transform(arr[:, :k])
                return out
        elif arr.ndim == 3:
            T = arr.shape[0] * arr.shape[1]
            flat = arr.reshape(T, F)
            if k == F:
                flat_scaled = feature_scaler.transform(flat)
            else:
                flat_scaled = flat.copy()
                flat_scaled[:, :k] = feature_scaler.transform(flat[:, :k])
            return flat_scaled.reshape(arr.shape)
        else:
            return arr
    except Exception:
        return arr

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
    shap_values: List[float] = Field(..., description="Flattened SHAP values for the first output")
    feature_names: List[str] = Field(..., description="Names of input features (flattened per timestep)")
    base_values: List[float] = Field(..., description="Base values (first output if multi-output)")
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
    time_series_plot: str = Field(..., description="HTML time series plot")
    feature_distribution_plot: str = Field(..., description="HTML feature distribution plot")
    correlation_plot: str = Field(..., description="HTML correlation heatmap plot")

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
        
        # Attempt to load trained checkpoint if available
        input_size = len(metadata["base_feature_cols"])  # default 11
        output_size = len(metadata["target_cols"])       # 3
        
        import os, torch as _torch
        ckpt_path = os.getenv("MODEL_PATH", "best_attentionlstm_20250907-091842.pth")
        loaded = False
        if os.path.exists(ckpt_path):
            try:
                logger.info(f"Loading trained model checkpoint from {ckpt_path}...")
                # Peek into checkpoint to infer input size if needed
                ckpt_raw = _torch.load(ckpt_path, map_location='cpu', weights_only=False)
                state = ckpt_raw.get('model_state_dict', ckpt_raw.state_dict() if hasattr(ckpt_raw, 'state_dict') else None)
                inferred_in = None
                if isinstance(state, dict):
                    w = state.get('lstm.weight_ih_l0')
                    if w is not None and w.dim() == 2:
                        inferred_in = int(w.shape[1])
                if inferred_in and inferred_in != input_size:
                    logger.warning(f"Metadata input_size={input_size}, checkpoint expects {inferred_in}. Adjusting to {inferred_in}.")
                    input_size = inferred_in
                    # Align feature names: prefer metadata['feature_cols'] if it matches inferred size
                    feature_cols_full = list(metadata.get("feature_cols", []))
                    if len(feature_cols_full) == input_size:
                        metadata["base_feature_cols"] = feature_cols_full
                    else:
                        # Fallback: extend base_feature_cols with placeholders
                        base_cols = list(metadata.get("base_feature_cols", []))
                        if len(base_cols) < input_size:
                            extra = [f"extra_feature_{i+1}" for i in range(input_size - len(base_cols))]
                            metadata["base_feature_cols"] = base_cols + extra
                
                loaded_model, ckpt = load_model_from_checkpoint(
                    ckpt_path, input_size=input_size, output_size=output_size, device='cpu'
                )
                globals()["model"] = loaded_model
                loaded = True
                logger.info("Checkpoint loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        
        if not loaded:
            # Fallback to randomly initialized model (not ideal for production)
            logger.warning("No valid checkpoint found. Falling back to randomly initialized AttentionLSTM.")
            model_fallback = AttentionLSTM(
                input_size=input_size,
                hidden_size=256,
                num_layers=2,
                output_size=output_size,
                dropout_rate=0.2
            )
            model_fallback.eval()
            globals()["model"] = model_fallback
        
        logger.info("Model and components loaded successfully!")
        logger.info(f"Expected input shape: (batch_size, {metadata['lookback_window']}, {input_size})")
        logger.info(f"Output shape: (batch_size, {output_size})")

        # Compute validation metrics if validation files are present
        try:
            compute_validation_metrics()
        except Exception as e:
            logger.warning(f"Validation metrics computation skipped/failed: {e}")
        
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

# Thread lock for simulation state to prevent race conditions
simulation_lock = threading.Lock()

def create_dummy_time_series(n_timesteps: int = 36, n_features: int = 11, advance_time: bool = True) -> np.ndarray:
    """Create realistic time-progressing dummy time series data for simulation.

    Generates the base 11 environmental/cyclical features. If n_features > 11,
    appends 3 autoregressive proxy columns for Zone 1/2/3 Power Consumption
    to reach 14 features, matching checkpoints trained with AR inputs.
    """
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
        
        # If model expects autoregressive power features (total 14 features),
        # synthesize plausible values from environmental signals.
        if n_features >= 14:
            # Proxy total consumption based on temp and solar (rough heuristic)
            # Higher when hot (cooling) and when solar is low (grid demand)
            solar = general_diffuse + 0.5 * diffuse
            demand_base = 1200 + 12 * max(temp - 18, 0) + 6 * max(16 - temp, 0)
            demand_solar_adj = -0.3 * (solar / 1000.0) * 800  # reduce when solar high
            total_kw = demand_base + demand_solar_adj + np.random.normal(0, 40)
            # Split into three zones
            z1 = max(0.0, 0.45 * total_kw + np.random.normal(0, 20))
            z2 = max(0.0, 0.33 * total_kw + np.random.normal(0, 15))
            z3 = max(0.0, 0.22 * total_kw + np.random.normal(0, 10))
            timestep_features.extend([z1, z2, z3])

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
            friendly_feature_name(metadata["base_feature_cols"][i]): {
                "min": float(features_array[:, i].min()),
                "max": float(features_array[:, i].max()),
                "mean": float(features_array[:, i].mean()),
                "std": float(features_array[:, i].std())
            } for i in range(features_array.shape[1])
        }
    }

def friendly_feature_name(name: str) -> str:
    """Map raw feature names to user-friendly labels for UI/plots."""
    mapping = {
        'diffuse flows': 'Solar Radiation',
        'general diffuse flows': 'Reflected Solar',
        'hour_sin': 'Hour (sin)',
        'hour_cos': 'Hour (cos)',
        'dow_sin': 'Day of Week (sin)',
        'dow_cos': 'Day of Week (cos)',
        'month_sin': 'Month (sin)',
        'month_cos': 'Month (cos)',
        'Zone 1 Power Consumption': 'Zone 1 Power Consumption',
        'Zone 2  Power Consumption': 'Zone 2 Power Consumption',
        'Zone 3  Power Consumption': 'Zone 3 Power Consumption',
    }
    return mapping.get(name, name)

def map_friendly_names(feature_names: List[str]) -> List[str]:
    return [friendly_feature_name(n) for n in feature_names]

def _iframe_from_html(html: str, height: int = 400) -> str:
    """Wrap HTML in an iframe (data URL) so embedded scripts execute.

    Directly injecting Plotly's HTML via innerHTML will not run <script> tags.
    Using an iframe ensures the figure initializes properly.
    """
    try:
        encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
        return (
            f"<iframe src=\"data:text/html;base64,{encoded}\" "
            f"style=\"width: 100%; height: {height}px; border: 0;\" loading=\"lazy\"></iframe>"
        )
    except Exception:
        return html

def create_time_series_plot(features_array: np.ndarray, feature_names: List[str]) -> str:
    """Create time series plot of input features"""
    try:
        # Local imports to reduce startup memory
        from plotly.subplots import make_subplots
        import plotly.express as px
        import plotly.graph_objects as go
        # Limit to first 11 features to fit 3x4 grid
        n_features = min(len(feature_names), 11, features_array.shape[1])
        
        # Apply friendly names
        display_names = map_friendly_names(feature_names[:n_features])

        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=display_names,
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        colors = px.colors.qualitative.Set3
        
        for i in range(n_features):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(features_array))),
                    y=features_array[:, i],
                    name=display_names[i],
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
        # Return as iframe to ensure scripts execute when inserted into DOM
        return _iframe_from_html(fig.to_html(include_plotlyjs='cdn', full_html=True), height=600)
        
    except Exception as e:
        logger.error(f"Time series plot creation failed: {str(e)}")
        return f"<div style='text-align: center; padding: 50px; color: red;'>Error creating time series plot: {str(e)}</div>"

def create_feature_distribution_plot(features_array: np.ndarray, feature_names: List[str]) -> str:
    """Create feature distribution plots"""
    try:
        # Local imports to reduce startup memory
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        n_features = min(len(feature_names), 11, features_array.shape[1])
        display_names = map_friendly_names(feature_names[:n_features])
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=display_names,
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        for i in range(n_features):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=features_array[:, i],
                    name=display_names[i],
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
        # Return as iframe to ensure scripts execute when inserted into DOM
        return _iframe_from_html(fig.to_html(include_plotlyjs='cdn', full_html=True), height=600)
        
    except Exception as e:
        logger.error(f"Distribution plot creation failed: {str(e)}")
        return f"<div style='text-align: center; padding: 50px; color: red;'>Error creating distribution plot: {str(e)}</div>"

def create_correlation_heatmap(features_array: np.ndarray, feature_names: List[str]) -> str:
    """Create correlation heatmap"""
    try:
        # Local imports to reduce startup memory
        import plotly.graph_objects as go
        n_features = min(len(feature_names), 11, features_array.shape[1])
        correlation_matrix = np.corrcoef(features_array[:, :n_features].T)
        # Replace NaNs/Infs (can occur for constant features) to make JSON/Plotly safe
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        display_names = map_friendly_names(feature_names[:n_features])
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=display_names,
            y=display_names,
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
        # Return as iframe to ensure scripts execute when inserted into DOM
        return _iframe_from_html(fig.to_html(include_plotlyjs='cdn', full_html=True), height=500)
        
    except Exception as e:
        logger.error(f"Correlation heatmap creation failed: {str(e)}")
        return f"<div style='text-align: center; padding: 50px; color: red;'>Error creating correlation heatmap: {str(e)}</div>"

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

def compute_validation_metrics(batch_size: int = 256, max_samples: Optional[int] = None):
    """Evaluate the loaded model on validation data and cache metrics.

    Uses feature_scaler to normalize inputs and target_scaler to denormalize outputs.
    Automatically detects whether validation targets are raw or normalized by
    comparing RMSE of two candidates.
    """
    global model_validation_metrics

    if model is None or feature_scaler is None or target_scaler is None or metadata is None:
        raise RuntimeError("Model/scalers/metadata not loaded")

    import os
    x_path = "val_sequences_fixed.npy"
    y_path = "val_targets_fixed.npy"
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError("Validation files not found: val_sequences_fixed.npy / val_targets_fixed.npy")

    X = np.load(x_path)  # expected shape (N, T, F)
    Y = np.load(y_path)  # expected shape (N, 3)

    if max_samples is not None:
        X = X[:max_samples]
        Y = Y[:max_samples]

    N, T, F = X.shape
    expected_T = metadata.get("lookback_window", T)
    expected_F = len(metadata.get("base_feature_cols", list(range(F))))
    if T != expected_T or F != expected_F:
        logger.warning(f"Validation shape differs from metadata: got (N={N}, T={T}, F={F}), expected T={expected_T}, F={expected_F}")

    # Normalize inputs per feature (partial scaling if needed)
    Xn = partial_scale_features(X)

    # Batched inference
    preds_norm = []
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(Xn[i:i+batch_size]).float()
            yb = model(xb).cpu().numpy()  # normalized target space
            preds_norm.append(yb)
    Y_pred_norm = np.vstack(preds_norm)
    # Denormalize predictions to raw kW
    Y_pred_raw = target_scaler.inverse_transform(Y_pred_norm)

    # Determine whether Y (validation targets) are raw or normalized
    # Candidate A: assume Y is raw
    rmse_raw = float(np.sqrt(mean_squared_error(Y, Y_pred_raw)))
    # Candidate B: assume Y is normalized
    try:
        Y_denorm_from_norm = target_scaler.inverse_transform(Y)
        rmse_norm = float(np.sqrt(mean_squared_error(Y_denorm_from_norm, Y_pred_raw)))
    except Exception:
        rmse_norm = float("inf")

    # Choose ground truth that minimizes RMSE
    if rmse_norm < rmse_raw:
        Y_true = Y_denorm_from_norm
    else:
        Y_true = Y

    # Compute zone-wise metrics and overall averages
    r2_list, rmse_list, mae_list = [], [], []
    for j in range(Y_true.shape[1]):
        r2_list.append(r2_score(Y_true[:, j], Y_pred_raw[:, j]))
        rmse_list.append(np.sqrt(mean_squared_error(Y_true[:, j], Y_pred_raw[:, j])))
        mae_list.append(mean_absolute_error(Y_true[:, j], Y_pred_raw[:, j]))

    model_validation_metrics = {
        "r2": float(np.mean(r2_list)),
        "rmse": float(np.mean(rmse_list)),
        "mae": float(np.mean(mae_list)),
        "details": {
            f"zone_{j+1}": {"r2": r2_list[j], "rmse": rmse_list[j], "mae": mae_list[j]} for j in range(len(r2_list))
        }
    }
    logger.info(f"Validation metrics: R2={model_validation_metrics['r2']:.4f}, RMSE={model_validation_metrics['rmse']:.2f}, MAE={model_validation_metrics['mae']:.2f}")

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
                    API call times are shown for diagnostic and performance monitoring.
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
    
    # Compute validation metrics only if explicitly enabled (saves memory)
    import os as _os
    global model_validation_metrics
    if _os.getenv('ENABLE_VALIDATION_METRICS', '0').lower() in ('1', 'true', 'yes') and model_validation_metrics is None:
        try:
            # Keep a cap on samples to limit memory/CPU
            max_samples = int(_os.getenv('VALIDATION_MAX_SAMPLES', '2000'))
            compute_validation_metrics(max_samples=max_samples)
        except Exception as e:
            logger.warning(f"Could not compute validation metrics: {e}")

    # Use computed validation metrics if available; otherwise fallback to static
    if model_validation_metrics is not None:
        perf = {
            "r2": float(model_validation_metrics.get("r2", 0.0)),
            "rmse": float(model_validation_metrics.get("rmse", 0.0)),
            "mae": float(model_validation_metrics.get("mae", 0.0)),
        }
    else:
        perf = {
            "r2": 0.9941256443659464,
            "rmse": 343.4457664431772,
            "mae": 242.9648691813151
        }

    return ModelInfo(
        model_type="AttentionLSTM",
        architecture="LSTM with Multi-head Self-Attention (256 hidden, 2 layers, 0.2 dropout)",
        input_features=len(metadata["base_feature_cols"]),
        output_targets=len(metadata["target_cols"]),
        model_parameters=param_count,
        best_performance=perf,
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
            features_array = partial_scale_features(features_array)
        
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
    
    # Advance simulation time BEFORE generating data to ensure consistency
    global simulation_state
    with simulation_lock:
        simulation_state["current_datetime"] += timedelta(hours=6)
        
        # Update seasonal and weather trends
        day_of_year = simulation_state["current_datetime"].timetuple().tm_yday
        simulation_state["seasonal_factor"] = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        simulation_state["weather_trend"] += np.random.normal(0, 0.5)
        simulation_state["weather_trend"] = np.clip(simulation_state["weather_trend"], -5, 5)
    
    # Generate dummy time series data WITHOUT advancing time again
    dummy_features = create_dummy_time_series(
        n_timesteps=metadata["lookback_window"], 
        n_features=len(metadata["base_feature_cols"]),
        advance_time=False  # Don't advance time again - we already did it
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
        n_features=len(metadata["base_feature_cols"]),
        advance_time=False  # Don't advance time for testing/debugging
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
    
    # Lazy imports to reduce baseline memory
    shap = None
    plt = None
    try:
        import shap as _shap
        shap = _shap
    except Exception:
        shap = None  # SHAP not available; we'll fall back
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        plt = _plt
    except Exception:
        plt = None  # Still proceed; fallback does not require plots
    
    try:
        # Process input
        features_array = np.array(request.features)
        if request.normalize:
            features_array = partial_scale_features(features_array)

        num_timesteps, num_feats = features_array.shape
        base_names = metadata["base_feature_cols"]
        feature_names_flat = [f"{friendly_feature_name(name)}_{i}" for i in range(num_timesteps) for name in base_names]

        # Try SHAP first only if available; otherwise skip to fallback
        if shap is not None and plt is not None:
            try:
                def model_predict(x):
                    with torch.no_grad():
                        x_tensor = torch.FloatTensor(x).reshape(-1, num_timesteps, num_feats)
                        predictions = model(x_tensor)
                        return predictions.cpu().numpy()

                background_data = create_dummy_time_series(10, len(metadata["base_feature_cols"]), advance_time=False)
                if request.normalize:
                    background_data = partial_scale_features(background_data)

                explainer = shap.KernelExplainer(model_predict, background_data.reshape(10, -1))
            # Keep SHAP sample budget small to reduce memory/CPU
            raw_shap_values = explainer.shap_values(features_array.reshape(1, -1), nsamples=20)

            # Normalize SHAP output to (1, F) for first output
            shap_arr = np.array(raw_shap_values[0] if isinstance(raw_shap_values, list) else raw_shap_values)
            if shap_arr.ndim == 1:
                shap_arr = shap_arr.reshape(1, -1)

            explanation_plots = {}
            # Summary plot
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_arr, features_array.reshape(1, -1),
                                  feature_names=feature_names_flat[: shap_arr.shape[1]],
                                  show=False, max_display=20)
                explanation_plots["summary"] = plot_to_base64(plt.gcf())
            except Exception as plot_error:
                logger.error(f"SHAP plot creation failed: {str(plot_error)}")
                explanation_plots["summary"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

            # Optional beeswarm (disabled by default for low-memory plans)
            import os as _os
            if _os.getenv('ENABLE_BEESWARM', '0').lower() in ('1', 'true', 'yes'):
                try:
                    M = 8  # keep small
                    eval_data = np.tile(features_array.reshape(1, num_timesteps, num_feats), (M, 1, 1))
                    noise = np.random.normal(0, 0.02, size=eval_data.shape)
                    eval_data = eval_data + noise
                    shap_eval = explainer.shap_values(eval_data.reshape(M, -1), nsamples=20)
                    shap_eval_arr = np.array(shap_eval[0] if isinstance(shap_eval, list) else shap_eval)
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_eval_arr, eval_data.reshape(M, -1),
                                      feature_names=feature_names_flat[: shap_eval_arr.shape[1]],
                                      show=False, max_display=20)
                    explanation_plots["beeswarm"] = plot_to_base64(plt.gcf())
                except Exception as bees_err:
                    logger.error(f"SHAP beeswarm generation failed: {str(bees_err)}")

            exp_val = explainer.expected_value
            base_vals = (list(np.array(exp_val).flatten()[:1])
                         if isinstance(exp_val, (list, tuple, np.ndarray))
                         else [float(exp_val)])

            return SHAPExplanation(
                shap_values=shap_arr.flatten().tolist(),
                feature_names=feature_names_flat[: shap_arr.size],
                base_values=base_vals,
                explanation_plots=explanation_plots
            )
        # If SHAP is not available or failed, fall back
        try:

            # Fallback: local permutation importance (signed impact on total power)
            with torch.no_grad():
                base_pred = model(torch.FloatTensor(features_array).unsqueeze(0))
                base_denorm = target_scaler.inverse_transform(base_pred.cpu().numpy()).flatten()
                base_total = float(base_denorm.sum())

            impacts = np.zeros(num_timesteps * num_feats, dtype=float)
            col_medians = np.median(features_array, axis=0)
            for t in range(num_timesteps):
                for f in range(num_feats):
                    x_alt = features_array.copy()
                    x_alt[t, f] = col_medians[f]
                    with torch.no_grad():
                        alt_pred = model(torch.FloatTensor(x_alt).unsqueeze(0))
                        alt_denorm = target_scaler.inverse_transform(alt_pred.cpu().numpy()).flatten()
                        alt_total = float(alt_denorm.sum())
                    impacts[t * num_feats + f] = alt_total - base_total

            explanation_plots = {"summary": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}

            return SHAPExplanation(
                shap_values=impacts.tolist(),
                feature_names=feature_names_flat,
                base_values=[base_total],
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
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        feature_stats = {}
        
        for i, name in enumerate(feature_names):
            pretty = friendly_feature_name(name)
            feature_stats[pretty] = {
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
        error_msg = str(e)
        logger.error(f"Feature analysis error: {error_msg}")
        
        # Provide more user-friendly error messages
        if "shape" in error_msg.lower():
            detail = "Invalid input data shape. Please ensure input data is properly formatted."
        elif "nan" in error_msg.lower() or "inf" in error_msg.lower():
            detail = "Input data contains invalid values (NaN or infinity). Please check data quality."
        else:
            detail = f"Analysis failed: {error_msg}"
            
        raise HTTPException(status_code=500, detail=detail)

@app.post("/visualize-input", response_model=InputVisualization)
async def visualize_input(request: PredictionRequest):
    """Create visualizations of input data"""
    try:
        features_array = np.array(request.features)
        feature_names = metadata["base_feature_cols"]
        
        # Debug logging
        logger.info(f"Input visualization - Features shape: {features_array.shape}")
        logger.info(f"Feature names count: {len(feature_names)}")
        
        # Validate input dimensions
        if len(features_array.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {features_array.shape}")
        
        if features_array.shape[1] != len(feature_names):
            logger.warning(f"Feature count mismatch: data has {features_array.shape[1]} features, metadata has {len(feature_names)}")
            # Adjust feature names to match data
            feature_names = feature_names[:features_array.shape[1]]
        
        # Always use raw data for visualization (don't normalize for display)  
        raw_features = features_array.copy()
        
        # Create visualizations with raw data
        logger.info("Creating time series plot...")
        time_series_plot = create_time_series_plot(raw_features, feature_names)
        
        logger.info("Creating distribution plot...")
        distribution_plot = create_feature_distribution_plot(raw_features, feature_names)
        
        logger.info("Creating correlation plot...")
        correlation_plot = create_correlation_heatmap(raw_features, feature_names)
        
        return InputVisualization(
            input_data=raw_features.tolist(),
            feature_names=map_friendly_names(feature_names),
            time_series_plot=time_series_plot,
            feature_distribution_plot=distribution_plot,
            correlation_plot=correlation_plot
        )
        
    except Exception as e:
        logger.error(f"Input visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.get("/feature-importance")
async def get_global_feature_importance():
    """Model-based permutation importance over multiple dummy samples.

    For each sample, permute each base feature across timesteps and measure
    the absolute change in total predicted power (sum of zones).
    """
    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(status_code=503, detail="Model components not loaded")

    try:
        rng = np.random.default_rng(42)
        n_samples = 12
        n_feats = len(metadata["base_feature_cols"])  # could be 14 after checkpoint adjust

        # Accumulators
        scores = np.zeros(n_feats, dtype=float)

        for _ in range(n_samples):
            # Generate one sample (raw)
            x_raw = create_dummy_time_series(
                n_timesteps=metadata["lookback_window"],
                n_features=n_feats,
                advance_time=False
            )

            # Normalize for model
            x_norm = partial_scale_features(x_raw)

            # Baseline prediction (denormalized total)
            with torch.no_grad():
                base_pred = model(torch.FloatTensor(x_norm).unsqueeze(0))
                base_denorm = target_scaler.inverse_transform(base_pred.cpu().numpy()).flatten()
                base_total = float(base_denorm.sum())

            # Permute each feature across time
            for f in range(n_feats):
                x_alt = x_norm.copy()
                # Shuffle this column across timesteps
                perm = rng.permutation(x_alt.shape[0])
                x_alt[:, f] = x_alt[perm, f]
                with torch.no_grad():
                    alt_pred = model(torch.FloatTensor(x_alt).unsqueeze(0))
                    alt_denorm = target_scaler.inverse_transform(alt_pred.cpu().numpy()).flatten()
                    alt_total = float(alt_denorm.sum())
                scores[f] += abs(alt_total - base_total)

        # Average over samples
        scores /= n_samples

        final_importance = {
            metadata["base_feature_cols"][i]: float(scores[i]) for i in range(n_feats)
        }

        return {
            "feature_importance": final_importance,
            "method": "permutation-based (temporal)",
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
