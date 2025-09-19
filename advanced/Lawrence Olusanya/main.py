
import fastapi
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# Define the paths to the saved model and scalers
# These paths are relative to the working directory inside the Docker container (/app)
model_save_path = 'best_power_consumption_model.keras'
#scaler_save_path = './' # Scalers are copied directly to the working directory

# Load the model and scalers once when the API starts
model = None
x_scaler = None
y_scaler = None

try:
    model = load_model(model_save_path)
    #x_scaler = joblib.load(scaler_save_path + 'x_scaler.pkl')
    #y_scaler = joblib.load(scaler_save_path + 'y_scaler.pkl')
    x_scaler = joblib.load('x_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    print("Model and scalers loaded successfully.")
except Exception as e:
    print(f"Error loading model or scalers: {e}")


#Define API Endpoints
app = FastAPI()

# Define the input data model
class TimeSeriesInput(BaseModel):
    # Assuming input_data is a list of lists representing the time series sequence
    # Each inner list is a timestep with features
    input_data: list[list[float]]
    # You might need to add window_size and n_features here if they are not fixed
    # or derive them from the input_data shape.
    window_size: int
    n_features: int


#Implement Prediction Logic
@app.post("/predict/")
async def predict_power_consumption(data: TimeSeriesInput):
    if model is None or x_scaler is None or y_scaler is None:
        return {"error": "Model or scalers not loaded. Check deployment environment."}

    try:
        # Receive input data (e.g., a JSON payload containing the time-series sequence).
        input_sequence = np.array(data.input_data)

        # Validate input shape
        if input_sequence.shape != (data.window_size, data.n_features):
            return {"error": f"Input shape mismatch. Expected ({data.window_size}, {data.n_features}), but got {input_sequence.shape}"}

        # Preprocess the input data using the loaded x_scaler
        # The scaler expects a 2D array, so we reshape the input_sequence
        input_sequence_scaled = x_scaler.transform(input_sequence.reshape(-1, data.n_features))
        # Reshape back to 3D for the model (batch_size, timesteps, features)
        input_sequence_scaled = input_sequence_scaled.reshape(1, data.window_size, data.n_features)

        # Use the loaded Keras model to make predictions on the preprocessed data.
        # Assuming the model predicts a single timestep output (n_output_features)
        # Determine the number of output features from the shape of y_scaler.data_min_
        # Using y_scaler.n_features_out_ is more reliable if available
        try:
            # Check if y_scaler has n_features_out_ attribute
            if hasattr(y_scaler, 'n_features_out_'):
                n_output_features = y_scaler.n_features_out_
            else:
                 # Fallback for older versions or different scaler types
                 # This assumes y_scaler was fitted on a 2D array where the second dimension is the number of features
                 n_output_features = y_scaler.data_min_.shape[0]
        except Exception as e:
             return {"error": f"Could not determine number of output features from scaler: {e}"}


        y_pred_scaled = model.predict(input_sequence_scaled) # shape: (1, output_timesteps, n_output_features)

        # Postprocess the predictions using the loaded y_scaler
        # The inverse transform requires a 2D array with all features
        # We need to pad the predicted output with zeros for the non-target features
        # Determine the original position of target features from the training phase
        # This might require knowing the original column order or indices used during training
        # For simplicity, assuming target features are the last ones

        # Reshape y_pred_scaled to 2D for inverse transform if it's 3D (batch, timesteps, features)
        y_pred_scaled_2d = y_pred_scaled.reshape(-1, n_output_features)

        # Inverse transform using the y_scaler
        # Note: MinMaxScaler inverse_transform expects input with the same number of features as fit was called on.
        # Since we only predicted the target features, we need to pad with zeros for the non-target features
        # and then inverse transform the full array.
        n_features_total = data.n_features # Total features the x_scaler was fitted on
        pad_cols = n_features_total - n_output_features

        if pad_cols < 0:
             return {"error": "Number of total features is less than number of output features."}

        # Create a zero array for padding
        zeros_for_pred_inv_transform = np.zeros((y_pred_scaled_2d.shape[0], pad_cols))

        # Concatenate zeros and predicted scaled values in the order expected by x_scaler
        # Assuming target features were the last columns when x_scaler was fitted
        full_scaled_output_for_inverse = np.concatenate([zeros_for_pred_inv_transform, y_pred_scaled_2d], axis=1)

        # Use x_scaler to inverse transform the full padded array
        # We then select only the target columns
        y_pred_inv = x_scaler.inverse_transform(full_scaled_output_for_inverse)[:, -n_output_features:]


        # Format the predictions (e.g., as JSON) and return them as the API response.
        # Convert to list for JSON serialization
        predictions = y_pred_inv.tolist()


        return {"predictions": predictions}

    except Exception as e:
        return {"error": f"An error occurred during prediction: {e}"}
