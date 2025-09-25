# Streamlit App for Power Consumption Forecasting
# This script creates a web-based user interface using the Streamlit library.
# It loads the pre-trained models and the scaler that were created by the
# `train_and_save_assets` script and uses them to make live predictions
# based on user input.

# --- Core Library Imports ---
import streamlit as st  # The main library for building the web app.
import pandas as pd  # For data manipulation and reading the uploaded CSV.
import numpy as np  # For numerical operations.
import torch  # The main PyTorch library.
import torch.nn as nn  # Contains the building blocks for neural networks.
import pickle  # Used to load the saved scaler object.
from datetime import timedelta  # For handling date and time calculations.

# --- 1. Load Deployment Assets ---
# This section is responsible for loading the pre-trained models and the data scaler
# from the files saved by the training script. This happens only once when the app starts.

# IMPORTANT: The model class definition must be identical to the one used during training.
# If this class differs, PyTorch will not be able to load the saved weights correctly.
class BiLSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(BiLSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.linear(last_time_step_out)
        return out

# Define the model parameters. These must also match the training script exactly.
INPUT_SIZE = 11  # Number of features after cyclical encoding.
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DROPOUT_RATE = 0.2

# The Streamlit decorator `@st.cache_resource` is used for caching. It tells Streamlit
# to run this function only once and store the result in memory. This is crucial for
# performance, as it prevents the app from reloading the large model files every
# time the user interacts with a widget.
@st.cache_resource
def load_assets():
    """
    Loads all necessary models (one for each zone) and the data scaler from disk.
    This function is cached to ensure assets are loaded only once.
    """
    # A dictionary to hold our loaded models, keyed by their user-friendly names.
    models = {}
    # Loop through the zone names to load each corresponding model file.
    for zone in ['Zone_1', 'Zone_2', 'Zone_3']:
        try:
            # First, instantiate the model architecture. It must be identical to the one used for training.
            model = BiLSTMForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_RATE)
            # Then, load the saved weights (the model's "brain") into the architecture.
            # `map_location=torch.device('cpu')` ensures the model loads correctly even on a machine without a GPU.
            model.load_state_dict(torch.load(f'model_{zone}.pth', map_location=torch.device('cpu')))
            # Set the model to evaluation mode. This disables layers like Dropout that behave differently during training.
            model.eval()
            # Store the ready-to-use model in our dictionary.
            models[zone.replace('_', ' ')] = model
        except FileNotFoundError:
            # If a model file is not found, display an error and stop.
            st.error(f"Error: `model_{zone}.pth` not found. Please run the training script first.")
            return None, None
            
    try:
        # Load the saved scaler object using pickle.
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        # If the scaler file is not found, display an error and stop.
        st.error("Error: `scaler.pkl` not found. Please run the training script first.")
        return None, None
        
    # Return the dictionary of models and the scaler object.
    return models, scaler


# Call the function to load the assets into memory.
models, scaler = load_assets()

# --- 2. Prediction Pipeline ---
# This function contains the end-to-end logic for taking raw data and producing a final prediction.
def make_prediction(input_df, model, scaler, model_selection, lookback_window=144):
    """
    Preprocesses a DataFrame of recent data, feeds it to a model to get a prediction,
    and inverse-transforms the result to its original scale.
    """
    # Ensure we have enough data to form a full sequence for the model.
    if len(input_df) < lookback_window:
        return None, f"Error: Input data must have at least {lookback_window} rows (24 hours)."

    # The model only needs the most recent `lookback_window` rows to make a prediction.
    # We create a copy to avoid SettingWithCopyWarning.
    sequence_df = input_df.tail(lookback_window).copy()

    # Align features with what the scaler expects
    expected_features = scaler.feature_names_in_
    sequence_df = sequence_df.reindex(columns=expected_features, fill_value=0)

    # Warn if any expected features were missing and had to be filled with zeros.
    missing = set(expected_features) - set(sequence_df.columns)
    if missing:
        st.warning(f"Missing features filled with 0: {missing}")

    # STEP 1: Feature Engineering
    # We must apply the exact same feature engineering as in the training script.
    sequence_df['hour_sin'] = np.sin(2 * np.pi * sequence_df.index.hour / 24.0)
    sequence_df['hour_cos'] = np.cos(2 * np.pi * sequence_df.index.hour / 24.0)
    sequence_df['dayofweek_sin'] = np.sin(2 * np.pi * sequence_df.index.dayofweek / 7.0)
    sequence_df['dayofweek_cos'] = np.cos(2 * np.pi * sequence_df.index.dayofweek / 7.0)

    # STEP 2: Scale the data
    # We use the loaded scaler to transform the new data.
    scaled_sequence = scaler.transform(sequence_df)

    # STEP 3: Convert to PyTorch Tensor
    # The data needs to be in the format PyTorch expects.
    # `unsqueeze(0)` adds a batch dimension of 1, as the model expects to process data in batches.
    sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)

    # STEP 4: Make the Prediction
    # `torch.no_grad()` is a performance optimization that disables gradient calculation.
    with torch.no_grad():
        # The model outputs a scaled prediction. `.item()` extracts the single scalar value.
        scaled_prediction = model(sequence_tensor).item()

    # STEP 5: Inverse Transform the Prediction
    # We must convert the scaled prediction back to its original units (kW).
    # Create a dummy array with the same number of columns as the original data.
    dummy_array = np.zeros((1, scaler.n_features_in_))
    # Find the correct column index for the zone we are predicting.
    target_col_name = f'Zone_{model_selection.split(" ")[1]}_Power_Consumption'
    if target_col_name == 'Zone_2_Power_Consumption':
        target_col_name = 'Zone_2__Power_Consumption' # Handle the double underscore case.

    if target_col_name == 'Zone_3_Power_Consumption':
        target_col_name = 'Zone_3__Power_Consumption' # Handle the double underscore case.

    # Get the target column index from the scaler's feature list, NOT the raw input_df.
    # This is the key change to prevent incorrect inverse transformations.
    try:
        # The scaler's `feature_names_in_` is a numpy array, so we convert it to a list to find the index.
        target_col_idx = list(scaler.feature_names_in_).index(target_col_name)
    except ValueError:
        return None, f"Error: Target column '{target_col_name}' not found in the scaler's features."
    
    # Place our scaled prediction into the correct column of the dummy array.
    dummy_array[0, target_col_idx] = scaled_prediction
    
    # Use the scaler's `inverse_transform` method and extract our final, human-readable prediction.
    final_prediction = scaler.inverse_transform(dummy_array)[0, target_col_idx]

    return final_prediction, None


# --- 3. Streamlit User Interface ---
# This section defines the layout and interactive components of the web app.

st.title("Tetuan City Power Consumption Forecast")

st.markdown("""
This app uses a pre-trained Bidirectional LSTM model to forecast power consumption
for the next 10 minutes in one of three zones in Tetuan City.

**Instructions:**
1.  Upload the `Tetuan City power consumption.csv` dataset.
2.  Select a date and time from the dataset to use as the prediction point.
3.  Choose the Zone you want to forecast.
4.  Click "Forecast" to see the result.
""")

# Create a file uploader widget.
uploaded_file = st.file_uploader("Upload the power consumption CSV file", type="csv")

# This block of code only runs if the user has uploaded a file AND the assets were loaded successfully.
if uploaded_file and models and scaler:
    try:
        # Load the uploaded CSV into a pandas DataFrame.
        df = pd.read_csv(uploaded_file)
        # Apply the same column cleaning and datetime conversion as the training script.
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        # --- NEW: More robust DatetimeIndex creation ---
        # Convert 'DateTime' column, coercing any parsing errors into NaT (Not a Time).
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        
        # Check for and remove any rows that failed to parse into a valid date.
        if df['DateTime'].isnull().any():
            st.warning("Some rows had invalid date formats and were removed.")
            df.dropna(subset=['DateTime'], inplace=True)
        # Set the 'DateTime' column as the DataFrame index.
        df.set_index('DateTime', inplace=True)
        # Add this line to explicitly sort the DataFrame by its index.
        # This is crucial for reliable time-series slicing with .loc.
        df.sort_index(inplace=True)
        # Verification step: Ensure the index is sorted before proceeding.
        if not df.index.is_monotonic_increasing:
            st.error("Fatal Error: The DataFrame index could not be sorted correctly. Please check the CSV file for timestamp issues like duplicates or incorrect formatting.")
            st.stop()

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        st.stop() # Stop the script if the CSV is invalid.

    st.success("Dataset loaded successfully!")
    
    # Create a slider for the user to select a point in time for the prediction.
    # We subtract 24 hours from the start date to ensure there's always enough lookback data.
    min_date = (df.index.min() + timedelta(hours=24)).to_pydatetime()
    max_date = df.index.max().to_pydatetime()

    selected_date = st.slider(
        "Select a prediction point (date and time)",
        min_value=min_date,
        max_value=max_date,
        value=min_date, # Default value
        step=timedelta(minutes=10), # The step size matches our data's frequency.
        format="YYYY-MM-DD HH:mm"
    )

    # Create a dropdown menu for selecting the zone.
    model_selection = st.selectbox("Select Zone to Forecast", list(models.keys()))

    # Create a button. The code inside this `if` block only runs when the button is clicked.
    if st.button("Forecast Next 10 Minutes"):
        # `st.spinner` shows a loading message while the prediction is being calculated.
        with st.spinner('Forecasting...'):
            # Get all data up to the point the user selected.
            input_data = df.loc[:selected_date].copy()
            
            # Select the correct pre-loaded model based on the user's dropdown choice.
            selected_model = models[model_selection]
            
            # Call our prediction pipeline function.
            prediction, error = make_prediction(input_data, selected_model, scaler, model_selection)

            if error:
                st.error(error)
            else:
                # Display the final prediction in a metric card for emphasis.
                st.subheader(f"Forecast for {model_selection}")
                st.metric(
                    label=f"Predicted Power Consumption at {selected_date + timedelta(minutes=10)}",
                    value=f"{prediction:,.2f} kW"
                )
                
                # Show the user the exact data that was used for the prediction.
                st.subheader("Input Data Used for Prediction (Last 24 hours)")
                st.dataframe(input_data.tail(144))

# This message is shown if the app starts but cannot find the necessary model/scaler files.
elif not models or not scaler:
    st.warning("Models or scaler not loaded. Please ensure the asset files are in the same directory and the training script has been run.")

