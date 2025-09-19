
# Specify a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI application file (assuming it's named main.py)
COPY main.py .

# Copy the saved model and scaler files
COPY best_power_consumption_model.keras .
COPY x_scaler.pkl .
COPY y_scaler.pkl .

# Expose the port that your FastAPI application will run on
EXPOSE 8000

# Define the command to run your FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
