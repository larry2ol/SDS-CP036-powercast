#  Energy Consumption Forecasting App
This project forecasts energy consumption in **three distinct zones** using time series modeling. It supports both **10-minute** and **hourly** prediction intervals, enabling granular insights into regional energy demand.
##  Overview
The application leverages historical input data to predict future energy usage across three zones. It uses a trained deep learning model (`model.keras`) and preprocessing scalers to normalize inputs and outputs. The app is containerized with Docker and designed for deployment on platforms like Render.
##  Features

- Forecast energy consumption for Zone A, Zone B, and Zone C
- Supports both 10-minute and hourly prediction intervals
- Scalable backend powered by FastAPI (or Flask)
- Dockerized for easy deployment
- RESTful API endpoints for integration
- Preprocessing with saved scalers (`x_scaler.pkt`, `y_scaler.pkt`)
- Optional Jupyter notebook (`forecast.ipynb`) for experimentation

