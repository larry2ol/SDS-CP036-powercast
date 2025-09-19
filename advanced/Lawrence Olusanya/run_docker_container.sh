#!/bin/bash

# Define the name for the Docker image
IMAGE_NAME="power-consumption-api"

# Define host and container ports
HOST_PORT=8000
CONTAINER_PORT=8000

# Run the Docker container in detached mode and map ports
docker run -d -p $HOST_PORT:$CONTAINER_PORT $IMAGE_NAME

echo "Docker container '$IMAGE_NAME' is starting on port $HOST_PORT."
