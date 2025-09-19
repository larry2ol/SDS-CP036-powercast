#!/bin/bash

# Define the name for the Docker image
IMAGE_NAME="power-consumption-api"

# Build the Docker image
# -t flag is used to tag the image with the specified name
# . at the end specifies the build context, which is the current directory
docker build -t $IMAGE_NAME .

echo "Docker image '$IMAGE_NAME' built successfully."
