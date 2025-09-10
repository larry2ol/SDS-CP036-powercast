#!/bin/bash

# Start script for Render.com deployment
echo "Starting Powercast API..."

# Set environment variables
export PYTHONPATH=/opt/render/project/src
export PORT=${PORT:-8000}

# Start the application
exec gunicorn app:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --log-level info \
    --access-logfile - \
    --error-logfile -