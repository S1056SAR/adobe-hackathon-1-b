# Use the official Python 3.10 slim image as a base
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyMuPDF and Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and the download script
COPY requirements.txt .
COPY download_models.py .

# Install all Python dependencies
# The --extra-index-url flag tells pip to use the PyTorch repo IN ADDITION to the default PyPI
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# --- This is the key change ---
# Run the download script to fetch and save models into the image
RUN python download_models.py

# Now, copy the rest of your application code
COPY src/ ./src/
# The prompt.txt will be mounted via volume, so no need to copy it
# COPY prompt.txt .

# Set an environment variable to ensure Python output is sent straight to the terminal
ENV PYTHONUNBUFFERED=1

# Specify the command to run on container startup
ENTRYPOINT ["python", "-m", "src.main"]
