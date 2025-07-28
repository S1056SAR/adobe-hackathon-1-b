# Use the official Python 3.10 slim image as a base
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyMuPDF, Tesseract OCR, and llama.cpp
# cmake is required for llama-cpp-python to build its C++ backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and the model download script
COPY requirements.txt .
COPY download_models.py .

# Install all Python dependencies
# bitsandbytes is removed as it's no longer used
# The --extra-index-url flag tells pip to use the PyTorch repo IN ADDITION to the default PyPI
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Create models directory (important for local_dir_use_symlinks=False for HuggingFace Hub)
RUN mkdir -p models/e5-small-v2 models/tinyllama-1.1b-gguf

# --- Download models in separate steps to manage memory ---
# Step 1: Download the smaller e5-small-v2 model
# This step is run via a Python snippet that calls the function from download_models.py
RUN python -c "from download_models import download_e5_model; download_e5_model()"

# Step 2: Download the larger TinyLlama GGUF model
# This step is run via a Python snippet that calls the function from download_models.py
RUN python -c "from download_models import download_tinyllama_model; download_tinyllama_model()"
# --- End of model download changes ---

# Now, copy the rest of your application code
COPY src/ ./src/

# Set an environment variable to ensure Python output is sent straight to the terminal
ENV PYTHONUNBUFFERED=1

# Specify the command to run on container startup
ENTRYPOINT ["python", "-m", "src.main"]
