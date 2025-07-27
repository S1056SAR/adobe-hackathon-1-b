FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy download script
COPY download_reliable_model.py /app/download_reliable_model.py

# Download model with better error visibility
RUN python download_reliable_model.py

# Debug: Show what was actually created
RUN echo "=== DEBUGGING MODEL DIRECTORY ===" && \
    ls -la /app/models/ && \
    echo "=== DISTILBERT DIRECTORY CONTENTS ===" && \
    ls -la /app/models/distilbert/ || echo "DistilBERT directory does not exist" && \
    echo "=== DISTILGPT2 DIRECTORY CONTENTS ===" && \
    ls -la /app/models/distilgpt2/ || echo "DistilGPT2 directory does not exist"

# Verify model exists (adjust path based on which model you used)
RUN test -f /app/models/distilbert/config.json || test -f /app/models/distilgpt2/config.json || (echo "❌ No model found!" && exit 1)

# Copy source code
COPY src/ /app/src/

# Set environment
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["python", "src/main.py"]
