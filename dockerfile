FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
import os; \
print('Downloading Qwen2.5-0.5B-Instruct model...'); \
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True); \
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True); \
os.makedirs('/app/models/qwen', exist_ok=True); \
tokenizer.save_pretrained('/app/models/qwen'); \
model.save_pretrained('/app/models/qwen'); \
print('Model cached successfully')"


# Copy source code
COPY ./src /app/src

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

CMD ["python", "src/main.py"]
