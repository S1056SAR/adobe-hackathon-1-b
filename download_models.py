# download_models.py
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os
from transformers import AutoTokenizer, AutoModelForCausalLM 

def download_e5_model():
    print("Downloading Sentence Transformer (e5-small-v2) model...")
    embedder = SentenceTransformer("intfloat/e5-small-v2")
    os.makedirs("./models/e5-small-v2", exist_ok=True)
    embedder.save("./models/e5-small-v2")
    print("e5-small-v2 model saved.")

def download_tinyllama_model():
    print("Downloading TinyLlama (1.1B GGUF) model...")
    # --- CRITICAL CHANGE: Use Q5_K_M quantization instead of Q4_K_M ---
    model_path = hf_hub_download(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", # Changed from Q4_K_M
        local_dir="./models/tinyllama-1.1b-gguf", 
        local_dir_use_symlinks=False 
    )
    print(f"TinyLlama GGUF model saved to: {model_path}")

if __name__ == "__main__":
    pass
