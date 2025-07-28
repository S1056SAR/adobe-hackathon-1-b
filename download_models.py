# download_models.py
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    """
    Downloads and saves all necessary models from Hugging Face.
    This script is run once during the Docker build process.
    """
    print("Downloading Sentence Transformer model...")
    # Download the embedding model
    embedder = SentenceTransformer("intfloat/e5-small-v2")
    embedder.save("./models/e5-small-v2")
    print("Embedding model saved.")

    print("Downloading TinyLlama model...")
    # Download the language model and its tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    model.save_pretrained("./models/tinyllama-1.1b")
    tokenizer.save_pretrained("./models/tinyllama-1.1b")
    print("TinyLlama model saved.")
    
    print("All models downloaded successfully.")

if __name__ == "__main__":
    main()
