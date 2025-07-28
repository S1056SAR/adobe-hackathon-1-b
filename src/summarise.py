# src/summarise.py

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class Summariser:
    def __init__(self, model_path):
        # --- This is the key change ---
        # Explicitly set the quantization type to "nf4" for CPU compatibility
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4"  # Use NormalFloat4 for CPU
        )

        # Load tokenizer and model from the local, pre-downloaded path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="cpu"
        )

    def refine(self, section, persona, job):
        prompt = (
            f"As a {persona} working to {job}, rewrite this excerpt for direct action:\n"
            f"{section['text']}\nSummary:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
        
        # Use torch.no_grad() for inference to save memory and computations
        with torch.no_grad():
            # Set pad_token_id to eos_token_id to prevent warnings
            out = self.model.generate(**inputs, max_new_tokens=80, pad_token_id=self.tokenizer.eos_token_id)
        
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Clean the output to remove the prompt from the generated text
        clean_text = text.replace(prompt, '').strip()
        return {"doc": section["doc"], "page": section["page"], "text": clean_text}

def refine_texts(ranked, persona, job):
    print("   - Initializing summarization model...")
    # Point to the local, pre-downloaded model directory
    summariser = Summariser("./models/tinyllama-1.1b")
    print("   - Model initialized. Starting text refinement...")
    
    refined_results = []
    for i, sec in enumerate(ranked):
        print(f"     - Refining section {i + 1}/{len(ranked)}...")
        refined_results.append(summariser.refine(sec, persona, job))
    
    return refined_results
