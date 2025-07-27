import os
import sys
import traceback
from transformers import AutoTokenizer, AutoModel

print("=== DISTILBERT DOWNLOAD WITH FULL DEBUGGING ===")

model_dir = "/app/models/distilbert"
os.makedirs(model_dir, exist_ok=True)

try:
    print("Step 1: Starting DistilBERT download...")
    print(f"Target directory: {model_dir}")
    
    # Download tokenizer first
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("✅ Tokenizer downloaded successfully")
    
    # Download model
    print("Downloading model...")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    print("✅ Model downloaded successfully")
    
    print("Step 2: Testing model functionality...")
    test_input = tokenizer("This is a test", return_tensors="pt")
    test_output = model(**test_input)
    print(f"✅ Model test successful! Output shape: {test_output.last_hidden_state.shape}")
    
    print("Step 3: Saving tokenizer...")
    tokenizer.save_pretrained(model_dir)
    print("✅ Tokenizer saved")
    
    print("Step 4: Saving model...")
    model.save_pretrained(model_dir)
    print("✅ Model saved")
    
    print("Step 5: Verifying saved files...")
    saved_files = os.listdir(model_dir)
    print(f"Files in {model_dir}: {saved_files}")
    
    # Check for required files (accept both pytorch_model.bin and model.safetensors)
    required_files = ["config.json", "tokenizer_config.json", "vocab.txt"]
    model_files = ["pytorch_model.bin", "model.safetensors"]  # Either format is acceptable
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} exists ({file_size} bytes)")
        else:
            missing_files.append(file)
            print(f"❌ {file} MISSING")
    
    # Check for model file (either format)
    model_file_found = False
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"✅ {model_file} exists ({file_size} bytes)")
            model_file_found = True
            break
    
    if not model_file_found:
        missing_files.append("pytorch_model.bin or model.safetensors")
        print(f"❌ No model file found (looking for pytorch_model.bin or model.safetensors)")
    
    if missing_files:
        print(f"💥 CRITICAL ERROR: Missing required files: {missing_files}")
        sys.exit(1)
    
    print("🏆 SUCCESS: DistilBERT model fully downloaded and verified!")
    print(f"📊 Model size: ~250MB")
    
except Exception as e:
    print(f"💥 CRITICAL ERROR during download: {e}")
    print("Full traceback:")
    traceback.print_exc()
    
    # Show what files actually exist
    if os.path.exists(model_dir):
        print(f"Files that were created: {os.listdir(model_dir)}")
    else:
        print("Model directory was not created")
    
    sys.exit(1)
