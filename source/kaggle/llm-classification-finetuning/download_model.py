"""
Download DistilBERT model from HuggingFace and save locally
This will be uploaded to Kaggle as a dataset
"""

from transformers import AutoTokenizer, AutoModel
import os

# Configuration
MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "models/distilbert-base-uncased"

print(f"Downloading {MODEL_NAME} from HuggingFace...")
print(f"This will be saved to: {SAVE_DIR}")
print("=" * 60)

# Create directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Download tokenizer
print("\n1. Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✓ Tokenizer saved to {SAVE_DIR}")

# Download model
print("\n2. Downloading model...")
model = AutoModel.from_pretrained(MODEL_NAME)
model.save_pretrained(SAVE_DIR)
print(f"✓ Model saved to {SAVE_DIR}")

# Verify saved files
print("\n3. Verifying saved files...")
saved_files = os.listdir(SAVE_DIR)
print(f"Saved files: {saved_files}")

required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'vocab.txt']
missing_files = [f for f in required_files if f not in saved_files]

if missing_files:
    print(f"\n⚠ WARNING: Missing files: {missing_files}")
else:
    print(f"\n✓ All required files present!")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE!")
print("=" * 60)
print(f"\nModel saved in: {os.path.abspath(SAVE_DIR)}")
print(f"Size: ~{sum(os.path.getsize(os.path.join(SAVE_DIR, f)) for f in os.listdir(SAVE_DIR)) / (1024*1024):.1f} MB")
print("\nNext steps:")
print("1. Zip the 'models' folder")
print("2. Upload to Kaggle as a new dataset")
print("3. Use the uploaded model in your notebook")
