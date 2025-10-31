"""
Download DistilBERT model from HuggingFace and save locally
This will be uploaded to Kaggle as a dataset
"""

from transformers import AutoTokenizer, AutoModel
import os
import zipfile

# Configuration
MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "models/distilbert-base-uncased"
ZIP_FILE = "models/distilbert-base-uncased.zip"

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

required_files = ['config.json', 'tokenizer_config.json', 'vocab.txt']
missing_files = [f for f in required_files if f not in saved_files]

if missing_files:
    print(f"\n⚠ WARNING: Missing files: {missing_files}")
else:
    print(f"\n✓ All required files present!")

# Calculate model size
model_size_mb = sum(os.path.getsize(os.path.join(SAVE_DIR, f)) for f in os.listdir(SAVE_DIR)) / (1024*1024)
print(f"\nModel size: ~{model_size_mb:.1f} MB")

# Create ZIP file
print("\n4. Creating ZIP file...")
print(f"Compressing models/ folder into {ZIP_FILE}...")

with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Walk through the models directory
    for root, dirs, files in os.walk('models'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = file_path  # Keep the 'models/' prefix in zip
            zipf.write(file_path, arcname)
            print(f"  Added: {arcname}")

zip_size_mb = os.path.getsize(ZIP_FILE) / (1024*1024)
print(f"\n✓ ZIP file created: {ZIP_FILE}")
print(f"✓ ZIP size: ~{zip_size_mb:.1f} MB")

print("\n" + "=" * 60)
print("DOWNLOAD AND COMPRESSION COMPLETE!")
print("=" * 60)
print(f"\nModel saved in: {os.path.abspath(SAVE_DIR)}")
print(f"ZIP file: {os.path.abspath(ZIP_FILE)}")
print(f"\nCompression ratio: {(1 - zip_size_mb/model_size_mb)*100:.1f}%")
print("\nNext steps:")
print("1. Upload the ZIP file to Kaggle as a new dataset:")
print(f"   - File: {ZIP_FILE}")
print("   - Title: distilbert-base-uncased")
print("2. Add the dataset to your Kaggle notebook")
print("3. Use the uploaded model in your notebook")