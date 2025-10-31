# ==============================================================================
# LLM Classification Fine-tuning Baseline
# Competition: https://www.kaggle.com/competitions/llm-classification-finetuning
#
# INSTRUCTIONS FOR KAGGLE:
# 1. Upload distilbert-base-uncased.zip as a Kaggle dataset
# 2. Add that dataset to this notebook
# 3. Update MODEL_PATH below to match your dataset path
# 4. Enable GPU (Settings → GPU T4 x2 or P100)
# 5. Run the notebook
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
print("Random seeds set to 42")

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    # UPDATE THIS: Path to your uploaded DistilBERT dataset
    # The path should point to the folder containing config.json
    # Check the exact path in your Kaggle Input panel
    MODEL_NAME = "/kaggle/input/distilbert-base-uncased/pytorch/default/1/models/distilbert-base-uncased"

    # If you see error, try to find the correct path by checking:
    # import os
    # for root, dirs, files in os.walk('/kaggle/input/distilbert-base-uncased'):
    #     if 'config.json' in files:
    #         print(f"Model path: {root}")
    #         break

    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 1  # Increase for better performance (e.g., 3-5 epochs)
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data paths
    TRAIN_DATA_PATH = "/kaggle/input/llm-classification-finetuning/train.csv"
    TEST_DATA_PATH = "/kaggle/input/llm-classification-finetuning/test.csv"
    SAMPLE_SUBMISSION_PATH = "/kaggle/input/llm-classification-finetuning/sample_submission.csv"

    # Output paths
    SUBMISSION_PATH = "/kaggle/working/submission.csv"
    MODEL_SAVE_PATH = "/kaggle/working/best_model.pt"

config = Config()
print(f"\nConfiguration:")
print(f"  Model: {config.MODEL_NAME}")
print(f"  Device: {config.DEVICE}")
print(f"  Batch Size: {config.BATCH_SIZE}")
print(f"  Epochs: {config.EPOCHS}")

# Verify model path exists
if not os.path.exists(config.MODEL_NAME):
    print(f"\n⚠ ERROR: Model path not found: {config.MODEL_NAME}")
    print("Please update MODEL_NAME to match your uploaded dataset path")
    print("\nAvailable inputs:")
    if os.path.exists('/kaggle/input'):
        for item in os.listdir('/kaggle/input'):
            print(f"  - /kaggle/input/{item}")
else:
    print(f"✓ Model path exists: {config.MODEL_NAME}")

# ==============================================================================
# Load Data
# ==============================================================================
print("\nLoading data...")
train_df = pd.read_csv(config.TRAIN_DATA_PATH)
test_df = pd.read_csv(config.TEST_DATA_PATH)
sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Split train/validation
train_data, val_data = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df['winner_model_a'].astype(str) + train_df['winner_model_b'].astype(str)
)
print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

# ==============================================================================
# Dataset Class
# ==============================================================================
class LLMComparisonDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{row['prompt']} [SEP] {row['response_a']} [SEP] {row['response_b']}"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        if not self.is_test:
            labels = torch.tensor([
                row['winner_model_a'],
                row['winner_model_b'],
                row['winner_tie']
            ], dtype=torch.float)
            item['labels'] = labels

        return item

print("Dataset class defined")

# ==============================================================================
# Model Class
# ==============================================================================
class LLMComparisonModel(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super(LLMComparisonModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        probs = self.softmax(logits)
        return probs

print("Model class defined")

# ==============================================================================
# Initialize Model
# ==============================================================================
print(f"\nLoading tokenizer and model from: {config.MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
model = LLMComparisonModel(config.MODEL_NAME)
model.to(config.DEVICE)
print(f"✓ Model loaded on: {config.DEVICE}")
print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==============================================================================
# Create Data Loaders
# ==============================================================================
train_dataset = LLMComparisonDataset(train_data, tokenizer, config.MAX_LENGTH)
val_dataset = LLMComparisonDataset(val_data, tokenizer, config.MAX_LENGTH)
test_dataset = LLMComparisonDataset(test_df, tokenizer, config.MAX_LENGTH, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# ==============================================================================
# Training Functions
# ==============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = nn.BCELoss()(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    loss = log_loss(actuals, predictions)

    return loss, predictions, actuals

print("Training functions defined")

# ==============================================================================
# Setup Optimizer and Scheduler
# ==============================================================================
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
total_steps = len(train_loader) * config.EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print(f"Optimizer: AdamW (lr={config.LEARNING_RATE})")
print(f"Total training steps: {total_steps}")

# ==============================================================================
# Training Loop
# ==============================================================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

best_val_loss = float('inf')

for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
    print("-" * 60)

    train_loss = train_epoch(model, train_loader, optimizer, scheduler, config.DEVICE)
    print(f"Training loss: {train_loss:.4f}")

    val_loss, val_preds, val_actuals = validate(model, val_loader, config.DEVICE)
    print(f"Validation loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"✓ Model saved with validation loss: {val_loss:.4f}")

print("\n" + "="*60)
print(f"TRAINING COMPLETED! Best validation loss: {best_val_loss:.4f}")
print("="*60)

# ==============================================================================
# Load Best Model and Make Predictions
# ==============================================================================
print("\nLoading best model for inference...")
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
model.eval()
print("✓ Best model loaded")

print("\nMaking predictions on test data...")
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        outputs = model(input_ids, attention_mask)
        predictions.append(outputs.cpu().numpy())

predictions = np.vstack(predictions)
print(f"✓ Predictions shape: {predictions.shape}")

# ==============================================================================
# Create Submission File
# ==============================================================================
submission = sample_submission.copy()
submission['winner_model_a'] = predictions[:, 0]
submission['winner_model_b'] = predictions[:, 1]
submission['winner_tie'] = predictions[:, 2]

submission.to_csv(config.SUBMISSION_PATH, index=False)

print("\n" + "="*60)
print("SUBMISSION FILE CREATED")
print("="*60)
print(f"Saved to: {config.SUBMISSION_PATH}")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))

# Verify probabilities sum to 1
prob_sums = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1)
print(f"\nProbability sums check:")
print(f"  Min: {prob_sums.min():.6f}")
print(f"  Max: {prob_sums.max():.6f}")
print(f"  Mean: {prob_sums.mean():.6f}")

if abs(prob_sums.mean() - 1.0) < 0.001:
    print("\n✓ All probabilities sum to ~1.0. Submission is valid!")
else:
    print("\n⚠ WARNING: Probabilities don't sum to 1.0!")

print("\n" + "="*60)
print("READY TO SUBMIT!")
print("="*60)
print("\nNow submit this notebook to the competition!")
