# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Korean study repository based on Sebastian Raschka's "Build a Large Language Model (From Scratch)" book. The repository contains:

1. **LLMs-from-scratch/** - Simplified version of the official code with chapter-by-chapter implementations
2. **Root directory** - Korean translation/study notes (build-a-large-language-model-from-scratch-ch##-ko.md) and English PDF extracts

Official repository: https://github.com/rasbt/LLMs-from-scratch

## Development Setup

### Environment Setup
```bash
# Navigate to the code directory
cd LLMs-from-scratch

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Linux/macOS

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running Jupyter Notebooks
```bash
# From LLMs-from-scratch/ directory
jupyter lab
```

### Deactivate Environment
```bash
deactivate
```

## Key Dependencies
- PyTorch (≥2.2.2) - Deep learning framework
- JupyterLab (≥4.0) - Interactive development
- tiktoken (≥0.5.1) - GPT tokenization
- matplotlib (≥3.7.1) - Visualization
- tensorflow (≥2.16.2 or 2.18.0) - For specific chapters
- tqdm (≥4.66.1) - Progress bars
- pandas (≥2.2.1) - Data manipulation

## Architecture Overview

### Core Model Components (Chapter 4)

The GPT model is built progressively through these key components:

1. **GPTDatasetV1** (`ch02/`) - Sliding window tokenization for training data
2. **MultiHeadAttention** (`ch03/`) - Causal multi-head self-attention mechanism
3. **TransformerBlock** (`ch04/gpt.py`) - Combines attention + feedforward with residual connections
4. **GPTModel** (`ch04/gpt.py`) - Complete transformer architecture with:
   - Token + positional embeddings
   - Stacked transformer blocks
   - Layer normalization
   - Output projection head

Configuration dictionary defines model size (e.g., GPT_CONFIG_124M for 124M parameter model).

### Training Pipeline (Chapter 5+)

Training scripts follow this pattern:
- `previous_chapters.py` - Imports core model components from earlier chapters
- `gpt_train.py` - Training loop with loss calculation and optimization
- `gpt_generate.py` - Text generation utilities
- `gpt_download.py` - Model checkpoint loading

### Fine-tuning Approaches

- **ch06/** - Classification fine-tuning (`gpt_class_finetune.py`)
- **ch07/** - Instruction fine-tuning (`gpt_instruction_finetuning.py`)
- **appendix-E/** - LoRA (Low-Rank Adaptation) efficient fine-tuning

### Distributed Training (Appendix A)

- `DDP-script.py` - PyTorch Distributed Data Parallel
- `DDP-script-torchrun.py` - Using torchrun for multi-GPU training

## Running Scripts

### Train a Model
```bash
cd LLMs-from-scratch/ch05
python gpt_train.py
```

### Generate Text
```bash
cd LLMs-from-scratch/ch05
python gpt_generate.py
```

### Fine-tune for Classification
```bash
cd LLMs-from-scratch/ch06
python gpt_class_finetune.py
```

### Instruction Fine-tuning
```bash
cd LLMs-from-scratch/ch07
python gpt_instruction_finetuning.py
```

## Code Organization Pattern

Each chapter follows this structure:
- `ch##/ch##.ipynb` - Main chapter notebook with step-by-step implementation
- `ch##/exercise-solutions.ipynb` - Solutions to chapter exercises
- `ch##/*.py` - Standalone scripts for training/inference
- `ch##/previous_chapters.py` - Imports components from earlier chapters

## Working with Korean Documentation

Korean study notes are in the root directory:
- `build-a-large-language-model-from-scratch-ch01-ko.md` - Introduction
- `build-a-large-language-model-from-scratch-ch02-ko.md` - Text data and tokenization
- `build-a-large-language-model-from-scratch-ch03-ko.md` - Attention mechanisms
- `build-a-large-language-model-from-scratch-ch04-ko.md` - GPT model implementation

English PDF extracts are also available for chapters 5-7.

## Important Notes

- GPU will be automatically used if available (PyTorch auto-detection)
- The repository structure is simplified from the official version for learning purposes
- All notebooks should be run from within the `LLMs-from-scratch/` directory
- Model checkpoints and datasets are downloaded automatically by scripts when needed
