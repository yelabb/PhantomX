# PhantomX

> ⚠️ **Experimental / Learning Project** - Not for production use

Personal research sandbox for exploring LaBraM-POYO neural decoding approaches for BCI applications.

## 🎯 Results

**Target achieved!** R² = 0.70 on MC_Maze velocity decoding with Progressive VQ-VAE

| Experiment | R² | Notes |
|------------|-----|-------|
| Baseline VQ-VAE | 0.007 | POYO tokenization destroys channel identity |
| Raw LSTM | 0.78 | Best possible (no tokenization) |
| Progressive VQ-VAE | **0.70** | Pre-train encoder, k-means init, finetune |

## Key Findings

1. **Temporal context is essential**: Single timestep R² ≈ 0.10, with 250ms history R² ≈ 0.78
2. **POYO trade-off**: Full permutation invariance → R² ≈ 0 (destroys velocity info)
3. **Codebook collapse**: Standard VQ training uses only 3-8% of codes
4. **Progressive training solves it**: Pre-train → k-means init → finetune achieves 86% codebook utilization

## What This Is

An experimental project exploring:
- VQ-VAE based neural codebooks
- POYO-style spike tokenization
- Test-time adaptation for signal drift
- Zero-shot velocity decoding from motor cortex data

## Main Documentation

📓 **[RESEARCH_LOG.md](RESEARCH_LOG.md)** - Detailed experiment notes, results, and analysis

## Project Structure

```
python/phantomx/     # Core modules
    model.py         # ProgressiveVQVAE (final architecture)
    trainer.py       # ProgressiveTrainer (3-phase training)
    tokenizer.py     # Spike tokenization
    data.py          # MC_Maze data loading
models/              # Trained model checkpoints
    exp9_progressive_vqvae.pt  # Best model (R²=0.70)
```

## Quick Start

```python
from phantomx.model import ProgressiveVQVAE, load_model
from phantomx.trainer import ProgressiveTrainer
from phantomx.data import MCMazeDataset, create_dataloaders

# Load data
dataset = MCMazeDataset("path/to/mc_maze.nwb")
train_loader, val_loader, test_loader = create_dataloaders(dataset)

# Create and train model
model = ProgressiveVQVAE(n_channels=142, window_size=10)
trainer = ProgressiveTrainer(model, train_loader, val_loader)
result = trainer.train()
print(f"Best R²: {result['best_r2']:.4f}")

# Evaluate
test_result = trainer.evaluate(test_loader)
print(f"Test R²: {test_result['r2']:.4f}")
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Current Status

✅ **Complete** - See [RESEARCH_LOG.md](RESEARCH_LOG.md) for full experiment details
