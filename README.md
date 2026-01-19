# PhantomX

> ⚠️ **Experimental / Learning Project** - Not for production use

Personal research sandbox for exploring LaBraM-POYO neural decoding approaches for BCI applications.

## 🎯 Results

**Target achieved!** R² = 0.71 on MC_Maze velocity decoding with Progressive VQ-VAE

| Model | R² | Codes Used | Training Time |
|-------|-----|------------|---------------|
| Progressive VQ-VAE | **0.71** | 218/256 | 174s |
| Transformer VQ-VAE | 0.55 | 163/256 | 1088s |
| Gumbel VQ-VAE | 0.00 | 1/256 | 121s |
| Baseline (no VQ) | 0.78 | - | - |

## Key Findings

1. **Temporal context is essential**: Single timestep R² ≈ 0.10, with 250ms history R² ≈ 0.78
2. **POYO trade-off**: Full permutation invariance → R² ≈ 0 (destroys velocity info)
3. **Codebook collapse**: Standard VQ training uses only 3-8% of codes
4. **Progressive training solves it**: Pre-train → k-means init → finetune achieves 85% codebook utilization
5. **Training strategy > Architecture**: Simple MLP beats Transformer when properly trained

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
python/phantomx/
    model.py           # ProgressiveVQVAE (final architecture)
    models_extended.py # TransformerVQVAE, GumbelVQVAE variants
    trainer.py         # ProgressiveTrainer (3-phase training)
    tta.py             # Test-Time Adaptation (TTAWrapper, OnlineTTA)
    tokenizer.py       # Spike tokenization
    data.py            # MC_Maze data loading
python/
    compare_models.py  # Run all model comparisons
models/
    exp9_progressive_vqvae.pt   # Best model (R²=0.71)
    comparison_results.json     # All experiment results
```

## Quick Start

```python
from phantomx.model import ProgressiveVQVAE
from phantomx.trainer import ProgressiveTrainer
from phantomx.data import MCMazeDataset

# Load data
dataset = MCMazeDataset("path/to/mc_maze.nwb")
train_loader, val_loader, test_loader = create_dataloaders(dataset)

# Create and train model
model = ProgressiveVQVAE(n_channels=142, window_size=10)
trainer = ProgressiveTrainer(model, train_loader, val_loader)
result = trainer.train()
print(f"Best R²: {result['best_r2']:.4f}")

# Test-Time Adaptation for new sessions
from phantomx.tta import OnlineTTA
tta = OnlineTTA(model)
predictions = tta.predict(new_data)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Current Status

✅ **Complete** - See [RESEARCH_LOG.md](RESEARCH_LOG.md) for full experiment details
