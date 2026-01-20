# PhantomX

> ⚠️ **Experimental / Learning Project** - Not for production use

> 📓 **[RESEARCH_LOG.md](RESEARCH_LOG.md)** - Detailed experiment notes, results, and analysis

Personal research sandbox for exploring LaBraM-POYO neural decoding approaches for BCI applications.

## 🎯 Results

**Gap to LSTM: 0.43%** - R² = 0.776 on MC_Maze velocity decoding

| Model | R² | R² vx | R² vy | Codes | Time |
|-------|-----|-------|-------|-------|------|
| **RVQ-4 (4 layers × 128)** | **0.776** | 0.80 | 0.75 | 454 | 6.2min |
| Residual Gumbel VQ | 0.771 | 0.78 | 0.77 | 167 | 6.5min |
| Deep CausalTransformer | 0.773 | 0.80 | 0.74 | 118 | 66min |
| Progressive VQ-VAE | 0.71 | 0.71 | 0.72 | 218 | 174s |
| **Raw LSTM (baseline)** | 0.78 | - | - | - | - |

**Pre-training encoder alone: R² = 0.784 (exceeds LSTM!)**

## Key Findings

1. **Temporal context is essential**: Single timestep R² ≈ 0.10, with 250ms history R² ≈ 0.78
2. **POYO trade-off**: Full permutation invariance → R² ≈ 0 (destroys velocity info)
3. **Codebook collapse**: Standard VQ training uses only 3-8% of codes
4. **Progressive training is key**: Pre-train → k-means init → finetune prevents collapse
5. **Residual VQ breaks Voronoi ceiling**: Multi-stage quantization captures fine details
6. **RVQ-4 optimal**: 4 layers × 128 codes, more layers = diminishing returns

## What This Is

An experimental project exploring:
- VQ-VAE based neural codebooks
- POYO-style spike tokenization
- Causal Transformer encoders with Gumbel-Softmax VQ
- Test-time adaptation for signal drift
- Zero-shot velocity decoding from motor cortex data

## Main Documentation

📓 **[RESEARCH_LOG.md](RESEARCH_LOG.md)** - Detailed experiment notes, results, and analysis

## Project Structure

```
python/phantomx/
    model.py           # ProgressiveVQVAE (MLP-based)
    models_extended.py # CausalTransformerVQVAE, GumbelVQVAE (best performers)
    trainer.py         # ProgressiveTrainer (3-phase training)
    tta.py             # Test-Time Adaptation (TTAWrapper, OnlineTTA)
    tokenizer/         # Spike tokenization
    data/              # MC_Maze data loading
python/
    exp10_beat_lstm.py # Latest: CausalTransformer + Gumbel experiments
    compare_models.py  # Model comparisons
models/
    exp9_progressive_vqvae.pt   # Progressive VQ-VAE (R²=0.71)
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

✅ **R² = 0.77 achieved** - Only 0.9% gap from raw LSTM baseline (0.78)

### Latest: Experiment 11 - Residual Gumbel VQ

- **Residual VQ** with learnable α preserves continuous nuance alongside discrete codes
- **Pre-training reaches R² = 0.77** (LSTM parity during encoder-only phase)
- **6.5 min training** on A100 GPU (Fly.io deployment)
- **Balanced vx/vy**: R² = 0.78 (vx), 0.77 (vy) - no more velocity bias
- VQ bottleneck accounts for remaining 1% gap

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for full experiment details
