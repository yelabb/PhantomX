# PhantomX

> ⚠️ **Experimental / Learning Project** - Not for production use

> 📓 **[RESEARCH_LOG.md](RESEARCH_LOG.md)** - Detailed experiment notes, results, and analysis

Personal research sandbox for exploring neural decoding approaches for BCI applications.

<img width="900" alt="image" src="https://github.com/user-attachments/assets/eef898bc-5c79-4ac7-a1e6-d34ec617fa86" />

## 🎯 Results

🏆 **Current Winner: [RVQ-4 (Residual Vector Quantization)](RESEARCH_LOG.md#experiment-12-residual-vector-quantization-rvq)** — R² = 0.776, only 0.43% gap to raw LSTM

| Model | R² | R² vx | R² vy | Codes | Time |
|-------|-----|-------|-------|-------|------|
| **[RVQ-4](RESEARCH_LOG.md#experiment-12-residual-vector-quantization-rvq)** | **0.776** | 0.80 | 0.75 | 454 | 6.2min |
| [Deep CausalTransformer](RESEARCH_LOG.md#experiment-11-beat-the-lstm---architecture-upgrade) | 0.773 | 0.80 | 0.74 | 118 | 66min |
| [Residual Gumbel VQ](RESEARCH_LOG.md#experiment-11-close-the-final-gap) | 0.771 | 0.78 | 0.77 | 167 | 6.5min |
| [Progressive VQ-VAE](RESEARCH_LOG.md#experiment-9-progressive-training-breakthrough) | 0.71 | 0.71 | 0.72 | 218 | 174s |
| [LADR-VQ v2](RESEARCH_LOG.md#experiment-18-ladr-vq-v2-teacher-student-distillation--lag-tuning) | 0.695 | 0.70 | 0.69 | 385 | 4.6min |
| [FSQ-VAE](RESEARCH_LOG.md#experiment-14-the-fsq-pivot-) | 0.64 | - | - | ~5 | 150ep |
| [Manifold FSQ](RESEARCH_LOG.md#experiment-15-manifold-fsq-vae-triple-loss) | 0.60 | - | - | - | 150ep |
| [Frankenstein](RESEARCH_LOG.md#experiment-16-the-frankenstein-pivot) | ~0.72 | - | - | - | ⏳ |
| **Raw LSTM (baseline)** | 0.78 | - | - | - | - |

**Pre-training encoder alone: R² = 0.784 (exceeds LSTM!)** — see [Exp 12 analysis](RESEARCH_LOG.md#experiment-12-residual-vector-quantization-rvq)

## Key Findings

1. **Temporal context is essential**: Single timestep R² ≈ 0.10, with 250ms history R² ≈ 0.78
2. **POYO trade-off**: Full permutation invariance → R² ≈ 0 (destroys velocity info)
3. **Codebook collapse**: Standard VQ training uses only 3-8% of codes
4. **Progressive training is key**: Pre-train → k-means init → finetune prevents collapse
5. **Residual VQ breaks Voronoi ceiling**: Multi-stage quantization captures fine details
6. **RVQ-4 optimal**: 4 layers × 128 codes, more layers = diminishing returns
7. **FSQ topology doesn't help**: Ordinal code structure underperforms discrete VQ (Exp 14)
8. **Distillation eliminates VQ tax**: Exp 18 proved 0% discretization loss with latent distillation
9. **Lag tuning (Δ=+1) hurts**: Predicting 25ms ahead decorrelates signal on MC_Maze

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

✅ **R² = 0.776 achieved** - Only 0.43% gap from raw LSTM baseline (0.78)

### Latest: Experiment 12 - Residual Vector Quantization (RVQ)

- **RVQ-4** (4 layers × 128 codes) breaks the "Voronoi Ceiling"
- **Pre-training reaches R² = 0.784** (exceeds LSTM!)
- **6.2 min training** on A100 GPU (Fly.io deployment)
- **Strong vx decoding**: R² = 0.80 (vx), 0.75 (vy)
- VQ bottleneck accounts for remaining 0.43% gap

### Failed: Experiment 13 - Wide-Window Mamba

- ❌ 80-bin (2s) context windows **hurt** performance (R² = 0.73)
- 250ms is the optimal window - more context = more noise
- Stateless Mamba on long windows doesn't leverage SSM advantages

### In Progress: Experiment 17 - Lag-Aware Distilled RVQ-4 (LADR-VQ)

- ⚠️ **BLOCKED** - RVQ initialization bug discovered
- Lag sweep complete: Δ=+1 (25ms ahead) shows best results
- Teacher R² = 0.67 (low due to initialization bug using 4 codes instead of 128)
- Fix required: Initialize RVQ codebooks AFTER encoder pre-training

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for full experiment details


## Dream
<img width="450" alt="image" src="https://github.com/user-attachments/assets/af41ec36-29ea-4560-93bc-007247c36227" />


