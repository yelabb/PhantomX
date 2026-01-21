# PhantomX

> ⚠️ **Experimental / Learning Project** - Not for production use

> 📓 **[RESEARCH_LOG.md](RESEARCH_LOG.md)** - Detailed experiment notes, results, and analysis

PhantomX — Neural Decoding as a Codec: Quantized Latent Representations for Robust BCI

<img width="900" alt="image" src="https://github.com/user-attachments/assets/eef898bc-5c79-4ac7-a1e6-d34ec617fa86" />

## 🎯 Results

🔬 **[Exp 23: Statistical Validation COMPLETE](RESEARCH_LOG.md#experiment-23-statistical-validation)** — LSTM wins!

### Validated Results (5 seeds each)

| Model | R² (mean ± std) | 95% CI | Verdict |
|-------|-----------------|--------|--------|
| 🥇 **LSTM (aug)** | **0.8015 ± 0.007** | [0.793, 0.810] | ✅ **WINNER** |
| 🥈 Wide Transformer (aug) | 0.7906 ± 0.034 | [0.749, 0.833] | ❌ High variance |
| LSTM (no aug) | ⏳ Pending | — | — |

**🔴 Key Finding**: Original claim REFUTED. LSTM beats Transformer when properly validated:
- LSTM is **1.4% better** (0.8015 vs 0.7906)
- LSTM is **5x more stable** (σ=0.007 vs σ=0.034)
- LSTM is **3.4x faster** to train

### Leaderboard (Validated)

| Rank | Model | R² | Notes |
|------|-------|-----|-------|
| 🥇 | **LSTM + Augmentation** | **0.8015 ± 0.007** | Stable, fast |
| 🥈 | Wide Transformer (384, 6L) | 0.7906 ± 0.034 | High variance |
| 🥉 | [Distilled RVQ (Exp 19)](RESEARCH_LOG.md#experiment-19-distilled-rvq-combining-best-of-exp-12--exp-18) | 0.784 | Best discrete VQ |
| 4 | [RVQ-4 (Exp 12)](RESEARCH_LOG.md#experiment-12-residual-vector-quantization-rvq) | 0.776 | Discrete VQ |

## Key Findings

1. **Temporal context is essential**: Single timestep R² ≈ 0.10, with 250ms history R² ≈ 0.78
2. **POYO trade-off**: Full permutation invariance → R² ≈ 0 (destroys velocity info)
3. **Codebook collapse**: Standard VQ training uses only 3-8% of codes
4. **Progressive training is key**: Pre-train → k-means init → finetune prevents collapse
5. **Residual VQ breaks Voronoi ceiling**: Multi-stage quantization captures fine details
6. **RVQ-4 optimal**: 4 layers × 128 codes, more layers = diminishing returns
7. **FSQ topology doesn't help**: Ordinal code structure underperforms discrete VQ (Exp 14)
8. **Distillation eliminates VQ tax**: Exp 18/19 proved 0% discretization loss with latent distillation
9. **Lag tuning (Δ=+1) hurts**: Predicting 25ms ahead decorrelates signal on MC_Maze
10. **Student can beat teacher**: Exp 19 student (0.783) exceeded teacher (0.780) — RVQ acts as regularizer
11. **β=0.5 is optimal for distillation**: Exp 20 sweep showed higher β degrades performance (U-shaped curve)
12. **🔴 Long context (2s) HURTS on MC_Maze**: Exp 21 showed slow pathway degrades R² by 2.8% — no exploitable preparatory dynamics
13. **250ms is optimal window**: Longer windows add noise, not signal for this dataset
14. **🎉 Width > Depth for Transformers**: Exp 21b showed 384×6L (0.806) beats 256×8L (0.793) and 512×10L (0.805)
15. **Too deep hurts**: 384×8L was WORST (0.752) — overfitting from excessive depth
16. **Data augmentation is CRITICAL during training**: Exp 21b used augment=True in sweep → 0.806. Exp 22 forgot augmentation → only 0.750
17. **Reproducibility requires matching ALL training conditions**: Architecture alone is insufficient — same augmentation, dropout, lr needed
18. **🔴 Exp 22 FAILED**: Teacher regressed 7% (0.806→0.750) without augmentation → Student only reached 0.741
19. **Excellent codebook utilization**: Exp 22 achieved 94.5% average usage (484/512 codes) — no collapse issue
20. **🧠 Inductive bias matters more than capacity**: Exp 23 showed LSTM (0.8015) beats Transformer (0.7906) because LSTM's sequential smoothing bias matches MC_Maze's simple reaching dynamics. Extra capacity without matching bias = variance, not performance.
21. **🔴 Exp 23 REFUTED Transformer claim**: Multi-seed validation showed Transformer is 1.4% worse, 5x less stable, and 3.4x slower than LSTM

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

## Author

📧 youssef@elabbassi.com

If you use this work in your research, please cite:

```bibtex
@software{phantomx,
  author = {Youssef El Abbassi}
  title = {PhantomX: Neural Decoding as a Codec},
  year = {2026},
  url = {https://github.com/yelabb/PhantomX}
}
```


