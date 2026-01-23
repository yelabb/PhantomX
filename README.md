# PhantomX

> ⚠️ **Experimental / Learning Project** - Not for production use

> 📓 **[RESEARCH_LOG.md](RESEARCH_LOG.md)** - Detailed experiment notes, results, and analysis


PhantomX — On the Limits of Discrete Representations for Neural Control. A systematic empirical study of tokenization, quantization, and inductive bias in BCI

<img width="900" alt="image" src="https://github.com/user-attachments/assets/249ccf49-29f7-435e-bad2-b94d91959da3" />


## 🎯 Results

🔬 **[Exp 25: Mamba on MC_RTT](RESEARCH_LOG.md#experiment-25-mamba-on-mc_rtt-the-navigation-filter)** — Mamba succeeds on continuous tracking!

### Latest: Exp 25 Mamba on MC_RTT (New Dataset)

| Model | Dataset | R² | Notes |
|-------|---------|-----|-------|
| **Mamba-4L (2s window)** | **MC_RTT** | **0.7474** | 🎯 Target exceeded! |
| LSTM (aug) | MC_Maze | 0.8000 | Validated baseline |
| Mamba-4L | MC_Maze | 0.68 | ❌ Failed (Exp 13) |

**Key Insight**: Same architecture succeeds/fails based on task structure:
- MC_Maze (discrete trials): Context = noise → LSTM wins
- MC_RTT (continuous tracking): Context = trajectory → Mamba shines

### Exp 24 Statistical Validation (5 seeds)

| Model | R² (mean ± std) | 95% CI | p-value |
|-------|-----------------|--------|--------|
| **LSTM (aug)** | **0.8000 ± 0.0088** | [0.789, 0.811] | — |
| Teacher (Transformer) | 0.7833 ± 0.0231 | [0.755, 0.812] | 0.092 |
| Student (RVQ-4) | 0.7762 ± 0.0208 | [0.750, 0.802] | 0.014 |

**Verdict**: ❌ RVQ Student does NOT beat LSTM (p = 0.014, LSTM wins)

### Validated Results (5 seeds each, Exp 23)

| Model | R² (mean ± std) | 95% CI | Verdict |
|-------|-----------------|--------|--------|
| 🥇 **LSTM (aug)** | **0.8015 ± 0.007** | [0.793, 0.810] | ✅ Practical winner |
| 🥈 LSTM (no aug) | 0.7936 ± 0.007 | [0.785, 0.802] | Solid baseline |
| 🥉 Wide Transformer (aug) | 0.7906 ± 0.034 | [0.749, 0.833] | ⚠️ High variance |

**Statistical Verdict**: ⚠️ **INCONCLUSIVE** (p = 0.44) — No significant difference between models

**Practical Verdict**: 🏆 **LSTM wins** — 5x more stable, 3.4x faster, equivalent performance

### Leaderboard (Validated, 5 seeds)

| Rank | Model | R² (mean ± std) | Notes |
|------|-------|-----------------|-------|
| 🥇 | **LSTM (aug)** | **0.8000 ± 0.009** | Stable, validated ✅ |
| 🥈 | Teacher (Transformer) | 0.7833 ± 0.023 | High variance |
| 🥉 | Student (RVQ-4) | 0.7762 ± 0.021 | Discretization tax: 0.71% |

*All validated with 5 seeds, paired t-test, and Cohen's d effect size*

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
22. **Multi-seed teacher selection improves distillation**: Exp 22c best seed (0.8162) vs mean (0.7910) shows ±2.5% seed variance
23. **Distillation preserves 99%+ of teacher performance**: Exp 22c student (0.8107) retained 99.3% of teacher (0.8162) with only 0.55% discretization tax
24. **Near-perfect codebook utilization achieved**: Exp 22c reached 98.4% usage (504/512 codes) with k-means init
25. **🔴 Exp 24 REFUTED Exp 22c**: Multi-seed validation (n=5) showed LSTM (0.8000) significantly beats RVQ Student (0.7762) with p=0.014
26. **Cherry-picking inflates results**: Exp 22c's single-split R²=0.8107 was not reproducible; true mean = 0.7762 ± 0.021
27. **Discretization tax is negligible but real**: 0.71% ± 1.01% across 5 seeds (not statistically significant, p=0.19)
28. **LSTM's inductive bias wins on MC_Maze**: Simple sequential dynamics favor LSTM's smoothing bias over Transformer's flexibility
29. **🎉 Mamba succeeds on MC_RTT**: R² = 0.7474 on continuous tracking task — same model that failed on MC_Maze (0.68) works when context = trajectory
30. **Task structure determines architecture**: MC_Maze (discrete) → context is noise; MC_RTT (continuous) → context is signal

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

✅ **Exp 25 In Progress** — Mamba succeeds on MC_RTT (R² = 0.7474)

### Latest: Mamba on MC_RTT (Exp 25)

| Model | Dataset | R² | Notes |
|-------|---------|-----|-------|
| **Mamba-4L (2s)** | **MC_RTT** | **0.7474** | 🎯 Continuous tracking |
| LSTM (aug) | MC_Maze | 0.8000 | Validated baseline |
| Mamba-4L | MC_Maze | 0.68 | ❌ Context = noise |

- **Hypothesis confirmed**: Mamba works when context = trajectory (not noise)
- **New dataset**: MC_RTT — 130 units, 649s continuous, random target tracking
- **Architecture**: Proper S6 with official Mamba initialization

### Validated Baselines (Exp 23, 5 seeds each)

| Model | R² (mean ± std) | Stability | Speed |
|-------|-----------------|-----------|-------|
| 🏆 **LSTM + Aug** | **0.8015 ± 0.007** | Rock solid | 66s |
| LSTM (no aug) | 0.7936 ± 0.007 | Solid | 71s |
| Transformer | 0.7906 ± 0.034 | Unstable | 224s |

- **Statistical verdict**: ⚠️ Inconclusive (p = 0.44)
- **Practical verdict**: 🏆 LSTM — 5x more stable, 3.4x faster

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for full experiment details







## Acknowledgments

This project was developed with assistance from AI coding assistants and workflows:
- Claude Opus 4.5 (Anthropic)
- Claude Sonnet 4.5 (Anthropic)
- Gemini 3.0 Pro (Google)
- GPT 5.2 (OpenAI)

All code was tested, and validated by the author.

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




https://github.com/user-attachments/assets/3efa947c-cd41-4606-8f38-a50026513ca1





