# PhantomX Research Log: LaBraM-POYO Exploration

## Project Goal
Implement and optimize LaBraM-POYO neural foundation model for BCI velocity decoding.

**Target**: R² ≥ 0.7 on MC_Maze zero-shot velocity decoding

---

## Experiment 1: Baseline VQ-VAE with POYO Tokenization
**Date**: 2026-01-19
**Config**:
- Tokenizer: SpikeTokenizer (sorted top-k order statistics)
- VQ-VAE: 256 codes, 64-dim embeddings
- Data: MC_Maze (142 channels, 11,746 samples @ 40Hz)
- Training: 100 epochs, batch_size=32

**Results**:
```
Direct Prediction R²: 0.0068
Linear Probe R² (z_q): 0.0063
Codebook Perplexity: 13.5 / 256
Codebook Utilization: 8.2%
```

**Analysis**:
- R² ≈ 0 means predictions are essentially random
- Codebook utilization is low (8.2%) - possible codebook collapse
- Perplexity 13.5 means only ~13 codes are being used effectively

**Hypothesis for failure**:
The POYO tokenizer uses **sorted order statistics** (top-k spike values in descending order).
This is designed for **permutation invariance** (electrode dropout robustness), but it
completely destroys **channel identity** - which is CRITICAL for velocity decoding!

In motor cortex, specific neurons encode specific velocity components (vx, vy).
By sorting, we lose which neuron fired how much.

**Next**: Experiment 2 - Test if channel identity matters by using raw spikes

---

## Experiment 2: Raw Spikes Baseline (Sanity Check)
**Date**: 2026-01-19
**Goal**: Verify MC_Maze data quality and establish baseline R² with full channel info

**Config**:
- Input: Raw normalized spike counts [142 channels]
- Model: Simple MLP decoder (no tokenization)
- Output: Velocity [vx, vy]

**Results**:
```
Ridge Regression R²: 0.0779
MLP Decoder R²:      0.1024
```

**Analysis**:
- R² ≈ 0.10 with full channel identity - still weak!
- This is NOT a tokenization problem - the issue is **temporal context**
- Single timestep neural activity is insufficient for velocity decoding
- Need to incorporate temporal history (common in Kalman filters, RNNs)

**Key Insight**: 
Motor cortex encodes velocity through **temporal patterns**, not just instantaneous rates.
Standard BCI decoders use 10-20 timesteps of history (~250-500ms at 40Hz).

**Next**: Experiment 3 - Add temporal context (sliding window)

---

## Experiment 3: Temporal Context
**Date**: 2026-01-19
**Goal**: Add temporal history to improve velocity decoding

**Config**:
- Input: Sliding window of spike counts [T x 142 channels]
- Window sizes to test: 5, 10, 20 timesteps (125ms, 250ms, 500ms)
- Model: MLP on flattened window

Running...
