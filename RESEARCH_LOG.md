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

**Results**:
```
    Window |     MLP R² |    LSTM R²
----------------------------------------
      5 (125ms) |     0.5765 |     0.5985
     10 (250ms) |     0.6834 |     0.7783  ← BEST
     20 (500ms) |     0.7249 |     0.7614
```

**Analysis**:
- **LSTM with 10-step window (250ms) achieves R² = 0.78!** ✓ EXCEEDS 0.7 TARGET
- Temporal context is CRITICAL - single timestep R² was only 0.10
- LSTM > MLP, suggesting temporal dynamics matter beyond simple concatenation
- 250ms is optimal - longer windows may overfit or include irrelevant history

**Key Insight**:
The MC_Maze data DOES contain strong velocity signal, but it requires:
1. Full channel identity (not sorted order statistics)
2. Temporal context (~250ms history)

**Problem for POYO**:
POYO's permutation-invariant tokenization destroys channel identity.
But POYO is designed for **cross-session transfer** where electrodes may shift/dropout.

**Challenge**: How to get BOTH permutation invariance AND velocity decoding?

**Next**: Experiment 4 - Test alternative tokenizations that preserve more information

---

## Experiment 4: Channel-Preserving POYO Variants
**Date**: 2026-01-19
**Goal**: Design tokenization that balances permutation invariance with velocity decoding

**Approaches to test**:
1. **Binned Histogram Tokens**: Instead of top-k values, use histogram of spike counts
2. **Positional POYO**: Add learnable position encodings after sorting
3. **Channel Groups**: Group channels by brain region, preserve within-group identity
4. **Temporal Tokens**: Tokenize temporal patterns instead of spatial

Running...
