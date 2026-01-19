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

**Results**:
```
Tokenization         |         R² |  Permutation Invariant
------------------------------------------------------------
Raw LSTM (baseline)  |     0.7783 |                     No
Histogram            |    -0.0020 |                    Yes
Sorted+Rank          |     0.0001 |                    Yes
Statistics           |     0.0155 |                    Yes
TemporalPattern      |     0.5566 |                Partial
```

**Analysis**:
- **Histogram, Sorted+Rank, Statistics**: All ~0 R² - completely lose velocity info
- **TemporalPattern**: R² = 0.56 - BEST permutation-invariant approach!

**Key Insight**:
Temporal patterns (mean, derivative, std, max over time window) preserve enough
channel-specific information for velocity decoding while being somewhat robust
to amplitude drift.

**Why TemporalPattern works**:
- Each channel has its own temporal dynamics (preserved)
- But the tokenization doesn't depend on absolute spike rates (amplitude invariant)
- Trade-off: Not fully permutation invariant (channels still identified by position)

**The Fundamental Trade-off**:
| Property | Raw Spikes | Full POYO | TemporalPattern |
|----------|------------|-----------|-----------------|
| R² | 0.78 | ~0 | 0.56 |
| Permutation Invariant | ❌ | ✅ | ❌ |
| Amplitude Invariant | ❌ | ✅ | ✅ |
| Cross-session Transfer | Poor | Good | Medium |

**Conclusion for LaBraM-POYO**:
The original POYO design prioritizes cross-session transfer over single-session accuracy.
For BCI applications where session-specific calibration is acceptable, we should:
1. Use temporal pattern tokenization (preserves channel identity)
2. Use VQ-VAE for learned discretization
3. Apply TTA for adaptation to new sessions

**Next**: Experiment 5 - Combine TemporalPattern tokenization with VQ-VAE

---

## Experiment 5: TemporalPattern VQ-VAE
**Date**: 2026-01-19
**Goal**: Integrate best tokenization with VQ-VAE for learned representations

**Config**:
- Tokenizer: TemporalPatternTokenizer (4 features x 142 channels)
- VQ-VAE: 256 codes, 64-dim embeddings
- LSTM decoder for velocity

**Results**:
```
Test R² (vx):      0.5728
Test R² (vy):      0.4354
Test R² (overall): 0.5041
Codebook utilization: 17/256 (6.6%)
```

**Analysis**:
- R² = 0.50 - decent but below TemporalPattern MLP (0.56) and LSTM (0.78)
- VQ bottleneck losing ~25% of the velocity information
- Low codebook utilization (6.6%) suggests the encoder is collapsing diversity
- Perplexity ~1.7 means only ~2 codes are being used frequently

**Hypothesis**: 
The information bottleneck from VQ is too aggressive for this task.
With 256 codes and 64-dim embeddings, we have only 256 discrete states
to represent the full range of neural-velocity mappings.

**Ideas to improve**:
1. Increase codebook size (512 or 1024 codes)
2. Use multiple VQ heads (like product quantization)
3. Add skip connection around VQ layer
4. Reduce commitment cost to allow more codebook exploration

**Next**: Experiment 6 - Improve VQ-VAE architecture

---

## Experiment 6: Improved VQ-VAE Architecture
**Date**: 2026-01-19
**Goal**: Reduce VQ information bottleneck while maintaining discrete representations

**Modifications**:
1. Increase codes to 512
2. Add residual connection around VQ
3. Lower commitment cost (0.1 instead of 0.25)
4. Add LayerNorm for training stability

Running...
