# PhantomX Research Log: LaBraM-POYO Exploration

> **[RESEACH_VULGARISATION.md](RESEACH_VULGARISATION.md)** for a beginner-friendly “0 → hero” walkthrough, then come back to this log for the full experiment trail.

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
1. Product Quantization (4 heads × 64 codes = 16M combinations)
2. Residual VQ (α * z_q + (1-α) * z_e)
3. Lower commitment cost (0.1)
4. LayerNorm for training stability

**Results**:
```
Config               |       R² |    R² vx |    R² vy
----------------------------------------------------
ProductVQ (pure)     |   0.5634 |   0.6359 |   0.4910
ResidualVQ_0.8       |   0.5213 |   0.5669 |   0.4756
ResidualVQ_0.5       |   0.5356 |   0.6111 |   0.4601
```
Best: ProductVQ with 208 unique code combinations

**Analysis**:
- Product VQ achieves R² = 0.56, slight improvement over single VQ
- Residual connection actually hurts - suggests discrete representation is valuable
- 208 unique combinations used (out of 16M possible)

---

## Experiment 7: Richer Temporal Representations
**Date**: 2026-01-19
**Goal**: Test different ways to encode temporal information

**Approaches**:
1. Full temporal (raw 10x142 windows)
2. Channel temporal (5 bins + stats per channel)
3. Temporal Conv (1D conv on time axis)
4. Full temporal + larger codebook (1024 codes, 256-dim)

**Results**:
```
Config                 |       R² |    R² vx |    R² vy |  Codes
----------------------------------------------------------------
FullTemporal           |   0.5360 |   0.6236 |   0.4483 |      8
ChannelTemporal        |   0.3277 |   0.2629 |   0.3926 |      7
TemporalConv           |  -0.0055 |  -0.0098 |  -0.0012 |      1
FullTemporal_Large     |   0.5554 |   0.5074 |   0.6035 |      8
```

**Analysis**:
- **Codebook collapse!** Only 7-8 codes being used out of 512-1024
- Larger codebook doesn't help without better training
- Conv encoder completely fails (1 code = all samples same)
- The VQ layer is a training bottleneck, not a capacity bottleneck

---

## Experiment 8: Addressing Codebook Collapse
**Date**: 2026-01-19
**Goal**: Fix codebook collapse with better VQ training techniques

**Approaches**:
1. EMA VQ (exponential moving average updates, k-means init, dead code revival)
2. Entropy-regularized VQ (encourage uniform code usage)

**Results**:
```
Config               |       R² |    R² vx |    R² vy |  Codes
--------------------------------------------------------------
EMA_VQ               |   0.6735 |   0.6834 |   0.6636 |     12
Entropy_VQ           |   0.5661 |   0.5794 |   0.5528 |      8
```

**Analysis**:
- **EMA VQ achieves R² = 0.67!** Significant improvement
- K-means initialization + EMA updates prevent collapse
- Still only 12 codes used - need better initialization strategy
- Entropy regularization destabilizes training

**Key Insight**:
EMA updates are gentler than gradient descent for codebook learning.
But we need to initialize from the actual encoder output distribution.

---

## Experiment 9: Progressive Training (BREAKTHROUGH!)
**Date**: 2026-01-19
**Goal**: Push to R² ≥ 0.70 with better training strategy

**Approach: Progressive VQ-VAE**
1. Phase 1: Pre-train encoder without VQ (30 epochs)
2. Phase 2: Initialize VQ codebook with k-means on encoder outputs
3. Phase 3: Finetune with VQ enabled (50 epochs)

**Results**:
```
============================================================
EXPERIMENT 9 RESULTS: Progressive VQ-VAE
============================================================
Test R² (overall): 0.7038  🎉
Test R² (vx):      0.6829
Test R² (vy):      0.7248
Codes used:        221/256 (86%)
Code entropy:      6.91 bits (max=7.79)
```

**Top 10 codes** (well-distributed!):
```
Code 211: 3.5%, Code 174: 2.6%, Code 231: 2.4%, Code 230: 2.4%
Code 40: 2.4%, Code 147: 2.4%, Code 81: 2.4%, Code 58: 2.4%
```

**Analysis**:
🎉 **SUCCESS! R² = 0.70 achieved!**
- Progressive training is the key breakthrough
- Pre-training lets encoder learn good representations FIRST
- K-means init ensures codebook covers the learned manifold
- 221/256 codes used = healthy codebook utilization (86%)
- Code entropy 6.91/7.79 = well-distributed code usage

**Why Progressive Training Works**:
1. Encoder converges to useful representation without VQ interference
2. K-means on converged encoder provides optimal codebook init
3. Finetuning with VQ preserves encoder quality while learning discretization

---

## Summary: LaBraM-POYO Findings

### Key Discoveries

1. **Temporal Context is Essential**
   - Single timestep: R² ≈ 0.10
   - 10 timesteps (250ms): R² ≈ 0.78
   - Motor cortex encodes velocity through temporal dynamics

2. **POYO Trade-off: Invariance vs. Information**
   - Full POYO (permutation invariant): R² ≈ 0
   - Raw spikes (no invariance): R² = 0.78
   - The sorted order statistics destroy channel identity

3. **VQ-VAE Challenges**
   - Codebook collapse is a major issue
   - Standard training uses only 3-8% of codes
   - EMA + k-means init improves to R² = 0.67

4. **Progressive Training is the Solution**
   - Pre-train encoder → k-means init → finetune with VQ
   - Achieves R² = 0.70 with 86% codebook utilization
   - Preserves encoder quality while adding discretization

### Final Architecture: PhantomX VQ-VAE

```
Input: Spike counts [10 timesteps × 142 channels]
   ↓
Encoder: MLP [1420 → 1024 → 512 → 256 → 128]
   ↓
VQ Layer: EMA VQ [256 codes × 128 dim]
   ↓
Decoder: MLP [128 → 256 → 128 → 2]
   ↓
Output: Velocity [vx, vy]
```

**Training Recipe**:
1. Pre-train encoder for 30 epochs (no VQ)
2. K-means init on encoder outputs
3. Finetune with VQ for 50 epochs
4. Cosine annealing LR schedule

### Implications for LaBraM-POYO

The original POYO design trades single-session accuracy for cross-session robustness.
For BCI applications:

- **If cross-session transfer is critical**: Use original POYO, accept lower R²
- **If session-specific calibration is OK**: Use temporal windows + Progressive VQ-VAE
- **Hybrid approach**: Pre-train on POYO for generalization, finetune with channel-specific features

### Future Work

1. Test on multiple sessions to validate transfer learning
2. Explore soft-VQ (Gumbel-softmax) for differentiable discretization
3. Add transformer encoder for longer temporal context
4. Investigate LaBraM pre-training on large neural datasets

---

## Experiment 10: Extended Architecture Comparison
**Date**: 2026-01-19
**Goal**: Compare Progressive VQ-VAE against new architectures

**Models Tested**:
1. **Progressive VQ-VAE** - MLP encoder, EMA VQ, progressive training
2. **Transformer VQ-VAE** - Self-attention encoder, EMA VQ
3. **Gumbel VQ-VAE** - MLP encoder, Gumbel-softmax VQ, end-to-end training
4. **Transformer + Gumbel** - Self-attention + Gumbel-softmax

**Results**:
```
Model                  |       R² |    R² vx |    R² vy |  Codes |     Time
----------------------------------------------------------------------     
Progressive            |   0.7143 |   0.7128 |   0.7158 |    218 |   173.9s
Transformer            |   0.5508 |   0.5257 |   0.5759 |    163 |  1087.6s
Gumbel                 |  -0.0012 |  -0.0021 |  -0.0003 |      1 |   120.6s
Transformer+Gumbel     |   0.5312 |   0.5875 |   0.4750 |    N/A |  1901.6s
```

**Analysis**:

1. **Progressive VQ-VAE remains the best**: R² = 0.71, fastest training (174s)
   - Progressive training is the key - not architecture complexity

2. **Transformer underperforms**: R² = 0.55, 6x slower
   - Self-attention doesn't help with 10-step windows (already short)
   - May need pre-training or longer sequences to benefit

3. **Gumbel VQ fails completely**: R² ≈ 0
   - End-to-end training causes codebook collapse (1 code!)
   - Temperature annealing doesn't prevent collapse
   - Needs progressive training like EMA VQ

4. **Transformer + Gumbel**: R² = 0.53
   - Better than pure Gumbel but still not competitive
   - Temperature stuck at 2.0 (not annealing properly)

**Key Insight**:
The training strategy (progressive) matters more than the architecture.
Even a simple MLP beats Transformer when properly trained.

### New Components Added

1. **TransformerEncoder** ([models_extended.py](python/phantomx/models_extended.py))
   - CLS token aggregation
   - Sinusoidal positional encoding
   - Multi-head self-attention

2. **GumbelVectorQuantizer** ([models_extended.py](python/phantomx/models_extended.py))
   - Differentiable soft-to-hard quantization
   - Temperature annealing
   - Diversity loss for code usage

3. **Test-Time Adaptation** ([tta.py](python/phantomx/tta.py))
   - TTAWrapper: Entropy minimization on code assignments
   - OnlineTTA: Sliding window buffer adaptation
   - SessionAdapter: Calibration-based adaptation

---

## Final Summary

### Best Configuration

| Component | Choice | Why |
|-----------|--------|-----|
| Encoder | MLP (1024→512→256→128) | Simple, fast, effective |
| VQ Type | EMA | Prevents collapse, stable |
| Training | Progressive (3-phase) | Prevents interference |
| Window | 10 steps (250ms) | Optimal temporal context |
| Codes | 256 | Good balance |

### Performance Achieved

- **R² = 0.71** on MC_Maze velocity decoding
- **218/256 codes used** (85% utilization)
- **174 seconds** training time on CPU

### Lessons Learned

1. **Training strategy > Architecture**: Progressive beats end-to-end
2. **Temporal context is critical**: 250ms history for motor cortex
3. **POYO trade-off is real**: Permutation invariance hurts velocity decoding
4. **Codebook collapse is the enemy**: Must use k-means init + EMA updates
5. **Simple works**: MLP beats Transformer on this task
