# PhantomX — Research Vulgarisation (0 → Hero)


https://github.com/user-attachments/assets/3c612eac-101e-4784-b703-067c2b1d79b2


> Goal: help you go from “I’m new” to “I can reproduce results, understand the trade-offs, and extend PhantomX responsibly.”
>
> Source of truth for claims in this document:
> - `README.md`
> - `RESEARCH_LOG.md`
> - (Helpful for running) `QUICK_REFERENCE.md` and `IMPLEMENTATION_LOG.md`

---

## 0) What PhantomX is (in one paragraph)

PhantomX is an **experimental research sandbox** exploring how to decode **2D hand velocity** from **motor cortex spiking activity** using ideas inspired by **LaBraM** (vector-quantized representations) and **POYO** (tokenizing spikes with robustness goals). The core result reported in this repo is that **training strategy** (especially *progressive training with k-means codebook init*) can make a VQ bottleneck work well: a **Progressive VQ‑VAE** reaches about **R² ≈ 0.71** on MC_Maze velocity decoding while using most of a 256‑code codebook.

**Latest milestone (Exp 12 - RVQ-4)**: Using **Residual Vector Quantization** (4 layers × 128 codes) with progressive training achieves **R² = 0.776**, closing 99% of the gap to raw LSTM (0.78). The encoder alone during pre-training actually **exceeds LSTM** at **R² = 0.784**!

If you only remember one idea: **motor cortex decoding needs temporal context (~250ms)**, and **progressive training with proper VQ initialization is the key to avoiding codebook collapse**.

---
## Quick Results Summary

| Experiment | Model | R² | Key Insight |
|------------|-------|-----|-------------|
| 1 | POYO VQ-VAE | ≈0 | ❌ Sorting destroys channel identity |
| 2 | Raw MLP | 0.10 | Need temporal context |
| 3 | LSTM (10 bins) | **0.78** | ✅ 250ms is optimal |
| 4 | POYO variants | 0.56 | Invariance vs information trade-off |
| 5-8 | Various VQ | 0.50-0.67 | Codebook collapse |
| 9 | Progressive VQ-VAE | **0.70** | 🎉 Progressive training breakthrough |
| 10 | Architecture comparison | 0.71 | Strategy > architecture |
| 11 | Causal Transformer + Gumbel | **0.77** | Attention helps |
| 12 | **RVQ-4** | **0.776** | 🏆 Current best - multi-stage quantization |
| 13 | Wide-window Mamba | 0.73 | ❌ Long windows hurt |
| 14-15 | FSQ-VAE | 0.60-0.64 | ❌ FSQ + auxiliary losses fail |
| 16 | Mamba + RVQ | 0.72 | ❌ Confirmed 250ms optimal |
| 17 | LADR-VQ | Blocked | ⚠️ RVQ init bug found |

**Target**: R² = 0.78 (LSTM baseline)  
**Current best**: R² = 0.776 (99% of target, only 0.43% gap!)  
**Pre-training alone**: R² = 0.784 (exceeds LSTM!)

---
## 1) The “why”: what problem are we trying to solve?

### The task: velocity decoding
- Input: spike counts from **142 channels** sampled at **40 Hz** (MC_Maze dataset).
- Output: velocity vector **(vx, vy)**.
- Metric: **$R^2$** (coefficient of determination). Higher is better; 0 means “no predictive value vs baseline,” negative means “worse than baseline.”

### The bigger research motivation
Real BCI systems have two practical headaches:
1. **Temporal dependence**: behavior is encoded in patterns over time, not a single instant.
2. **Non-stationarity**: signals drift (electrode changes, impedance, neuron turnover). You want representations + adaptation that remain useful.

PhantomX explores a path that looks like:
- Build a **learned discrete “neural alphabet”** (a codebook)
- Decode kinematics from those codes
- Optionally apply **Test-Time Adaptation (TTA)** to keep performance under drift

---

## 2) Minimal neuroscience / BCI background (only what you need here)

### Spikes, channels, and “identity”
- A **channel** is an electrode/contact (or a preprocessed unit). In this repo, channels behave like consistent signal sources.
- “Channel identity matters” means: *which channel fired* carries information. If you scramble channels, you often destroy decodability.

### Why time matters
In `RESEARCH_LOG.md`, a single timestep gives **R² ≈ 0.10**, but adding about **250 ms** of history (10 steps @ 40Hz) can reach **R² ≈ 0.78**.

Interpretation: velocity is encoded in **temporal dynamics**, not just instantaneous firing rates.

### What POYO-style tokenization is trying to achieve
POYO-style designs often aim for robustness like:
- **Permutation invariance** (re-ordering channels doesn’t change representation)
- **Electrode dropout** robustness

But in this repo’s experiments, full permutation invariance can be catastrophic for velocity decoding because it destroys channel identity.

---

## 3) Minimal ML background (only what you need here)

### VQ / VQ-VAE in one minute
A Vector Quantizer (VQ) takes a continuous vector $z_e$ and replaces it by the nearest entry in a learnable table (a **codebook**) producing $z_q$.

Why do this?
- Discrete codes can be **compact**, **interpretable**, and can help with **transfer**.

### The main failure mode: codebook collapse
“Collapse” means the model uses only a tiny fraction of codes (e.g., 1–12 out of 256). In `RESEARCH_LOG.md`, standard training often used **3–8%** of codes.

Why it matters:
- If only a few codes are used, your discrete representation can’t carry enough information.

### The key fix in PhantomX: progressive training
The successful recipe in `RESEARCH_LOG.md`:
1. **Pre-train** encoder/decoder without VQ (learn a good continuous representation first)
2. Run **k-means** on encoder outputs to initialize the codebook
3. **Finetune with VQ enabled** (often using EMA-style updates)

This raises utilization to ~**85%+** of codes and hits **R² ≈ 0.70–0.71**.

---

## 4) The story of the repo (experiments as a learning path)

This is the shortest path to “I understand what happened.”
### The Experiment Journey (17 experiments)

```
Exp 1-2: ❌ Failures (R²≈0)
   ↓ Learn: POYO sorting destroys channel identity
   
Exp 3: ✅ Baseline (R²=0.78 LSTM)
   ↓ Learn: 250ms temporal context is crucial
   
Exp 4: Testing POYO variants (R²=0.56 best)
   ↓ Learn: Invariance vs information trade-off
   
Exp 5-8: ❌ VQ struggles (R²≈0.50-0.67)
   ↓ Learn: Codebook collapse from end-to-end training
   
Exp 9: 🎉 BREAKTHROUGH (R²=0.70)
   ↓ Learn: Progressive training is the KEY
   
Exp 10: Architecture comparison (R²=0.71)
   ↓ Learn: Training strategy > architecture complexity
   
Exp 11: Close to LSTM (R²=0.77)
   ↓ Learn: Causal Transformer + Gumbel-Softmax works
   
Exp 12: 🏆 CURRENT BEST (R²=0.776)
   ↓ Learn: RVQ-4 breaks Voronoi Ceiling
   
Exp 13, 16: ❌ Long windows fail (R²≈0.72)
   ↓ Learn: 2s windows dilute signal, 250ms is optimal
   
Exp 14-15: ❌ FSQ fails (R²≈0.60)
   ↓ Learn: Auxiliary losses hurt, topology doesn't help
   
Exp 17: ⚠️ BLOCKED (RVQ init bug found)
   ↓ Learn: Lag-aware decoding, initialization timing matters
```

### Detailed experiment breakdown:
### Step A — Learn the baseline reality (Experiment 2 & 3)
- Experiment 2 (`RESEARCH_LOG.md`): raw spikes + simple models → around **R² ≈ 0.10** for a single timestep.
- Experiment 3: add temporal context (sliding window). With 10 steps (~250ms):
  - MLP: ~0.68
  - LSTM: ~0.78 (best)

Takeaway: **don’t judge tokenizers or VQ until you have temporal context**.

### Step B — Understand the POYO trade-off (Experiment 1 & 4)
- Full permutation-invariant sorting (“order statistics”) → **R² ≈ 0**.
- Some partial/temporal-feature tokenizations can recover some signal (e.g. ~0.56), but not full raw performance.

Takeaway: **invariance vs information is a real trade-off**.

### Step C — Learn why VQ is hard (Experiments 5–8)
- VQ bottlenecks can reduce performance if the codebook collapses.
- EMA and better initialization help, but may still underuse codes.

Takeaway: **capacity is not the only issue; training dynamics are**.

### Step D — Reproduce the breakthrough (Experiment 9)
- Progressive training + k-means init → about **R² ≈ 0.70** with **~86%** code usage.

Takeaway: **strategy beats architecture**.

### Step E — Architecture comparison (Experiment 10)
- Progressive MLP-based VQ-VAE remains best in these tests (R² = 0.71).
- Transformers were slower and worse on short windows.
- Gumbel-softmax variants collapsed without progressive tricks.

Takeaway: **complexity doesn't automatically help**. The training strategy (progressive) matters more than architecture choice.
### Step F — Beat the LSTM milestone (Experiments 11-12)

**Experiment 11: Causal Transformer + Gumbel-Softmax**
- Combined **Causal Transformer** encoder with **Progressive Gumbel-Softmax** VQ
- Key insight: The encoder alone can reach **R² = 0.78** (LSTM parity!) during pre-training
- After VQ finetuning: **R² = 0.77** (only 0.7% gap remaining!)
- Deep Causal Transformer (6 layers) performed best
- Gumbel-Softmax with progressive training prevents collapse (118+ codes used)

**Experiment 12: Residual Vector Quantization (RVQ-4)** 🏆
- Implemented **multi-stage quantization**: 4 layers × 128 codes each
- Each layer quantizes the **residual error** from the previous layer
- Result: **R² = 0.776** (only 0.43% gap to LSTM!)
- Pre-training encoder reached **R² = 0.784** (EXCEEDS LSTM baseline!)
- Per-layer code usage: 118/121/113/102 codes (excellent distribution)
- Final residual norm: 0.57 (well-compressed)

**Why RVQ breaks the "Voronoi Ceiling"**:
- Single-stage VQ creates hard boundaries (Voronoi cells) that lose fine details
- RVQ refines iteratively: Layer 1 captures 70%, Layer 2 adds 15%, Layer 3 adds 8%, Layer 4 adds 5%
- This breaks through the discrete bottleneck while maintaining interpretability

Takeaway: **Multi-stage quantization + progressive training closes 99% of the gap to continuous LSTM**.

### Step G — Failed experiments teach important lessons (Experiments 13-16)

**Experiment 13: Wide-Window Mamba (FAILED)**
- Hypothesis: Use 2-second context (80 bins) instead of 250ms (10 bins)
- Result: **R² = 0.73** - 5 points BELOW 250ms baseline!
- Lesson: **More context ≠ better context**. Motor velocity encoding is LOCAL (~250ms). Longer windows add noise, not signal.

**Experiments 14-15: FSQ-VAE Series (FAILED)**
- Tried Finite Scalar Quantization (FSQ) with topology-preserving codes
- Added dual-head decoder (velocity + spike reconstruction)
- Added dynamics loss for smooth latent trajectories
- Results: R² = 0.644 (Exp 14) and R² = 0.597 (Exp 15)
- Lessons:
  - **Topology preservation doesn't help discrete velocity decoding**
  - **Auxiliary losses compete with main objective** if not carefully designed
  - **End-to-end training causes collapse** - progressive training is essential
  - **For supervised tasks, focus on the target** - don't add losses that pull away from it

**Experiment 16: Frankenstein (Mamba + RVQ-4) (FAILED)**
- Tried combining 2s Mamba encoder with RVQ-4 quantizer
- Pre-training peaked at R² = 0.716, then overfitted
- Confirmed Exp 13's finding: 2s windows dilute the signal
- Lesson: **The bottleneck is VQ discretization, not temporal context**. 250ms is optimal.

**Experiment 17: LADR-VQ (IN PROGRESS - BLOCKED)**
- Lag-sweep analysis: Predicting 25ms ahead (Δ=+1) is optimal
- Discovered critical bug: RVQ k-means initialization happens BEFORE encoder pre-training
- This collapses codebooks to only 4 clusters instead of 128
- Status: Blocked until RVQ initialization is fixed

Takeaway: **Failed experiments are crucial** - they validated that 250ms windows, progressive training, and RVQ are the winning combination.
---

## 5) Repo map: “where do I look for what?”

### High-level docs
- `README.md`: what the project is + headline results
- `RESEARCH_LOG.md`: the experiment-by-experiment narrative and numbers (17 experiments!)
- `QUICK_REFERENCE.md`: run commands + integration notes
- `IMPLEMENTATION_LOG.md`: implementation inventory + test status

### Where the code lives
Most work is under `python/`:
- `python/phantomx/`: the Python package (models, data loader, tokenizer, TTA)
- `python/exp*.py`: experiment scripts matching the research log
  - `exp9_progressive_vqvae.py`: Breakthrough experiment (R² = 0.71)
  - `exp11_close_gap.py`: Causal Transformer + Gumbel-Softmax (R² = 0.77)
  - `exp12_residual_vq.py`: **RVQ-4 - Current best (R² = 0.776)** 🏆
  - `exp13_wide_window_mamba.py`: Failed 2s window experiment
  - `exp14_fsq_pivot.py` / `exp15_manifold_quantization.py`: Failed FSQ experiments
  - `exp16_frankenstein.py`: Failed Mamba+RVQ combo
  - `exp17_ladr_vq.py`: Lag-sweep + distillation (in progress, blocked)
- `python/compare_models.py`: architecture comparison harness
- `python/train_labram.py` / `python/test_zero_shot.py`: training/testing entrypoints referenced by quick reference

---

## 6) Hands-on: from zero to running experiments (Windows-friendly)

### 6.1 Create an environment and install deps
From the PhantomX repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e python/
```

If imports fail, the editable install (`pip install -e python/`) is usually the missing step.

### 6.2 Make sure you have the dataset
Many scripts expect an MC_Maze NWB file.

Common paths seen in this repo:
- `../PhantomLink/data/mc_maze.nwb` (from `QUICK_REFERENCE.md`)
- A hard-coded local path in some experiment scripts (example: `python/exp3_temporal.py` has `DATA_PATH = "c:/Users/guzzi/.../mc_maze.nwb"`).

Practical advice:
- Pick **one canonical location** for `mc_maze.nwb`.
- Update the `DATA_PATH` constant inside the scripts you run (or refactor later to use CLI args).

### 6.3 Run the “sanity ladder” (recommended order)
1) Temporal context sanity check:
```bash
python python/exp3_temporal.py
```
Expected: LSTM with 10-step window (250ms) achieves R² ≈ 0.78

2) Breakthrough reproduction (Progressive VQ-VAE):
```bash
python python/exp9_progressive_vqvae.py
```
Expected: R² ≈ 0.70 with ~86% code usage

3) Current best model (RVQ-4):
```bash
python python/exp12_residual_vq.py
```
Expected: R² ≈ 0.776 with multi-stage quantization

4) Compare architectures (optional):
```bash
python python/compare_models.py
```

**Note**: Many experiments (11-17) were run on Fly.io A100 GPU. See [docs/FLY_GPU.md](docs/FLY_GPU.md) for cloud deployment setup if needed.

---

## 7) “0 → Hero” mental model (how to think like the repo)

When you touch PhantomX, keep these questions in mind:

### Q1: Are we giving the model enough information?
- If you remove channel identity (full permutation invariance), do you still expect velocity decoding? In this repo: usually **no**.
- If you remove time history, do you still expect good decoding? In this repo: **no**.
- How much time history? **250ms (10 bins @ 40Hz) is optimal**. More context (2s) actually hurts!

### Q2: Are we training the discrete bottleneck correctly?
If VQ collapses:
- It might not be "the wrong architecture."
- It might be "the training dynamics are wrong."

Progressive training + k-means init is the repo's main antidote. **This is the #1 breakthrough insight.**

### Q3: What type of quantization should I use?
The experiments tested many approaches:
- **Single-stage VQ** (Exp 5-9): Works with progressive training (R² = 0.71)
- **Gumbel-Softmax VQ** (Exp 11): Soft quantization helps (R² = 0.77)
- **RVQ (4 layers)** (Exp 12): **Best approach** - refines iteratively (R² = 0.776) 🏆
- **FSQ** (Exp 14-15): Failed - topology preservation doesn't help velocity decoding
- **Product VQ** (Exp 6, 11): Mixed results - added complexity without clear gains

**Takeaway**: Multi-stage RVQ with progressive training is the winning formula.

### Q4: Does more context always help?
**No!** Experiment 13 and 16 proved that 2-second windows (80 bins) are worse than 250ms (10 bins).
- Motor cortex velocity encoding is LOCAL (~250ms)
- Longer windows include irrelevant history (pauses, direction changes)
- This dilutes the signal with noise

**Optimal window**: 10 bins (250ms) @ 40Hz sampling rate.

### Q5: What about auxiliary losses?
Experiments 14-15 tested:
- Spike reconstruction loss (dual-head decoder)
- Dynamics loss (smooth latent trajectories)

**Both hurt performance!** They compete with the velocity objective rather than help it.

**Lesson**: For supervised tasks, focus on the target. Only add auxiliary losses if they're carefully designed to align with the main objective.

### Q6: What is the representation *for*?
- If you want **single-session accuracy**: Progressive RVQ-4 achieves 99% of LSTM performance
- If you want **robustness / transfer**: you may need invariances and/or adaptation (TTA)
- The current models prioritize accuracy over robustness - cross-session transfer is future work

---

## 8) Glossary (quick definitions)

- **Channel**: one electrode stream / feature dimension (142 here).
- **Temporal window**: concatenating/processing multiple timesteps (e.g. 10 steps = 250ms at 40Hz).
- **Permutation invariance**: swapping channel order does not change representation.
- **VQ / Codebook**: discretization via nearest neighbor to one of $K$ embedding vectors.
- **RVQ (Residual Vector Quantization)**: Multi-stage VQ where each layer quantizes the residual error from the previous layer.
- **FSQ (Finite Scalar Quantization)**: Topology-preserving quantization that rounds continuous values to discrete levels (failed in Exp 14-15).
- **Codebook collapse**: only a few codes are used.
- **Perplexity**: effective number of codes being used (higher is healthier).
- **EMA VQ**: codebook updated via exponential moving averages (often more stable).
- **k-means init**: initialize codebook centers from actual encoder outputs (crucial for preventing collapse).
- **Progressive training**: pretrain without VQ → init codebook → finetune with VQ (the #1 breakthrough).
- **TTA (Test-Time Adaptation)**: adapt model online at inference to handle drift.
- **Causal Transformer**: Transformer with masked attention (each position sees only past).
- **Gumbel-Softmax**: Differentiable approximation to discrete sampling; allows soft-to-hard annealing.
- **Mamba (S6)**: State-space model with linear-time complexity (tested in Exp 12-13, 16 - failed due to context issues).
- **Voronoi Ceiling**: The performance limit from single-stage VQ creating hard boundaries; broken by RVQ.
- **Context Dilution**: Performance drop when using too much temporal context (e.g., 2s vs 250ms).

---

## 9) Next “hero” steps (what to do after you reproduce results)

Choose one direction:

1) **Understand the winning combination**
   - Study [exp12_residual_vq.py](python/exp12_residual_vq.py) - the current best model
   - Key components: Progressive training, RVQ-4 (4×128 codes), 250ms windows
   - Pre-training phase is crucial - encoder alone exceeds LSTM!

2) **Learn from the failures**
   - Read Experiments 13-16 in [RESEARCH_LOG.md](RESEARCH_LOG.md)
   - Understand why longer windows (2s) hurt performance
   - See why FSQ and auxiliary losses failed
   - Failed experiments teach what NOT to do

3) **Make experiments reproducible**
   - Replace hard-coded `DATA_PATH` in experiment scripts with a CLI flag or environment variable.
   - Add experiment configuration files (YAML/JSON)

4) **Close the final gap (0.43%)**
   - The encoder pre-training reaches R² = 0.784 (beats LSTM!)
   - The gap comes from VQ discretization loss
   - Ideas: Softer quantization, larger codebooks, slower temperature annealing
   - See incomplete Experiment 17 for lag-aware distillation approach

5) **Stress test robustness**
   - Evaluate electrode dropout sensitivity explicitly.
   - Evaluate drift by scaling channels or adding offsets, then try TTA.
   - Cross-session transfer (the original BCI goal)

6) **Bridge to the broader Phantom stack**
   - The repo contains notes on integration with PhantomLink (see `QUICK_REFERENCE.md` and `IMPLEMENTATION_LOG.md`).

7) **Deploy to production**
   - See [docs/FLY_GPU.md](docs/FLY_GPU.md) for cloud GPU deployment
   - Experiments 11-17 were run on Fly.io A100-40GB GPU

---

## 10) If you get stuck (common failure points)

- `ModuleNotFoundError: phantomx` → run `pip install -e python/` from the repo root.
- Dataset path errors → search for `DATA_PATH =` in `python/exp*.py` and update it.
- Bad R² unexpectedly → first confirm you're using a temporal window (10 bins = 250ms); single timestep is expected to be weak.
- Model not converging → check if you're using progressive training (pretrain → k-means → finetune).
- Codebook collapse (low perplexity) → ensure k-means initialization happens AFTER encoder pre-training, not before.
- Too much overfitting → try shorter windows (250ms, not 2s), remove auxiliary losses, add early stopping.
- GPU memory issues → reduce batch size or use gradient accumulation.
- Fly.io deployment issues → see [docs/FLY_GPU.md](docs/FLY_GPU.md) for troubleshooting.

---

## 11) Key Insights Summary (What Actually Worked)

This section distills the core lessons from 17 experiments:

### ✅ What Works

1. **Progressive Training** (Exp 9) - THE breakthrough
   - Pre-train encoder without VQ
   - Initialize codebook with k-means on encoder outputs
   - Finetune with VQ enabled
   - This single change: R² 0.50 → 0.70

2. **250ms Temporal Windows** (Exp 3, confirmed in 13, 16)
   - 10 bins @ 40Hz is optimal
   - Motor velocity encoding is LOCAL, not long-range
   - More context (2s) dilutes signal with noise

3. **Multi-stage RVQ** (Exp 12) - Current best
   - 4 layers × 128 codes each
   - Quantize residual error iteratively
   - Breaks the "Voronoi Ceiling"
   - R² = 0.776 (99% of LSTM performance)

4. **Channel Identity Matters** (Exp 1, 4)
   - Full permutation invariance destroys velocity decoding
   - Specific neurons encode specific velocity components
   - Robustness vs accuracy is a real trade-off

5. **Focus on the Target** (Exp 14-15)
   - Single objective (velocity MSE) works best
   - Auxiliary losses (reconstruction, dynamics) compete and hurt
   - For supervised tasks, don't overcomplicate

### ❌ What Doesn't Work

1. **End-to-end Training** (Exp 1, 5-8, 14-15)
   - Causes codebook collapse
   - Low perplexity (few codes used)
   - Progressive training is essential

2. **Long Context Windows** (Exp 13, 16)
   - 2s (80 bins) performs worse than 250ms (10 bins)
   - "Context Dilution" - longer ≠ better
   - The signal is local, not long-range

3. **FSQ Quantization** (Exp 14-15)
   - Topology preservation doesn't help discrete velocity decoding
   - Still collapses despite "implicit" large codebook
   - VQ/RVQ with k-means init performs better

4. **Auxiliary Losses** (Exp 14-15)
   - Spike reconstruction loss: competes with velocity
   - Dynamics loss: penalizes velocity changes!
   - Added complexity without benefit

5. **Stateful Training with Shuffled Data** (Exp 12-13)
   - DataLoader shuffle breaks hidden state continuity
   - Mamba/LSTM can't learn temporal dependencies
   - Use stateless training or sequential sampling

### 🎯 The Winning Formula (RVQ-4)

```
Input (250ms window) → Pre-trained Encoder → RVQ-4 (4×128) → Decoder → Velocity
                                ↑
                         K-means Init AFTER Pre-training
```

**Performance**: R² = 0.776 (gap to LSTM: 0.43%)

**Why it works**:
1. Pre-training learns good continuous representations (R² = 0.784)
2. K-means provides optimal codebook initialization
3. RVQ refines iteratively, breaking through discrete bottleneck
4. 250ms window captures optimal temporal context

---

### Suggested reading order
1. `README.md` - Overview and headline results
2. `RESEARCH_VULGARISATION.md` (this file) - Beginner-friendly walkthrough
3. `RESEARCH_LOG.md` - Full experiment details
   - Focus on: Experiments 2-3 (baselines), 9 (breakthrough), 12 (current best)
   - Learn from failures: 13-16 (what doesn't work)
4. Code dive:
   - `python/exp3_temporal.py` - Temporal context matters
   - `python/exp9_progressive_vqvae.py` - Progressive training breakthrough
   - `python/exp12_residual_vq.py` - **Current best model**
5. `QUICK_REFERENCE.md` - Run commands and integration
6. `docs/FLY_GPU.md` - Cloud deployment (if needed)

---

> Reminder: this is a learning/research repo. Treat results as exploratory and validate carefully before any real-world claims.
