# PhantomX — Research Vulgarisation (0 → Hero)

> Goal: help you go from “I’m new” to “I can reproduce results, understand the trade-offs, and extend PhantomX responsibly.”
>
> Source of truth for claims in this document:
> - `README.md`
> - `RESEARCH_LOG.md`
> - (Helpful for running) `QUICK_REFERENCE.md` and `IMPLEMENTATION_LOG.md`

---

## 0) What PhantomX is (in one paragraph)

PhantomX is an **experimental research sandbox** exploring how to decode **2D hand velocity** from **motor cortex spiking activity** using ideas inspired by **LaBraM** (vector-quantized representations) and **POYO** (tokenizing spikes with robustness goals). The core result reported in this repo is that **training strategy** (especially *progressive training with k-means codebook init*) can make a VQ bottleneck work well: a **Progressive VQ‑VAE** reaches about **R² ≈ 0.71** on MC_Maze velocity decoding while using most of a 256‑code codebook.

If you only remember one idea: **motor cortex decoding needs temporal context**, and **VQ codebooks collapse unless you train them carefully**.

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
- Progressive MLP-based VQ-VAE remains best in these tests.
- Transformers were slower and worse on short windows.
- Gumbel-softmax variants collapsed without progressive tricks.

Takeaway: **complexity doesn’t automatically help**.

---

## 5) Repo map: “where do I look for what?”

### High-level docs
- `README.md`: what the project is + headline results
- `RESEARCH_LOG.md`: the experiment-by-experiment narrative and numbers
- `QUICK_REFERENCE.md`: run commands + integration notes
- `IMPLEMENTATION_LOG.md`: implementation inventory + test status

### Where the code lives
Most work is under `python/`:
- `python/phantomx/`: the Python package (models, data loader, tokenizer, TTA)
- `python/exp*.py`: experiment scripts matching the research log
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

2) Breakthrough reproduction:
```bash
python python/exp9_progressive_vqvae.py
```

3) Compare architectures:
```bash
python python/compare_models.py
```

Expected outcome (from docs): best model ends around **R² ≈ 0.71** with high code usage.

---

## 7) “0 → Hero” mental model (how to think like the repo)

When you touch PhantomX, keep these questions in mind:

### Q1: Are we giving the model enough information?
- If you remove channel identity (full permutation invariance), do you still expect velocity decoding? In this repo: usually **no**.
- If you remove time history, do you still expect good decoding? In this repo: **no**.

### Q2: Are we training the discrete bottleneck correctly?
If VQ collapses:
- It might not be “the wrong architecture.”
- It might be “the training dynamics are wrong.”

Progressive training + k-means init is the repo’s main antidote.

### Q3: What is the representation *for*?
- If you want **single-session accuracy**, raw + temporal context can be great.
- If you want **robustness / transfer**, you may need invariances and/or adaptation (TTA).

---

## 8) Glossary (quick definitions)

- **Channel**: one electrode stream / feature dimension (142 here).
- **Temporal window**: concatenating/processing multiple timesteps (e.g. 10 steps = 250ms at 40Hz).
- **Permutation invariance**: swapping channel order does not change representation.
- **VQ / Codebook**: discretization via nearest neighbor to one of $K$ embedding vectors.
- **Codebook collapse**: only a few codes are used.
- **Perplexity**: effective number of codes being used (higher is healthier).
- **EMA VQ**: codebook updated via exponential moving averages (often more stable).
- **k-means init**: initialize codebook centers from actual encoder outputs.
- **Progressive training**: pretrain without VQ → init codebook → finetune with VQ.
- **TTA (Test-Time Adaptation)**: adapt model online at inference to handle drift.

---

## 9) Next “hero” steps (what to do after you reproduce results)

Choose one direction:

1) **Make experiments reproducible**
- Replace hard-coded `DATA_PATH` in experiment scripts with a CLI flag or environment variable.

2) **Stress test robustness**
- Evaluate electrode dropout sensitivity explicitly.
- Evaluate drift by scaling channels or adding offsets, then try TTA.

3) **Bridge to the broader Phantom stack**
- The repo contains notes on integration with PhantomLink (see `QUICK_REFERENCE.md` and `IMPLEMENTATION_LOG.md`).

---

## 10) If you get stuck (common failure points)

- `ModuleNotFoundError: phantomx` → run `pip install -e python/` from the repo root.
- Dataset path errors → search for `DATA_PATH =` in `python/exp*.py` and update it.
- Bad R² unexpectedly → first confirm you’re using a temporal window; single timestep is expected to be weak.

---

### Suggested reading order
1. `README.md`
2. `RESEARCH_LOG.md` (focus on Experiments 2–3, then 9–10)
3. `python/exp3_temporal.py` and `python/exp9_progressive_vqvae.py`
4. `QUICK_REFERENCE.md`

---

> Reminder: this is a learning/research repo. Treat results as exploratory and validate carefully before any real-world claims.
