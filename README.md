> **⚠️ EARLY DEVELOPMENT - EDUCATIONAL PURPOSE ONLY**  
> This project is in early development and created for personal learning purposes. Not intended for production use.

# PhantomX: LaBraM-POYO Neural Foundation Model

**Population-geometry BCI decoding with electrode-dropout robustness**

## Overview

PhantomX implements the LaBraM-POYO stack for Brain-Computer Interface neural decoding:

- **POYO Neural Tokenization** (NeurIPS '24): Discrete spike-to-token conversion, invariant to electrode dropout/shift
- **LaBraM VQ-VAE Pre-training** (ICLR '24): Vector-quantized autoencoder for universal latent neural dynamics codebook, enabling zero-shot decoding
- **Test-Time Adaptation (TTA)**: Unsupervised entropy minimization for real-time drift correction

## Key Features

✅ **Electrode-dropout robust**: 50% channel loss → system continues working  
✅ **Zero-shot decoding**: Pre-trained codebook transfers to new sessions without calibration  
✅ **Population geometry**: Understands neural manifolds, not individual wires  
✅ **Real-time adaptation**: Online gradient updates counter signal drift  

## Architecture

```
Spikes [142 channels] 
    ↓
POYO Tokenizer (spike → discrete tokens)
    ↓
VQ-VAE Encoder (tokens → latent codes)
    ↓
Codebook Lookup (codes → quantized embeddings)
    ↓
VQ-VAE Decoder (embeddings → kinematics)
    ↓
TTA Optimizer (entropy minimization)
```

## Installation

```bash
cd PhantomX
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Train VQ-VAE Codebook

```python
from phantomx.vqvae import VQVAETrainer
from phantomx.data import MCMazeDataLoader

# Load MC_Maze dataset via PhantomLink
loader = MCMazeDataLoader("data/mc_maze.nwb")

# Train universal codebook
trainer = VQVAETrainer(
    n_channels=142,
    n_codes=256,
    embedding_dim=64
)
trainer.train(loader, epochs=100)
trainer.save("models/mc_maze_codebook_256.pt")

# PhantomX: LaBraM-POYO Neural Foundation Model

**Population-geometry BCI decoding with electrode-dropout robustness**

## Overview

PhantomX implements the LaBraM-POYO stack for Brain-Computer Interface neural decoding:

- **POYO Neural Tokenization** (NeurIPS '24): Discrete spike-to-token conversion, invariant to electrode dropout/shift
- **LaBraM VQ-VAE Pre-training** (ICLR '24): Vector-quantized autoencoder for universal latent neural dynamics codebook, enabling zero-shot decoding
- **Test-Time Adaptation (TTA)**: Unsupervised entropy minimization for real-time drift correction

## Key Features

✅ **Electrode-dropout robust**: 50% channel loss → system continues working  
✅ **Zero-shot decoding**: Pre-trained codebook transfers to new sessions without calibration  
✅ **Population geometry**: Understands neural manifolds, not individual wires  
✅ **Real-time adaptation**: Online gradient updates counter signal drift  

## Architecture

```
Spikes [142 channels] 
    ↓
POYO Tokenizer (spike → discrete tokens)
    ↓
VQ-VAE Encoder (tokens → latent codes)
    ↓
Codebook Lookup (codes → quantized embeddings)
    ↓
VQ-VAE Decoder (embeddings → kinematics)
    ↓
TTA Optimizer (entropy minimization)
```

## Installation

```bash
cd PhantomX
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Train VQ-VAE Codebook

```python
from phantomx.vqvae import VQVAETrainer
from phantomx.data import MCMazeDataLoader

# Load MC_Maze dataset via PhantomLink
loader = MCMazeDataLoader("data/mc_maze.nwb")

# Train universal codebook
trainer = VQVAETrainer(
    n_channels=142,
    n_codes=256,
    embedding_dim=64
)
trainer.train(loader, epochs=100)
trainer.save("models/mc_maze_codebook_256.pt")
```

### 2. Zero-Shot Inference

```python
from phantomx.inference import LabramDecoder

decoder = LabramDecoder.load("models/mc_maze_codebook_256.pt")

# Real-time decoding (integrates with PhantomLink streaming)
for packet in live_stream:
    spikes = packet.spikes.spike_counts  # [142]
    kinematics = decoder.decode(spikes)   # {vx, vy}
    print(f"Velocity: ({kinematics['vx']:.2f}, {kinematics['vy']:.2f})")
```

### 3. Test-Time Adaptation

```python
from phantomx.tta import EntropyMinimizer

# Enable online adaptation
tta = EntropyMinimizer(decoder, lr=1e-4)

for packet in drifted_stream:
    spikes = packet.spikes.spike_counts
    
    # Decode with TTA
    kinematics = tta.adapt_and_decode(spikes)
    # Decoder weights updated automatically
```

## Integration with Phantom Stack

| Component | Integration Point | Purpose |
|-----------|-------------------|---------|
| **PhantomLink** | `playback_engine.py` | Spike data streaming |
| **PhantomCore** | ONNX export (future) | Low-latency C++ inference |
| **PhantomLoop** | Token visualization | Real-time codebook monitoring |
| **PhantomCodec** | Token compression | Efficient transmission |

## Project Structure

```
PhantomX/
├── python/
│   ├── phantomx/
│   │   ├── tokenizer/       # POYO tokenization
│   │   ├── vqvae/           # LaBraM VQ-VAE
│   │   ├── tta/             # Test-time adaptation
│   │   ├── data/            # Data loaders
│   │   └── inference/       # Runtime decoder
│   └── setup.py
├── models/                   # Saved codebooks
├── notebooks/                # Analysis & experiments
├── tests/                    # Unit tests
└── requirements.txt
```

## Performance Benchmarks

| Metric | Target | Status |
|--------|--------|--------|
| Codebook size | 256-1024 codes | 🟡 Tuning |
| Reconstruction MSE | < 0.1 cm² | 🟡 Training |
| Zero-shot R² | > 0.7 | ⚪ TBD |
| TTA drift correction | > 20% improvement | ⚪ TBD |
| Electrode dropout (50%) | > 0.6 R² | ⚪ TBD |

## Research References

- **POYO** (NeurIPS 2024): "Neural Population Dynamics as Discrete Token Sequences"
- **LaBraM** (ICLR 2024): "Large Brain Model: Universal Neural Dynamics via Vector Quantization"
- **TTA** (ICML 2025): "Test-Time Entropy Minimization for Non-Stationary Brain Signals"

## License

MIT License - See [LICENSE](../LICENSE)

## Citation

```bibtex
@software{phantomx2026,
  title={PhantomX: LaBraM-POYO Neural Foundation Model},
  author={Youssef El abbassi},
  year={2026},
  url={https://github.com/yelabb/PhantomX}
}
```
