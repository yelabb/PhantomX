# PhantomX Quick Reference

## Installation

```bash
cd PhantomX
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e python/
```

## Training

```bash
# Train universal codebook on MC_Maze
python python/train_labram.py \
    --data_path ../PhantomLink/data/mc_maze.nwb \
    --epochs 100 \
    --batch_size 32 \
    --num_codes 256 \
    --save_dir models/
```

## Zero-Shot Inference

```bash
# Test on held-out trials
python python/test_zero_shot.py \
    --model_path models/best_model.pt \
    --data_path ../PhantomLink/data/mc_maze.nwb \
    --use_tta  # Enable test-time adaptation
```

## Python API

### Basic Usage

```python
from phantomx.inference import LabramDecoder
import numpy as np

# Load pre-trained decoder
decoder = LabramDecoder.load("models/best_model.pt")

# Decode spike counts
spikes = np.random.poisson(2.0, size=142)  # [142 channels]
velocity = decoder.decode(spikes)           # [vx, vy]
print(f"Velocity: ({velocity[0]:.3f}, {velocity[1]:.3f})")
```

### With Test-Time Adaptation

```python
decoder = LabramDecoder.load(
    "models/best_model.pt",
    use_tta=True,
    tta_strategy='entropy'
)

# Decoder adapts automatically with each call
for packet in live_stream:
    velocity = decoder.decode(packet.spikes)
```

### Batch Processing

```python
# Process 1000 samples efficiently
spike_batch = np.random.poisson(2.0, size=(1000, 142))
velocities = decoder.decode_batch(spike_batch, batch_size=32)
# velocities.shape: (1000, 2)
```

## Integration with PhantomLink

```python
from phantomlink.labram_integration import LabramDecoderMiddleware

# In playback_engine.py
middleware = LabramDecoderMiddleware(
    model_path="../PhantomX/models/best_model.pt",
    use_tta=True
)

# Decode packets
for packet in stream:
    velocity = middleware.decode_packet(packet.spikes.spike_counts)
    # velocity: {'vx': 0.123, 'vy': -0.456}
```

## Model Architecture

```
Input: Spike Counts [142 channels]
    ↓
SpikeTokenizer
    ↓ [16 tokens]
SpikeEncoder (MLP)
    ↓ [64-dim continuous latent]
VectorQuantizer
    ↓ [64-dim quantized → codebook lookup]
Codebook [256 codes × 64 dims]
    ↓
KinematicsDecoder (MLP)
    ↓
Output: Velocity [vx, vy]
```

## Training Configuration

Default hyperparameters:

```python
{
    'n_tokens': 16,              # Tokenizer output size
    'token_dim': 256,            # Vocabulary size
    'embedding_dim': 64,         # Latent dimension
    'num_codes': 256,            # Codebook size
    'commitment_cost': 0.25,     # β for commitment loss
    'learning_rate': 3e-4,       # AdamW LR
    'batch_size': 32,            # Training batch size
    'epochs': 100                # Training epochs
}
```

## Performance Metrics

### Latency
- **CPU**: ~1-3ms per packet
- **GPU**: <1ms per packet
- **With TTA**: +1-2ms overhead

### Accuracy (Expected)
- **Zero-shot R²**: >0.7 on held-out trials
- **With TTA**: +20% improvement on drifted data
- **50% electrode dropout**: >0.6 R²

## Key Features

✅ **Electrode-dropout robust**: 50% channel loss → system keeps working  
✅ **Zero-shot decoding**: Pre-trained codebook transfers to new sessions  
✅ **Population geometry**: Understands neural manifolds, not individual wires  
✅ **Real-time adaptation**: Online gradient updates counter signal drift  

## File Structure

```
PhantomX/
├── python/
│   ├── phantomx/           # Main package
│   │   ├── tokenizer/      # POYO tokenization
│   │   ├── vqvae/          # LaBraM VQ-VAE
│   │   ├── tta/            # Test-time adaptation
│   │   ├── data/           # Data loaders
│   │   └── inference/      # Runtime decoder
│   ├── train_labram.py     # Training script
│   └── test_zero_shot.py   # Testing script
├── models/                  # Saved codebooks
├── notebooks/               # Jupyter notebooks
├── docs/                    # Documentation
└── tests/                   # Unit tests
```

## Common Issues

### Import Error
```bash
# If "ModuleNotFoundError: No module named 'phantomx'"
cd PhantomX/python
pip install -e .
```

### High Latency
```python
# Use GPU acceleration
decoder = LabramDecoder.load(model_path, device='cuda')

# Or disable TTA for faster inference
decoder = LabramDecoder.load(model_path, use_tta=False)
```

### TTA Instability
```python
# Reset TTA state periodically
if frame_count % 1200 == 0:  # Every 30s at 40Hz
    decoder.reset_tta()
```

## Citation

```bibtex
@software{phantomx2026,
  title={PhantomX: LaBraM-POYO Neural Foundation Model},
  author={Youssef El abbassi},
  year={2026},
  url={https://github.com/yelabb/PhantomX}
}
```

## Support

- **Documentation**: See [README.md](README.md) for detailed architecture
- **Integration**: See [PHANTOMLINK_INTEGRATION.md](docs/PHANTOMLINK_INTEGRATION.md)
- **Examples**: See [notebooks/quick_start.ipynb](notebooks/quick_start.ipynb)
- **Implementation**: See [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md)

---

**Quick Links:**
- [Train your first model](#training)
- [Test zero-shot decoding](#zero-shot-inference)
- [Integrate with PhantomLink](#integration-with-phantomlink)
