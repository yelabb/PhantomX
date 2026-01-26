# Migration Guide: v0.1 → v0.2

This guide helps you migrate from the old "one script per experiment" approach to the new configuration-driven system.

## Key Changes

### 1. Single Entry Point

**Before (v0.1):**
```bash
python python/exp25_mamba_mcrtt.py
python python/exp23_statistical_validation.py
python python/exp22c_multiseed_teacher.py
```

**After (v0.2):**
```bash
python train.py experiment=exp25_mamba
python train.py experiment=exp23_validation
python train.py experiment=exp22c_teacher
```

### 2. Configuration Files Replace Hardcoded Params

**Before:** Hyperparameters scattered in `exp25_mamba_mcrtt.py` (lines 50-100)
```python
d_model = 128
n_layers = 4
window_size = 80  # 2 seconds
batch_size = 32
learning_rate = 1e-3
```

**After:** Centralized in `configs/experiment/exp25_mamba.yaml`
```yaml
model:
  d_model: 128
  n_layers: 4
  window_size: 80

dataset:
  batch_size: 32

trainer:
  learning_rate: 1e-3
```

### 3. Import Paths Changed

**Before:**
```python
from python.phantomx.model import ProgressiveVQVAE
from python.phantomx.trainer import ProgressiveTrainer
```

**After:**
```python
from src.models import build_model, ProgressiveVQVAE
from src.trainer import Trainer, ProgressiveTrainer
from src.datamodules import build_datamodule
```

### 4. Experiment Tracking

**Before:** Manual README.md tables

**After:** Automatic WandB logging
```bash
# Disable if needed
python train.py logging.use_wandb=false
```

## Mapping Old Experiments to New Configs

| Old Script | New Command |
|------------|-------------|
| `exp25_mamba_mcrtt.py` | `python train.py experiment=exp25_mamba` |
| `exp22c_multiseed_teacher.py` | `python train.py experiment=exp22c_teacher` |
| `exp23_statistical_validation.py` | `python train.py experiment=exp23_validation` |
| `exp10_beat_lstm.py` | `python train.py experiment=lstm_baseline` |

## Creating New Experiments

### 1. Create an experiment config

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_
defaults:
  - override /model: mamba
  - override /dataset: mc_rtt
  - override /augmentation: strong

experiment_name: "my_experiment"
notes: "Testing new idea..."

model:
  n_layers: 6
  d_model: 256

seed: 42
```

### 2. Run it

```bash
python train.py experiment=my_experiment
```

### 3. Compare variations

```bash
python train.py experiment=my_experiment --multirun model.n_layers=4,6,8
```

## Common Migration Issues

### Issue: ModuleNotFoundError

**Solution:** The new structure uses `src/` instead of `python/phantomx/`
```bash
# Make sure you're in the project root
cd PhantomX
pip install -e .
```

### Issue: Config not found

**Solution:** Run from project root where `configs/` is located
```bash
# Wrong (inside configs/)
cd configs && python ../train.py  # ❌

# Right (project root)
python train.py  # ✓
```

### Issue: Different results than before

**Checklist:**
1. Same seed? (`seed: 42`)
2. Same augmentation? (`augmentation: standard` vs `none`)
3. Same window size? (`model.window_size`)
4. Same batch size? (`dataset.batch_size`)

Use the config to ensure reproducibility:
```bash
python train.py experiment=exp25_mamba --cfg job
```

## Backward Compatibility

The old `python/phantomx/` code still works. You can gradually migrate:

```python
# This still works
from python.phantomx.model import ProgressiveVQVAE
```

But we recommend migrating to enjoy:
- ✅ Automatic config saving
- ✅ WandB integration
- ✅ Reproducible experiments
- ✅ Easy hyperparameter sweeps
- ✅ No more code duplication
