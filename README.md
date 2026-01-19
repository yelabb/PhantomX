# PhantomX

> ⚠️ **Experimental / Learning Project** - Not for production use

Personal research sandbox for exploring LaBraM-POYO neural decoding approaches for BCI applications.

## What This Is

An experimental project exploring:
- VQ-VAE based neural codebooks
- POYO-style spike tokenization
- Test-time adaptation for signal drift
- Zero-shot velocity decoding from motor cortex data

## Main Documentation

📓 **[RESEARCH_LOG.md](RESEARCH_LOG.md)** - Detailed experiment notes, results, and analysis

## Project Structure

```
python/phantomx/     # Core modules (tokenizer, vqvae, tta, inference)
data/                # MC_Maze dataset (.nwb)
models/              # Trained model checkpoints
notebooks/           # Quick start notebook
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Current Status

Work in progress - see [RESEARCH_LOG.md](RESEARCH_LOG.md) for latest experiments and findings
