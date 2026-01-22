"""
Experiment Configuration System

Load experiments from YAML configs and run them.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "mc_maze"
    window_size: int = 10
    batch_size: int = 64
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    temporal_split: bool = True


@dataclass
class EncoderConfig:
    """Encoder configuration."""
    name: str = "lstm"
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    # Additional kwargs passed to encoder
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderConfig:
    """Decoder configuration."""
    name: str = "mlp"
    n_layers: int = 2
    dropout: float = 0.1
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantizerConfig:
    """Quantizer configuration (optional)."""
    enabled: bool = False
    name: str = "rvq"
    codebook_size: int = 128
    n_layers: int = 4
    commitment_cost: float = 0.25
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # "cosine", "step", "none"
    early_stopping: int = 20
    seeds: List[int] = field(default_factory=lambda: [42])
    
    # Progressive training (for VQ-VAE)
    progressive: bool = False
    pretrain_epochs: int = 30


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str
    description: str = ""
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Output directory
    output_dir: str = "results"
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        config = cls(name=data.get('name', 'unnamed'))
        config.description = data.get('description', '')
        
        if 'dataset' in data:
            config.dataset = DatasetConfig(**data['dataset'])
        if 'encoder' in data:
            config.encoder = EncoderConfig(**data['encoder'])
        if 'decoder' in data:
            config.decoder = DecoderConfig(**data['decoder'])
        if 'quantizer' in data:
            config.quantizer = QuantizerConfig(**data['quantizer'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'output_dir' in data:
            config.output_dir = data['output_dir']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'dataset': self.dataset.__dict__,
            'encoder': self.encoder.__dict__,
            'decoder': self.decoder.__dict__,
            'quantizer': self.quantizer.__dict__,
            'training': self.training.__dict__,
            'output_dir': self.output_dir,
        }
    
    def save(self, path: str):
        """Save configuration to YAML."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"ExperimentConfig(name='{self.name}', encoder='{self.encoder.name}', quantizer={self.quantizer.enabled})"


# Preset configurations
PRESETS = {
    "lstm_baseline": ExperimentConfig(
        name="lstm_baseline",
        description="LSTM baseline for velocity decoding",
        encoder=EncoderConfig(name="lstm", hidden_dim=256, n_layers=2),
        decoder=DecoderConfig(name="mlp", n_layers=2),
        quantizer=QuantizerConfig(enabled=False),
    ),
    
    "transformer_baseline": ExperimentConfig(
        name="transformer_baseline",
        description="Transformer baseline",
        encoder=EncoderConfig(name="transformer", hidden_dim=256, n_layers=2, kwargs={"n_heads": 4}),
        decoder=DecoderConfig(name="mlp", n_layers=2),
        quantizer=QuantizerConfig(enabled=False),
    ),
    
    "vqvae_progressive": ExperimentConfig(
        name="vqvae_progressive",
        description="Progressive VQ-VAE with LSTM encoder",
        encoder=EncoderConfig(name="lstm", hidden_dim=256, n_layers=2),
        decoder=DecoderConfig(name="mlp", n_layers=2),
        quantizer=QuantizerConfig(enabled=True, name="vq", codebook_size=256),
        training=TrainingConfig(progressive=True, pretrain_epochs=30),
    ),
    
    "rvq_distillation": ExperimentConfig(
        name="rvq_distillation",
        description="RVQ student distilled from transformer teacher",
        encoder=EncoderConfig(name="mlp", hidden_dim=256, n_layers=3),
        decoder=DecoderConfig(name="mlp", n_layers=2),
        quantizer=QuantizerConfig(enabled=True, name="rvq", codebook_size=128, n_layers=4),
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    """Get a preset configuration."""
    if name not in PRESETS:
        raise KeyError(f"Preset '{name}' not found. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def list_presets():
    """List available preset configurations."""
    print("Available experiment presets:")
    for name, config in PRESETS.items():
        print(f"  {name}: {config.description}")
