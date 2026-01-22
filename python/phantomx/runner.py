"""
Experiment Runner

Runs experiments from configuration with proper logging, seeding, and result saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from .config import ExperimentConfig
from .datasets import get_dataset
from .datasets.torch_dataset import create_dataloaders
from .models import get_encoder, get_decoder, get_quantizer


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    seed: int
    train_r2: float
    val_r2: float
    test_r2: float
    test_r2_x: float
    test_r2_y: float
    codebook_utilization: Optional[float] = None
    training_time: float = 0.0
    epochs_trained: int = 0
    best_epoch: int = 0


class NeuralDecoder(nn.Module):
    """Unified model: Encoder -> (Quantizer) -> Decoder."""
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        
        Returns:
            predictions: [batch, n_targets]
            info: Dict with quantizer outputs, etc.
        """
        # Encode
        z = self.encoder(x)
        
        info = {}
        
        # Quantize (optional)
        if self.quantizer is not None:
            q_out = self.quantizer(z)
            z = q_out.z_q
            info['indices'] = q_out.indices
            info['commitment_loss'] = q_out.losses.get('commitment', 0.0)
        
        # Decode
        predictions = self.decoder(z)
        
        return predictions, info


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentRunner:
    """
    Runs experiments from configuration.
    
    Usage:
        config = ExperimentConfig.from_yaml("config.yaml")
        runner = ExperimentRunner(config)
        results = runner.run()
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Run experiment for all seeds."""
        results = []
        
        print("=" * 60)
        print(f"Experiment: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"Dataset: {self.config.dataset.name}")
        print(f"Encoder: {self.config.encoder.name}")
        print(f"Quantizer: {self.config.quantizer.name if self.config.quantizer.enabled else 'None'}")
        print(f"Seeds: {self.config.training.seeds}")
        print("=" * 60)
        
        for seed in self.config.training.seeds:
            print(f"\n--- Seed {seed} ---")
            result = self._run_single_seed(seed)
            results.append(result)
            print(f"Test R²: {result.test_r2:.4f} (vx: {result.test_r2_x:.4f}, vy: {result.test_r2_y:.4f})")
        
        # Aggregate results
        summary = self._aggregate_results(results)
        
        # Save results
        self._save_results(results, summary)
        
        return summary
    
    def _run_single_seed(self, seed: int) -> ExperimentResult:
        """Run experiment for a single seed."""
        set_seed(seed)
        start_time = time.time()
        
        # Load dataset
        dataset = get_dataset(self.config.dataset.name)
        loaders = create_dataloaders(
            dataset,
            window_size=self.config.dataset.window_size,
            batch_size=self.config.dataset.batch_size,
            train_ratio=self.config.dataset.train_ratio,
            val_ratio=self.config.dataset.val_ratio,
            temporal_split=self.config.dataset.temporal_split,
            seed=seed,
        )
        
        # Build model
        model = self._build_model(dataset.n_channels, dataset.n_targets)
        model = model.to(self.device)
        
        # Train
        best_model, history = self._train(model, loaders)
        
        # Evaluate
        test_r2, test_r2_x, test_r2_y = self._evaluate(best_model, loaders['test'])
        val_r2, _, _ = self._evaluate(best_model, loaders['val'])
        train_r2, _, _ = self._evaluate(best_model, loaders['train'])
        
        # Codebook utilization
        codebook_util = None
        if self.config.quantizer.enabled:
            codebook_util = self._compute_codebook_utilization(best_model, loaders['test'])
        
        training_time = time.time() - start_time
        
        return ExperimentResult(
            seed=seed,
            train_r2=train_r2,
            val_r2=val_r2,
            test_r2=test_r2,
            test_r2_x=test_r2_x,
            test_r2_y=test_r2_y,
            codebook_utilization=codebook_util,
            training_time=training_time,
            epochs_trained=len(history['train_loss']),
            best_epoch=history.get('best_epoch', 0),
        )
    
    def _build_model(self, n_channels: int, n_targets: int) -> NeuralDecoder:
        """Build model from configuration."""
        encoder_kwargs = {
            'input_dim': n_channels,
            'hidden_dim': self.config.encoder.hidden_dim,
            'n_layers': self.config.encoder.n_layers,
            'dropout': self.config.encoder.dropout,
            'window_size': self.config.dataset.window_size,
            **self.config.encoder.kwargs,
        }
        encoder = get_encoder(self.config.encoder.name, **encoder_kwargs)
        
        decoder_kwargs = {
            'hidden_dim': self.config.encoder.hidden_dim,
            'n_targets': n_targets,
            'n_layers': self.config.decoder.n_layers,
            'dropout': self.config.decoder.dropout,
            **self.config.decoder.kwargs,
        }
        decoder = get_decoder(self.config.decoder.name, **decoder_kwargs)
        
        quantizer = None
        if self.config.quantizer.enabled:
            quantizer_kwargs = {
                'hidden_dim': self.config.encoder.hidden_dim,
                'codebook_size': self.config.quantizer.codebook_size,
                **self.config.quantizer.kwargs,
            }
            if self.config.quantizer.name == 'rvq':
                quantizer_kwargs['n_layers'] = self.config.quantizer.n_layers
            quantizer = get_quantizer(self.config.quantizer.name, **quantizer_kwargs)
        
        return NeuralDecoder(encoder, decoder, quantizer)
    
    def _train(self, model: nn.Module, loaders: Dict[str, DataLoader]) -> Tuple[nn.Module, Dict]:
        """Train the model."""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Scheduler
        if self.config.training.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.epochs
            )
        else:
            scheduler = None
        
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_r2': [], 'best_epoch': 0}
        best_r2 = -float('inf')
        best_state = None
        patience = 0
        
        for epoch in range(self.config.training.epochs):
            # Train epoch
            model.train()
            train_loss = 0.0
            
            for X, y in loaders['train']:
                X, y = X.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                pred, info = model(X)
                
                loss = criterion(pred, y)
                if 'commitment_loss' in info:
                    loss = loss + info['commitment_loss']
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(loaders['train'])
            
            # Validate
            val_r2, _, _ = self._evaluate(model, loaders['val'])
            
            history['train_loss'].append(train_loss)
            history['val_r2'].append(val_r2)
            
            # Best model
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                history['best_epoch'] = epoch
                patience = 0
            else:
                patience += 1
            
            if scheduler:
                scheduler.step()
            
            # Early stopping
            if patience >= self.config.training.early_stopping:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
            
            # Progress
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: loss={train_loss:.4f}, val_r2={val_r2:.4f}")
        
        # Load best model
        if best_state:
            model.load_state_dict(best_state)
        
        return model, history
    
    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate model and return R² scores."""
        model.eval()
        
        all_preds = []
        all_targets = []
        
        for X, y in loader:
            X = X.to(self.device)
            pred, _ = model(X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        r2_x = compute_r2(targets[:, 0], preds[:, 0])
        r2_y = compute_r2(targets[:, 1], preds[:, 1])
        r2_overall = (r2_x + r2_y) / 2
        
        return r2_overall, r2_x, r2_y
    
    @torch.no_grad()
    def _compute_codebook_utilization(self, model: nn.Module, loader: DataLoader) -> float:
        """Compute fraction of codebook used."""
        model.eval()
        
        all_indices = []
        for X, y in loader:
            X = X.to(self.device)
            _, info = model(X)
            if 'indices' in info:
                all_indices.append(info['indices'].cpu().numpy())
        
        if not all_indices:
            return 0.0
        
        indices = np.concatenate(all_indices, axis=0)
        unique = len(np.unique(indices))
        total = self.config.quantizer.codebook_size
        if indices.ndim > 1:
            total *= indices.shape[1]  # For RVQ
        
        return unique / total
    
    def _aggregate_results(self, results: list) -> Dict[str, Any]:
        """Aggregate results across seeds."""
        test_r2s = [r.test_r2 for r in results]
        
        return {
            'config': self.config.to_dict(),
            'n_seeds': len(results),
            'test_r2_mean': np.mean(test_r2s),
            'test_r2_std': np.std(test_r2s),
            'test_r2_min': np.min(test_r2s),
            'test_r2_max': np.max(test_r2s),
            'results': [asdict(r) for r in results],
        }
    
    def _save_results(self, results: list, summary: Dict):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save config
        config_path = self.output_dir / "config.yaml"
        self.config.save(str(config_path))
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  Test R² = {summary['test_r2_mean']:.4f} ± {summary['test_r2_std']:.4f}")


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Convenience function to run an experiment."""
    runner = ExperimentRunner(config)
    return runner.run()
