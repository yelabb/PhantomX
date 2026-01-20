"""
FSQ-VAE Trainer

Training pipeline for FSQ-VAE with dual-head decoder.

Key Differences from VQ-VAE Trainer:
1. No commitment loss tracking (FSQ eliminates it)
2. Tracks both kinematics and reconstruction losses
3. Supports variable reconstruction weight (λ)
4. Monitors FSQ-specific metrics (codebook coverage)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Union
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from .fsq_vae import FSQVAE, FSQVAEWithCausalTransformer


class FSQVAETrainer:
    """
    Trainer for FSQ-VAE model with dual-head decoder.
    
    Implements:
    - Training with combined kinematics + reconstruction loss
    - Dynamic reconstruction weight scheduling (optional)
    - Codebook usage monitoring (no collapse by design!)
    - R² metric tracking for velocity prediction
    """
    
    def __init__(
        self,
        model: Union[FSQVAE, FSQVAEWithCausalTransformer],
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        reconstruction_weight: float = 0.5,
        use_weight_schedule: bool = False,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: FSQ-VAE model to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            reconstruction_weight: Initial weight for reconstruction loss (λ)
            use_weight_schedule: Whether to schedule λ during training
            device: Device to train on
        """
        self.model = model
        self.reconstruction_weight = reconstruction_weight
        self.initial_reconstruction_weight = reconstruction_weight
        self.use_weight_schedule = use_weight_schedule
        
        # Override model's reconstruction weight
        self.model.reconstruction_weight = reconstruction_weight
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'kinematics_loss': [],
            'reconstruction_loss': [],
            'perplexity': [],
            'codebook_usage': [],
            'r2_score': [],
            'reconstruction_weight': []
        }
    
    def _update_reconstruction_weight(self, epoch: int, total_epochs: int) -> None:
        """
        Optionally schedule the reconstruction weight.
        
        Strategy: Start with high λ (encourage rich representations),
        gradually decrease to focus on kinematics task.
        """
        if not self.use_weight_schedule:
            return
        
        # Linear decay from initial to 0.1
        progress = epoch / total_epochs
        min_weight = 0.1
        self.reconstruction_weight = (
            self.initial_reconstruction_weight * (1 - 0.8 * progress)
        )
        self.reconstruction_weight = max(self.reconstruction_weight, min_weight)
        self.model.reconstruction_weight = self.reconstruction_weight
    
    def compute_r2(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Compute R² score for velocity prediction.
        
        R² = 1 - SS_res / SS_tot
        """
        with torch.no_grad():
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - targets.mean()) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            return r2.item()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epoch: int = 0,
        is_causal_transformer: bool = False
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epoch: Current epoch number
            is_causal_transformer: Whether model uses raw spikes
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        # Metrics accumulators
        total_loss = 0.0
        total_kin_loss = 0.0
        total_recon_loss = 0.0
        total_perplexity = 0.0
        total_codebook_usage = 0.0
        all_predictions = []
        all_targets = []
        n_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            if is_causal_transformer:
                # Raw spike input
                spikes = batch['spikes'].to(self.device)
                targets = batch['kinematics'].to(self.device)
                output = self.model(spikes, targets)
            else:
                # Tokenized input
                tokens = batch['tokens'].to(self.device)
                targets = batch['kinematics'].to(self.device)
                
                # Handle reconstruction targets
                if 'spikes' in batch:
                    recon_targets = batch['spikes'].to(self.device)
                elif 'reconstruction_targets' in batch:
                    recon_targets = batch['reconstruction_targets'].to(self.device)
                else:
                    # Fall back to using tokens as reconstruction targets
                    recon_targets = tokens
                
                output = self.model(tokens, targets, recon_targets)
            
            loss = output['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_kin_loss += output['kinematics_loss'].item()
            total_recon_loss += output.get('reconstruction_loss', torch.tensor(0.0)).item()
            total_perplexity += output['perplexity'].item()
            total_codebook_usage += output['codebook_usage']
            n_batches += 1
            
            # Store for R² calculation
            all_predictions.append(output['kinematics_pred'].detach())
            all_targets.append(targets.detach())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'kin': f"{output['kinematics_loss'].item():.4f}",
                'recon': f"{output.get('reconstruction_loss', torch.tensor(0.0)).item():.4f}",
                'perp': f"{output['perplexity'].item():.1f}"
            })
        
        # Compute epoch metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        r2 = self.compute_r2(all_predictions, all_targets)
        
        metrics = {
            'train_loss': total_loss / n_batches,
            'kinematics_loss': total_kin_loss / n_batches,
            'reconstruction_loss': total_recon_loss / n_batches,
            'perplexity': total_perplexity / n_batches,
            'codebook_usage': total_codebook_usage / n_batches,
            'r2_score': r2,
            'reconstruction_weight': self.reconstruction_weight
        }
        
        # Validation
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader, is_causal_transformer)
            metrics['val_loss'] = val_metrics['loss']
            metrics['val_r2'] = val_metrics['r2_score']
        
        # Update scheduler
        self.scheduler.step()
        
        # Store history
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        return metrics
    
    def evaluate(
        self,
        data_loader: DataLoader,
        is_causal_transformer: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            data_loader: Data loader for evaluation
            is_causal_transformer: Whether model uses raw spikes
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_kin_loss = 0.0
        total_recon_loss = 0.0
        all_predictions = []
        all_targets = []
        n_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if is_causal_transformer:
                    spikes = batch['spikes'].to(self.device)
                    targets = batch['kinematics'].to(self.device)
                    output = self.model(spikes, targets)
                else:
                    tokens = batch['tokens'].to(self.device)
                    targets = batch['kinematics'].to(self.device)
                    
                    if 'spikes' in batch:
                        recon_targets = batch['spikes'].to(self.device)
                    elif 'reconstruction_targets' in batch:
                        recon_targets = batch['reconstruction_targets'].to(self.device)
                    else:
                        recon_targets = tokens
                    
                    output = self.model(tokens, targets, recon_targets)
                
                total_loss += output['total_loss'].item()
                total_kin_loss += output['kinematics_loss'].item()
                total_recon_loss += output.get('reconstruction_loss', torch.tensor(0.0)).item()
                
                all_predictions.append(output['kinematics_pred'])
                all_targets.append(targets)
                n_batches += 1
        
        # Compute R²
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        r2 = self.compute_r2(all_predictions, all_targets)
        
        return {
            'loss': total_loss / n_batches,
            'kinematics_loss': total_kin_loss / n_batches,
            'reconstruction_loss': total_recon_loss / n_batches,
            'r2_score': r2
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_dir: str = 'checkpoints',
        save_every: int = 10,
        is_causal_transformer: bool = False
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            is_causal_transformer: Whether model uses raw spikes
            
        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"=" * 60)
        print(f"FSQ-VAE Training")
        print(f"=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Codebook size: {self.model.codebook_size}")
        print(f"Initial reconstruction weight (λ): {self.reconstruction_weight}")
        print(f"Weight scheduling: {self.use_weight_schedule}")
        print(f"=" * 60)
        
        best_val_loss = float('inf')
        best_val_r2 = -float('inf')
        
        for epoch in range(1, epochs + 1):
            # Update reconstruction weight if scheduling
            self._update_reconstruction_weight(epoch, epochs)
            
            # Train epoch
            metrics = self.train_epoch(
                train_loader, val_loader, epoch, is_causal_transformer
            )
            
            # Print metrics
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train Loss: {metrics['train_loss']:.4f}")
            print(f"  Kinematics Loss: {metrics['kinematics_loss']:.4f}")
            print(f"  Reconstruction Loss: {metrics['reconstruction_loss']:.4f}")
            print(f"  Perplexity: {metrics['perplexity']:.1f}")
            print(f"  Codebook Usage: {metrics['codebook_usage']:.1%}")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  λ (recon weight): {metrics['reconstruction_weight']:.3f}")
            
            if 'val_loss' in metrics:
                print(f"  Val Loss: {metrics['val_loss']:.4f}")
                print(f"  Val R²: {metrics['val_r2']:.4f}")
                
                # Save best model (by R²)
                if metrics['val_r2'] > best_val_r2:
                    best_val_r2 = metrics['val_r2']
                    self.save_checkpoint(save_path / 'best_model.pt', epoch, metrics)
                    print(f"  ✓ New best model (R² = {best_val_r2:.4f})!")
            
            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch_{epoch}.pt', epoch, metrics)
        
        # Save final model
        self.save_checkpoint(save_path / 'final_model.pt', epochs, metrics)
        
        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"Best Validation R²: {best_val_r2:.4f}")
        print(f"Model saved to: {save_path}")
        print(f"{'=' * 60}")
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': self.model.config if hasattr(self.model, 'config') else {}
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint


def compute_baseline_comparison(
    fsq_r2: float,
    lstm_r2: float = 0.78
) -> str:
    """
    Generate a comparison summary against LSTM baseline.
    
    Args:
        fsq_r2: R² achieved by FSQ-VAE
        lstm_r2: R² of LSTM baseline (default: 0.78)
        
    Returns:
        Formatted comparison string
    """
    gap = lstm_r2 - fsq_r2
    gap_pct = (gap / lstm_r2) * 100
    
    if fsq_r2 > lstm_r2:
        status = "✓ BEAT LSTM BASELINE!"
        improvement = (fsq_r2 - lstm_r2) / lstm_r2 * 100
        details = f"Improvement: +{improvement:.1f}%"
    else:
        status = "✗ Still below LSTM baseline"
        details = f"Gap: {gap:.4f} ({gap_pct:.1f}% below)"
    
    return f"""
{'=' * 50}
BASELINE COMPARISON
{'=' * 50}
LSTM Baseline R²:  {lstm_r2:.4f}
FSQ-VAE R²:        {fsq_r2:.4f}
{'=' * 50}
{status}
{details}
{'=' * 50}
"""
