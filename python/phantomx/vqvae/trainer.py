"""
VQ-VAE Training Pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from .vqvae import VQVAE


class VQVAETrainer:
    """
    Trainer for LaBraM VQ-VAE model.
    
    Implements:
    - Training loop with reconstruction + commitment losses
    - Codebook perplexity monitoring
    - Learning rate scheduling
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: VQVAE,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: VQ-VAE model to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        
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
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated based on epochs
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'commitment_loss': [],
            'perplexity': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        # Metrics accumulators
        total_loss = 0.0
        total_recon_loss = 0.0
        total_commit_loss = 0.0
        total_perplexity = 0.0
        n_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            tokens = batch['tokens'].to(self.device)
            targets = batch['kinematics'].to(self.device)
            
            # Forward pass
            output = self.model(tokens, targets)
            loss = output['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += output['reconstruction_loss'].item()
            total_commit_loss += output['commitment_loss'].item()
            total_perplexity += output['perplexity'].item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{output['reconstruction_loss'].item():.4f}",
                'perp': f"{output['perplexity'].item():.1f}"
            })
        
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / n_batches,
            'reconstruction_loss': total_recon_loss / n_batches,
            'commitment_loss': total_commit_loss / n_batches,
            'perplexity': total_perplexity / n_batches
        }
        
        # Validation
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)
            metrics['val_loss'] = val_metrics['loss']
        
        # Update scheduler
        self.scheduler.step()
        
        # Store history
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        return metrics
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                tokens = batch['tokens'].to(self.device)
                targets = batch['kinematics'].to(self.device)
                
                output = self.model(tokens, targets)
                
                total_loss += output['total_loss'].item()
                total_recon_loss += output['reconstruction_loss'].item()
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'reconstruction_loss': total_recon_loss / n_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_dir: str = 'checkpoints',
        save_every: int = 10
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Update scheduler T_max
        self.scheduler.T_max = epochs
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Train epoch
            metrics = self.train_epoch(train_loader, val_loader, epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train Loss: {metrics['train_loss']:.4f}")
            print(f"  Recon Loss: {metrics['reconstruction_loss']:.4f}")
            print(f"  Commit Loss: {metrics['commitment_loss']:.4f}")
            print(f"  Perplexity: {metrics['perplexity']:.1f}")
            
            if 'val_loss' in metrics:
                print(f"  Val Loss: {metrics['val_loss']:.4f}")
                
                # Save best model
                if metrics['val_loss'] < best_val_loss:
                    best_val_loss = metrics['val_loss']
                    self.save_checkpoint(save_path / 'best_model.pt', epoch, metrics)
                    print(f"  ✓ New best model saved!")
            
            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch_{epoch}.pt', epoch, metrics)
        
        # Save final model
        self.save_checkpoint(save_path / 'final_model.pt', epochs, metrics)
        
        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
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
