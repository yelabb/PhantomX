"""
Progressive VQ-VAE Trainer

Implements the three-phase training approach:
1. Pre-train encoder without VQ
2. K-means initialize codebook
3. Finetune with VQ enabled
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from typing import Optional
from pathlib import Path


class ProgressiveTrainer:
    """
    Progressive trainer for VQ-VAE.
    
    Training phases:
    1. Pre-train encoder (learns good representations)
    2. Initialize VQ codebook with k-means
    3. Finetune with VQ (learns discretization)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        pretrain_epochs: int = 30,
        finetune_epochs: int = 50,
        pretrain_lr: float = 1e-3,
        finetune_lr: float = 3e-4,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.weight_decay = weight_decay
        
        self.history = {
            'pretrain_r2': [],
            'finetune_r2': [],
            'perplexity': []
        }
    
    def train(self, verbose: bool = True) -> dict:
        """
        Run full progressive training.
        
        Returns:
            Dict with best validation R² and training history
        """
        # Phase 1: Pre-train
        if verbose:
            print("\n[Phase 1] Pre-training encoder without VQ...")
        pretrain_r2 = self._pretrain(verbose)
        
        # Phase 2: Init codebook
        if verbose:
            print("\n[Phase 2] Initializing codebook with k-means...")
        self._init_codebook()
        
        # Phase 3: Finetune
        if verbose:
            print("\n[Phase 3] Finetuning with VQ...")
        best_r2, best_state = self._finetune(verbose)
        
        # Load best model
        self.model.load_state_dict(best_state)
        
        return {
            'pretrain_r2': pretrain_r2,
            'best_r2': best_r2,
            'history': self.history
        }
    
    def _pretrain(self, verbose: bool) -> float:
        """Pre-train encoder without VQ layer."""
        self.model.use_vq = False
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.pretrain_lr,
            weight_decay=self.weight_decay
        )
        
        best_r2 = -float('inf')
        
        for epoch in range(self.pretrain_epochs):
            self.model.train()
            for batch in self.train_loader:
                window = batch['window'].to(self.device)
                velocity = batch['velocity'].to(self.device)
                
                optimizer.zero_grad()
                output = self.model(window, velocity)
                output['total_loss'].backward()
                optimizer.step()
            
            val_r2 = self._validate()
            self.history['pretrain_r2'].append(val_r2)
            
            if val_r2 > best_r2:
                best_r2 = val_r2
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.pretrain_epochs}: val_r2={val_r2:.4f}")
        
        if verbose:
            print(f"  Best pretrain R²: {best_r2:.4f}")
        
        return best_r2
    
    def _init_codebook(self) -> None:
        """Initialize VQ codebook with k-means on encoder outputs."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in self.train_loader:
                window = batch['window'].to(self.device)
                z_e = self.model.encode(window)
                embeddings.append(z_e.cpu())
        
        z_all = torch.cat(embeddings, dim=0)
        self.model.vq.init_from_data(z_all)
    
    def _finetune(self, verbose: bool) -> tuple:
        """Finetune model with VQ enabled."""
        self.model.use_vq = True
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.finetune_lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.finetune_epochs
        )
        
        best_r2 = -float('inf')
        best_state = None
        patience = 0
        
        for epoch in range(self.finetune_epochs):
            self.model.train()
            for batch in self.train_loader:
                window = batch['window'].to(self.device)
                velocity = batch['velocity'].to(self.device)
                
                optimizer.zero_grad()
                output = self.model(window, velocity)
                output['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            val_r2, perplexity = self._validate(return_perplexity=True)
            self.history['finetune_r2'].append(val_r2)
            self.history['perplexity'].append(perplexity)
            
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.finetune_epochs}: val_r2={val_r2:.4f}, perp={perplexity:.1f}")
            
            if patience >= 15:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        return best_r2, best_state
    
    def _validate(self, return_perplexity: bool = False):
        """Compute validation R² and optionally perplexity."""
        self.model.eval()
        preds, targets = [], []
        perp_sum = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                window = batch['window'].to(self.device)
                velocity = batch['velocity'].to(self.device)
                output = self.model(window)
                preds.append(output['velocity_pred'].cpu().numpy())
                targets.append(velocity.cpu().numpy())
                if 'perplexity' in output:
                    perp_sum += output['perplexity'].item()
        
        r2 = r2_score(np.concatenate(targets), np.concatenate(preds))
        
        if return_perplexity:
            perp_avg = perp_sum / len(self.val_loader)
            return r2, perp_avg
        return r2
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate model on test set."""
        self.model.eval()
        preds, targets = [], []
        indices = []
        
        with torch.no_grad():
            for batch in test_loader:
                window = batch['window'].to(self.device)
                output = self.model(window)
                preds.append(output['velocity_pred'].cpu().numpy())
                targets.append(batch['velocity'].numpy())
                if 'indices' in output:
                    indices.append(output['indices'].cpu().numpy())
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        
        result = {
            'r2': r2_score(targets, preds),
            'r2_vx': r2_score(targets[:, 0], preds[:, 0]),
            'r2_vy': r2_score(targets[:, 1], preds[:, 1]),
        }
        
        if indices:
            all_indices = np.concatenate(indices)
            result['n_codes_used'] = len(np.unique(all_indices))
            result['indices'] = all_indices
        
        return result
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
