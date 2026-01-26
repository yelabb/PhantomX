"""
PhantomX Unified Trainer

Generic trainer supporting all model types and training modes.
Replaces the scattered training logic in exp*.py files.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from omegaconf import DictConfig
from tqdm import tqdm

from .utils.metrics import compute_metrics, MetricTracker
from .utils.logging import log_metrics


@dataclass
class TrainerState:
    """Trainer state for checkpointing."""
    epoch: int
    global_step: int
    best_metric: float
    best_epoch: int


class Trainer:
    """
    Unified trainer for PhantomX models.
    
    Features:
    - Standard training loop with validation
    - Progressive training for VQ-VAE
    - Early stopping and checkpointing
    - Metric tracking and logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        test_loader: Optional[DataLoader] = None,
        device: str = "auto",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Get trainer config
        trainer_cfg = cfg.get('trainer', {})
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=trainer_cfg.get('learning_rate', 1e-3),
            weight_decay=trainer_cfg.get('weight_decay', 1e-4),
            betas=tuple(trainer_cfg.get('betas', [0.9, 0.999])),
        )
        
        # LR Scheduler
        max_epochs = trainer_cfg.get('max_epochs', 100)
        scheduler_type = trainer_cfg.get('lr_scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=trainer_cfg.get('lr_min', 1e-6)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
            )
        else:
            self.scheduler = None
        
        # Training config
        self.max_epochs = max_epochs
        self.gradient_clip_val = trainer_cfg.get('gradient_clip_val', 1.0)
        self.log_every_n_steps = cfg.get('logging', {}).get('log_every_n_steps', 10)
        
        # Early stopping
        self.early_stopping = trainer_cfg.get('early_stopping', True)
        self.patience = trainer_cfg.get('patience', 20)
        self.monitor = trainer_cfg.get('monitor', 'val/r2')
        
        # Metric tracker
        self.metric_tracker = MetricTracker(
            metric_name='r2',
            mode='max'
        )
        
        # State
        self.state = TrainerState(
            epoch=0,
            global_step=0,
            best_metric=-float('inf'),
            best_epoch=0,
        )
        self.best_model_state = None
        
        # Output directory
        self.output_dir = Path(cfg.get('output_dir', 'logs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dict with training results
        """
        print(f"\n🚀 Starting training on {self.device}")
        print(f"   Model: {type(self.model).__name__}")
        print(f"   Epochs: {self.max_epochs}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            self.state.epoch = epoch
            
            # Training
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = self._validate()
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            log_metrics(all_metrics, step=epoch)
            
            # Update LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val/r2'])
                else:
                    self.scheduler.step()
            
            # Track best model
            r2 = val_metrics['val/r2']
            is_best = self.metric_tracker.update(r2, epoch)
            
            if is_best:
                self.state.best_metric = r2
                self.state.best_epoch = epoch
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save_checkpoint('best_model.pt')
            
            # Log progress
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | Train Loss: {train_metrics['train/loss']:.4f} | "
                  f"Val R²: {r2:.4f} | LR: {lr:.2e}" + (" ⭐" if is_best else ""))
            
            # Early stopping
            if self.early_stopping and self.metric_tracker.should_stop(self.patience):
                print(f"\n⚠️ Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        training_time = time.time() - start_time
        
        # Final test
        test_metrics = {}
        if self.test_loader is not None:
            test_metrics = self._test()
            print(f"\n🎯 Test R²: {test_metrics['test/r2']:.4f}")
        
        results = {
            'best_r2': self.state.best_metric,
            'best_epoch': self.state.best_epoch,
            'training_time_s': training_time,
            **test_metrics,
        }
        
        print(f"\n✅ Training complete!")
        print(f"   Best R²: {self.state.best_metric:.4f} (epoch {self.state.best_epoch})")
        print(f"   Time: {training_time/60:.1f} minutes")
        
        return results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch}", leave=False)
        
        for batch_idx, (neural, target) in enumerate(pbar):
            neural = neural.to(self.device)
            target = target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output, info = self.model(neural)
            
            # Loss
            loss = F.mse_loss(output, target)
            
            # Add VQ loss if present
            if 'commitment_loss' in info:
                loss = loss + info['commitment_loss']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
            
            # Track
            total_loss += loss.item()
            all_preds.append(output.detach())
            all_targets.append(target.detach())
            
            self.state.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(preds, targets, prefix='train/')
        metrics['train/loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        for neural, target in self.val_loader:
            neural = neural.to(self.device)
            target = target.to(self.device)
            
            output, _ = self.model(neural)
            loss = F.mse_loss(output, target)
            
            total_loss += loss.item()
            all_preds.append(output)
            all_targets.append(target)
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics(preds, targets, prefix='val/')
        metrics['val/loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    @torch.no_grad()
    def _test(self) -> Dict[str, float]:
        """Run test evaluation."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        for neural, target in self.test_loader:
            neural = neural.to(self.device)
            target = target.to(self.device)
            
            output, _ = self.model(neural)
            
            all_preds.append(output)
            all_targets.append(target)
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        return compute_metrics(preds, targets, prefix='test/')
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'epoch': self.state.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.state.best_metric,
        }, path)


class ProgressiveTrainer(Trainer):
    """
    Progressive trainer for VQ-VAE models.
    
    Three phases:
    1. Pre-train encoder (VQ disabled)
    2. K-means codebook initialization
    3. Fine-tune with VQ enabled
    """
    
    def train(self) -> Dict[str, Any]:
        """Run progressive training."""
        trainer_cfg = self.cfg.get('trainer', {})
        
        # Phase 1: Pre-train
        print("\n[Phase 1/3] Pre-training encoder...")
        if hasattr(self.model, 'use_vq'):
            self.model.use_vq = False
        
        pretrain_epochs = trainer_cfg.get('pretrain', {}).get('epochs', 30)
        self.max_epochs = pretrain_epochs
        pretrain_results = super().train()
        
        # Phase 2: Init codebook
        print("\n[Phase 2/3] Initializing codebook with k-means...")
        if hasattr(self.model, 'init_codebook'):
            self.model.init_codebook(self.train_loader)
        
        # Phase 3: Fine-tune
        print("\n[Phase 3/3] Fine-tuning with VQ...")
        if hasattr(self.model, 'use_vq'):
            self.model.use_vq = True
        
        # Reset optimizer with lower LR
        finetune_lr = trainer_cfg.get('finetune', {}).get('learning_rate', 3e-4)
        finetune_epochs = trainer_cfg.get('finetune', {}).get('epochs', 50)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=finetune_lr,
            weight_decay=trainer_cfg.get('weight_decay', 1e-4),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=finetune_epochs)
        
        # Reset state for fine-tuning
        self.max_epochs = finetune_epochs
        self.state = TrainerState(epoch=0, global_step=0, best_metric=-float('inf'), best_epoch=0)
        self.metric_tracker = MetricTracker('r2', 'max')
        
        finetune_results = super().train()
        
        return {
            'pretrain_r2': pretrain_results.get('best_r2', 0),
            'finetune_r2': finetune_results.get('best_r2', 0),
            **finetune_results,
        }
