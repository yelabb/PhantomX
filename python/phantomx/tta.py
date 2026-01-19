"""
Test-Time Adaptation (TTA) for PhantomX

Handles signal drift and electrode changes between sessions by adapting
the model at inference time using unsupervised objectives.

Approaches:
1. Entropy Minimization - Encourage confident predictions
2. Codebook Alignment - Match new data to existing codes
3. Self-Training - Use high-confidence predictions as pseudo-labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List
import numpy as np
from sklearn.metrics import r2_score


class TTAWrapper(nn.Module):
    """
    Wrapper for test-time adaptation of VQ-VAE models.
    
    Adapts encoder and/or decoder while keeping VQ codebook frozen
    to maintain discrete representations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        adapt_encoder: bool = True,
        adapt_decoder: bool = False,
        lr: float = 1e-4,
        momentum: float = 0.9
    ):
        super().__init__()
        self.model = model
        self.adapt_encoder = adapt_encoder
        self.adapt_decoder = adapt_decoder
        
        # Collect trainable parameters
        params = []
        if adapt_encoder:
            params.extend(model.encoder.parameters())
        if adapt_decoder:
            params.extend(model.decoder.parameters())
        
        if params:
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
        else:
            self.optimizer = None
        
        # Store original state for reset
        self._original_state = {
            k: v.clone() for k, v in model.state_dict().items()
        }
    
    def reset(self):
        """Reset model to original state."""
        self.model.load_state_dict(self._original_state)
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass (no adaptation)."""
        return self.model(x)
    
    def adapt(self, x: torch.Tensor, n_steps: int = 1) -> dict:
        """
        Adapt model on new data and return predictions.
        
        Uses entropy minimization on VQ assignments.
        """
        self.model.train()
        
        for _ in range(n_steps):
            if self.optimizer:
                self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(x)
            
            # Adaptation loss: entropy minimization on code assignments
            if hasattr(self.model, 'vq') and hasattr(self.model.vq, 'to_logits'):
                # Gumbel VQ: use logits
                logits = self.model.vq.to_logits(output['z_e'])
                probs = F.softmax(logits, dim=-1)
            else:
                # EMA VQ: use distances as negative logits
                z_e = output['z_e']
                distances = torch.cdist(z_e, self.model.vq.embeddings)
                probs = F.softmax(-distances / 0.1, dim=-1)  # temperature
            
            # Entropy: lower is more confident
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            if self.optimizer:
                entropy.backward()
                self.optimizer.step()
        
        # Final prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output


class OnlineTTA:
    """
    Online test-time adaptation with sliding window.
    
    Accumulates a buffer of recent samples and periodically adapts.
    """
    
    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 128,
        adapt_every: int = 32,
        adapt_steps: int = 3,
        lr: float = 1e-4
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.adapt_every = adapt_every
        self.adapt_steps = adapt_steps
        
        self.buffer: List[torch.Tensor] = []
        self.sample_count = 0
        
        # Only adapt encoder
        self.optimizer = torch.optim.SGD(
            model.encoder.parameters(), 
            lr=lr, 
            momentum=0.9
        )
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict with online adaptation.
        
        Args:
            x: Input batch [batch, window_size, n_channels]
        
        Returns:
            Velocity predictions [batch, 2]
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        # Add to buffer
        self.buffer.append(x.detach().cpu())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        self.sample_count += x.size(0)
        
        # Adapt periodically
        if self.sample_count >= self.adapt_every and len(self.buffer) >= 4:
            self._adapt()
            self.sample_count = 0
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output['velocity_pred']
    
    def _adapt(self):
        """Run adaptation step on buffer."""
        self.model.train()
        
        # Sample from buffer
        buffer_data = torch.cat(self.buffer, dim=0)
        indices = torch.randperm(len(buffer_data))[:64]
        batch = buffer_data[indices].to(next(self.model.parameters()).device)
        
        for _ in range(self.adapt_steps):
            self.optimizer.zero_grad()
            
            z_e = self.model.encode(batch)
            distances = torch.cdist(z_e, self.model.vq.embeddings)
            probs = F.softmax(-distances / 0.1, dim=-1)
            
            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            # Diversity: encourage using different codes
            avg_probs = probs.mean(0)
            diversity = torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            
            loss = entropy + 0.5 * diversity
            loss.backward()
            self.optimizer.step()


class SessionAdapter:
    """
    Adapts model to a new recording session.
    
    Uses a small calibration set with ground truth to fine-tune.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_calibration_samples: int = 100,
        adapt_epochs: int = 10,
        lr: float = 1e-4,
        freeze_vq: bool = True
    ):
        self.model = model
        self.n_calibration = n_calibration_samples
        self.adapt_epochs = adapt_epochs
        self.lr = lr
        self.freeze_vq = freeze_vq
    
    def adapt(
        self,
        calibration_loader: DataLoader,
        device: str = 'cuda'
    ) -> dict:
        """
        Adapt model using calibration data.
        
        Args:
            calibration_loader: DataLoader with 'window' and 'velocity' keys
            device: Device to use
        
        Returns:
            Dict with adaptation metrics
        """
        self.model = self.model.to(device)
        
        # Freeze VQ codebook
        if self.freeze_vq and hasattr(self.model, 'vq'):
            for p in self.model.vq.parameters():
                p.requires_grad = False
        
        # Optimize encoder and decoder
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        
        losses = []
        
        for epoch in range(self.adapt_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in calibration_loader:
                window = batch['window'].to(device)
                velocity = batch['velocity'].to(device)
                
                optimizer.zero_grad()
                output = self.model(window, velocity)
                loss = output['total_loss']
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(calibration_loader))
        
        # Evaluate on calibration data
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in calibration_loader:
                window = batch['window'].to(device)
                output = self.model(window)
                preds.append(output['velocity_pred'].cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        
        return {
            'final_loss': losses[-1],
            'calibration_r2': r2_score(targets, preds),
            'loss_history': losses
        }


def test_tta():
    """Quick test of TTA components."""
    from .model import ProgressiveVQVAE
    
    # Create model
    model = ProgressiveVQVAE(n_channels=142, window_size=10)
    model.use_vq = True
    model.vq._initialized = True
    
    # Test TTAWrapper
    tta = TTAWrapper(model)
    x = torch.randn(8, 10, 142)
    output = tta.adapt(x, n_steps=2)
    print(f"TTA output shape: {output['velocity_pred'].shape}")
    
    # Test OnlineTTA
    online = OnlineTTA(model)
    for _ in range(5):
        x = torch.randn(16, 10, 142)
        pred = online.predict(x)
        print(f"Online pred shape: {pred.shape}")
    
    print("TTA tests passed!")


if __name__ == '__main__':
    test_tta()
