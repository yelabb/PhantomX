"""
FSQ Test-Time Adaptation (TTA) Optimizer

Standard entropy minimization fails for FSQ (scalars have no entropy).
Instead, we minimize quantization error to "snap" encoder predictions
to valid FSQ grid states.

Algorithm:
1. Forward: z_pred = Encoder(x)
2. Loss: L_TTA = mean((z_pred - round(z_pred))^2)
3. Update: Gradient step on Encoder weights

Effect: Reduces uncertainty by forcing encoder to output values
that lie exactly on the FSQ grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class FSQTestTimeAdapter:
    """
    Test-Time Adaptation for FSQ models.
    
    Minimizes quantization error to snap encoder outputs
    to the discrete FSQ grid.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        fsq_module: nn.Module,
        learning_rate: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        max_steps: int = 10,
        convergence_threshold: float = 1e-4,
        update_batch_norm: bool = False
    ):
        """
        Initialize FSQ TTA optimizer.
        
        Args:
            encoder: Encoder network to adapt
            fsq_module: FSQ quantization module
            learning_rate: Learning rate for adaptation
            momentum: SGD momentum
            weight_decay: L2 regularization
            max_steps: Maximum adaptation steps per sample
            convergence_threshold: Stop if loss change < threshold
            update_batch_norm: Whether to update BatchNorm statistics
        """
        self.encoder = encoder
        self.fsq_module = fsq_module
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.update_batch_norm = update_batch_norm
        
        # Create optimizer for encoder parameters
        self.optimizer = torch.optim.SGD(
            encoder.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Store original state for reset
        self._original_state = None
    
    def save_state(self) -> None:
        """Save encoder state before adaptation."""
        self._original_state = {
            'encoder': {k: v.clone() for k, v in self.encoder.state_dict().items()},
            'optimizer': self.optimizer.state_dict()
        }
    
    def restore_state(self) -> None:
        """Restore encoder to original state."""
        if self._original_state is not None:
            self.encoder.load_state_dict(self._original_state['encoder'])
            self.optimizer.load_state_dict(self._original_state['optimizer'])
    
    def compute_quant_error(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Compute quantization error loss.
        
        L_TTA = mean((z_pred - round(z_pred))^2)
        
        Args:
            z_e: [batch, d_model] encoder output
            
        Returns:
            quant_error: scalar quantization error loss
        """
        # Get pre-quantized continuous values
        z_proj = self.fsq_module.projection(z_e)  # [batch, n_dims]
        z_bounded = torch.tanh(z_proj)
        z_scaled = z_bounded * self.fsq_module._half_levels
        
        # Quantization error: distance to nearest grid point
        z_quantized = torch.round(z_scaled)
        quant_error = F.mse_loss(z_scaled, z_quantized)
        
        return quant_error
    
    def adapt(
        self,
        x: torch.Tensor,
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Perform test-time adaptation on a single sample/batch.
        
        Args:
            x: Input tensor (format depends on encoder)
            return_metrics: Whether to return adaptation metrics
            
        Returns:
            z_e: Adapted encoder output
            metrics: Optional dict with adaptation stats
        """
        self.save_state()
        
        # Set encoder to training mode for gradients
        was_training = self.encoder.training
        self.encoder.train()
        
        # Optionally freeze batch norm
        if not self.update_batch_norm:
            for module in self.encoder.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    module.eval()
        
        metrics = {'initial_error': None, 'final_error': None, 'steps': 0}
        prev_loss = float('inf')
        
        for step in range(self.max_steps):
            self.optimizer.zero_grad()
            
            # Forward through encoder
            z_e = self.encoder(x)
            
            # Compute quantization error
            loss = self.compute_quant_error(z_e)
            
            if step == 0:
                metrics['initial_error'] = loss.item()
            
            # Check convergence
            if abs(prev_loss - loss.item()) < self.convergence_threshold:
                break
            
            prev_loss = loss.item()
            
            # Backward and update
            loss.backward()
            self.optimizer.step()
            metrics['steps'] = step + 1
        
        # Final forward pass
        with torch.no_grad():
            z_e = self.encoder(x)
        
        metrics['final_error'] = self.compute_quant_error(z_e).item()
        metrics['error_reduction'] = metrics['initial_error'] - metrics['final_error']
        
        # Restore training mode
        if not was_training:
            self.encoder.eval()
        
        if return_metrics:
            return z_e, metrics
        return z_e, None
    
    def adapt_and_predict(
        self,
        x: torch.Tensor,
        decoder: nn.Module,
        fsq_module: nn.Module
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Adapt encoder and make prediction.
        
        Args:
            x: Input tensor
            decoder: Decoder network
            fsq_module: FSQ module for quantization
            
        Returns:
            prediction: Decoder output
            metrics: Adaptation metrics
        """
        z_e, metrics = self.adapt(x, return_metrics=True)
        
        # Quantize and decode
        z_q, fsq_info = fsq_module(z_e)
        prediction = decoder(z_q)
        
        metrics['perplexity'] = fsq_info['perplexity'].item()
        
        return prediction, metrics


class OnlineFSQAdapter:
    """
    Online FSQ adaptation with exponential moving average.
    
    Maintains running statistics and adapts gradually over time,
    suitable for streaming BCI applications.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        fsq_module: nn.Module,
        ema_decay: float = 0.99,
        adaptation_rate: float = 0.01
    ):
        """
        Initialize online adapter.
        
        Args:
            encoder: Encoder network
            fsq_module: FSQ quantization module
            ema_decay: EMA decay for running statistics
            adaptation_rate: Learning rate for online updates
        """
        self.encoder = encoder
        self.fsq_module = fsq_module
        self.ema_decay = ema_decay
        self.adaptation_rate = adaptation_rate
        
        # Running statistics
        self.running_quant_error = None
    
    def update(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Perform one online adaptation step.
        
        Args:
            x: Input tensor
            
        Returns:
            metrics: Dict with current statistics
        """
        self.encoder.train()
        
        # Forward
        z_e = self.encoder(x)
        
        # Compute quantization error
        z_proj = self.fsq_module.projection(z_e)
        z_bounded = torch.tanh(z_proj)
        z_scaled = z_bounded * self.fsq_module._half_levels
        z_quantized = torch.round(z_scaled)
        
        quant_error = F.mse_loss(z_scaled, z_quantized)
        
        # Update running statistics
        if self.running_quant_error is None:
            self.running_quant_error = quant_error.item()
        else:
            self.running_quant_error = (
                self.ema_decay * self.running_quant_error +
                (1 - self.ema_decay) * quant_error.item()
            )
        
        # Gradient step (small online update)
        quant_error.backward()
        
        with torch.no_grad():
            for param in self.encoder.parameters():
                if param.grad is not None:
                    param.data -= self.adaptation_rate * param.grad
                    param.grad.zero_()
        
        return {
            'quant_error': quant_error.item(),
            'running_quant_error': self.running_quant_error
        }
