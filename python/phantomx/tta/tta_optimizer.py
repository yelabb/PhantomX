"""
TTA Optimizer Wrapper

Provides high-level interface for test-time adaptation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from .entropy_minimizer import EntropyMinimizer, OnlineRLSAdapter


class TTAOptimizer:
    """
    Unified interface for test-time adaptation strategies.
    
    Supports:
    - Entropy minimization (gradient-based)
    - RLS adaptation (closed-form)
    - Hybrid approach
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = 'entropy',
        learning_rate: float = 1e-4,
        forgetting_factor: float = 0.995,
        device: str = None
    ):
        """
        Initialize TTA optimizer.
        
        Args:
            model: VQ-VAE model to adapt
            strategy: 'entropy', 'rls', or 'hybrid'
            learning_rate: Learning rate for entropy minimization
            forgetting_factor: Forgetting factor for RLS
            device: Device to run on
        """
        self.model = model
        self.strategy = strategy
        
        # Initialize adaptation strategy
        if strategy == 'entropy':
            self.adapter = EntropyMinimizer(
                model=model,
                learning_rate=learning_rate,
                device=device
            )
        elif strategy == 'rls':
            self.adapter = OnlineRLSAdapter(
                model=model,
                forgetting_factor=forgetting_factor
            )
        elif strategy == 'hybrid':
            # Use both: entropy for decoder, RLS for final layer
            self.entropy_adapter = EntropyMinimizer(
                model=model,
                learning_rate=learning_rate,
                device=device
            )
            self.rls_adapter = OnlineRLSAdapter(
                model=model,
                forgetting_factor=forgetting_factor
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def adapt_and_decode(
        self,
        tokens: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        return_info: bool = False
    ) -> torch.Tensor:
        """
        Decode with adaptation.
        
        Args:
            tokens: [batch, n_tokens] discrete token IDs
            target: [batch, output_dim] optional ground truth (for supervised adaptation)
            return_info: If True, return adaptation statistics
            
        Returns:
            kinematics: [batch, output_dim] predicted kinematics
            info: (optional) Dictionary with adaptation statistics
        """
        if self.strategy == 'entropy':
            return self.adapter.adapt_and_decode(tokens, return_info)
        
        elif self.strategy == 'rls':
            # RLS requires ground truth for supervised update
            if target is None:
                # Fall back to model without adaptation
                with torch.no_grad():
                    output = self.model(tokens)
                    return output['kinematics_pred']
            
            # Get quantized latent
            with torch.no_grad():
                z_e = self.model.encoder(tokens)
                z_q, _ = self.model.quantizer(z_e)
            
            # RLS update and prediction
            prediction = self.adapter.update(z_q.squeeze(), target.squeeze())
            
            if return_info:
                return prediction.unsqueeze(0), {'strategy': 'rls'}
            return prediction.unsqueeze(0)
        
        elif self.strategy == 'hybrid':
            # First: Entropy minimization on decoder
            kinematics, info = self.entropy_adapter.adapt_and_decode(tokens, return_info=True)
            
            # Second: RLS fine-tuning (if target available)
            if target is not None:
                with torch.no_grad():
                    z_e = self.model.encoder(tokens)
                    z_q, _ = self.model.quantizer(z_e)
                kinematics = self.rls_adapter.update(z_q.squeeze(), target.squeeze())
                kinematics = kinematics.unsqueeze(0)
            
            if return_info:
                info['strategy'] = 'hybrid'
                return kinematics, info
            return kinematics
    
    def reset(self):
        """Reset adaptation state"""
        if hasattr(self.adapter, 'reset_statistics'):
            self.adapter.reset_statistics()
    
    def get_statistics(self) -> Dict:
        """Get adaptation statistics"""
        if hasattr(self.adapter, 'get_statistics'):
            return self.adapter.get_statistics()
        return {}
