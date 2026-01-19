"""
Test-Time Entropy Minimization

Implements unsupervised adaptation for handling signal drift.

Reference: "Test-Time Entropy Minimization for Non-Stationary Brain Signals" (ICML 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class EntropyMinimizer:
    """
    Test-time adaptation via entropy minimization.
    
    During runtime (without ground truth labels), we minimize the entropy
    of the model's predictions to adapt to distribution shift.
    
    Intuition: Confident predictions → low entropy
    Drift causes uncertain predictions → high entropy
    Minimizing entropy → adapt to new distribution
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        momentum: float = 0.9,
        temperature: float = 1.0,
        adaptation_steps: int = 1,
        device: str = None
    ):
        """
        Initialize entropy minimizer.
        
        Args:
            model: VQ-VAE model to adapt
            learning_rate: Learning rate for adaptation
            momentum: Momentum for optimizer
            temperature: Temperature for entropy calculation
            adaptation_steps: Number of gradient steps per sample
            device: Device to run on
        """
        self.model = model
        self.temperature = temperature
        self.adaptation_steps = adaptation_steps
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Only update decoder parameters (keep encoder/codebook frozen)
        # This prevents catastrophic forgetting of the universal codebook
        self.params_to_adapt = list(model.decoder.parameters())
        
        # Optimizer for test-time adaptation
        self.optimizer = torch.optim.SGD(
            self.params_to_adapt,
            lr=learning_rate,
            momentum=momentum
        )
        
        # Statistics for monitoring
        self.entropy_history = []
        self.prediction_history = []
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction entropy.
        
        Args:
            logits: [batch, n_classes] unnormalized predictions
            
        Returns:
            entropy: Scalar entropy value
        """
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Entropy: H = -Σ p(x) log p(x)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        return torch.mean(entropy)
    
    def adapt_and_decode(
        self,
        tokens: torch.Tensor,
        return_info: bool = False
    ) -> torch.Tensor:
        """
        Decode with test-time adaptation.
        
        Args:
            tokens: [batch, n_tokens] or [n_tokens] discrete token IDs
            return_info: If True, return adaptation statistics
            
        Returns:
            kinematics: [batch, output_dim] predicted kinematics
            info: (optional) Dictionary with adaptation statistics
        """
        # Handle single sample
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        
        tokens = tokens.to(self.device)
        
        # Initial forward pass (without adaptation)
        self.model.eval()
        with torch.no_grad():
            output_initial = self.model(tokens)
            kinematics_initial = output_initial['kinematics_pred']
        
        # Test-time adaptation
        for step in range(self.adaptation_steps):
            self.model.train()  # Enable dropout for regularization
            
            # Forward pass
            output = self.model(tokens)
            kinematics = output['kinematics_pred']
            
            # Compute entropy loss
            # For regression, we can use prediction variance as a proxy for entropy
            entropy_loss = self._compute_regression_entropy(kinematics)
            
            # Backward pass (update only decoder)
            self.optimizer.zero_grad()
            entropy_loss.backward()
            self.optimizer.step()
        
        # Final forward pass (evaluation mode)
        self.model.eval()
        with torch.no_grad():
            output_final = self.model(tokens)
            kinematics_final = output_final['kinematics_pred']
        
        # Track statistics
        self.entropy_history.append(entropy_loss.item())
        self.prediction_history.append(kinematics_final.cpu().numpy())
        
        if return_info:
            info = {
                'entropy_loss': entropy_loss.item(),
                'kinematics_initial': kinematics_initial,
                'kinematics_adapted': kinematics_final,
                'improvement': torch.mean(torch.abs(kinematics_final - kinematics_initial)).item()
            }
            return kinematics_final, info
        
        return kinematics_final
    
    def _compute_regression_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy-like objective for regression.
        
        For continuous outputs, we encourage:
        1. Low prediction variance (confidence)
        2. Temporal smoothness
        3. Consistency with prior predictions
        
        Args:
            predictions: [batch, output_dim] predicted kinematics
            
        Returns:
            loss: Scalar loss value
        """
        # 1. Variance penalty (encourage confident predictions)
        variance = torch.var(predictions, dim=0).mean()
        
        # 2. Temporal smoothness (if we have history)
        smoothness_loss = 0.0
        if len(self.prediction_history) > 0:
            prev_pred = torch.from_numpy(self.prediction_history[-1]).to(predictions.device)
            smoothness_loss = F.mse_loss(predictions, prev_pred)
        
        # 3. Magnitude regularization (prevent extreme predictions)
        magnitude_loss = torch.mean(predictions ** 2)
        
        # Combine losses
        loss = variance + 0.1 * smoothness_loss + 0.01 * magnitude_loss
        
        return loss
    
    def reset_statistics(self):
        """Reset tracking statistics"""
        self.entropy_history = []
        self.prediction_history = []
    
    def get_statistics(self) -> Dict:
        """Get adaptation statistics"""
        return {
            'mean_entropy': np.mean(self.entropy_history) if self.entropy_history else 0.0,
            'std_entropy': np.std(self.entropy_history) if self.entropy_history else 0.0,
            'n_samples_adapted': len(self.entropy_history)
        }


class OnlineRLSAdapter:
    """
    Alternative: Recursive Least Squares (RLS) adaptation.
    
    This is similar to PhantomCore's adaptive_decoder.cpp but works
    with discrete tokens.
    """
    
    def __init__(
        self,
        model: nn.Module,
        forgetting_factor: float = 0.995,
        regularization: float = 1.0
    ):
        """
        Initialize RLS adapter.
        
        Args:
            model: VQ-VAE model
            forgetting_factor: λ ∈ [0.99, 0.999] (higher = slower forgetting)
            regularization: Initial covariance matrix scale
        """
        self.model = model
        self.forgetting_factor = forgetting_factor
        
        # Initialize covariance matrix P
        # P represents uncertainty in parameter estimates
        embedding_dim = model.embedding_dim
        output_dim = model.output_dim
        
        self.P = torch.eye(embedding_dim) * regularization  # [D, D]
        self.W = torch.zeros(output_dim, embedding_dim)  # [2, D] for (vx, vy)
        
    def update(
        self,
        z_q: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        RLS update with new observation.
        
        Args:
            z_q: [embedding_dim] quantized latent vector
            target: [output_dim] true kinematics
            
        Returns:
            prediction: [output_dim] predicted kinematics
        """
        # Prediction
        prediction = self.W @ z_q  # [2]
        
        # Error
        error = target - prediction  # [2]
        
        # RLS update
        Pz = self.P @ z_q  # [D]
        lambda_val = self.forgetting_factor
        denom = lambda_val + z_q @ Pz  # scalar
        
        # Kalman gain
        K = Pz / denom  # [D]
        
        # Update weights
        self.W = self.W + error.unsqueeze(1) @ K.unsqueeze(0)  # [2, D]
        
        # Update covariance (Woodbury matrix identity)
        self.P = (self.P - torch.outer(K, Pz)) / lambda_val
        
        return prediction
