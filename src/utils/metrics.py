"""
Evaluation Metrics

Standardized metrics for neural decoding evaluation.
Primary metric: R² (coefficient of determination)
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    prefix: str = "",
    per_dim: bool = True,
) -> Dict[str, float]:
    """
    Compute standard evaluation metrics.
    
    Args:
        predictions: Model predictions [N, D] or [N,]
        targets: Ground truth [N, D] or [N,]
        prefix: Prefix for metric names (e.g., "val/", "test/")
        per_dim: Compute per-dimension metrics for multi-dim outputs
        
    Returns:
        Dictionary of metrics
        
    Example:
        >>> preds = model(neural_data)
        >>> metrics = compute_metrics(preds, velocities, prefix="val/")
        >>> print(metrics)
        {'val/r2': 0.71, 'val/r2_x': 0.72, 'val/r2_y': 0.70, ...}
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Flatten if 1D
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    
    metrics = {}
    
    # Overall R² (macro average)
    r2_overall = r2_score(targets, predictions, multioutput='uniform_average')
    metrics[f"{prefix}r2"] = float(r2_overall)
    
    # Overall MSE and MAE
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    metrics[f"{prefix}mse"] = float(mse)
    metrics[f"{prefix}mae"] = float(mae)
    metrics[f"{prefix}rmse"] = float(np.sqrt(mse))
    
    # Per-dimension metrics (for velocity: x, y)
    if per_dim and predictions.shape[1] > 1:
        dim_names = ['x', 'y', 'z', 'w'][:predictions.shape[1]]
        
        for i, name in enumerate(dim_names):
            r2_dim = r2_score(targets[:, i], predictions[:, i])
            mse_dim = mean_squared_error(targets[:, i], predictions[:, i])
            
            metrics[f"{prefix}r2_{name}"] = float(r2_dim)
            metrics[f"{prefix}mse_{name}"] = float(mse_dim)
    
    # Variance explained (alternative to R²)
    var_explained = 1 - np.var(targets - predictions) / np.var(targets)
    metrics[f"{prefix}var_explained"] = float(var_explained)
    
    return metrics


def compute_r2(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Simple R² computation (convenience function).
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        
    Returns:
        R² score (float)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    return float(r2_score(targets.flatten(), predictions.flatten()))


def compute_bits_per_spike(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    spike_counts: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute bits per spike metric (NLB standard).
    
    Args:
        predictions: Predicted spike rates (log-rates)
        targets: Actual spike counts
        spike_counts: Total spike counts for normalization
        
    Returns:
        Bits per spike
    """
    # This is a placeholder - implement based on NLB evaluation code
    raise NotImplementedError("Bits per spike not yet implemented")


class MetricTracker:
    """
    Track metrics over training epochs.
    
    Useful for early stopping and best model selection.
    """
    
    def __init__(self, metric_name: str = "r2", mode: str = "max"):
        """
        Args:
            metric_name: Name of metric to track
            mode: "max" for metrics like R², "min" for loss
        """
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = -float('inf') if mode == "max" else float('inf')
        self.best_epoch = 0
        self.history = []
    
    def update(self, value: float, epoch: int) -> bool:
        """
        Update tracker with new value.
        
        Returns:
            True if this is a new best value
        """
        self.history.append({"epoch": epoch, "value": value})
        
        is_best = False
        if self.mode == "max":
            if value > self.best_value:
                self.best_value = value
                self.best_epoch = epoch
                is_best = True
        else:
            if value < self.best_value:
                self.best_value = value
                self.best_epoch = epoch
                is_best = True
        
        return is_best
    
    def should_stop(self, patience: int) -> bool:
        """Check if training should stop based on patience."""
        if len(self.history) < patience:
            return False
        return self.history[-1]["epoch"] - self.best_epoch >= patience
