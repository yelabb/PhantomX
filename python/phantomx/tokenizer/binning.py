"""
Population-level spike binning utilities
"""

import numpy as np
from typing import Tuple


class PopulationBinner:
    """
    Bins spike counts into time windows with population-level features.
    
    This is used before tokenization to aggregate spikes over time.
    """
    
    def __init__(self, bin_size_ms: float = 25.0, sampling_rate_hz: float = 40.0):
        """
        Initialize binner.
        
        Args:
            bin_size_ms: Time bin size in milliseconds
            sampling_rate_hz: Sampling rate of input data (Hz)
        """
        self.bin_size_ms = bin_size_ms
        self.sampling_rate_hz = sampling_rate_hz
        
        # Convert to samples
        self.bin_size_samples = int(bin_size_ms / 1000.0 * sampling_rate_hz)
    
    def bin_spikes(
        self,
        spike_times: np.ndarray,
        spike_channels: np.ndarray,
        n_channels: int,
        duration_s: float
    ) -> np.ndarray:
        """
        Bin spike times into discrete time windows.
        
        Args:
            spike_times: [n_spikes] spike timestamps (seconds)
            spike_channels: [n_spikes] spike channel IDs
            n_channels: Total number of channels
            duration_s: Total recording duration (seconds)
            
        Returns:
            spike_counts: [n_bins, n_channels] binned spike counts
        """
        n_bins = int(duration_s * 1000.0 / self.bin_size_ms)
        spike_counts = np.zeros((n_bins, n_channels), dtype=np.float32)
        
        # Assign spikes to bins
        bin_indices = (spike_times * 1000.0 / self.bin_size_ms).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Count spikes per bin per channel
        for bin_idx, channel_idx in zip(bin_indices, spike_channels):
            if 0 <= channel_idx < n_channels:
                spike_counts[bin_idx, channel_idx] += 1
        
        return spike_counts
    
    def compute_population_rate(self, spike_counts: np.ndarray) -> np.ndarray:
        """
        Compute population firing rate.
        
        Args:
            spike_counts: [n_bins, n_channels] spike counts
            
        Returns:
            population_rate: [n_bins] total spikes per bin
        """
        return np.sum(spike_counts, axis=1)
    
    def compute_participation_ratio(self, spike_counts: np.ndarray) -> np.ndarray:
        """
        Compute participation ratio (measure of population synchrony).
        
        PR = (Σ λᵢ)² / Σ λᵢ²
        where λᵢ are the eigenvalues of the spike count covariance matrix.
        
        High PR → many neurons participating
        Low PR → few neurons dominating
        
        Args:
            spike_counts: [n_bins, n_channels] spike counts
            
        Returns:
            pr: [n_bins] participation ratio per bin
        """
        # Compute covariance over channels
        cov = np.cov(spike_counts.T)  # [n_channels, n_channels]
        
        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
        
        # Participation ratio
        pr = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)
        
        return pr
