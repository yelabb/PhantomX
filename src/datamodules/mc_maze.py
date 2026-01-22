"""
MC_Maze DataModule

Motor cortex recordings during maze reaching task.
Discrete reaching movements with pauses.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from .base import BaseDataModule
from . import register_datamodule


@register_datamodule('mc_maze')
class MCMazeDataModule(BaseDataModule):
    """
    MC_Maze DataModule.
    
    Motor cortex recordings during a delayed center-out reaching task.
    137 sorted units, discrete trials.
    """
    
    @property
    def name(self) -> str:
        return "mc_maze"
    
    def _get_expected_n_channels(self) -> int:
        """Expected channel count (actual may differ based on data file)."""
        return 142  # Can vary; actual determined from data
    
    @property
    def output_dim(self) -> int:
        return 2  # Velocity (vx, vy)
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load MC_Maze data from NWB file."""
        filepath = self.data_dir / "mc_maze.nwb"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"MC_Maze data not found at {filepath}. "
                "Download from DANDI Archive or run `python upload_data.py`"
            )
        
        try:
            from pynwb import NWBHDF5IO
        except ImportError:
            raise ImportError("PyNWB required. Install with: pip install pynwb")
        
        print(f"Loading MC_Maze from: {filepath}")
        
        bin_size_ms = 25.0  # 40 Hz
        
        with NWBHDF5IO(str(filepath), mode='r', load_namespaces=True) as io:
            nwb = io.read()
            
            # Get neural units
            units = nwb.units
            n_units = len(units.id[:])
            
            # Get behavior
            behavior = nwb.processing.get('behavior')
            hand_vel = behavior.data_interfaces.get('hand_vel')
            cursor_pos = behavior.data_interfaces.get('cursor_pos')
            
            timestamps = cursor_pos.timestamps[:]
            velocity = hand_vel.data[:].astype(np.float32)
            
            # Bin spikes
            duration = timestamps[-1] - timestamps[0]
            n_bins = int(duration * 1000 / bin_size_ms)
            
            spike_counts = np.zeros((n_bins, n_units), dtype=np.float32)
            
            for unit_idx in range(n_units):
                spike_times = units.get_unit_spike_times(unit_idx)
                if spike_times is not None and len(spike_times) > 0:
                    bin_indices = ((spike_times - timestamps[0]) * 1000 / bin_size_ms).astype(np.int32)
                    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                    for idx in bin_indices:
                        spike_counts[idx, unit_idx] += 1
            
            # Bin velocities
            bin_samples = int(bin_size_ms)
            rate = 1000  # 1000 Hz behavior data
            binned_vel = np.zeros((n_bins, 2), dtype=np.float32)
            
            for i in range(n_bins):
                start_time = timestamps[0] + i * bin_size_ms / 1000
                end_time = start_time + bin_size_ms / 1000
                mask = (timestamps >= start_time) & (timestamps < end_time)
                if mask.any():
                    binned_vel[i] = velocity[mask].mean(axis=0)
        
        print(f"  Loaded: {spike_counts.shape[0]} bins, {spike_counts.shape[1]} channels")
        
        return spike_counts, binned_vel
