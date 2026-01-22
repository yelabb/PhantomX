"""
Example: Using the PhantomX Dataset System

This shows how to use the unified dataset API for experiments.
"""

from phantomx.datasets import get_dataset, list_datasets
from phantomx.datasets.torch_dataset import create_dataloaders
import numpy as np


def main():
    # List available datasets
    print("=" * 60)
    print("Available Datasets")
    print("=" * 60)
    datasets = list_datasets(verbose=False)
    for name in datasets:
        ds = get_dataset(name)
        status = "✓" if ds.is_cached else "○"
        print(f"  {status} {name}")
    
    # Load MC_Maze dataset
    print("\n" + "=" * 60)
    print("Loading MC_Maze")
    print("=" * 60)
    
    dataset = get_dataset("mc_maze")
    print(dataset.summary())
    
    # Load the actual data (downloads if not cached)
    neural, targets = dataset.load()
    print(f"\nNeural data shape: {neural.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Neural stats: mean={neural.mean():.3f}, std={neural.std():.3f}")
    print(f"Velocity stats: mean={targets.mean(axis=0)}, std={targets.std(axis=0)}")
    
    # Get train/val/test splits
    print("\n" + "=" * 60)
    print("Creating DataLoaders")
    print("=" * 60)
    
    loaders = create_dataloaders(
        dataset,
        window_size=10,
        batch_size=64,
        train_ratio=0.7,
        val_ratio=0.15,
        temporal_split=True  # Prevents data leakage
    )
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # Sample a batch
    batch = next(iter(loaders['train']))
    X, y = batch
    print(f"\nBatch shapes: X={X.shape}, y={y.shape}")
    # X: [batch, window_size, n_channels]
    # y: [batch, n_targets]
    
    print("\n✓ Dataset system working correctly!")


if __name__ == "__main__":
    main()
