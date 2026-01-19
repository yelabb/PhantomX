"""
Zero-Shot Inference Example

Test pre-trained codebook on held-out trials.

Usage:
    python test_zero_shot.py --model_path checkpoints/best_model.pt --data_path ../PhantomLink/data/mc_maze.nwb
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from phantomx.inference import LabramDecoder
from phantomx.tokenizer import SpikeTokenizer
from phantomx.data import MCMazeDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Test zero-shot decoding")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to MC_Maze NWB file')
    parser.add_argument('--use_tta', action='store_true',
                        help='Enable test-time adaptation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on')
    
    return parser.parse_args()


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute decoding metrics"""
    # Mean squared error
    mse = np.mean((predictions - targets) ** 2)
    
    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Mean absolute error
    mae = np.mean(np.abs(predictions - targets))
    
    return {
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }


def main():
    args = parse_args()
    
    print("="*60)
    print("Zero-Shot Decoding Test")
    print("="*60)
    
    # Load decoder
    print("[1/3] Loading pre-trained decoder...")
    decoder = LabramDecoder.load(
        args.model_path,
        device=args.device,
        use_tta=args.use_tta
    )
    
    # Load test data
    print("[2/3] Loading test data...")
    data_loader = MCMazeDataLoader(
        data_path=args.data_path,
        tokenizer=decoder.tokenizer,
        batch_size=64
    )
    _, _, test_loader = data_loader.get_loaders()
    
    # Run inference
    print("[3/3] Running zero-shot inference...")
    all_predictions = []
    all_targets = []
    
    for batch in test_loader:
        spike_counts = batch['spike_counts'].numpy()
        targets = batch['kinematics'].numpy()
        
        # Decode
        predictions = decoder.decode_batch(spike_counts)
        
        all_predictions.append(predictions)
        all_targets.append(targets)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    print("\n" + "="*60)
    print("Results:")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    
    # Decoder statistics
    stats = decoder.get_statistics()
    print(f"\nDecoder Statistics:")
    print(f"  Inference count: {stats['inference_count']}")
    print(f"  Mean latency: {stats['mean_latency_ms']:.2f} ms")
    print(f"  Codebook size: {stats['num_codes']}")
    print("="*60)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Velocity X
    axes[0, 0].scatter(all_targets[:, 0], all_predictions[:, 0], alpha=0.3, s=1)
    axes[0, 0].plot([all_targets[:, 0].min(), all_targets[:, 0].max()],
                     [all_targets[:, 0].min(), all_targets[:, 0].max()], 'r--')
    axes[0, 0].set_xlabel('True vx')
    axes[0, 0].set_ylabel('Predicted vx')
    axes[0, 0].set_title('Velocity X')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity Y
    axes[0, 1].scatter(all_targets[:, 1], all_predictions[:, 1], alpha=0.3, s=1)
    axes[0, 1].plot([all_targets[:, 1].min(), all_targets[:, 1].max()],
                     [all_targets[:, 1].min(), all_targets[:, 1].max()], 'r--')
    axes[0, 1].set_xlabel('True vy')
    axes[0, 1].set_ylabel('Predicted vy')
    axes[0, 1].set_title('Velocity Y')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time series (first 1000 samples)
    n_plot = min(1000, len(all_predictions))
    axes[1, 0].plot(all_targets[:n_plot, 0], label='True', alpha=0.7)
    axes[1, 0].plot(all_predictions[:n_plot, 0], label='Predicted', alpha=0.7)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Velocity X')
    axes[1, 0].set_title('Time Series (vx)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(all_targets[:n_plot, 1], label='True', alpha=0.7)
    axes[1, 1].plot(all_predictions[:n_plot, 1], label='Predicted', alpha=0.7)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Velocity Y')
    axes[1, 1].set_title('Time Series (vy)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('zero_shot_results.png', dpi=150)
    print("\nPlot saved to: zero_shot_results.png")


if __name__ == '__main__':
    main()
