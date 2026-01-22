#!/usr/bin/env python
"""
Evaluate a trained PhantomX model.

Usage:
    python evaluate.py checkpoint_path [--dataset mc_maze|mc_rtt]
    
Example:
    python evaluate.py logs/exp25_mamba/best_model.pt --dataset mc_rtt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from omegaconf import OmegaConf

from src.models import build_model
from src.datamodules import build_datamodule
from src.utils.metrics import compute_metrics
from src.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhantomX model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="mc_maze", choices=["mc_maze", "mc_rtt"])
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        cfg = OmegaConf.create(checkpoint["config"])
    else:
        # Load default configs
        cfg = OmegaConf.load("configs/config.yaml")
        model_cfg = OmegaConf.load(f"configs/model/vqvae.yaml")
        dataset_cfg = OmegaConf.load(f"configs/dataset/{args.dataset}.yaml")
        cfg = OmegaConf.merge(cfg, {"model": model_cfg, "dataset": dataset_cfg})
    
    # Override dataset if specified
    if args.dataset:
        dataset_cfg = OmegaConf.load(f"configs/dataset/{args.dataset}.yaml")
        cfg.dataset = OmegaConf.merge(cfg.dataset, dataset_cfg)
    
    seed_everything(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Device: {device}")
    
    # Load data
    print(f"\n📊 Loading {args.dataset} dataset...")
    datamodule = build_datamodule(cfg)
    datamodule.setup()
    
    if args.split == "test":
        dataloader = datamodule.test_dataloader()
    else:
        dataloader = datamodule.val_dataloader()
    
    # Build model
    print(f"\n🔧 Building model...")
    model = build_model(cfg, n_channels=datamodule.n_channels)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Evaluate
    print(f"\n🎯 Evaluating on {args.split} set...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for neural, target in dataloader:
            neural = neural.to(device)
            output, _ = model(neural)
            
            all_preds.append(output.cpu())
            all_targets.append(target)
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(preds, targets, prefix="")
    
    # Print results
    print("\n" + "=" * 40)
    print("Results")
    print("=" * 40)
    print(f"  R²:      {metrics['r2']:.4f}")
    print(f"  R² (x):  {metrics.get('r2_x', 'N/A'):.4f}" if 'r2_x' in metrics else "")
    print(f"  R² (y):  {metrics.get('r2_y', 'N/A'):.4f}" if 'r2_y' in metrics else "")
    print(f"  MSE:     {metrics['mse']:.6f}")
    print(f"  RMSE:    {metrics['rmse']:.6f}")
    print("=" * 40)
    
    return metrics["r2"]


if __name__ == "__main__":
    main()
