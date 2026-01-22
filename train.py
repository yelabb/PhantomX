#!/usr/bin/env python
"""
PhantomX Unified Training Entry Point

This is the SINGLE entry point for all experiments.
Replaces the 25+ exp*.py scripts with configuration-driven training.

Usage:
    # Default training (VQ-VAE on MC_Maze)
    python train.py

    # Specific experiment
    python train.py experiment=exp25_mamba

    # Override model/dataset
    python train.py model=mamba dataset=mc_rtt

    # Override hyperparameters
    python train.py model=lstm model.hidden_dim=512 trainer.learning_rate=1e-4

    # Multi-seed validation
    python train.py experiment=exp23_validation --multirun seed=42,123,456

    # Sweep
    python train.py --multirun model=vqvae,lstm,mamba dataset=mc_maze

Examples:
    # Replicate Exp 25: Mamba on MC_RTT
    python train.py experiment=exp25_mamba

    # Replicate Exp 22c: Multi-seed teacher
    python train.py experiment=exp22c_teacher

    # Quick test
    python train.py trainer.max_epochs=5 logging.use_wandb=false
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf

# Import after hydra to avoid issues
import torch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Best validation R² (for Hydra sweeps)
    """
    # Import here to avoid circular imports
    from src.utils.seeding import seed_everything, get_git_hash
    from src.utils.logging import init_logger, finish_logger
    from src.models import build_model
    from src.datamodules import build_datamodule
    from src.trainer import Trainer, ProgressiveTrainer
    
    # Print config
    print("=" * 60)
    print("PhantomX Training")
    print("=" * 60)
    print(f"Experiment: {cfg.get('experiment_name', 'default')}")
    print(f"Model: {cfg.model.name}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)
    
    # Seed everything for reproducibility
    seed_everything(
        seed=cfg.seed,
        deterministic=cfg.reproducibility.get('deterministic', True)
    )
    
    # Log git hash
    git_hash = get_git_hash()
    if git_hash:
        print(f"Git commit: {git_hash}")
    
    # Initialize logger
    logger = init_logger(cfg)
    
    try:
        # Build datamodule
        print("\n📊 Loading data...")
        datamodule = build_datamodule(cfg)
        datamodule.setup()
        
        # Get window size from model config
        window_size = cfg.model.get('window_size', 10)
        
        # Update datamodule window size if needed
        if hasattr(datamodule, 'window_size') and datamodule.window_size != window_size:
            print(f"   Updating window size: {datamodule.window_size} → {window_size}")
            datamodule.window_size = window_size
            datamodule.setup()  # Recreate datasets
        
        # Build model
        print("\n🔧 Building model...")
        model = build_model(cfg, n_channels=datamodule.n_channels)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parameters: {n_params:,}")
        
        # Create trainer
        print("\n🏋️ Setting up trainer...")
        trainer_name = cfg.trainer.get('name', 'default')
        
        if trainer_name == 'progressive' and hasattr(model, 'use_vq'):
            trainer = ProgressiveTrainer(
                model=model,
                train_loader=datamodule.train_dataloader(),
                val_loader=datamodule.val_dataloader(),
                test_loader=datamodule.test_dataloader(),
                cfg=cfg,
            )
        else:
            trainer = Trainer(
                model=model,
                train_loader=datamodule.train_dataloader(),
                val_loader=datamodule.val_dataloader(),
                test_loader=datamodule.test_dataloader(),
                cfg=cfg,
            )
        
        # Train
        results = trainer.train()
        
        # Log summary
        logger.log_summary({
            'best_r2': results['best_r2'],
            'best_epoch': results['best_epoch'],
            'training_time_s': results['training_time_s'],
            'n_parameters': n_params,
            'git_hash': git_hash,
        })
        
        # Save final model
        model_path = Path(cfg.output_dir) / 'final_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': OmegaConf.to_container(cfg),
            'results': results,
        }, model_path)
        logger.save_model(model_path)
        
        print(f"\n📁 Results saved to: {cfg.output_dir}")
        
        return results['best_r2']
    
    finally:
        finish_logger()


if __name__ == "__main__":
    main()
