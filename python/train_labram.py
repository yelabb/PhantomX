"""
Training Script for LaBraM VQ-VAE

Train universal neural codebook on MC_Maze dataset.

Usage:
    python train_labram.py --data_path ../PhantomLink/data/mc_maze.nwb --epochs 100
"""

import argparse
import torch
from pathlib import Path

from phantomx.tokenizer import SpikeTokenizer
from phantomx.vqvae import VQVAE, VQVAETrainer
from phantomx.data import MCMazeDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train LaBraM VQ-VAE")
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to MC_Maze NWB file')
    
    # Model architecture
    parser.add_argument('--n_tokens', type=int, default=16,
                        help='Number of input tokens')
    parser.add_argument('--token_dim', type=int, default=256,
                        help='Token vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Latent embedding dimension')
    parser.add_argument('--num_codes', type=int, default=256,
                        help='Number of codebook entries')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                        help='Commitment loss weight (beta)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Data splits
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Training data fraction')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation data fraction')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Hardware
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda/cpu/None for auto)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("LaBraM VQ-VAE Training")
    print("="*60)
    print(f"Data: {args.data_path}")
    print(f"Codebook size: {args.num_codes}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    # Initialize tokenizer
    print("\n[1/5] Initializing tokenizer...")
    tokenizer = SpikeTokenizer(
        n_channels=142,
        quantization_levels=args.n_tokens,
        use_population_norm=True,
        dropout_invariant=True
    )
    
    # Create data loaders
    print("[2/5] Loading MC_Maze dataset...")
    data_loader = MCMazeDataLoader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    
    print(f"  Train samples: {len(data_loader.train_dataset)}")
    print(f"  Val samples: {len(data_loader.val_dataset)}")
    print(f"  Test samples: {len(data_loader.test_dataset)}")
    
    # Create model
    print("[3/5] Creating VQ-VAE model...")
    model = VQVAE(
        n_tokens=args.n_tokens,
        token_dim=args.token_dim,
        embedding_dim=args.embedding_dim,
        num_codes=args.num_codes,
        commitment_cost=args.commitment_cost,
        output_dim=2,  # (vx, vy)
        use_transformer_encoder=False,  # Start with MLP
        use_recurrent_decoder=False
    )
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("[4/5] Initializing trainer...")
    trainer = VQVAETrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device
    )
    
    # Train
    print("[5/5] Training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=args.save_every
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Final perplexity: {history['perplexity'][-1]:.1f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
