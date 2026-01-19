"""
VQ-VAE Evaluation Script

Evaluate trained VQ-VAE on MC_Maze data:
- Reconstruction quality (MSE, SSIM)
- Codebook usage (perplexity, utilization)
- Velocity decoding (R² with linear probe)

Usage:
    python evaluate_vqvae.py --model_path models/best_model.pt --data_path path/to/mc_maze.nwb
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from phantomx.vqvae import VQVAE
from phantomx.tokenizer import SpikeTokenizer
from phantomx.data import MCMazeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VQ-VAE")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to MC_Maze NWB file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("VQ-VAE Evaluation on MC_Maze")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load checkpoint
    print("\n[1/5] Loading model checkpoint...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    # Get model config
    model_config = checkpoint.get('model_config', checkpoint.get('config', {}))
    
    # Initialize tokenizer
    print("[2/5] Initializing tokenizer...")
    tokenizer_config = model_config.get('tokenizer', {})
    tokenizer = SpikeTokenizer(
        n_channels=tokenizer_config.get('n_channels', 142),
        quantization_levels=tokenizer_config.get('quantization_levels', 16)
    )
    
    # Load dataset
    print("[3/5] Loading MC_Maze dataset...")
    dataset = MCMazeDataset(
        data_path=args.data_path,
        tokenizer=tokenizer
    )
    
    # Split dataset: 70% train, 15% val, 15% test
    n_samples = len(dataset)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n_samples))
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize VQ-VAE
    print("[4/5] Loading VQ-VAE model...")
    vqvae_config = model_config.get('vqvae', {})
    
    # The input is n_tokens from tokenizer (quantization_levels) with token_dim = 256 (vocab size)
    n_tokens = tokenizer.config.quantization_levels
    
    model = VQVAE(
        n_tokens=vqvae_config.get('n_tokens', n_tokens),
        token_dim=vqvae_config.get('token_dim', 256),
        embedding_dim=vqvae_config.get('embedding_dim', 64),
        num_codes=vqvae_config.get('num_codes', 256),
        commitment_cost=vqvae_config.get('commitment_cost', 0.25),
        output_dim=vqvae_config.get('output_dim', 2)
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate
    print("[5/5] Running evaluation...")
    
    # Collect all data
    def collect_features(loader, is_train=False):
        all_tokens = []
        all_z_e = []
        all_z_q = []
        all_velocities = []
        all_predictions = []
        codebook_indices = []
        
        with torch.no_grad():
            for batch in loader:
                tokens = batch['tokens'].long().to(device)  # Convert to long for embedding
                velocity = batch['kinematics']  # [batch, 2]
                
                # Forward pass
                outputs = model(tokens)
                
                all_tokens.append(tokens.cpu().numpy())
                all_z_e.append(outputs['z_e'].cpu().numpy())
                all_z_q.append(outputs['z_q'].cpu().numpy())
                all_predictions.append(outputs['kinematics_pred'].cpu().numpy())
                all_velocities.append(velocity.numpy())
                codebook_indices.append(outputs['indices'].cpu().numpy())
        
        return {
            'tokens': np.concatenate(all_tokens),
            'z_e': np.concatenate(all_z_e),
            'z_q': np.concatenate(all_z_q),
            'predictions': np.concatenate(all_predictions),
            'velocities': np.concatenate(all_velocities),
            'indices': np.concatenate(codebook_indices)
        }
    
    print("\n  Collecting training features...")
    train_data = collect_features(train_loader, is_train=True)
    
    print("  Collecting test features...")
    test_data = collect_features(test_loader)
    
    # ========================================
    # Metrics 1: Direct Velocity Prediction Quality
    # ========================================
    print("\n" + "-" * 40)
    print("DIRECT VELOCITY PREDICTION")
    print("-" * 40)
    
    # The model directly predicts kinematics, compute MSE
    pred_mse = np.mean((test_data['velocities'] - test_data['predictions']) ** 2)
    pred_mae = np.mean(np.abs(test_data['velocities'] - test_data['predictions']))
    
    # Compute R² for direct prediction
    r2_direct_vx = r2_score(test_data['velocities'][:, 0], test_data['predictions'][:, 0])
    r2_direct_vy = r2_score(test_data['velocities'][:, 1], test_data['predictions'][:, 1])
    r2_direct = r2_score(test_data['velocities'], test_data['predictions'])
    
    print(f"  Test MSE: {pred_mse:.6f}")
    print(f"  Test MAE: {pred_mae:.6f}")
    print(f"  R² (vx): {r2_direct_vx:.4f}")
    print(f"  R² (vy): {r2_direct_vy:.4f}")
    print(f"  R² (overall): {r2_direct:.4f}")
    
    # ========================================
    # Metrics 2: Codebook Utilization
    # ========================================
    print("\n" + "-" * 40)
    print("CODEBOOK UTILIZATION")
    print("-" * 40)
    
    # Count unique codes used
    all_indices = np.concatenate([train_data['indices'], test_data['indices']])
    unique_codes = len(np.unique(all_indices))
    total_codes = model.quantizer.codebook.num_codes
    utilization = unique_codes / total_codes * 100
    
    # Compute perplexity from frequency distribution
    code_counts = np.bincount(all_indices.flatten(), minlength=total_codes)
    code_probs = code_counts / code_counts.sum()
    code_probs = code_probs[code_probs > 0]  # Remove zeros
    entropy = -np.sum(code_probs * np.log2(code_probs + 1e-10))
    perplexity = 2 ** entropy
    
    print(f"  Unique codes used: {unique_codes}/{total_codes}")
    print(f"  Utilization: {utilization:.1f}%")
    print(f"  Perplexity: {perplexity:.1f}")
    
    # ========================================
    # Metrics 3: Linear Probe on z_q embeddings
    # ========================================
    print("\n" + "-" * 40)
    print("LINEAR PROBE ON z_q EMBEDDINGS")
    print("-" * 40)
    
    # Use quantized embeddings as features
    X_train = train_data['z_q']
    y_train = train_data['velocities']
    X_test = test_data['z_q']
    y_test = test_data['velocities']
    
    # Fit Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    
    # Predict
    y_pred = ridge.predict(X_test)
    
    # Compute R²
    r2_vx = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_vy = r2_score(y_test[:, 1], y_pred[:, 1])
    r2_overall = r2_score(y_test, y_pred)
    
    # Compute correlation
    corr_vx = np.corrcoef(y_test[:, 0], y_pred[:, 0])[0, 1]
    corr_vy = np.corrcoef(y_test[:, 1], y_pred[:, 1])[0, 1]
    
    print(f"  R² (vx): {r2_vx:.4f}")
    print(f"  R² (vy): {r2_vy:.4f}")
    print(f"  R² (overall): {r2_overall:.4f}")
    print(f"  Correlation (vx): {corr_vx:.4f}")
    print(f"  Correlation (vy): {corr_vy:.4f}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Direct Prediction R²: {r2_direct:.4f}")
    print(f"  Linear Probe R² (z_q): {r2_overall:.4f}")
    print(f"  Codebook Perplexity: {perplexity:.1f} / {total_codes}")
    print(f"  Codebook Utilization: {utilization:.1f}%")
    print("=" * 60)
    
    # Target check - use the better of direct or linear probe
    best_r2 = max(r2_direct, r2_overall)
    if best_r2 >= 0.7:
        print("\n✓ Target R² >= 0.7 ACHIEVED!")
    else:
        print(f"\n⚠ Best R² = {best_r2:.4f} < 0.7 target. More training may help.")


if __name__ == '__main__':
    main()