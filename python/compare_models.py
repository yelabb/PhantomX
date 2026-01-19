"""
PhantomX Comparison Experiments

Compare all model variants:
1. Progressive VQ-VAE (baseline, R²=0.70)
2. Transformer VQ-VAE
3. Gumbel VQ-VAE
4. Transformer + Gumbel VQ-VAE
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phantomx.data import MCMazeDataset
from phantomx.tokenizer import SpikeTokenizer
from phantomx.model import ProgressiveVQVAE
from phantomx.models_extended import TransformerVQVAE, GumbelVQVAE
from phantomx.trainer import ProgressiveTrainer


# ============================================================
# Dataset
# ============================================================

class SlidingWindowDataset(Dataset):
    def __init__(self, spike_counts, velocities, window_size=10):
        n = len(spike_counts) - window_size + 1
        self.windows = np.stack([spike_counts[i:i+window_size] for i in range(n)])
        self.velocities = velocities[window_size-1:window_size-1+n]
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'window': torch.tensor(self.windows[idx], dtype=torch.float32),
            'velocity': torch.tensor(self.velocities[idx], dtype=torch.float32)
        }


# ============================================================
# Training Functions
# ============================================================

def train_progressive(model, train_loader, val_loader, device, pretrain_epochs=30, finetune_epochs=50):
    """Progressive training for EMA VQ models."""
    
    # Phase 1: Pre-train
    print("  Phase 1: Pre-training encoder...")
    model.use_vq = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(pretrain_epochs):
        model.train()
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            optimizer.zero_grad()
            output = model(window, velocity)
            output['total_loss'].backward()
            optimizer.step()
    
    # Phase 2: Init codebook
    print("  Phase 2: K-means initialization...")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in train_loader:
            window = batch['window'].to(device)
            z_e = model.encode(window)
            embeddings.append(z_e.cpu())
    z_all = torch.cat(embeddings, dim=0)
    model.vq.init_from_data(z_all)
    
    # Phase 3: Finetune
    print("  Phase 3: Finetuning with VQ...")
    model.use_vq = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
    
    best_r2 = -float('inf')
    best_state = None
    patience = 0
    
    for epoch in range(finetune_epochs):
        model.train()
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            optimizer.zero_grad()
            output = model(window, velocity)
            output['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                preds.append(output['velocity_pred'].cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        val_r2 = r2_score(np.concatenate(targets), np.concatenate(preds))
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if patience >= 15:
            break
    
    model.load_state_dict(best_state)
    return model


def train_gumbel(model, train_loader, val_loader, device, epochs=100):
    """End-to-end training for Gumbel VQ models."""
    
    print("  Training end-to-end with Gumbel-Softmax...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_r2 = -float('inf')
    best_state = None
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            optimizer.zero_grad()
            output = model(window, velocity)
            output['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                preds.append(output['velocity_pred'].cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        val_r2 = r2_score(np.concatenate(targets), np.concatenate(preds))
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 20 == 0:
            temp = model.vq.temperature.item() if hasattr(model.vq, 'temperature') else 0
            print(f"    Epoch {epoch+1}: val_r2={val_r2:.4f}, temp={temp:.3f}")
        
        if patience >= 20:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    preds, targets = [], []
    indices = []
    
    with torch.no_grad():
        for batch in test_loader:
            window = batch['window'].to(device)
            output = model(window)
            preds.append(output['velocity_pred'].cpu().numpy())
            targets.append(batch['velocity'].numpy())
            if 'indices' in output:
                indices.append(output['indices'].cpu().numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    result = {
        'r2': r2_score(targets, preds),
        'r2_vx': r2_score(targets[:, 0], preds[:, 0]),
        'r2_vy': r2_score(targets[:, 1], preds[:, 1]),
    }
    
    if indices:
        all_indices = np.concatenate(indices)
        result['n_codes'] = len(np.unique(all_indices))
    
    return result


# ============================================================
# Main
# ============================================================

def main():
    DATA_PATH = "c:/Users/guzzi/Desktop/Projects/DEV-ACTIF/NeuraLink/PhantomLink/data/raw/mc_maze.nwb"
    WINDOW_SIZE = 10
    
    print("=" * 70)
    print("PhantomX Model Comparison")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading MC_Maze data...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_dataset = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    dataset = SlidingWindowDataset(
        mc_dataset.spike_counts,
        mc_dataset.velocities,
        window_size=WINDOW_SIZE
    )
    
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_loader = DataLoader(Subset(dataset, range(n_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(n_train, n_train + n_val)), batch_size=128)
    test_loader = DataLoader(Subset(dataset, range(n_train + n_val, n)), batch_size=128)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")
    
    results = {}
    
    # ========================================
    # Model 1: Progressive VQ-VAE (baseline)
    # ========================================
    print("\n[2/5] Training Progressive VQ-VAE (baseline)...")
    model1 = ProgressiveVQVAE(n_channels=142, window_size=WINDOW_SIZE).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    start = time.time()
    model1 = train_progressive(model1, train_loader, val_loader, device)
    train_time1 = time.time() - start
    
    results['Progressive'] = evaluate(model1, test_loader, device)
    results['Progressive']['time'] = train_time1
    print(f"  Test R²: {results['Progressive']['r2']:.4f} ({train_time1:.1f}s)")
    
    # ========================================
    # Model 2: Transformer VQ-VAE
    # ========================================
    print("\n[3/5] Training Transformer VQ-VAE...")
    model2 = TransformerVQVAE(n_channels=142, window_size=WINDOW_SIZE, vq_type='ema').to(device)
    print(f"  Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    start = time.time()
    model2 = train_progressive(model2, train_loader, val_loader, device)
    train_time2 = time.time() - start
    
    results['Transformer'] = evaluate(model2, test_loader, device)
    results['Transformer']['time'] = train_time2
    print(f"  Test R²: {results['Transformer']['r2']:.4f} ({train_time2:.1f}s)")
    
    # ========================================
    # Model 3: Gumbel VQ-VAE
    # ========================================
    print("\n[4/5] Training Gumbel VQ-VAE...")
    model3 = GumbelVQVAE(n_channels=142, window_size=WINDOW_SIZE).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    
    start = time.time()
    model3 = train_gumbel(model3, train_loader, val_loader, device)
    train_time3 = time.time() - start
    
    results['Gumbel'] = evaluate(model3, test_loader, device)
    results['Gumbel']['time'] = train_time3
    print(f"  Test R²: {results['Gumbel']['r2']:.4f} ({train_time3:.1f}s)")
    
    # ========================================
    # Model 4: Transformer + Gumbel
    # ========================================
    print("\n[5/5] Training Transformer + Gumbel VQ-VAE...")
    model4 = TransformerVQVAE(n_channels=142, window_size=WINDOW_SIZE, vq_type='gumbel').to(device)
    print(f"  Parameters: {sum(p.numel() for p in model4.parameters()):,}")
    
    start = time.time()
    model4 = train_gumbel(model4, train_loader, val_loader, device)
    train_time4 = time.time() - start
    
    results['Transformer+Gumbel'] = evaluate(model4, test_loader, device)
    results['Transformer+Gumbel']['time'] = train_time4
    print(f"  Test R²: {results['Transformer+Gumbel']['r2']:.4f} ({train_time4:.1f}s)")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<22} | {'R²':>8} | {'R² vx':>8} | {'R² vy':>8} | {'Codes':>6} | {'Time':>8}")
    print("-" * 70)
    
    for name, res in results.items():
        codes = res.get('n_codes', 'N/A')
        codes_str = f"{codes}" if isinstance(codes, int) else codes
        print(f"{name:<22} | {res['r2']:>8.4f} | {res['r2_vx']:>8.4f} | {res['r2_vy']:>8.4f} | {codes_str:>6} | {res['time']:>7.1f}s")
    
    best_name = max(results, key=lambda k: results[k]['r2'])
    print("-" * 70)
    print(f"Best: {best_name} (R² = {results[best_name]['r2']:.4f})")
    print("=" * 70)
    
    # Save results
    import json
    results_json = {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv 
                        for kk, vv in v.items()} 
                    for k, v in results.items()}
    
    with open('models/comparison_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\nResults saved to models/comparison_results.json")


if __name__ == '__main__':
    main()
