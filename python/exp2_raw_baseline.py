"""
Experiment 2: Raw Spikes Baseline

Test if MC_Maze data contains velocity information by using 
raw spike counts (no tokenization) with a simple MLP.

This establishes the CEILING R² we can achieve with full channel identity.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from phantomx.data import MCMazeDataset
from phantomx.tokenizer import SpikeTokenizer


class RawSpikeDataset(Dataset):
    """Dataset that returns raw spike counts (no tokenization)."""
    
    def __init__(self, mc_maze_dataset):
        self.spike_counts = mc_maze_dataset.spike_counts
        self.velocities = mc_maze_dataset.velocities
    
    def __len__(self):
        return len(self.spike_counts)
    
    def __getitem__(self, idx):
        return {
            'spikes': torch.tensor(self.spike_counts[idx], dtype=torch.float32),
            'velocity': torch.tensor(self.velocities[idx], dtype=torch.float32)
        }


class MLPDecoder(nn.Module):
    """Simple MLP for velocity decoding."""
    
    def __init__(self, input_dim=142, hidden_dims=[256, 128], output_dim=2, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def main():
    DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
    
    print("=" * 60)
    print("Experiment 2: Raw Spikes Baseline")
    print("=" * 60)
    
    # Load data using existing loader (for normalization)
    print("\n[1/4] Loading MC_Maze data...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_dataset = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    # Create raw dataset
    dataset = RawSpikeDataset(mc_dataset)
    
    # Split: 70% train, 15% val, 15% test
    n_samples = len(dataset)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_dataset = Subset(dataset, list(range(n_train)))
    val_dataset = Subset(dataset, list(range(n_train, n_train + n_val)))
    test_dataset = Subset(dataset, list(range(n_train + n_val, n_samples)))
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # ========================================
    # Method A: Ridge Regression Baseline
    # ========================================
    print("\n[2/4] Ridge Regression Baseline...")
    
    # Collect all data
    X_train = np.array([dataset[i]['spikes'].numpy() for i in range(n_train)])
    y_train = np.array([dataset[i]['velocity'].numpy() for i in range(n_train)])
    X_test = np.array([dataset[i]['spikes'].numpy() for i in range(n_train + n_val, n_samples)])
    y_test = np.array([dataset[i]['velocity'].numpy() for i in range(n_train + n_val, n_samples)])
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    r2_ridge_vx = r2_score(y_test[:, 0], y_pred_ridge[:, 0])
    r2_ridge_vy = r2_score(y_test[:, 1], y_pred_ridge[:, 1])
    r2_ridge = r2_score(y_test, y_pred_ridge)
    
    print(f"  Ridge R² (vx): {r2_ridge_vx:.4f}")
    print(f"  Ridge R² (vy): {r2_ridge_vy:.4f}")
    print(f"  Ridge R² (overall): {r2_ridge:.4f}")
    
    # ========================================
    # Method B: MLP Decoder
    # ========================================
    print("\n[3/4] Training MLP Decoder...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPDecoder(input_dim=142, hidden_dims=[256, 128], output_dim=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            spikes = batch['spikes'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            pred = model(spikes)
            loss = criterion(pred, velocity)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                spikes = batch['spikes'].to(device)
                velocity = batch['velocity'].to(device)
                pred = model(spikes)
                val_loss += criterion(pred, velocity).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    model.eval()
    
    print("\n[4/4] Evaluating MLP...")
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            spikes = batch['spikes'].to(device)
            velocity = batch['velocity']
            pred = model(spikes).cpu()
            all_preds.append(pred.numpy())
            all_targets.append(velocity.numpy())
    
    y_pred_mlp = np.concatenate(all_preds)
    y_test_mlp = np.concatenate(all_targets)
    
    r2_mlp_vx = r2_score(y_test_mlp[:, 0], y_pred_mlp[:, 0])
    r2_mlp_vy = r2_score(y_test_mlp[:, 1], y_pred_mlp[:, 1])
    r2_mlp = r2_score(y_test_mlp, y_pred_mlp)
    
    print(f"  MLP R² (vx): {r2_mlp_vx:.4f}")
    print(f"  MLP R² (vy): {r2_mlp_vy:.4f}")
    print(f"  MLP R² (overall): {r2_mlp:.4f}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 60)
    print(f"  Ridge Regression R²: {r2_ridge:.4f}")
    print(f"  MLP Decoder R²:      {r2_mlp:.4f}")
    print("=" * 60)
    
    best_r2 = max(r2_ridge, r2_mlp)
    if best_r2 >= 0.5:
        print(f"\n✓ Data contains strong velocity signal (R² = {best_r2:.4f})")
        print("  → POYO tokenization is losing this information")
    elif best_r2 >= 0.2:
        print(f"\n~ Data contains moderate velocity signal (R² = {best_r2:.4f})")
    else:
        print(f"\n⚠ Data has weak velocity signal (R² = {best_r2:.4f})")
        print("  → May need better preprocessing or temporal context")


if __name__ == '__main__':
    main()
