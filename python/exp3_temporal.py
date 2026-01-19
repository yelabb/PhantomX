"""
Experiment 3: Temporal Context

Test different temporal window sizes for velocity decoding.
Hypothesis: Motor cortex needs temporal history for velocity encoding.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phantomx.data import MCMazeDataset
from phantomx.tokenizer import SpikeTokenizer


class TemporalSpikeDataset(Dataset):
    """Dataset with sliding window of spike history."""
    
    def __init__(self, spike_counts, velocities, window_size=10):
        self.spike_counts = spike_counts
        self.velocities = velocities
        self.window_size = window_size
        self.n_valid = len(spike_counts) - window_size + 1
    
    def __len__(self):
        return self.n_valid
    
    def __getitem__(self, idx):
        # Get window of spike counts ending at idx + window_size - 1
        window = self.spike_counts[idx:idx + self.window_size]  # [T, 142]
        velocity = self.velocities[idx + self.window_size - 1]  # Current velocity
        
        return {
            'spikes': torch.tensor(window, dtype=torch.float32),
            'velocity': torch.tensor(velocity, dtype=torch.float32)
        }


class TemporalMLP(nn.Module):
    """MLP that takes flattened temporal window."""
    
    def __init__(self, n_channels=142, window_size=10, hidden_dims=[512, 256], output_dim=2):
        super().__init__()
        input_dim = n_channels * window_size
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [batch, window, channels]
        x_flat = x.view(x.size(0), -1)  # [batch, window * channels]
        return self.network(x_flat)


class TemporalLSTM(nn.Module):
    """LSTM for temporal velocity decoding."""
    
    def __init__(self, n_channels=142, hidden_dim=128, n_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_dim, n_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: [batch, window, channels]
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use last hidden state


def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs=50):
    """Train model and return test R²."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            spikes = batch['spikes'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            pred = model(spikes)
            loss = criterion(pred, velocity)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                spikes = batch['spikes'].to(device)
                velocity = batch['velocity'].to(device)
                val_loss += criterion(model(spikes), velocity).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break
    
    # Load best and evaluate
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            spikes = batch['spikes'].to(device)
            all_preds.append(model(spikes).cpu().numpy())
            all_targets.append(batch['velocity'].numpy())
    
    y_pred = np.concatenate(all_preds)
    y_test = np.concatenate(all_targets)
    
    return {
        'r2_vx': r2_score(y_test[:, 0], y_pred[:, 0]),
        'r2_vy': r2_score(y_test[:, 1], y_pred[:, 1]),
        'r2_overall': r2_score(y_test, y_pred)
    }


def main():
    DATA_PATH = "c:/Users/guzzi/Desktop/Projects/DEV-ACTIF/NeuraLink/PhantomLink/data/raw/mc_maze.nwb"
    
    print("=" * 60)
    print("Experiment 3: Temporal Context")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading MC_Maze data...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_dataset = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different window sizes
    window_sizes = [5, 10, 20]
    results = {}
    
    for window_size in window_sizes:
        print(f"\n[2/3] Testing window_size={window_size} ({window_size * 25}ms)...")
        
        dataset = TemporalSpikeDataset(
            mc_dataset.spike_counts, 
            mc_dataset.velocities,
            window_size=window_size
        )
        
        # Split
        n = len(dataset)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        
        train_loader = DataLoader(Subset(dataset, range(n_train)), batch_size=64, shuffle=True)
        val_loader = DataLoader(Subset(dataset, range(n_train, n_train + n_val)), batch_size=64)
        test_loader = DataLoader(Subset(dataset, range(n_train + n_val, n)), batch_size=64)
        
        # MLP
        print(f"    Training MLP (input_dim={142 * window_size})...")
        mlp = TemporalMLP(n_channels=142, window_size=window_size).to(device)
        mlp_results = train_and_evaluate(mlp, train_loader, val_loader, test_loader, device)
        
        # LSTM
        print(f"    Training LSTM...")
        lstm = TemporalLSTM(n_channels=142).to(device)
        lstm_results = train_and_evaluate(lstm, train_loader, val_loader, test_loader, device)
        
        results[window_size] = {
            'mlp': mlp_results,
            'lstm': lstm_results
        }
        
        print(f"    MLP R²:  {mlp_results['r2_overall']:.4f}")
        print(f"    LSTM R²: {lstm_results['r2_overall']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 RESULTS")
    print("=" * 60)
    print(f"{'Window':>10} | {'MLP R²':>10} | {'LSTM R²':>10}")
    print("-" * 40)
    for ws in window_sizes:
        mlp_r2 = results[ws]['mlp']['r2_overall']
        lstm_r2 = results[ws]['lstm']['r2_overall']
        print(f"{ws:>7} ({ws*25:>3}ms) | {mlp_r2:>10.4f} | {lstm_r2:>10.4f}")
    print("=" * 60)
    
    best_r2 = max(
        max(results[ws]['mlp']['r2_overall'], results[ws]['lstm']['r2_overall'])
        for ws in window_sizes
    )
    
    if best_r2 >= 0.5:
        print(f"\n✓ Temporal context significantly improves decoding (R² = {best_r2:.4f})")
    elif best_r2 >= 0.2:
        print(f"\n~ Moderate improvement with temporal context (R² = {best_r2:.4f})")
    else:
        print(f"\n⚠ Still weak signal (R² = {best_r2:.4f})")


if __name__ == '__main__':
    main()
