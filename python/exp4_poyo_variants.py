"""
Experiment 4: Channel-Preserving POYO Variants

Test tokenization strategies that balance:
- Permutation invariance (POYO goal: electrode dropout robustness)
- Velocity information (need channel identity for decoding)

Key insight: LaBraM's original POYO uses sorted order statistics which
completely destroys spatial information. We need alternatives.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phantomx.data import MCMazeDataset
from phantomx.tokenizer import SpikeTokenizer


class TemporalDataset(Dataset):
    """Base dataset with temporal windows."""
    
    def __init__(self, spike_counts, velocities, window_size=10):
        self.spike_counts = spike_counts
        self.velocities = velocities
        self.window_size = window_size
    
    def __len__(self):
        return len(self.spike_counts) - self.window_size + 1
    
    def __getitem__(self, idx):
        window = self.spike_counts[idx:idx + self.window_size]
        velocity = self.velocities[idx + self.window_size - 1]
        return {
            'spikes': torch.tensor(window, dtype=torch.float32),
            'velocity': torch.tensor(velocity, dtype=torch.float32)
        }


# ============================================================
# Tokenization Strategies
# ============================================================

class HistogramTokenizer:
    """
    Approach 1: Binned Histogram Tokens
    
    Instead of keeping top-k values (which loses channel info),
    create a histogram of spike counts across all channels.
    
    This is permutation invariant (channel order doesn't matter)
    but preserves the DISTRIBUTION of neural activity.
    """
    
    def __init__(self, n_bins=32, min_val=-3, max_val=3):
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    def tokenize(self, spikes):
        """Convert spike counts to histogram."""
        # spikes: [n_channels] or [T, n_channels]
        if spikes.ndim == 1:
            hist, _ = np.histogram(spikes, bins=self.bin_edges)
            return hist.astype(np.float32) / len(spikes)  # Normalize
        else:
            # Temporal: histogram per timestep
            T, C = spikes.shape
            hists = np.zeros((T, self.n_bins), dtype=np.float32)
            for t in range(T):
                hists[t], _ = np.histogram(spikes[t], bins=self.bin_edges)
                hists[t] /= C
            return hists


class SortedRankTokenizer:
    """
    Approach 2: Sorted with Rank Encoding
    
    Sort channels by activity, but include the RANK as a feature.
    This way the model learns what "top 10% of channels" means.
    """
    
    def __init__(self, n_channels=142):
        self.n_channels = n_channels
        # Rank encoding: normalized position [0, 1]
        self.ranks = np.linspace(0, 1, n_channels).astype(np.float32)
    
    def tokenize(self, spikes):
        """Return sorted values with their ranks."""
        if spikes.ndim == 1:
            sorted_vals = np.sort(spikes)[::-1]  # Descending
            return np.stack([sorted_vals, self.ranks], axis=-1)  # [C, 2]
        else:
            T, C = spikes.shape
            output = np.zeros((T, C, 2), dtype=np.float32)
            for t in range(T):
                output[t, :, 0] = np.sort(spikes[t])[::-1]
                output[t, :, 1] = self.ranks
            return output


class StatisticsTokenizer:
    """
    Approach 3: Population Statistics
    
    Compute permutation-invariant statistics over channels:
    - Mean, std, skewness, kurtosis
    - Percentiles (10, 25, 50, 75, 90)
    - Number of active channels above threshold
    """
    
    def __init__(self, percentiles=[10, 25, 50, 75, 90]):
        self.percentiles = percentiles
    
    def tokenize(self, spikes):
        """Extract population statistics."""
        if spikes.ndim == 1:
            return self._compute_stats(spikes)
        else:
            T = spikes.shape[0]
            stats = [self._compute_stats(spikes[t]) for t in range(T)]
            return np.stack(stats)
    
    def _compute_stats(self, x):
        stats = [
            np.mean(x),
            np.std(x),
            np.median(x),
            np.sum(x > 0) / len(x),  # Fraction active
            np.sum(x > 1) / len(x),  # Fraction highly active
        ]
        # Add percentiles
        stats.extend([np.percentile(x, p) for p in self.percentiles])
        return np.array(stats, dtype=np.float32)


class TemporalPatternTokenizer:
    """
    Approach 4: Temporal Pattern Tokens
    
    Instead of tokenizing spatial patterns, tokenize TEMPORAL patterns.
    For each channel, extract temporal features over the window.
    This preserves some channel identity while being robust to amplitude changes.
    """
    
    def __init__(self, n_channels=142):
        self.n_channels = n_channels
    
    def tokenize(self, spikes):
        """Extract temporal features per channel."""
        # spikes: [T, C]
        T, C = spikes.shape
        
        features = []
        # Mean activity
        features.append(np.mean(spikes, axis=0))
        # Temporal derivative (velocity of neural activity)
        features.append(np.mean(np.diff(spikes, axis=0), axis=0))
        # Temporal std (variability)
        features.append(np.std(spikes, axis=0))
        # Max over time
        features.append(np.max(spikes, axis=0))
        
        return np.stack(features, axis=0).astype(np.float32)  # [4, C]


# ============================================================
# Models
# ============================================================

class HistogramLSTM(nn.Module):
    """LSTM on histogram tokens."""
    
    def __init__(self, n_bins=32, hidden_dim=64, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(n_bins, hidden_dim, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class RankSortedLSTM(nn.Module):
    """LSTM on sorted values with rank encoding."""
    
    def __init__(self, n_channels=142, hidden_dim=128, output_dim=2):
        super().__init__()
        # Input: [T, C, 2] -> flatten to [T, C*2]
        self.lstm = nn.LSTM(n_channels * 2, hidden_dim, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        B, T, C, F = x.shape
        x = x.view(B, T, C * F)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class StatsLSTM(nn.Module):
    """LSTM on population statistics."""
    
    def __init__(self, n_stats=10, hidden_dim=64, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(n_stats, hidden_dim, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TemporalPatternMLP(nn.Module):
    """MLP on temporal pattern tokens."""
    
    def __init__(self, n_features=4, n_channels=142, hidden_dim=256, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# Training & Evaluation
# ============================================================

def create_tokenized_dataset(spike_counts, velocities, tokenizer, window_size=10):
    """Create dataset with pre-computed tokens."""
    n_samples = len(spike_counts) - window_size + 1
    
    tokens_list = []
    velocity_list = []
    
    for i in range(n_samples):
        window = spike_counts[i:i + window_size]
        tokens = tokenizer.tokenize(window)
        tokens_list.append(tokens)
        velocity_list.append(velocities[i + window_size - 1])
    
    return list(zip(tokens_list, velocity_list))


def train_and_evaluate(model, train_data, val_data, test_data, device, epochs=50):
    """Train model and return test R²."""
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, 
                              collate_fn=lambda x: (
                                  torch.stack([torch.tensor(i[0]) for i in x]),
                                  torch.stack([torch.tensor(i[1]) for i in x])
                              ))
    val_loader = DataLoader(val_data, batch_size=64,
                            collate_fn=lambda x: (
                                torch.stack([torch.tensor(i[0]) for i in x]),
                                torch.stack([torch.tensor(i[1]) for i in x])
                            ))
    test_loader = DataLoader(test_data, batch_size=64,
                             collate_fn=lambda x: (
                                 torch.stack([torch.tensor(i[0]) for i in x]),
                                 torch.stack([torch.tensor(i[1]) for i in x])
                             ))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for tokens, velocity in train_loader:
            tokens = tokens.float().to(device)
            velocity = velocity.float().to(device)
            
            optimizer.zero_grad()
            pred = model(tokens)
            loss = criterion(pred, velocity)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tokens, velocity in val_loader:
                tokens = tokens.float().to(device)
                velocity = velocity.float().to(device)
                val_loss += criterion(model(tokens), velocity).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break
    
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for tokens, velocity in test_loader:
            tokens = tokens.float().to(device)
            all_preds.append(model(tokens).cpu().numpy())
            all_targets.append(velocity.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_test = np.concatenate(all_targets)
    
    return r2_score(y_test, y_pred)


def main():
    DATA_PATH = "c:/Users/guzzi/Desktop/Projects/DEV-ACTIF/NeuraLink/PhantomLink/data/raw/mc_maze.nwb"
    WINDOW_SIZE = 10
    
    print("=" * 60)
    print("Experiment 4: Channel-Preserving POYO Variants")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading MC_Maze data...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_dataset = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Split indices
    n = len(mc_dataset.spike_counts) - WINDOW_SIZE + 1
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    results = {}
    
    # ========================================
    # Approach 1: Histogram Tokens
    # ========================================
    print("\n[2/5] Testing Histogram Tokenizer...")
    hist_tokenizer = HistogramTokenizer(n_bins=32)
    hist_data = create_tokenized_dataset(
        mc_dataset.spike_counts, mc_dataset.velocities, hist_tokenizer, WINDOW_SIZE
    )
    
    hist_model = HistogramLSTM(n_bins=32).to(device)
    hist_r2 = train_and_evaluate(
        hist_model,
        hist_data[:n_train],
        hist_data[n_train:n_train+n_val],
        hist_data[n_train+n_val:],
        device
    )
    results['Histogram'] = hist_r2
    print(f"  Histogram LSTM R²: {hist_r2:.4f}")
    
    # ========================================
    # Approach 2: Sorted + Rank Encoding
    # ========================================
    print("\n[3/5] Testing Sorted+Rank Tokenizer...")
    rank_tokenizer = SortedRankTokenizer(n_channels=142)
    rank_data = create_tokenized_dataset(
        mc_dataset.spike_counts, mc_dataset.velocities, rank_tokenizer, WINDOW_SIZE
    )
    
    rank_model = RankSortedLSTM(n_channels=142).to(device)
    rank_r2 = train_and_evaluate(
        rank_model,
        rank_data[:n_train],
        rank_data[n_train:n_train+n_val],
        rank_data[n_train+n_val:],
        device
    )
    results['Sorted+Rank'] = rank_r2
    print(f"  Sorted+Rank LSTM R²: {rank_r2:.4f}")
    
    # ========================================
    # Approach 3: Population Statistics
    # ========================================
    print("\n[4/5] Testing Statistics Tokenizer...")
    stats_tokenizer = StatisticsTokenizer()
    stats_data = create_tokenized_dataset(
        mc_dataset.spike_counts, mc_dataset.velocities, stats_tokenizer, WINDOW_SIZE
    )
    
    n_stats = len(stats_data[0][0][0])  # Get feature dimension
    stats_model = StatsLSTM(n_stats=n_stats).to(device)
    stats_r2 = train_and_evaluate(
        stats_model,
        stats_data[:n_train],
        stats_data[n_train:n_train+n_val],
        stats_data[n_train+n_val:],
        device
    )
    results['Statistics'] = stats_r2
    print(f"  Statistics LSTM R²: {stats_r2:.4f}")
    
    # ========================================
    # Approach 4: Temporal Pattern Tokens
    # ========================================
    print("\n[5/5] Testing Temporal Pattern Tokenizer...")
    temp_tokenizer = TemporalPatternTokenizer(n_channels=142)
    temp_data = create_tokenized_dataset(
        mc_dataset.spike_counts, mc_dataset.velocities, temp_tokenizer, WINDOW_SIZE
    )
    
    temp_model = TemporalPatternMLP(n_features=4, n_channels=142).to(device)
    temp_r2 = train_and_evaluate(
        temp_model,
        temp_data[:n_train],
        temp_data[n_train:n_train+n_val],
        temp_data[n_train+n_val:],
        device
    )
    results['TemporalPattern'] = temp_r2
    print(f"  Temporal Pattern MLP R²: {temp_r2:.4f}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 4 RESULTS")
    print("=" * 60)
    print(f"{'Tokenization':<20} | {'R²':>10} | {'Permutation Invariant':>22}")
    print("-" * 60)
    print(f"{'Raw LSTM (baseline)':<20} | {'0.7783':>10} | {'No':>22}")
    for name, r2 in results.items():
        invariant = 'Yes' if name != 'TemporalPattern' else 'Partial'
        print(f"{name:<20} | {r2:>10.4f} | {invariant:>22}")
    print("=" * 60)
    
    best_name = max(results, key=results.get)
    best_r2 = results[best_name]
    
    print(f"\nBest permutation-invariant approach: {best_name} (R² = {best_r2:.4f})")
    
    if best_r2 >= 0.5:
        print("✓ This approach achieves reasonable velocity decoding!")
    else:
        print("⚠ Permutation invariance significantly hurts velocity decoding")


if __name__ == '__main__':
    main()
