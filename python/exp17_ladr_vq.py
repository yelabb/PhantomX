"""
Experiment 17: Lag-Aware Distilled RVQ-4 (LADR-VQ)

Goal: Beat LSTM fairly by fixing benchmark noise, sweeping lag, and distilling
teacher (continuous) into RVQ-4 student.

Pipeline:
  0) Lock benchmark: fixed split, preprocessing, metric
  1) Lag sweep (Δ in [-5, +5]) for Teacher + LSTM
  2) Train Teacher (no VQ) at best Δ*
  3) Train Student (RVQ-4) with distillation
  4) Optional dequant repair MLP
  5) Seed sweep (N=10) and report mean±std + paired win-rate
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, Subset

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from phantomx.tokenizer import SpikeTokenizer
from phantomx.data import MCMazeDataset

# Reuse Exp12 architecture
from exp12_residual_vq import CausalTransformerEncoder, ResidualVectorQuantizer

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def r2(preds: np.ndarray, targets: np.ndarray) -> float:
    return r2_score(targets, preds)


@dataclass
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15


@dataclass
class TrainConfig:
    window_size: int = 10
    d_model: int = 256
    embedding_dim: int = 128
    num_layers: int = 6
    num_quantizers: int = 4
    num_codes: int = 128
    batch_size: int = 64
    pretrain_epochs: int = 50
    finetune_epochs: int = 100
    lr_pretrain: float = 3e-4
    lr_encoder: float = 1e-5
    lr_vq: float = 5e-5
    lr_decoder: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 30
    huber_delta: Optional[float] = None


@dataclass
class DistillConfig:
    alpha: float = 1.0
    beta: float = 0.0
    use_latent_match: bool = False
    use_repair: bool = False


# ============================================================
# Dataset with Lagged Target
# ============================================================

class LaggedWindowDataset(Dataset):
    """
    Windowed dataset with lagged target.

    For each window ending at t, target is velocity[t + delta].
    """

    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 10,
        delta: int = 0,
    ):
        self.window_size = window_size
        self.delta = int(delta)

        n = len(spike_counts)
        window_ends = np.arange(window_size - 1, n, dtype=np.int64)
        target_idx = window_ends + self.delta
        valid = (target_idx >= 0) & (target_idx < n)

        window_ends = window_ends[valid]
        target_idx = target_idx[valid]
        window_starts = window_ends - (window_size - 1)

        self.windows = np.stack([
            spike_counts[i:i + window_size]
            for i in window_starts
        ])
        self.targets = velocities[target_idx]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "window": torch.tensor(self.windows[idx], dtype=torch.float32),
            "velocity": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


# ============================================================
# Models
# ============================================================

class DistilledRVQModel(nn.Module):
    """
    CausalTransformer + RVQ-4 with optional dequant repair.
    """

    def __init__(
        self,
        n_channels: int = 142,
        window_size: int = 10,
        d_model: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 6,
        num_quantizers: int = 4,
        num_codes: int = 128,
        use_repair: bool = False,
        huber_delta: Optional[float] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.use_vq = False
        self.use_repair = use_repair

        self.encoder = CausalTransformerEncoder(
            n_channels=n_channels,
            d_model=d_model,
            nhead=8,
            num_layers=num_layers,
            output_dim=embedding_dim,
        )

        self.vq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_codes=num_codes,
            embedding_dim=embedding_dim,
        )

        self.repair = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        ) if use_repair else None

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        self.loss_fn = nn.HuberLoss(delta=huber_delta) if huber_delta else nn.MSELoss()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict:
        z_e = self.encode(x)

        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {
                "indices": torch.zeros(x.size(0), dtype=torch.long, device=x.device),
                "perplexity": torch.tensor(0.0, device=x.device),
                "commitment_loss": torch.tensor(0.0, device=x.device),
                "residual_norm": torch.tensor(0.0, device=x.device),
            }

        z_hat = z_q + self.repair(z_q) if self.repair is not None else z_q
        velocity_pred = self.decoder(z_hat)

        output = {
            "velocity_pred": velocity_pred,
            "z_e": z_e,
            "z_q": z_q,
            "z_hat": z_hat,
            **vq_info,
        }

        if targets is not None:
            recon_loss = self.loss_fn(velocity_pred, targets)
            commitment_loss = vq_info.get("commitment_loss", 0.0)
            total_loss = recon_loss + (commitment_loss if isinstance(commitment_loss, torch.Tensor) else 0.0)
            output["recon_loss"] = recon_loss
            output["total_loss"] = total_loss

        return output


class LSTMBaseline(nn.Module):
    def __init__(self, n_channels=142, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.decoder(out[:, -1, :])


# ============================================================
# Training / Evaluation
# ============================================================

def train_teacher(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    model.use_vq = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_pretrain,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.pretrain_epochs)

    best_val_r2 = -float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, cfg.pretrain_epochs + 1):
        model.train()
        for batch in train_loader:
            window = batch["window"].to(device)
            velocity = batch["velocity"].to(device)
            optimizer.zero_grad()
            output = model(window, velocity)
            output["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch["window"].to(device))
                val_preds.append(output["velocity_pred"].cpu())
                val_targets.append(batch["velocity"])

        val_r2 = r2(torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy())
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [Teacher] Epoch {epoch:3d}: val_R²={val_r2:.4f} (best={best_val_r2:.4f})")

        if patience >= cfg.patience:
            print(f"    [Teacher] Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return best_val_r2


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["window"].to(device))
            # Handle both dict (DistilledRVQModel) and tensor (LSTMBaseline) outputs
            if isinstance(output, dict):
                preds.append(output["velocity_pred"].cpu())
            else:
                preds.append(output.cpu())
            targets.append(batch["velocity"])
    return r2(torch.cat(preds).numpy(), torch.cat(targets).numpy())


def init_rvq_from_train(model: DistilledRVQModel, train_loader: DataLoader, device: torch.device) -> None:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in train_loader:
            z_e = model.encode(batch["window"].to(device))
            embeddings.append(z_e)
    model.vq.init_from_data(torch.cat(embeddings))


def train_student(
    student: DistilledRVQModel,
    teacher: DistilledRVQModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    distill: DistillConfig,
) -> float:
    student.use_vq = True
    teacher.eval()

    optimizer = torch.optim.AdamW(
        [
            {"params": student.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": student.vq.parameters(), "lr": cfg.lr_vq},
            {"params": student.decoder.parameters(), "lr": cfg.lr_decoder},
            *([{"params": student.repair.parameters(), "lr": cfg.lr_decoder}] if student.repair else []),
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.finetune_epochs)

    best_val_r2 = -float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, cfg.finetune_epochs + 1):
        student.train()
        for batch in train_loader:
            window = batch["window"].to(device)
            velocity = batch["velocity"].to(device)

            with torch.no_grad():
                teacher_out = teacher(window)
                v_teacher = teacher_out["velocity_pred"].detach()
                z_teacher = teacher_out["z_e"].detach()

            optimizer.zero_grad()
            student_out = student(window, velocity)
            v_student = student_out["velocity_pred"]
            z_q = student_out["z_q"]

            loss_gt = F.mse_loss(v_student, velocity)
            loss_distill = F.mse_loss(v_student, v_teacher)
            loss_lat = F.mse_loss(z_q, z_teacher) if distill.use_latent_match else 0.0

            total = loss_gt + distill.alpha * loss_distill + distill.beta * loss_lat
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        student.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                output = student(batch["window"].to(device))
                val_preds.append(output["velocity_pred"].cpu())
                val_targets.append(batch["velocity"])

        val_r2 = r2(torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy())
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [Student] Epoch {epoch:3d}: val_R²={val_r2:.4f} (best={best_val_r2:.4f})")

        if patience >= cfg.patience:
            print(f"    [Student] Early stopping at epoch {epoch}")
            break

    if best_state:
        student.load_state_dict(best_state)

    return best_val_r2


def eval_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
) -> Tuple[float, float]:
    model = LSTMBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_r2 = -float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["window"].to(device))
            loss = F.mse_loss(pred, batch["velocity"].to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["window"].to(device))
                preds.append(pred.cpu())
                targets.append(batch["velocity"])

        r2_val = r2(torch.cat(preds).numpy(), torch.cat(targets).numpy())
        if r2_val > best_r2:
            best_r2 = r2_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    test_r2 = eval_model(model, test_loader, device)
    return best_r2, test_r2


# ============================================================
# Benchmark Locking / Data
# ============================================================

def load_and_normalize(split: SplitConfig) -> Tuple[np.ndarray, np.ndarray]:
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_maze = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    spikes = mc_maze.spike_counts
    velocities = mc_maze.velocities

    # Train-only normalization (prevent leakage)
    train_end = int(split.train_frac * len(spikes))
    spike_mean = spikes[:train_end].mean(0, keepdims=True)
    spike_std = spikes[:train_end].std(0, keepdims=True) + 1e-6
    vel_mean = velocities[:train_end].mean(0, keepdims=True)
    vel_std = velocities[:train_end].std(0, keepdims=True) + 1e-6

    spikes = (spikes - spike_mean) / spike_std
    velocities = (velocities - vel_mean) / vel_std

    return spikes, velocities


def make_loaders(
    spikes: np.ndarray,
    velocities: np.ndarray,
    window_size: int,
    delta: int,
    batch_size: int,
    split: SplitConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = LaggedWindowDataset(spikes, velocities, window_size=window_size, delta=delta)
    n_total = len(dataset)
    n_train = int(split.train_frac * n_total)
    n_val = int(split.val_frac * n_total)
    n_test = n_total - n_train - n_val

    train_ds = Subset(dataset, list(range(0, n_train)))
    val_ds = Subset(dataset, list(range(n_train, n_train + n_val)))
    test_ds = Subset(dataset, list(range(n_train + n_val, n_train + n_val + n_test)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ============================================================
# Lag Sweep + Seed Sweep
# ============================================================

def lag_sweep(
    spikes: np.ndarray,
    velocities: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    split: SplitConfig,
    deltas: List[int],
    seeds: List[int],
) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}

    for delta in deltas:
        print(f"\n[Δ={delta}] Lag sweep")
        teacher_scores = []
        lstm_scores = []

        for seed in seeds:
            set_seed(seed)
            train_loader, val_loader, test_loader = make_loaders(
                spikes, velocities, cfg.window_size, delta, cfg.batch_size, split
            )

            teacher = DistilledRVQModel(
                window_size=cfg.window_size,
                d_model=cfg.d_model,
                embedding_dim=cfg.embedding_dim,
                num_layers=cfg.num_layers,
                num_quantizers=cfg.num_quantizers,
                num_codes=cfg.num_codes,
                use_repair=False,
                huber_delta=cfg.huber_delta,
            ).to(device)

            teacher_r2 = train_teacher(teacher, train_loader, val_loader, device, cfg)
            teacher_scores.append(teacher_r2)

            lstm_r2, _ = eval_lstm(train_loader, val_loader, test_loader, device)
            lstm_scores.append(lstm_r2)

        results[delta] = {
            "teacher_r2_mean": float(np.mean(teacher_scores)),
            "teacher_r2_std": float(np.std(teacher_scores, ddof=1)) if len(teacher_scores) > 1 else 0.0,
            "lstm_r2_mean": float(np.mean(lstm_scores)),
            "lstm_r2_std": float(np.std(lstm_scores, ddof=1)) if len(lstm_scores) > 1 else 0.0,
        }

    return results


def run_seed_sweep(
    spikes: np.ndarray,
    velocities: np.ndarray,
    cfg: TrainConfig,
    distill: DistillConfig,
    split: SplitConfig,
    delta: int,
    seeds: List[int],
    device: torch.device,
) -> Dict:
    teacher_scores = []
    student_scores = []
    lstm_scores = []

    for seed in seeds:
        print(f"\n[Seed {seed}] Running teacher/student/LSTM")
        set_seed(seed)
        train_loader, val_loader, test_loader = make_loaders(
            spikes, velocities, cfg.window_size, delta, cfg.batch_size, split
        )

        teacher = DistilledRVQModel(
            window_size=cfg.window_size,
            d_model=cfg.d_model,
            embedding_dim=cfg.embedding_dim,
            num_layers=cfg.num_layers,
            num_quantizers=cfg.num_quantizers,
            num_codes=cfg.num_codes,
            use_repair=False,
            huber_delta=cfg.huber_delta,
        ).to(device)

        teacher_r2 = train_teacher(teacher, train_loader, val_loader, device, cfg)
        teacher_test_r2 = eval_model(teacher, test_loader, device)
        teacher_scores.append(teacher_test_r2)

        student = DistilledRVQModel(
            window_size=cfg.window_size,
            d_model=cfg.d_model,
            embedding_dim=cfg.embedding_dim,
            num_layers=cfg.num_layers,
            num_quantizers=cfg.num_quantizers,
            num_codes=cfg.num_codes,
            use_repair=distill.use_repair,
            huber_delta=cfg.huber_delta,
        ).to(device)

        # Initialize student from teacher weights (encoder + decoder)
        student.encoder.load_state_dict(teacher.encoder.state_dict())
        student.decoder.load_state_dict(teacher.decoder.state_dict())

        init_rvq_from_train(student, train_loader, device)
        student_r2 = train_student(student, teacher, train_loader, val_loader, device, cfg, distill)
        student_test_r2 = eval_model(student, test_loader, device)
        student_scores.append(student_test_r2)

        _, lstm_test_r2 = eval_lstm(train_loader, val_loader, test_loader, device)
        lstm_scores.append(lstm_test_r2)

    def summarize(scores: List[float]) -> Dict[str, float]:
        arr = np.array(scores)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }

    # Paired win-rate: Student vs LSTM on TEST
    wins = sum(1 for s, l in zip(student_scores, lstm_scores) if s > l)
    win_rate = wins / max(1, len(seeds))

    return {
        "teacher": summarize(teacher_scores),
        "student": summarize(student_scores),
        "lstm": summarize(lstm_scores),
        "student_vs_lstm_win_rate": float(win_rate),
        "raw": {
            "teacher": teacher_scores,
            "student": student_scores,
            "lstm": lstm_scores,
        },
    }


# ============================================================
# Main
# ============================================================

def run_experiment():
    print("\n" + "=" * 70)
    print("EXPERIMENT 17: Lag-Aware Distilled RVQ-4 (LADR-VQ)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TrainConfig()
    distill = DistillConfig()
    split = SplitConfig()

    # Load data once
    spikes, velocities = load_and_normalize(split)

    # Smoke check: RVQ init exists
    try:
        _smoke_model = DistilledRVQModel().to(device)
        dummy = torch.randn(8, cfg.window_size, 142, device=device)
        with torch.no_grad():
            z_e = _smoke_model.encode(dummy)
        _smoke_model.vq.init_from_data(z_e)
    except Exception as exc:
        raise RuntimeError(
            "RVQ init_from_data failed. Verify ResidualVectorQuantizer implements init_from_data."
        ) from exc

    # Step 1: Lag sweep (teacher + LSTM)
    deltas = list(range(-5, 6))
    lag_results = lag_sweep(
        spikes, velocities, cfg, device, split, deltas=deltas, seeds=[0, 1, 2]
    )

    # Choose Δ* by teacher
    best_delta = max(lag_results, key=lambda d: lag_results[d]["teacher_r2_mean"])
    print(f"\nBest Δ* by teacher: {best_delta} bins")

    # Step 0/5: Seed sweep at Δ*
    seeds = list(range(10))
    sweep_results = run_seed_sweep(
        spikes, velocities, cfg, distill, split, best_delta, seeds, device
    )

    out = {
        "config": cfg.__dict__,
        "distill": distill.__dict__,
        "delta_star": best_delta,
        "lag_sweep": lag_results,
        "seed_sweep": sweep_results,
    }

    out_path = RESULTS_DIR / f"exp17_ladr_vq_results_delta_{best_delta}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "-" * 70)
    print("Seed sweep summary:")
    print(f"  Teacher: {sweep_results['teacher']['mean']:.4f} ± {sweep_results['teacher']['std']:.4f}")
    print(f"  Student: {sweep_results['student']['mean']:.4f} ± {sweep_results['student']['std']:.4f}")
    print(f"  LSTM:    {sweep_results['lstm']['mean']:.4f} ± {sweep_results['lstm']['std']:.4f}")
    print(f"  Student vs LSTM win-rate: {sweep_results['student_vs_lstm_win_rate']:.2f}")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_experiment()
