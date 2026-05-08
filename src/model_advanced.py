"""Advanced model: LSTM (sequence-to-one) with quantile heads for water-level forecasting.

Project 20 - Vernagtferner Glacier Water Level Forecasting (Liora Phase 1).

Architecture follows Kratzert and colleagues 2018 (HESS) for the LSTM rainfall-runoff
backbone, extended with three quantile output heads (0.1, 0.5, 0.9) trained jointly
under the pinball loss for prediction intervals on the high-flow tail. Optional
Temporal Fusion Transformer variant is sketched at the end of this file as a
Phase 2 follow-up (Lim and colleagues 2021, IJF).

This script is NOT executed in Phase 1. The main session runs:
    python src/model_advanced.py
once data has been downloaded into ../data/raw/ and processed.

Dependencies expected by main session:
    torch >= 2.0, pandas, numpy, scikit-learn, joblib.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # torch optional at scaffold time
    torch = None  # type: ignore
    nn = None  # type: ignore

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DELIVERABLES = PROJECT_DIR / "deliverables"
DELIVERABLES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LSTMConfig:
    seq_len: int = 30          # 30-day input window
    horizon: int = 1           # 1-day-ahead target
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    quantiles: tuple = (0.1, 0.5, 0.9)
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GlacierSeqDataset(Dataset if torch is not None else object):  # type: ignore[misc]
    """Sequence-to-one dataset.

    X shape: (seq_len, n_features). y shape: (1,) - water level at t + horizon.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len - self.horizon + 1)

    def __getitem__(self, idx: int):
        x = self.X[idx: idx + self.seq_len]
        target = self.y[idx + self.seq_len + self.horizon - 1]
        return x, np.float32(target)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if torch is not None:

    class GlacierLSTM(nn.Module):
        """LSTM with one shared trunk and one linear head per quantile."""

        def __init__(self, n_features: int, cfg: LSTMConfig):
            super().__init__()
            self.cfg = cfg
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.heads = nn.ModuleList([nn.Linear(cfg.hidden_size, 1) for _ in cfg.quantiles])

        def forward(self, x):
            out, _ = self.lstm(x)
            last = self.dropout(out[:, -1, :])
            preds = [head(last).squeeze(-1) for head in self.heads]
            return torch.stack(preds, dim=-1)  # (batch, n_quantiles)


    def pinball_loss(pred: "torch.Tensor", target: "torch.Tensor", quantiles: Iterable[float]) -> "torch.Tensor":
        losses = []
        for i, q in enumerate(quantiles):
            err = target - pred[..., i]
            losses.append(torch.maximum(q * err, (q - 1) * err).mean())
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Hydrological metrics
# ---------------------------------------------------------------------------

def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    r = np.corrcoef(obs, pred)[0, 1] if obs.std() > 0 and pred.std() > 0 else 0.0
    alpha = pred.std() / obs.std() if obs.std() > 0 else 0.0
    beta = pred.mean() / obs.mean() if obs.mean() != 0 else 0.0
    return float(1.0 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "t_mean_C", "t_max_C", "t_min_C",
    "precip_mm", "rad_global_Wm2", "snow_depth_cm",
    "rh_mean", "wind_speed_mean",
    "glacier_area_km2",
    "doy_sin", "doy_cos",
]

TARGET = "water_level_cm"


def add_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    return out


def standardise(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0, 1.0, std)
    return (arr - mean) / std_safe


def load_processed() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "vernagt_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the processed table first with notebooks/02_features.ipynb."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fit_lstm(cfg: LSTMConfig | None = None) -> dict:
    if torch is None:
        raise ImportError("torch is required for the advanced LSTM model. Install with `pip install torch`.")

    cfg = cfg or LSTMConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = load_processed()
    df = add_seasonal(df)

    feat = [c for c in FEATURE_COLS if c in df.columns]
    df_model = df.dropna(subset=feat + [TARGET]).reset_index(drop=True)

    last_year = int(df_model["date"].dt.year.max())
    train_mask = df_model["date"].dt.year < last_year - 1
    val_mask = df_model["date"].dt.year == last_year - 1
    test_mask = df_model["date"].dt.year == last_year

    Xtr = df_model.loc[train_mask, feat].to_numpy()
    Xva = df_model.loc[val_mask, feat].to_numpy()
    Xte = df_model.loc[test_mask, feat].to_numpy()
    ytr = df_model.loc[train_mask, TARGET].to_numpy()
    yva = df_model.loc[val_mask, TARGET].to_numpy()
    yte = df_model.loc[test_mask, TARGET].to_numpy()

    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
    Xtr_n = standardise(Xtr, mean, std)
    Xva_n = standardise(Xva, mean, std)
    Xte_n = standardise(Xte, mean, std)

    y_mean, y_std = float(ytr.mean()), float(ytr.std() or 1.0)
    ytr_n = (ytr - y_mean) / y_std
    yva_n = (yva - y_mean) / y_std
    yte_n = (yte - y_mean) / y_std

    train_ds = GlacierSeqDataset(Xtr_n, ytr_n, cfg.seq_len, cfg.horizon)
    val_ds = GlacierSeqDataset(Xva_n, yva_n, cfg.seq_len, cfg.horizon)
    test_ds = GlacierSeqDataset(Xte_n, yte_n, cfg.seq_len, cfg.horizon)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GlacierLSTM(n_features=len(feat), cfg=cfg).to(cfg.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = pinball_loss(pred, yb, cfg.quantiles)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                val_losses.append(pinball_loss(model(xb), yb, cfg.quantiles).item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test-set evaluation: median (q=0.5) head as the point forecast
    model.eval()
    obs_all, pred_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(cfg.device)
            preds_n = model(xb).cpu().numpy()
            median_idx = list(cfg.quantiles).index(0.5) if 0.5 in cfg.quantiles else preds_n.shape[-1] // 2
            preds = preds_n[..., median_idx] * y_std + y_mean
            pred_all.append(preds)
            obs_all.append(yb.numpy() * y_std + y_mean)
    if pred_all:
        pred = np.concatenate(pred_all)
        obs = np.concatenate(obs_all)
        metrics = {
            "rmse_cm": float(np.sqrt(np.mean((obs - pred) ** 2))),
            "mae_cm": float(np.mean(np.abs(obs - pred))),
            "nse": nse(obs, pred),
            "kge": kge(obs, pred),
            "n_test": int(len(obs)),
        }
    else:
        metrics = {"warning": "test set empty"}

    out = {
        "config": cfg.__dict__,
        "feature_columns": feat,
        "metrics_test_median": metrics,
        "best_val_pinball": best_val,
        "history_tail": history[-5:],
    }

    torch.save(model.state_dict(), DELIVERABLES / "lstm_advanced.pt")
    with open(DELIVERABLES / "lstm_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main():
    out = fit_lstm()
    print(json.dumps(out["metrics_test_median"], indent=2))
    print(f"LSTM checkpoint saved to {DELIVERABLES / 'lstm_advanced.pt'}")


# ---------------------------------------------------------------------------
# Phase 2 sketch: Temporal Fusion Transformer variant
# ---------------------------------------------------------------------------
# from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
# Multi-horizon (1, 7, 30 day) forecasts with attention weights over climate
# covariates would replace the LSTM in a follow-up. See Lim and colleagues
# 2021 (DOI:10.1016/j.ijforecast.2021.03.012).


if __name__ == "__main__":
    main()
