"""Baseline model: linear regression with seasonal and lagged climate features.

Project 20 - Vernagtferner Glacier Water Level Forecasting (Project layout

The baseline is intentionally simple and physically motivated:
  - Daily aggregation of the 5-minute hydro-met record.
  - Lagged temperature and precipitation at 1d, 7d, 30d windows.
  - Cyclical encoding of day-of-year (sin/cos) to capture the strong
    glacier-melt seasonality without committing to a specific basis.
  - Snow-depth and global-radiation covariates when available.
  - Diagnostics: NSE, KGE, RMSE, MAE, residual ACF, residual QQ.

This script is NOT executed in v1.0. The main session runs:
    python src/model_baseline.py
once data has been downloaded into ../data/raw/.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DELIVERABLES = PROJECT_DIR / "deliverables"
DELIVERABLES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed() -> pd.DataFrame:
    """Load the processed daily-resolution training table.

    Expected columns:
      - date (datetime, daily)
      - water_level_cm (target)
      - t_mean_C, t_max_C, t_min_C (DWD or BAdW air-temperature aggregates)
      - precip_mm (24-hour sum)
      - rad_global_Wm2 (mean global radiation)
      - snow_depth_cm (end-of-day snow depth)
      - glacier_area_km2 (annual WGMS forward-filled)
    """
    path = DATA_DIR / "processed" / "vernagt_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the processed table first with notebooks/02_features.ipynb."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

LAG_DAYS = (1, 3, 7, 30)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append lag and seasonal features to the daily frame."""
    out = df.copy()

    # Cyclical encoding of day-of-year
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    # Cumulative degree days from Oct 1 of the current mass-balance year
    if "t_mean_C" in out.columns:
        mb_year_start = out["date"].apply(
            lambda d: pd.Timestamp(year=d.year if d.month >= 10 else d.year - 1, month=10, day=1)
        )
        days_into_mb = (out["date"] - mb_year_start).dt.days
        pdd = (out["t_mean_C"].clip(lower=0)).copy()
        pdd_cum = pd.Series(0.0, index=out.index)
        cum = 0.0
        prev_year = None
        for i, (start, val) in enumerate(zip(mb_year_start, pdd.fillna(0))):
            if start != prev_year:
                cum = 0.0
                prev_year = start
            cum += val
            pdd_cum.iat[i] = cum
        out["pdd_cum_C"] = pdd_cum
        out["mb_day_idx"] = days_into_mb

    # Rolling lag aggregates
    for col in ("t_mean_C", "precip_mm", "rad_global_Wm2", "snow_depth_cm"):
        if col not in out.columns:
            continue
        for k in LAG_DAYS:
            out[f"{col}_lag{k}d_mean"] = (
                out[col].shift(1).rolling(window=k, min_periods=max(1, k // 2)).mean()
            )
        out[f"{col}_lag1d"] = out[col].shift(1)

    return out


# ---------------------------------------------------------------------------
# Hydrological metrics
# ---------------------------------------------------------------------------

def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (Nash and Sutcliffe 1970, DOI 10.1016/0022-1694(70)90255-6)."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency (Gupta and colleagues 2009)."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    r = np.corrcoef(obs, pred)[0, 1] if obs.std() > 0 and pred.std() > 0 else 0.0
    alpha = pred.std() / obs.std() if obs.std() > 0 else 0.0
    beta = pred.mean() / obs.mean() if obs.mean() != 0 else 0.0
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def forward_chain_split(df: pd.DataFrame, val_year: int, test_year: int) -> Split:
    """Forward-chained split: train on years < val_year, validate on val_year, test on test_year."""
    years = df["date"].dt.year
    train = np.where(years < val_year)[0]
    val = np.where(years == val_year)[0]
    test = np.where(years == test_year)[0]
    return Split(train_idx=train, val_idx=val, test_idx=test)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

FEATURE_COLS_DEFAULT = [
    "doy_sin", "doy_cos", "pdd_cum_C",
    "t_mean_C_lag1d", "t_mean_C_lag1d_mean", "t_mean_C_lag3d_mean",
    "t_mean_C_lag7d_mean", "t_mean_C_lag30d_mean",
    "precip_mm_lag1d", "precip_mm_lag1d_mean", "precip_mm_lag7d_mean",
    "rad_global_Wm2_lag1d_mean", "rad_global_Wm2_lag7d_mean",
    "snow_depth_cm_lag1d", "snow_depth_cm_lag30d_mean",
    "glacier_area_km2",
]


def _select_available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def fit_baseline(df: pd.DataFrame, target: str, val_year: int, test_year: int) -> dict:
    feat_cols = _select_available(df, FEATURE_COLS_DEFAULT)
    df_model = df.dropna(subset=feat_cols + [target]).reset_index(drop=True)
    X = df_model[feat_cols].to_numpy()
    y = df_model[target].to_numpy()
    split = forward_chain_split(df_model, val_year=val_year, test_year=test_year)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearRegression()),
    ])
    pipe.fit(X[split.train_idx], y[split.train_idx])

    metrics = {}
    for name, idx in (("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)):
        if len(idx) == 0:
            continue
        yp = pipe.predict(X[idx])
        ya = y[idx]
        metrics[name] = {
            "n": int(len(idx)),
            "rmse_cm": float(np.sqrt(mean_squared_error(ya, yp))),
            "mae_cm": float(mean_absolute_error(ya, yp)),
            "nse": nse(ya, yp),
            "kge": kge(ya, yp),
        }

    out = {
        "feature_columns": feat_cols,
        "split": {"val_year": val_year, "test_year": test_year},
        "metrics": metrics,
        "n_train": int(len(split.train_idx)),
        "n_val": int(len(split.val_idx)),
        "n_test": int(len(split.test_idx)),
    }

    joblib.dump(pipe, DELIVERABLES / "baseline_linear.pkl")
    with open(DELIVERABLES / "baseline_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main():
    df = load_processed()
    df = build_features(df)
    target = "water_level_cm"
    last_year = int(df["date"].dt.year.max())
    out = fit_baseline(df, target=target, val_year=last_year - 1, test_year=last_year)
    print(json.dumps(out["metrics"], indent=2))
    print(f"Baseline model saved to {DELIVERABLES / 'baseline_linear.pkl'}")


if __name__ == "__main__":
    main()
