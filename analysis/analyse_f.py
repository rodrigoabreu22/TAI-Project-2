#!/usr/bin/env python3
"""
Deep-dive analysis of File F: is there exploitable spatial correlation,
or is it at its entropy limit?

Compares:
  1. Current GAP predictor vs simpler alternatives
  2. Autocorrelation of raw pixels and residuals (horizontal + vertical)
  3. Row-mean profile (detects smooth gradients GAP might miss)
  4. Distribution shape (Gaussian noise vs structured residuals)
"""

import numpy as np
from pathlib import Path

WIDTH, HEIGHT = 1500, 1500
DATA_DIR = Path(__file__).parent / "data"


def load(name: str) -> np.ndarray:
    raw = np.frombuffer((DATA_DIR / name).read_bytes(), dtype=np.uint8)
    px = (raw[0::2].astype(np.uint16) << 8) | raw[1::2].astype(np.uint16)
    return px.reshape(HEIGHT, WIDTH)


def gap_predict(img: np.ndarray) -> np.ndarray:
    i = img.astype(np.int32)
    W  = np.zeros_like(i);  W[:, 1:]   = i[:, :-1]
    WW = W.copy();           WW[:, 2:]  = i[:, :-2]
    N  = W.copy();           N[1:, :]   = i[:-1, :]
    NN = N.copy();           NN[2:, :]  = i[:-2, :]
    NW = W.copy();           NW[1:, 1:] = i[:-1, :-1]
    NE = N.copy();           NE[1:, :-1]= i[:-1, 1:]
    THRESH = 128
    dh = np.abs(W - WW) + np.abs(N - NW) + np.abs(N - NE)
    dv = np.abs(N - NN) + np.abs(W - NW) + np.abs(NE - N)
    med = W + N - NW
    med = np.where(NW >= np.maximum(W, N), np.minimum(W, N), med)
    med = np.where(NW <= np.minimum(W, N), np.maximum(W, N), med)
    pred = np.where(dv - dh > THRESH, W, np.where(dh - dv > THRESH, N, med))
    pred[0, 0] = 0
    return pred


def autocorr(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Normalised autocorrelation at lags 1..max_lag for a 1D array."""
    x = x.astype(np.float64)
    x -= x.mean()
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag)
    return np.array([np.mean(x[lag:] * x[:-lag]) / var for lag in range(1, max_lag + 1)])


DIVIDER = "─" * 80

def main():
    for name in ["F", "A", "B"]:   # F = target; A and B as calibration references
        img = load(name)
        img32 = img.astype(np.int32)

        # ── Basic stats ────────────────────────────────────────────────────
        mean, std = float(img.mean()), float(img.std())
        print(f"\n{'═'*80}")
        print(f"  File {name}   mean={mean:.0f}  std={std:.1f}  "
              f"range={img.min()}–{img.max()}")
        print('═'*80)

        # ── Predictor comparison ───────────────────────────────────────────
        pred_gap  = gap_predict(img32)
        pred_W    = np.zeros_like(img32); pred_W[:, 1:] = img32[:, :-1]
        pred_N    = np.zeros_like(img32)
        pred_N[0, :] = img32[0, :]       # first row: predict self (residual=0)
        pred_N[1:, :] = img32[:-1, :]
        pred_mean = np.full_like(img32, int(round(mean)))

        print("\nPredictor comparison (MAE over all pixels):")
        for label, pred in [("GAP (current)", pred_gap),
                             ("W only",        pred_W),
                             ("N only",        pred_N),
                             ("Global mean",   pred_mean)]:
            resid = (img32 - pred) & 0xFFFF
            signed = np.where(resid <= 32767, resid, resid - 65536)
            mae = float(np.abs(signed).mean())
            rmse = float(np.sqrt(np.mean(signed.astype(np.float64)**2)))
            print(f"  {label:<20}  MAE={mae:8.2f}  RMSE={rmse:9.2f}")

        # ── Autocorrelation of raw pixels ──────────────────────────────────
        print("\nAutocorrelation of RAW pixels (lag 1–5):")
        # horizontal: along each row
        h_ac = np.mean([autocorr(img[r, :], 5) for r in range(0, HEIGHT, 10)], axis=0)
        # vertical: along each column
        v_ac = np.mean([autocorr(img[:, c], 5) for c in range(0, WIDTH, 10)], axis=0)
        print(f"  Horizontal (within row):  " +
              "  ".join(f"r{i+1}={h_ac[i]:+.4f}" for i in range(5)))
        print(f"  Vertical   (across rows): " +
              "  ".join(f"r{i+1}={v_ac[i]:+.4f}" for i in range(5)))

        # ── Autocorrelation of GAP residuals ───────────────────────────────
        resid_gap = (img32 - pred_gap) & 0xFFFF
        signed_gap = np.where(resid_gap <= 32767, resid_gap, resid_gap - 65536)
        signed_gap = signed_gap.astype(np.float64)

        print("\nAutocorrelation of GAP residuals (lag 1–5):")
        h_ac_r = np.mean([autocorr(signed_gap[r, :], 5) for r in range(0, HEIGHT, 10)], axis=0)
        v_ac_r = np.mean([autocorr(signed_gap[:, c], 5) for c in range(0, WIDTH, 10)], axis=0)
        print(f"  Horizontal (within row):  " +
              "  ".join(f"r{i+1}={h_ac_r[i]:+.4f}" for i in range(5)))
        print(f"  Vertical   (across rows): " +
              "  ".join(f"r{i+1}={v_ac_r[i]:+.4f}" for i in range(5)))

        # ── Row-mean and row-std profile (detect smooth gradients) ─────────
        row_means = img.mean(axis=1)
        row_stds  = img.std(axis=1)
        print(f"\nRow mean profile (every 150 rows):")
        print("  row  " + "  ".join(f"{r:4d}:{row_means[r]:7.1f}±{row_stds[r]:5.1f}"
                                     for r in range(0, HEIGHT, 150)))

        # ── Distribution shape: compare to Gaussian ────────────────────────
        s = signed_gap.ravel()
        kurtosis = float(np.mean((s - s.mean())**4) / np.std(s)**4)
        print(f"\nResidual distribution shape:")
        print(f"  mean={s.mean():.2f}  std={s.std():.2f}  "
              f"kurtosis={kurtosis:.3f}  (Gaussian=3.0)")
        # Percentile spread
        p = np.percentile(np.abs(s), [50, 75, 90, 99])
        print(f"  |residual| percentiles: 50%={p[0]:.0f}  75%={p[1]:.0f}  "
              f"90%={p[2]:.0f}  99%={p[3]:.0f}")

    print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    main()
