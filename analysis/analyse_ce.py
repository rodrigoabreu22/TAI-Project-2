#!/usr/bin/env python3
"""
Deep-dive analysis of Files C and E to understand why H(residuals) > H(raw pixels).
Checks autocorrelation, predictor comparison, and distribution shape.
Includes File G as a control (similar stats to C/E but predictor seems to help).
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


def entropy_bits(vals: np.ndarray, nbins: int) -> float:
    counts, _ = np.histogram(vals.ravel(), bins=nbins, range=(0, nbins))
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def autocorr(x: np.ndarray, max_lag: int = 5) -> np.ndarray:
    x = x.astype(np.float64) - x.mean()
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag)
    return np.array([np.mean(x[lag:] * x[:-lag]) / var for lag in range(1, max_lag + 1)])


def main():
    # C and E are the focus; G is a control (same easy profile but predictor helps)
    for name in ["C", "E", "G", "D"]:
        img = load(name)
        img32 = img.astype(np.int32)
        mean_px = float(img.mean())
        std_px  = float(img.std())

        print(f"\n{'═'*80}")
        print(f"  File {name}   mean={mean_px:.1f}  std={std_px:.2f}  "
              f"range={img.min()}–{img.max()}")
        print('═'*80)

        # ── Predictor comparison ───────────────────────────────────────────
        pred_gap  = gap_predict(img32)
        pred_W    = np.zeros_like(img32); pred_W[:, 1:] = img32[:, :-1]
        pred_N    = np.zeros_like(img32); pred_N[0, :] = img32[0, :]; pred_N[1:, :] = img32[:-1, :]
        pred_mean = np.full_like(img32, int(round(mean_px)))

        print("\nPredictor comparison (MAE over all pixels):")
        for label, pred in [("GAP (current)", pred_gap),
                             ("W only",        pred_W),
                             ("N only",        pred_N),
                             ("Global mean",   pred_mean)]:
            resid = (img32 - pred) & 0xFFFF
            signed = np.where(resid <= 32767, resid, resid - 65536)
            mae  = float(np.abs(signed).mean())
            rmse = float(np.sqrt(np.mean(signed.astype(np.float64)**2)))
            print(f"  {label:<20}  MAE={mae:7.3f}  RMSE={rmse:8.3f}")

        # ── Raw pixel entropy vs residual entropy ──────────────────────────
        # Raw pixels: H(px) — this is what direct entropy coding of raw pixels achieves per pixel
        H_raw = entropy_bits(img, nbins=65536)

        # GAP residuals after zigzag split into hi+lo
        resid_gap = (img32 - pred_gap) & 0xFFFF
        z = np.where(resid_gap <= 32767, resid_gap * 2, (65536 - resid_gap) * 2 - 1).astype(np.uint16)
        hi = (z >> 8).astype(np.uint8)
        lo = (z & 0xFF).astype(np.uint8)
        frac_hi0 = float((hi == 0).mean())
        H_hi = entropy_bits(hi, 256)
        H_lo_hi0 = entropy_bits(lo[hi == 0], 256) if (hi == 0).any() else 0.0
        H_lo_hip = entropy_bits(lo[hi > 0],  256) if (hi > 0).any() else 0.0
        H_resid_per_px = H_hi + frac_hi0 * H_lo_hi0 + (1 - frac_hi0) * H_lo_hip

        print(f"\nEntropy analysis (bits per pixel):")
        print(f"  H(raw pixels):              {H_raw:.4f} bpp  →  bpb lower bound = {H_raw/2:.4f}")
        print(f"  H(GAP residuals, hi+lo):    {H_resid_per_px:.4f} bpp  →  est bpb = {H_resid_per_px/2:.4f}")
        print(f"  Overhead vs direct coding:  {(H_resid_per_px - H_raw):+.4f} bpp  "
              f"({'PREDICTOR HURTS' if H_resid_per_px > H_raw else 'predictor helps'})")
        print(f"  frac hi=0: {frac_hi0:.4f}   H(hi)={H_hi:.4f}   "
              f"H(lo|hi=0)={H_lo_hi0:.4f}   H(lo|hi>0)={H_lo_hip:.4f}")

        # ── Autocorrelation of RAW pixels ──────────────────────────────────
        print("\nAutocorrelation of RAW pixels (lag 1–5):")
        h_ac = np.mean([autocorr(img[r, :]) for r in range(0, HEIGHT, 10)], axis=0)
        v_ac = np.mean([autocorr(img[:, c]) for c in range(0, WIDTH, 10)], axis=0)
        print(f"  Horizontal: " + "  ".join(f"r{i+1}={h_ac[i]:+.4f}" for i in range(5)))
        print(f"  Vertical:   " + "  ".join(f"r{i+1}={v_ac[i]:+.4f}" for i in range(5)))

        # ── Autocorrelation of GAP residuals ───────────────────────────────
        signed_gap = np.where(resid_gap <= 32767, resid_gap, resid_gap - 65536).astype(np.float64)
        print("\nAutocorrelation of GAP residuals (lag 1–5):")
        h_ac_r = np.mean([autocorr(signed_gap[r, :]) for r in range(0, HEIGHT, 10)], axis=0)
        v_ac_r = np.mean([autocorr(signed_gap[:, c]) for c in range(0, WIDTH, 10)], axis=0)
        print(f"  Horizontal: " + "  ".join(f"r{i+1}={h_ac_r[i]:+.4f}" for i in range(5)))
        print(f"  Vertical:   " + "  ".join(f"r{i+1}={v_ac_r[i]:+.4f}" for i in range(5)))

        # ── Distribution shape ─────────────────────────────────────────────
        s = signed_gap.ravel()
        kurtosis = float(np.mean((s - s.mean())**4) / np.std(s)**4)
        print(f"\nResidual distribution:  mean={s.mean():.3f}  std={s.std():.3f}  "
              f"kurtosis={kurtosis:.1f}  (Gaussian=3.0, Laplacian=6.0)")
        p = np.percentile(np.abs(s), [50, 90, 99, 99.9])
        print(f"  |residual| pct:  50%={p[0]:.1f}  90%={p[1]:.1f}  "
              f"99%={p[2]:.1f}  99.9%={p[3]:.1f}")

        # ── Row mean profile ────────────────────────────────────────────────
        row_means = img.mean(axis=1)
        row_stds  = img.std(axis=1)
        print(f"\nRow mean profile (every 150 rows):")
        print("  " + "  ".join(f"r{r}:{row_means[r]:.1f}±{row_stds[r]:.1f}"
                                for r in range(0, HEIGHT, 150)))

    print()


if __name__ == "__main__":
    main()
