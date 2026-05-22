#!/usr/bin/env python3
"""
Analyse raw 16-bit astronomical images (A–H).

Mirrors the compressor pipeline exactly (GAP predictor, zigzag, hi/lo split,
16-context hi model) so entropy estimates are directly comparable to the
actual benchmark bpb figures.
"""

import math
import numpy as np
from pathlib import Path

WIDTH, HEIGHT = 1500, 1500
DATA_DIR = Path(__file__).parent / "data"
FILES = list("ABCDEFGH")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load(name: str) -> np.ndarray:
    raw = np.frombuffer((DATA_DIR / name).read_bytes(), dtype=np.uint8)
    px = (raw[0::2].astype(np.uint16) << 8) | raw[1::2].astype(np.uint16)
    return px.reshape(HEIGHT, WIDTH)


# ---------------------------------------------------------------------------
# Vectorised GAP predictor (mirrors C++ exactly)
# ---------------------------------------------------------------------------

def make_neighbors(img: np.ndarray):
    """Return (W, WW, N, NN, NW, NE) neighbor arrays with correct boundaries."""
    i = img.astype(np.int32)

    W  = np.zeros_like(i);  W[:, 1:]   = i[:, :-1]
    WW = W.copy();           WW[:, 2:]  = i[:, :-2]
    N  = W.copy();           N[1:, :]   = i[:-1, :]   # N=W for row 0
    NN = N.copy();           NN[2:, :]  = i[:-2, :]   # NN=N for row 0-1
    NW = W.copy();           NW[1:, 1:] = i[:-1, :-1] # NW=W for row/col 0
    NE = N.copy();           NE[1:, :-1]= i[:-1, 1:]  # NE=N for row 0 or last col

    return W, WW, N, NN, NW, NE


def gap_predict(img: np.ndarray) -> np.ndarray:
    W, WW, N, NN, NW, NE = make_neighbors(img.astype(np.int32))
    THRESH = 128
    dh = np.abs(W - WW) + np.abs(N - NW) + np.abs(N - NE)
    dv = np.abs(N - NN) + np.abs(W - NW) + np.abs(NE - N)

    # MED
    med = W + N - NW
    med = np.where(NW >= np.maximum(W, N), np.minimum(W, N), med)
    med = np.where(NW <= np.minimum(W, N), np.maximum(W, N), med)

    pred = np.where(dv - dh > THRESH, W, np.where(dh - dv > THRESH, N, med))
    pred[0, 0] = 0
    return pred


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def entropy(arr: np.ndarray, nbins: int = 256) -> float:
    counts, _ = np.histogram(arr.ravel(), bins=nbins, range=(0, nbins))
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def cond_entropy(vals: np.ndarray, ctx: np.ndarray, n_ctx: int) -> float:
    """H(vals | ctx)  — weighted sum of per-context entropies."""
    total = len(vals)
    h = 0.0
    for c in range(n_ctx):
        mask = ctx == c
        n = mask.sum()
        if n == 0:
            continue
        h += (n / total) * entropy(vals[mask])
    return h


def quantize_hi(h: np.ndarray) -> np.ndarray:
    q = np.zeros_like(h, dtype=np.int32)
    q[h > 0] = 1; q[h > 2] = 2; q[h > 7] = 3
    return q


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyse(name: str) -> dict:
    img = load(name)
    img32 = img.astype(np.int32)

    # ── Pixel-level stats ──────────────────────────────────────────────────
    pmin, pmax = int(img.min()), int(img.max())
    pmean, pstd = float(img.mean()), float(img.std())
    bits_used = int(math.ceil(math.log2(pmax + 1))) if pmax > 0 else 1
    raw_entropy = entropy(img, nbins=65536)   # bits per pixel of raw values

    # ── Residuals after GAP ────────────────────────────────────────────────
    pred  = gap_predict(img32)
    resid = (img32 - pred) & 0xFFFF           # wrapping uint16

    # Zigzag
    z  = np.where(resid <= 32767, resid * 2, (65536 - resid) * 2 - 1)
    z  = z.astype(np.uint16)
    hi = (z >> 8).astype(np.uint8)
    lo = (z & 0xFF).astype(np.uint8)

    # ── Hi-byte analysis ───────────────────────────────────────────────────
    frac_hi0 = float((hi == 0).mean())
    H_hi = entropy(hi)                         # unconditional H(hi)

    # 16-context model
    hi2d  = hi.reshape(HEIGHT, WIDTH)
    hi_W  = np.zeros_like(hi2d); hi_W[:, 1:]  = hi2d[:, :-1]
    hi_N  = np.zeros_like(hi2d); hi_N[1:, :]  = hi2d[:-1, :]
    ctx16 = (quantize_hi(hi_W) * 4 + quantize_hi(hi_N)).ravel()  # 0..15

    H_hi_ctx16 = cond_entropy(hi.ravel(), ctx16, 16)
    ctx_gain   = H_hi - H_hi_ctx16              # bits/pixel saved by 16 contexts

    # Context population balance (are all 16 contexts actually used?)
    ctx_counts = np.bincount(ctx16, minlength=16)
    ctx_frac   = ctx_counts / ctx_counts.sum()

    # ── Lo-byte analysis ───────────────────────────────────────────────────
    lo_hi0 = lo[hi == 0]
    lo_hip = lo[hi > 0]
    H_lo_hi0 = entropy(lo_hi0) if len(lo_hi0) > 0 else 0.0
    H_lo_hip = entropy(lo_hip) if len(lo_hip) > 0 else 0.0

    # Estimated achievable bpb (lower bound; assumes perfect per-context coding)
    # bits per pixel = H_hi_ctx16 + frac_hi0*H_lo_hi0 + (1-frac_hi0)*H_lo_hip
    # bpb = bits_per_pixel / 2  (original file has 2 bytes per pixel)
    est_bpb = (H_hi_ctx16 + frac_hi0 * H_lo_hi0 + (1 - frac_hi0) * H_lo_hip) / 2

    # ── Residual shape ─────────────────────────────────────────────────────
    signed = np.where(resid <= 32767, resid.astype(np.int32),
                      resid.astype(np.int32) - 65536)
    mae  = float(np.abs(signed).mean())
    # Fraction of residuals that are exactly 0
    frac_zero = float((signed == 0).mean())

    # ── Per-context hi entropy spread (min/max) ────────────────────────────
    ctx_hi_entropies = []
    for c in range(16):
        mask = ctx16 == c
        if mask.sum() > 0:
            ctx_hi_entropies.append(entropy(hi.ravel()[mask]))
    ctx_h_min = min(ctx_hi_entropies)
    ctx_h_max = max(ctx_hi_entropies)

    return dict(
        name=name,
        pmin=pmin, pmax=pmax, pmean=pmean, pstd=pstd,
        bits_used=bits_used,
        raw_entropy=raw_entropy,
        frac_hi0=frac_hi0,
        frac_zero=frac_zero,
        mae=mae,
        H_hi=H_hi,
        H_hi_ctx16=H_hi_ctx16,
        ctx_gain=ctx_gain,
        ctx_h_min=ctx_h_min,
        ctx_h_max=ctx_h_max,
        H_lo_hi0=H_lo_hi0,
        H_lo_hip=H_lo_hip,
        est_bpb=est_bpb,
        ctx_frac=ctx_frac,
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

DIVIDER = "─" * 110

def print_results(results: list[dict]):
    print("\n" + DIVIDER)
    print("PIXEL VALUE STATISTICS")
    print(DIVIDER)
    print(f"{'File':>4}  {'min':>6}  {'max':>6}  {'mean':>8}  {'std':>8}  "
          f"{'bits used':>9}  {'raw entropy':>11}")
    print(DIVIDER)
    for r in results:
        print(f"{r['name']:>4}  {r['pmin']:>6}  {r['pmax']:>6}  "
              f"{r['pmean']:>8.1f}  {r['pstd']:>8.1f}  "
              f"{r['bits_used']:>9}  {r['raw_entropy']:>11.4f} bpp")

    print("\n" + DIVIDER)
    print("RESIDUAL ANALYSIS (after GAP predictor)")
    print(DIVIDER)
    print(f"{'File':>4}  {'frac hi=0':>9}  {'frac res=0':>10}  {'MAE resid':>10}  "
          f"{'H(hi)':>7}  {'H(hi|ctx16)':>12}  {'ctx gain':>9}")
    print(DIVIDER)
    for r in results:
        print(f"{r['name']:>4}  {r['frac_hi0']:>9.3f}  {r['frac_zero']:>10.3f}  "
              f"{r['mae']:>10.2f}  "
              f"{r['H_hi']:>7.4f}  {r['H_hi_ctx16']:>12.4f}  "
              f"{r['ctx_gain']:>9.4f} b/px")

    print("\n" + DIVIDER)
    print("ENTROPY BUDGET & BPB ESTIMATE")
    print(DIVIDER)
    print(f"{'File':>4}  {'H(lo|hi=0)':>10}  {'H(lo|hi>0)':>10}  "
          f"{'est bpb':>8}  {'ctx_h spread':>13}")
    print(DIVIDER)
    for r in results:
        spread = f"{r['ctx_h_min']:.3f}–{r['ctx_h_max']:.3f}"
        print(f"{r['name']:>4}  {r['H_lo_hi0']:>10.4f}  {r['H_lo_hip']:>10.4f}  "
              f"{r['est_bpb']:>8.4f}  {spread:>13}")

    print("\n" + DIVIDER)
    print("CONTEXT POPULATION (fraction of pixels in each of 16 hi contexts)")
    print("ctx = quantize(hi_W)*4 + quantize(hi_N)   [q: 0=hi0, 1=hi1-2, 2=hi3-7, 3=hi8+]")
    print(DIVIDER)
    header = f"{'File':>4}  " + "  ".join(f"c{i:02d}" for i in range(16))
    print(header)
    print(DIVIDER)
    for r in results:
        row = f"{r['name']:>4}  " + "  ".join(f"{v:.3f}" for v in r['ctx_frac'])
        print(row)

    print("\n" + DIVIDER)
    print("SUMMARY")
    print(DIVIDER)
    est_bpbs = [r['est_bpb'] for r in results]
    ctx_gains = [r['ctx_gain'] for r in results]
    frac_hi0s = [r['frac_hi0'] for r in results]
    print(f"  Estimated bpb range:  {min(est_bpbs):.4f} – {max(est_bpbs):.4f}  "
          f"(mean {sum(est_bpbs)/len(est_bpbs):.4f})")
    print(f"  Actual bpb (benchmk): 3.5950  (all 8 files combined)")
    print(f"  Context gain range:   {min(ctx_gains):.4f} – {max(ctx_gains):.4f} b/px "
          f"(mean {sum(ctx_gains)/len(ctx_gains):.4f})")
    print(f"  Fraction hi=0 range:  {min(frac_hi0s):.3f} – {max(frac_hi0s):.3f} "
          f"(mean {sum(frac_hi0s)/len(frac_hi0s):.3f})")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    print("Loading and analysing files", end="", flush=True)
    results = []
    for name in FILES:
        r = analyse(name)
        results.append(r)
        print(f" {name}", end="", flush=True)
    print()
    print_results(results)
