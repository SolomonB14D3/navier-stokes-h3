#!/usr/bin/env python3
"""
TEST: ALIGNMENT-STRETCHING CORRELATION

Key hypothesis: Regions with higher icosahedral alignment should have LESS stretching.

If true → geometric constraint is the MECHANISM for reduced stretching
If false → alignment and stretching are independent

Expected: NEGATIVE correlation (high alignment → low stretching)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import time
import mlx.core as mx

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4

# Icosahedral axes
ICOSA_5FOLD = np.array([
    [1, PHI, 0], [-1, PHI, 0], [1, -PHI, 0], [-1, -PHI, 0],
    [0, 1, PHI], [0, -1, PHI], [0, 1, -PHI], [0, -1, -PHI],
    [PHI, 0, 1], [-PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, -1]
]) / np.sqrt(1 + PHI**2)

ICOSA_2FOLD = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [PHI, 1/PHI, 0], [PHI, -1/PHI, 0], [-PHI, 1/PHI, 0], [-PHI, -1/PHI, 0],
    [0, PHI, 1/PHI], [0, PHI, -1/PHI], [0, -PHI, 1/PHI], [0, -PHI, -1/PHI],
    [1/PHI, 0, PHI], [-1/PHI, 0, PHI], [1/PHI, 0, -PHI], [-1/PHI, 0, -PHI]
])
ICOSA_2FOLD = ICOSA_2FOLD / np.linalg.norm(ICOSA_2FOLD, axis=1, keepdims=True)
ICOSA_AXES = np.vstack([ICOSA_5FOLD, ICOSA_2FOLD])

print(f"Testing correlation between icosahedral alignment and vortex stretching")


class NSSolver:
    def __init__(self, n=64, nu=0.001, dt=0.0001):
        self.n, self.nu, self.dt = n, nu, dt
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)
        self.visc = mx.exp(-nu * self.k2 * dt)

    def get_fields(self, wh):
        """Get velocity and vorticity in physical space."""
        wx_hat, wy_hat, wz_hat = wh
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe

        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        return (ux_hat, uy_hat, uz_hat), (wx, wy, wz)

    def compute_stretching_field(self, wh):
        """Compute local stretching S = ω_i S_ij ω_j / |ω|² at each point."""
        (ux_hat, uy_hat, uz_hat), (wx, wy, wz) = self.get_fields(wh)

        # Velocity gradients
        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real

        # Strain tensor components
        S_xx, S_yy, S_zz = dux_dx, duy_dy, duz_dz
        S_xy = 0.5 * (dux_dy + duy_dx)
        S_xz = 0.5 * (dux_dz + duz_dx)
        S_yz = 0.5 * (duy_dz + duz_dy)

        # ω_i S_ij ω_j
        omega_S_omega = (wx * (S_xx * wx + S_xy * wy + S_xz * wz) +
                         wy * (S_xy * wx + S_yy * wy + S_yz * wz) +
                         wz * (S_xz * wx + S_yz * wy + S_zz * wz))

        omega_sq = wx**2 + wy**2 + wz**2

        # Normalized stretching (avoid division by zero)
        S_normalized = omega_S_omega / (omega_sq + 1e-10)

        return np.array(S_normalized), np.array(omega_sq)

    def compute_icosa_alignment(self, wh):
        """Compute alignment of vorticity to nearest icosahedral axis."""
        _, (wx, wy, wz) = self.get_fields(wh)

        omega = np.stack([np.array(wx), np.array(wy), np.array(wz)], axis=-1)
        omega_norm = np.linalg.norm(omega, axis=-1, keepdims=True)
        omega_hat = omega / (omega_norm + 1e-10)

        # Max |cos θ| to any icosahedral axis
        # Shape: (n, n, n, 3) @ (27, 3).T = (n, n, n, 27)
        dots = np.abs(omega_hat @ ICOSA_AXES.T)
        cos_theta = np.max(dots, axis=-1)  # (n, n, n)

        return cos_theta, np.squeeze(omega_norm)

    def step(self, wh):
        wx_hat, wy_hat, wz_hat = wh
        (ux_hat, uy_hat, uz_hat), (wx, wy, wz) = self.get_fields(wh)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real

        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        nlx = nlx + wx * mx.fft.ifftn(1j * self.kx * ux_hat).real + wy * mx.fft.ifftn(1j * self.ky * ux_hat).real + wz * mx.fft.ifftn(1j * self.kz * ux_hat).real
        nly = nly + wx * mx.fft.ifftn(1j * self.kx * uy_hat).real + wy * mx.fft.ifftn(1j * self.ky * uy_hat).real + wz * mx.fft.ifftn(1j * self.kz * uy_hat).real
        nlz = nlz + wx * mx.fft.ifftn(1j * self.kx * uz_hat).real + wy * mx.fft.ifftn(1j * self.ky * uz_hat).real + wz * mx.fft.ifftn(1j * self.kz * uz_hat).real

        return ((wx_hat + self.dt * mx.fft.fftn(nlx)) * self.visc,
                (wy_hat + self.dt * mx.fft.fftn(nly)) * self.visc,
                (wz_hat + self.dt * mx.fft.fftn(nlz)) * self.visc)


def taylor_green(n, scale=5.0):
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return np.stack([scale * np.cos(X) * np.sin(Y) * np.sin(Z),
                     scale * np.sin(X) * np.cos(Y) * np.sin(Z),
                     -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  ALIGNMENT-STRETCHING CORRELATION TEST")
    print("=" * 70)
    print("""
  Hypothesis: Higher icosahedral alignment → LESS stretching

  If correlation is NEGATIVE: Geometric constraint is the mechanism
  If correlation is ~0 or positive: Alignment doesn't reduce stretching
    """)

    n = 64
    solver = NSSolver(n=n)
    omega_init = taylor_green(n, scale=5.0)
    wh = tuple(mx.fft.fftn(mx.array(omega_init[..., i])) for i in range(3))

    # Run to high-enstrophy state
    print("Evolving to high-enstrophy state...")
    for step in range(8000):
        wh = solver.step(wh)
        mx.eval(*wh)
        if step % 2000 == 0:
            Z = 0.5 * float(mx.mean(sum(mx.fft.ifftn(h).real**2 for h in wh)).item())
            print(f"  step {step}, Z = {Z:.1f}")

    # Measure at peak
    print("\nMeasuring alignment-stretching correlation...")

    S_field, omega_sq = solver.compute_stretching_field(wh)
    cos_theta, omega_norm = solver.compute_icosa_alignment(wh)

    # Focus on high-vorticity regions
    threshold = 2.0 * np.mean(omega_norm)
    mask = omega_norm > threshold

    if np.sum(mask) < 1000:
        print("Not enough high-vorticity points!")
        return

    flat_cos = cos_theta[mask].flatten()
    flat_S = S_field[mask].flatten()
    flat_omega = omega_norm[mask].flatten()

    # Remove NaN/inf
    valid = np.isfinite(flat_cos) & np.isfinite(flat_S)
    flat_cos = flat_cos[valid]
    flat_S = flat_S[valid]

    print(f"\nAnalyzing {len(flat_cos)} high-vorticity points")

    # Correlation
    corr_pearson, p_pearson = pearsonr(flat_cos, flat_S)
    corr_spearman, p_spearman = spearmanr(flat_cos, flat_S)

    print(f"\n  Pearson correlation:  r = {corr_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"  Spearman correlation: ρ = {corr_spearman:.4f} (p = {p_spearman:.2e})")

    # Binned analysis
    bins = np.linspace(np.min(flat_cos), np.max(flat_cos), 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_S_per_bin = []
    std_S_per_bin = []

    for i in range(len(bins) - 1):
        in_bin = (flat_cos >= bins[i]) & (flat_cos < bins[i+1])
        if np.sum(in_bin) > 10:
            mean_S_per_bin.append(np.mean(flat_S[in_bin]))
            std_S_per_bin.append(np.std(flat_S[in_bin]))
        else:
            mean_S_per_bin.append(np.nan)
            std_S_per_bin.append(np.nan)

    print(f"\n  Mean S per cos θ bin:")
    for i, (c, s) in enumerate(zip(bin_centers, mean_S_per_bin)):
        print(f"    cos θ = {c:.3f}: S = {s:.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    if corr_pearson < -0.1 and p_pearson < 0.01:
        print(f"\n  ✓ NEGATIVE correlation (r = {corr_pearson:.3f})")
        print("  Higher icosahedral alignment → LESS stretching")
        print("  This confirms the geometric constraint mechanism!")
    elif corr_pearson > 0.1 and p_pearson < 0.01:
        print(f"\n  ✗ POSITIVE correlation (r = {corr_pearson:.3f})")
        print("  Higher alignment → MORE stretching (unexpected)")
    else:
        print(f"\n  ~ WEAK/NO correlation (r = {corr_pearson:.3f})")
        print("  Alignment and stretching appear independent")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot
    ax = axes[0, 0]
    idx = np.random.choice(len(flat_cos), min(5000, len(flat_cos)), replace=False)
    ax.scatter(flat_cos[idx], flat_S[idx], alpha=0.3, s=5)
    ax.set_xlabel('Icosahedral Alignment (cos θ)')
    ax.set_ylabel('Local Stretching S')
    ax.set_title(f'Alignment vs Stretching (r = {corr_pearson:.3f})')
    ax.grid(True, alpha=0.3)

    # Binned mean
    ax = axes[0, 1]
    ax.errorbar(bin_centers, mean_S_per_bin, yerr=std_S_per_bin,
                fmt='o-', capsize=3, linewidth=2)
    ax.set_xlabel('Icosahedral Alignment (cos θ)')
    ax.set_ylabel('Mean Stretching S')
    ax.set_title('Binned Mean Stretching vs Alignment')
    ax.grid(True, alpha=0.3)

    # Histograms
    ax = axes[1, 0]
    ax.hist(flat_cos, bins=50, density=True, alpha=0.7)
    ax.set_xlabel('Icosahedral Alignment (cos θ)')
    ax.set_ylabel('Density')
    ax.set_title('Alignment Distribution')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(flat_S, bins=50, density=True, alpha=0.7)
    ax.set_xlabel('Local Stretching S')
    ax.set_ylabel('Density')
    ax.set_title('Stretching Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_alignment_stretching_correlation.png', dpi=150)
    print("\nSaved: test_alignment_stretching_correlation.png")

    return {'corr': corr_pearson, 'p': p_pearson, 'bins': bin_centers, 'mean_S': mean_S_per_bin}


if __name__ == "__main__":
    results = main()
