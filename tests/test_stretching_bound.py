#!/usr/bin/env python3
"""
TEST: IS STRETCHING BOUNDED IN ALIGNED REGIONS?

The proof claims: alignment to icosahedral axes BOUNDS (not reduces) stretching.

Better test: For regions with high alignment, is max(S) bounded?
Compare max stretching in:
- Highly aligned regions (cos θ > 0.95)
- Poorly aligned regions (cos θ < 0.9)

If proof is correct: max(S) should be LOWER in aligned regions despite
mean(S) potentially being higher (because that's where vortices are).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import mlx.core as mx

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
        wx_hat, wy_hat, wz_hat = wh
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real
        return (ux_hat, uy_hat, uz_hat), (wx, wy, wz)

    def compute_strain_alignment(self, wh):
        """Compute vorticity-STRAIN alignment A = (ω^T S ω) / (|ω|² ||S||)."""
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

        # Strain tensor
        S_xx, S_yy, S_zz = dux_dx, duy_dy, duz_dz
        S_xy = 0.5 * (dux_dy + duy_dx)
        S_xz = 0.5 * (dux_dz + duz_dx)
        S_yz = 0.5 * (duy_dz + duz_dy)

        # ω^T S ω
        omega_S_omega = (wx * (S_xx * wx + S_xy * wy + S_xz * wz) +
                         wy * (S_xy * wx + S_yy * wy + S_yz * wz) +
                         wz * (S_xz * wx + S_yz * wy + S_zz * wz))

        omega_sq = wx**2 + wy**2 + wz**2
        S_norm = mx.sqrt(S_xx**2 + S_yy**2 + S_zz**2 +
                         2*(S_xy**2 + S_xz**2 + S_yz**2))

        # Alignment factor A = (ω^T S ω) / (|ω|² ||S||)
        A = omega_S_omega / (omega_sq * S_norm + 1e-10)

        return np.array(A), np.array(omega_sq), np.array(omega_S_omega)

    def compute_icosa_alignment(self, wh):
        _, (wx, wy, wz) = self.get_fields(wh)
        omega = np.stack([np.array(wx), np.array(wy), np.array(wz)], axis=-1)
        omega_norm = np.linalg.norm(omega, axis=-1, keepdims=True)
        omega_hat = omega / (omega_norm + 1e-10)
        dots = np.abs(omega_hat @ ICOSA_AXES.T)
        cos_theta = np.max(dots, axis=-1)
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
    print("  STRETCHING BOUND TEST")
    print("=" * 70)
    print(f"""
  Better test: Is the MAXIMUM stretching bounded in aligned regions?

  The proof claims alignment BOUNDS stretching, not reduces it.
  Bound: A ≤ (1 - δ₀) = {1 - DELTA_0:.4f}

  Compare max(A) in highly-aligned vs poorly-aligned regions.
    """)

    n = 64
    solver = NSSolver(n=n)
    omega_init = taylor_green(n, scale=5.0)
    wh = tuple(mx.fft.fftn(mx.array(omega_init[..., i])) for i in range(3))

    print("Evolving flow...")
    for step in range(8000):
        wh = solver.step(wh)
        mx.eval(*wh)
        if step % 2000 == 0:
            Z = 0.5 * float(mx.mean(sum(mx.fft.ifftn(h).real**2 for h in wh)).item())
            print(f"  step {step}, Z = {Z:.1f}")

    print("\nAnalyzing stretching bounds...")

    # Get strain alignment (this is the A from the proof)
    A, omega_sq, omega_S_omega = solver.compute_strain_alignment(wh)

    # Get icosahedral alignment
    cos_theta, omega_norm = solver.compute_icosa_alignment(wh)

    # Focus on high-vorticity regions
    threshold = 2.0 * np.mean(omega_norm)
    mask = omega_norm > threshold

    A_flat = A[mask].flatten()
    cos_theta_flat = cos_theta[mask].flatten()
    omega_flat = omega_norm[mask].flatten()

    # Remove invalid
    valid = np.isfinite(A_flat) & np.isfinite(cos_theta_flat)
    A_flat = A_flat[valid]
    cos_theta_flat = cos_theta_flat[valid]

    print(f"\nAnalyzing {len(A_flat)} high-vorticity points")
    print(f"Theoretical bound on A: {1 - DELTA_0:.4f}")

    # Compare highly-aligned vs less-aligned
    high_align_mask = cos_theta_flat > 0.98
    mid_align_mask = (cos_theta_flat > 0.9) & (cos_theta_flat <= 0.98)
    low_align_mask = cos_theta_flat <= 0.9

    print(f"\n  Regions with cos θ > 0.98 (highly aligned):")
    if np.sum(high_align_mask) > 10:
        A_high = A_flat[high_align_mask]
        print(f"    N = {np.sum(high_align_mask)}")
        print(f"    max(A) = {np.max(A_high):.4f}")
        print(f"    99th percentile = {np.percentile(A_high, 99):.4f}")
        print(f"    mean(A) = {np.mean(A_high):.4f}")
        print(f"    Fraction > (1-δ₀): {np.mean(A_high > 1 - DELTA_0):.2%}")
    else:
        print("    Insufficient points")

    print(f"\n  Regions with 0.9 < cos θ ≤ 0.98 (mid aligned):")
    if np.sum(mid_align_mask) > 10:
        A_mid = A_flat[mid_align_mask]
        print(f"    N = {np.sum(mid_align_mask)}")
        print(f"    max(A) = {np.max(A_mid):.4f}")
        print(f"    99th percentile = {np.percentile(A_mid, 99):.4f}")
        print(f"    mean(A) = {np.mean(A_mid):.4f}")
        print(f"    Fraction > (1-δ₀): {np.mean(A_mid > 1 - DELTA_0):.2%}")
    else:
        print("    Insufficient points")

    print(f"\n  All high-vorticity regions:")
    print(f"    max(A) = {np.max(A_flat):.4f}")
    print(f"    99th percentile = {np.percentile(A_flat, 99):.4f}")
    print(f"    Fraction > (1-δ₀): {np.mean(A_flat > 1 - DELTA_0):.2%}")

    # Key test: Is max(A) LOWER in highly aligned regions?
    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)

    if np.sum(high_align_mask) > 10 and np.sum(mid_align_mask) > 10:
        max_A_high = np.max(A_flat[high_align_mask])
        max_A_mid = np.max(A_flat[mid_align_mask])

        print(f"\n  Max stretching (A):")
        print(f"    Highly aligned (cos θ > 0.98): {max_A_high:.4f}")
        print(f"    Mid aligned (cos θ < 0.98):    {max_A_mid:.4f}")
        print(f"    Theoretical bound (1-δ₀):      {1 - DELTA_0:.4f}")

        if max_A_high < max_A_mid:
            print(f"\n  ✓ Max stretching LOWER in aligned regions!")
            print(f"    Reduction: {(1 - max_A_high/max_A_mid)*100:.1f}%")
        else:
            print(f"\n  ✗ Max stretching NOT lower in aligned regions")

        # Does alignment correlate with respecting the bound?
        frac_exceed_high = np.mean(A_flat[high_align_mask] > 1 - DELTA_0)
        frac_exceed_mid = np.mean(A_flat[mid_align_mask] > 1 - DELTA_0)

        print(f"\n  Fraction exceeding (1-δ₀) bound:")
        print(f"    Highly aligned: {frac_exceed_high:.2%}")
        print(f"    Mid aligned:    {frac_exceed_mid:.2%}")

        if frac_exceed_high < frac_exceed_mid:
            print(f"\n  ✓ Aligned regions BETTER respect the bound!")
        else:
            print(f"\n  ~ Alignment doesn't reduce bound violations")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    idx = np.random.choice(len(A_flat), min(5000, len(A_flat)), replace=False)
    sc = ax.scatter(cos_theta_flat[idx], A_flat[idx], c=omega_flat[valid][idx],
                    alpha=0.4, s=10, cmap='viridis')
    ax.axhline(1 - DELTA_0, color='r', linestyle='--', label=f'Bound = {1-DELTA_0:.3f}')
    ax.set_xlabel('Icosahedral Alignment (cos θ)')
    ax.set_ylabel('Vorticity-Strain Alignment (A)')
    ax.set_title('A vs Icosahedral Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='|ω|')

    ax = axes[0, 1]
    bins = np.linspace(0.85, 1.0, 16)
    for i in range(len(bins)-1):
        in_bin = (cos_theta_flat >= bins[i]) & (cos_theta_flat < bins[i+1])
        if np.sum(in_bin) > 10:
            ax.scatter(bins[i] + 0.005, np.max(A_flat[in_bin]), s=50, c='blue')
            ax.scatter(bins[i] + 0.005, np.percentile(A_flat[in_bin], 99),
                       s=30, c='green', marker='^')
    ax.axhline(1 - DELTA_0, color='r', linestyle='--', label=f'Bound = {1-DELTA_0:.3f}')
    ax.set_xlabel('Icosahedral Alignment (cos θ)')
    ax.set_ylabel('Max A (blue) / 99th pct (green)')
    ax.set_title('Max Stretching per Alignment Bin')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(A_flat[high_align_mask] if np.sum(high_align_mask) > 10 else [],
            bins=50, density=True, alpha=0.5, label='cos θ > 0.98')
    ax.hist(A_flat[mid_align_mask] if np.sum(mid_align_mask) > 10 else [],
            bins=50, density=True, alpha=0.5, label='cos θ ≤ 0.98')
    ax.axvline(1 - DELTA_0, color='r', linestyle='--', label=f'Bound')
    ax.set_xlabel('Vorticity-Strain Alignment (A)')
    ax.set_ylabel('Density')
    ax.set_title('A Distribution by Alignment Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    exceed = A_flat > 1 - DELTA_0
    ax.scatter(cos_theta_flat[~exceed], A_flat[~exceed], alpha=0.3, s=5, c='blue', label='Below bound')
    ax.scatter(cos_theta_flat[exceed], A_flat[exceed], alpha=0.5, s=15, c='red', label='Exceeds bound')
    ax.axhline(1 - DELTA_0, color='k', linestyle='--')
    ax.set_xlabel('Icosahedral Alignment')
    ax.set_ylabel('A')
    ax.set_title(f'Bound Violations ({np.mean(exceed):.1%} exceed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_stretching_bound.png', dpi=150)
    print("\nSaved: test_stretching_bound.png")


if __name__ == "__main__":
    main()
