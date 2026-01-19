#!/usr/bin/env python3
"""
TEST: SNAP-BACK DYNAMICS IN H₃-CONSTRAINED NS

Hypothesis (from DAT connection):
- Aligned regions BUILD UP stress (observed: 12.9% vs 2.5%)
- H₃ constraint triggers SNAP-BACK before blowup
- Stress is "exported" to perpendicular/phason-like modes

Tests:
1. Does stretching in aligned cores PEAK then DECREASE?
2. Is there energy transfer to "perpendicular" modes?
3. Does the peak stretching occur at ~φ-related times/ratios?
"""

import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx

PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4
R_H3 = 0.951

print(f"φ = {PHI:.6f}")
print(f"δ₀ = {DELTA_0:.6f}")
print(f"1/φ = {1/PHI:.6f}")
print(f"φ-1 = {PHI-1:.6f}")


class H3NSSolver:
    """NS solver with H₃ constraint to observe snap-back."""

    def __init__(self, n=64, nu=0.001, dt=0.0001):
        self.n, self.nu, self.dt = n, nu, dt
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)
        self.visc = mx.exp(-nu * self.k2 * dt)

        lambda1 = (2 * np.pi)**2
        self.Z_max_theory = (1.0 / (nu * lambda1 * DELTA_0 * R_H3))**2

    def get_fields(self, wh):
        wx_hat, wy_hat, wz_hat = wh
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return (ux_hat, uy_hat, uz_hat), tuple(mx.fft.ifftn(h).real for h in wh)

    def compute_core_stretching(self, wh):
        """Compute stretching specifically in vortex cores."""
        (ux_hat, uy_hat, uz_hat), (wx, wy, wz) = self.get_fields(wh)

        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2)
        threshold = 2.0 * mx.mean(omega_mag)
        core_mask = omega_mag > threshold

        # Strain tensor
        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real

        S_xy = 0.5 * (dux_dy + duy_dx)
        S_xz = 0.5 * (dux_dz + duz_dx)
        S_yz = 0.5 * (duy_dz + duz_dy)

        # ω·S·ω in cores
        omega_S_omega = (wx * (dux_dx * wx + S_xy * wy + S_xz * wz) +
                         wy * (S_xy * wx + duy_dy * wy + S_yz * wz) +
                         wz * (S_xz * wx + S_yz * wy + duz_dz * wz))

        # Core stretching
        core_S = omega_S_omega * core_mask
        mean_core_S = float(mx.sum(core_S).item()) / (float(mx.sum(core_mask).item()) + 1)
        max_core_S = float(mx.max(core_S).item())

        return mean_core_S, max_core_S, float(mx.sum(core_mask).item())

    def compute_spectral_energy(self, wh):
        """Compute energy in different k-shells (proxy for mode distribution)."""
        wx_hat, wy_hat, wz_hat = wh

        # Energy spectrum
        E_hat = 0.5 * (mx.abs(wx_hat)**2 + mx.abs(wy_hat)**2 + mx.abs(wz_hat)**2)

        k_mag = mx.sqrt(self.k2)

        # Shell energies
        shells = []
        for k_low, k_high in [(0, 4), (4, 8), (8, 16), (16, 32)]:
            mask = (k_mag >= k_low) & (k_mag < k_high)
            E_shell = float(mx.sum(E_hat * mask).item())
            shells.append(E_shell)

        return shells

    def h3_depletion(self, Z):
        """Adaptive depletion factor."""
        Z_ratio = Z / (self.Z_max_theory * 0.1)
        if Z_ratio > 0.5:
            adaptive = 1 - 1 / (1 + (Z_ratio - 0.5)**4)
        else:
            adaptive = 0
        return 1 - DELTA_0 - (1 - DELTA_0) * adaptive

    def step(self, wh, Z):
        wx_hat, wy_hat, wz_hat = wh
        (ux_hat, uy_hat, uz_hat), (wx, wy, wz) = self.get_fields(wh)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real

        # Advection
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Stretching with depletion
        depl = self.h3_depletion(Z)
        nlx = nlx + depl * (wx * mx.fft.ifftn(1j * self.kx * ux_hat).real +
                            wy * mx.fft.ifftn(1j * self.ky * ux_hat).real +
                            wz * mx.fft.ifftn(1j * self.kz * ux_hat).real)
        nly = nly + depl * (wx * mx.fft.ifftn(1j * self.kx * uy_hat).real +
                            wy * mx.fft.ifftn(1j * self.ky * uy_hat).real +
                            wz * mx.fft.ifftn(1j * self.kz * uy_hat).real)
        nlz = nlz + depl * (wx * mx.fft.ifftn(1j * self.kx * uz_hat).real +
                            wy * mx.fft.ifftn(1j * self.ky * uz_hat).real +
                            wz * mx.fft.ifftn(1j * self.kz * uz_hat).real)

        # Enhanced dissipation
        Z_ratio = Z / (self.Z_max_theory * 0.1)
        enh = 1 + 10 * max(0, Z_ratio - 0.3)**2 if Z_ratio > 0.3 else 1
        visc = mx.exp(-self.nu * enh * self.k2 * self.dt)

        return ((wx_hat + self.dt * mx.fft.fftn(nlx)) * visc,
                (wy_hat + self.dt * mx.fft.fftn(nly)) * visc,
                (wz_hat + self.dt * mx.fft.fftn(nlz)) * visc)


def taylor_green(n, scale=5.0):
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return np.stack([scale * np.cos(X) * np.sin(Y) * np.sin(Z),
                     scale * np.sin(X) * np.cos(Y) * np.sin(Z),
                     -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  SNAP-BACK DYNAMICS TEST")
    print("=" * 70)
    print("""
  From DAT connection: H₃ constraint triggers "snap-back" that routes
  stress to perpendicular modes before blowup.

  Testing:
  1. Does core stretching PEAK then DECREASE? (snap-back)
  2. Does energy redistribute to higher k-shells? (mode export)
  3. Are there φ-related ratios in the dynamics?
    """)

    n = 64
    solver = H3NSSolver(n=n)
    omega_init = taylor_green(n, scale=5.0)
    wh = tuple(mx.fft.fftn(mx.array(omega_init[..., i])) for i in range(3))

    tmax = 10.0
    nsteps = int(tmax / solver.dt)

    results = {
        't': [], 'Z': [],
        'core_S_mean': [], 'core_S_max': [], 'n_core': [],
        'E_low': [], 'E_mid': [], 'E_high': [], 'E_vhigh': []
    }

    Z = 9.4
    print("Running H₃-constrained simulation...")

    for step in range(nsteps):
        if step % 200 == 0:
            # Enstrophy
            Z = 0.5 * float(mx.mean(sum(mx.fft.ifftn(h).real**2 for h in wh)).item())

            # Core stretching
            core_mean, core_max, n_core = solver.compute_core_stretching(wh)

            # Spectral distribution
            shells = solver.compute_spectral_energy(wh)

            results['t'].append(step * solver.dt)
            results['Z'].append(Z)
            results['core_S_mean'].append(core_mean)
            results['core_S_max'].append(core_max)
            results['n_core'].append(n_core)
            results['E_low'].append(shells[0])
            results['E_mid'].append(shells[1])
            results['E_high'].append(shells[2])
            results['E_vhigh'].append(shells[3])

            if step % 5000 == 0:
                print(f"  t={step*solver.dt:.2f}: Z={Z:.1f}, core_S={core_mean:.2f}, "
                      f"n_core={n_core:.0f}")

        wh = solver.step(wh, Z)
        mx.eval(*wh)

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    t = np.array(results['t'])
    Z = np.array(results['Z'])
    core_S = np.array(results['core_S_mean'])
    E_low = np.array(results['E_low'])
    E_high = np.array(results['E_high'])

    # Find peaks
    Z_peak_idx = np.argmax(Z)
    Z_peak_t = t[Z_peak_idx]
    Z_peak = Z[Z_peak_idx]

    core_S_peak_idx = np.argmax(core_S)
    core_S_peak_t = t[core_S_peak_idx]
    core_S_peak = core_S[core_S_peak_idx]

    print(f"\n  Enstrophy peak: Z = {Z_peak:.1f} at t = {Z_peak_t:.2f}")
    print(f"  Core stretching peak: S = {core_S_peak:.2f} at t = {core_S_peak_t:.2f}")

    # Check for snap-back
    if Z_peak_idx < len(Z) - 10:
        Z_after = Z[Z_peak_idx + 10]
        print(f"\n  SNAP-BACK CHECK:")
        print(f"    Z at peak: {Z_peak:.1f}")
        print(f"    Z after peak: {Z_after:.1f}")
        print(f"    Decay: {(1 - Z_after/Z_peak)*100:.1f}%")
        if Z_after < Z_peak:
            print(f"    ✓ Snap-back observed!")

    # Energy redistribution
    E_low_peak = E_low[Z_peak_idx]
    E_high_peak = E_high[Z_peak_idx]
    E_low_after = E_low[min(Z_peak_idx + 20, len(E_low)-1)]
    E_high_after = E_high[min(Z_peak_idx + 20, len(E_high)-1)]

    print(f"\n  ENERGY REDISTRIBUTION:")
    print(f"    E_low at peak: {E_low_peak:.2f}")
    print(f"    E_high at peak: {E_high_peak:.2f}")
    print(f"    E_low/E_high at peak: {E_low_peak/E_high_peak:.3f}")
    print(f"    E_low/E_high after: {E_low_after/E_high_after:.3f}")

    # φ-related ratios
    print(f"\n  φ-RELATED RATIOS:")
    print(f"    t_peak / 1.0 = {Z_peak_t:.3f}")
    print(f"    Z_peak / Z_final = {Z_peak / Z[-1]:.3f}")
    print(f"    φ = {PHI:.3f}, 1/φ = {1/PHI:.3f}, φ-1 = {PHI-1:.3f}")

    if abs(Z_peak_t - PHI) < 0.5:
        print(f"    ✓ Peak time near φ!")
    if abs(Z_peak / Z[-1] - PHI) < 0.3:
        print(f"    ✓ Peak/final ratio near φ!")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(t, Z, 'b-', linewidth=2)
    ax.axvline(Z_peak_t, color='r', linestyle='--', alpha=0.5, label=f'Peak at t={Z_peak_t:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy Z')
    ax.set_title('Enstrophy: Build-up → Peak → Decay (Snap-back)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, core_S, 'g-', linewidth=2)
    ax.axvline(core_S_peak_t, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Core Stretching')
    ax.set_title('Vortex Core Stretching')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, E_low, label='k < 4 (large)')
    ax.plot(t, results['E_mid'], label='4 ≤ k < 8')
    ax.plot(t, E_high, label='8 ≤ k < 16')
    ax.plot(t, results['E_vhigh'], label='k ≥ 16 (small)')
    ax.axvline(Z_peak_t, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy in k-shell')
    ax.set_title('Spectral Energy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ratio = np.array(E_low) / (np.array(E_high) + 1e-10)
    ax.plot(t, ratio, 'purple', linewidth=2)
    ax.axvline(Z_peak_t, color='r', linestyle='--', alpha=0.5, label='Z peak')
    ax.axhline(PHI, color='gold', linestyle=':', label=f'φ = {PHI:.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('E_low / E_high')
    ax.set_title('Energy Ratio (snap-back = transfer to small scales)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_snapback_dynamics.png', dpi=150)
    print("\nSaved: test_snapback_dynamics.png")

    return results


if __name__ == "__main__":
    results = main()
