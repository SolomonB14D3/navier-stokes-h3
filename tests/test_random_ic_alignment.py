#!/usr/bin/env python3
"""
TEST: ICOSAHEDRAL ALIGNMENT WITH RANDOM INITIAL CONDITIONS

The vortex tube geometry test showed 98% alignment with icosahedral axes,
but used Taylor-Green initial conditions with 12-fold perturbation.

This test uses RANDOM initial vorticity to rule out symmetry bias.
If alignment still emerges, it's truly intrinsic to NS dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import mlx.core as mx

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4

# Icosahedral axes (same as before)
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

print(f"Testing with {len(ICOSA_AXES)} icosahedral axes")


class NSSolver:
    """Standard NS solver."""

    def __init__(self, n=64, viscosity=0.001, dt=0.0001):
        self.n = n
        self.nu = viscosity
        self.dt = dt

        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)
        self.visc_decay = mx.exp(-self.nu * self.k2 * self.dt)

    def step(self, wx_hat, wy_hat, wz_hat):
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        for u_hat, (sx, sy, sz) in [(ux_hat, (1,0,0)), (uy_hat, (0,1,0)), (uz_hat, (0,0,1))]:
            dudx = mx.fft.ifftn(1j * self.kx * u_hat).real
            dudy = mx.fft.ifftn(1j * self.ky * u_hat).real
            dudz = mx.fft.ifftn(1j * self.kz * u_hat).real
            stretch = wx * dudx + wy * dudy + wz * dudz
            if sx: nlx = nlx + stretch
            if sy: nly = nly + stretch
            if sz: nlz = nlz + stretch

        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        return ((wx_hat + self.dt * nlx_hat) * self.visc_decay,
                (wy_hat + self.dt * nly_hat) * self.visc_decay,
                (wz_hat + self.dt * nlz_hat) * self.visc_decay)


def random_solenoidal_field(n, scale=5.0, seed=None):
    """
    Generate random divergence-free vorticity field.
    Uses random Fourier modes with solenoidal projection.
    """
    if seed is not None:
        np.random.seed(seed)

    # Random complex Fourier coefficients
    shape = (n, n, n)

    # Generate random vector potential A, then ω = ∇ × A
    Ax = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    Ay = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    Az = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    # Low-pass filter (smooth field)
    k = np.fft.fftfreq(n, d=1/n) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    k_filter = np.exp(-(k_mag / (n/4))**2)

    Ax_hat = np.fft.fftn(Ax) * k_filter
    Ay_hat = np.fft.fftn(Ay) * k_filter
    Az_hat = np.fft.fftn(Az) * k_filter

    # ω = ∇ × A in Fourier space
    wx_hat = 1j * ky * Az_hat - 1j * kz * Ay_hat
    wy_hat = 1j * kz * Ax_hat - 1j * kx * Az_hat
    wz_hat = 1j * kx * Ay_hat - 1j * ky * Ax_hat

    wx = np.real(np.fft.ifftn(wx_hat))
    wy = np.real(np.fft.ifftn(wy_hat))
    wz = np.real(np.fft.ifftn(wz_hat))

    # Normalize to desired scale
    omega_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    current_max = np.max(omega_mag)
    factor = scale / current_max

    return np.stack([wx * factor, wy * factor, wz * factor], axis=-1)


def compute_icosa_alignment(omega):
    """Compute alignment of vorticity directions to icosahedral axes."""
    omega_norm = np.linalg.norm(omega, axis=-1)
    threshold = 2.0 * np.mean(omega_norm)
    mask = omega_norm > threshold

    if np.sum(mask) < 100:
        return None, None

    # Normalized vorticity directions
    directions = omega[mask] / (omega_norm[mask, np.newaxis] + 1e-10)

    # Max alignment to any icosahedral axis
    dots = np.abs(directions @ ICOSA_AXES.T)
    max_alignment = np.max(dots, axis=1)

    return max_alignment, np.sum(mask)


def run_single_test(seed, n=64, tmax=1.0):
    """Run single test with random IC."""
    omega_init = random_solenoidal_field(n, scale=5.0, seed=seed)

    # Check initial alignment (should be ~0.5 for random)
    init_align, _ = compute_icosa_alignment(omega_init)
    init_mean = np.mean(init_align) if init_align is not None else 0.5

    # Run NS
    solver = NSSolver(n=n, viscosity=0.001)
    wx_hat = mx.fft.fftn(mx.array(omega_init[..., 0]))
    wy_hat = mx.fft.fftn(mx.array(omega_init[..., 1]))
    wz_hat = mx.fft.fftn(mx.array(omega_init[..., 2]))

    nsteps = int(tmax / solver.dt)
    alignments = []
    times = []

    for step in range(nsteps):
        if step % 200 == 0:
            wx = np.array(mx.fft.ifftn(wx_hat).real)
            wy = np.array(mx.fft.ifftn(wy_hat).real)
            wz = np.array(mx.fft.ifftn(wz_hat).real)
            omega = np.stack([wx, wy, wz], axis=-1)

            Z = 0.5 * np.mean(wx**2 + wy**2 + wz**2)
            if np.isnan(Z) or Z > 1e6:
                break

            align, n_pts = compute_icosa_alignment(omega)
            if align is not None:
                alignments.append(np.mean(align))
                times.append(step * solver.dt)

        wx_hat, wy_hat, wz_hat = solver.step(wx_hat, wy_hat, wz_hat)
        mx.eval(wx_hat, wy_hat, wz_hat)

    final_align = alignments[-1] if alignments else init_mean
    return init_mean, final_align, times, alignments


def main():
    print("\n" + "=" * 70)
    print("  RANDOM INITIAL CONDITIONS - ICOSAHEDRAL ALIGNMENT TEST")
    print("=" * 70)
    print("""
  Question: Does icosahedral alignment emerge from RANDOM initial data?

  If yes → alignment is INTRINSIC to NS dynamics
  If no  → previous result was artifact of Taylor-Green symmetry

  Random directions have mean alignment ≈ 0.5 to icosahedral axes.
  Taylor-Green showed alignment ≈ 0.98.
    """)

    n_trials = 10
    n = 64
    tmax = 1.0

    print(f"Running {n_trials} trials with random ICs (n={n}, t_max={tmax})")

    results = []
    for i in range(n_trials):
        print(f"\n  Trial {i+1}/{n_trials}...", end=" ", flush=True)
        init_align, final_align, times, alignments = run_single_test(seed=42+i, n=n, tmax=tmax)
        print(f"init={init_align:.3f} → final={final_align:.3f}")
        results.append({
            'init': init_align,
            'final': final_align,
            'times': times,
            'alignments': alignments
        })

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    init_aligns = [r['init'] for r in results]
    final_aligns = [r['final'] for r in results]

    print(f"\nInitial alignment (random):  {np.mean(init_aligns):.4f} ± {np.std(init_aligns):.4f}")
    print(f"Final alignment (evolved):   {np.mean(final_aligns):.4f} ± {np.std(final_aligns):.4f}")
    print(f"Random expectation:          ~0.50")
    print(f"Taylor-Green result:         0.98")

    # Did alignment increase?
    improvement = np.mean(final_aligns) - np.mean(init_aligns)
    print(f"\nAlignment change: {improvement:+.4f}")

    if np.mean(final_aligns) > 0.7:
        print("\n✓ STRONG alignment emerges - intrinsic to NS dynamics!")
    elif np.mean(final_aligns) > 0.55:
        print("\n~ WEAK alignment increase - some preference for icosahedral")
    else:
        print("\n✗ NO significant alignment - Taylor-Green was biased")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Alignment evolution for each trial
    ax = axes[0]
    for i, r in enumerate(results):
        if r['times']:
            ax.plot(r['times'], r['alignments'], alpha=0.5, label=f'Trial {i+1}')
    ax.axhline(0.5, color='gray', linestyle=':', label='Random')
    ax.axhline(0.98, color='r', linestyle='--', label='Taylor-Green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Icosahedral Alignment')
    ax.set_title('Alignment Evolution (Random ICs)')
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, alpha=0.3)

    # Initial vs final
    ax = axes[1]
    ax.scatter(init_aligns, final_aligns, s=100, alpha=0.7)
    ax.plot([0.4, 1], [0.4, 1], 'k--', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Initial Alignment')
    ax.set_ylabel('Final Alignment')
    ax.set_title('Initial vs Final Alignment')
    ax.set_xlim([0.4, 0.7])
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[2]
    ax.hist(init_aligns, bins=10, alpha=0.5, label='Initial', density=True)
    ax.hist(final_aligns, bins=10, alpha=0.5, label='Final', density=True)
    ax.axvline(0.5, color='gray', linestyle=':', label='Random')
    ax.axvline(0.98, color='r', linestyle='--', label='Taylor-Green')
    ax.set_xlabel('Alignment')
    ax.set_ylabel('Density')
    ax.set_title('Alignment Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_random_ic_alignment.png', dpi=150)
    print("\nSaved: test_random_ic_alignment.png")

    return results


if __name__ == "__main__":
    results = main()
