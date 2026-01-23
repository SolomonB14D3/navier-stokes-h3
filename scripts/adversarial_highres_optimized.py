#!/usr/bin/env python3
"""
OPTIMIZED HIGH-RESOLUTION ADVERSARIAL TEST

Based on the verified solver, tests "Max Strained" IC at n=64,128,256,512
to check convergence of the 11% overshoot.

Optimizations:
- Compute enstrophy only every 20 steps (like original)
- Use adaptive depletion + enhanced dissipation (verified version)
- Shorter simulation time (peak occurs around t=3-5)
"""

import mlx.core as mx
import numpy as np
import time
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4
R_H3 = 0.951

print("=" * 70)
print("  OPTIMIZED ADVERSARIAL CONVERGENCE TEST")
print("=" * 70)
print(f"MLX Device: {mx.default_device()}")
print(f"δ₀ = {DELTA_0:.6f}")


class H3NavierStokesOptimized:
    """Optimized solver matching the verified version."""

    def __init__(self, n, viscosity=0.001, delta0=DELTA_0):
        self.n = n
        self.nu = viscosity
        self.delta0 = delta0
        self.dt = 0.0001 * (64 / n)  # CFL-stable timestep

        # Wavenumbers
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)

        # Theoretical bound (matches verified version)
        lambda1 = (2 * np.pi)**2
        C_bound = 1.0
        self.Z_theoretical_max = (C_bound / (viscosity * lambda1 * delta0 * R_H3))**2

        # Reference bound from n=256 validation
        self.Z_ref_bound = 547.0

        self.visc_decay_base = mx.exp(-self.nu * self.k2 * self.dt)
        mx.eval(self.kx, self.ky, self.kz, self.k2, self.k2_safe, self.visc_decay_base)

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat):
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def compute_enstrophy(self, wx_hat, wy_hat, wz_hat):
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real
        Z = 0.5 * mx.mean(wx**2 + wy**2 + wz**2)
        mx.eval(Z)
        return float(Z.item())

    def h3_depletion(self, omega_mag, Z_current):
        """Adaptive H₃ depletion (from verified solver).

        Uses Z_theoretical_max for scaling, not Z_ref_bound.
        """
        base_depletion = self.delta0
        # Use theoretical max (~7429), not reference bound (547)
        Z_bound_strict = self.Z_theoretical_max * 0.1
        Z_ratio = Z_current / Z_bound_strict

        if Z_ratio > 0.5:
            adaptive_factor = 1 - 1 / (1 + (Z_ratio - 0.5)**4)
        else:
            adaptive_factor = 0

        total_depletion = base_depletion + (1 - base_depletion) * adaptive_factor
        return mx.array(1 - total_depletion)

    def enhanced_dissipation(self, Z_current):
        """Enhanced dissipation at high enstrophy."""
        # Use theoretical max (~7429), not reference bound (547)
        Z_bound_strict = self.Z_theoretical_max * 0.1
        Z_ratio = Z_current / Z_bound_strict

        if Z_ratio > 0.3:
            enhancement = 1 + 10 * (Z_ratio - 0.3)**2
        else:
            enhancement = 1.0

        return mx.exp(-self.nu * enhancement * self.k2 * self.dt)

    def step(self, wx_hat, wy_hat, wz_hat, Z_current):
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2)

        # Advection
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Stretching
        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real

        stretch_x = wx * dux_dx + wy * dux_dy + wz * dux_dz
        stretch_y = wx * duy_dx + wy * duy_dy + wz * duy_dz
        stretch_z = wx * duz_dx + wy * duz_dy + wz * duz_dz

        # Apply depletion
        depletion_factor = self.h3_depletion(omega_mag, Z_current)
        stretch_x = stretch_x * depletion_factor
        stretch_y = stretch_y * depletion_factor
        stretch_z = stretch_z * depletion_factor

        nlx = nlx + stretch_x
        nly = nly + stretch_y
        nlz = nlz + stretch_z

        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        visc_decay = self.enhanced_dissipation(Z_current)

        wx_hat_new = (wx_hat + self.dt * nlx_hat) * visc_decay
        wy_hat_new = (wy_hat + self.dt * nly_hat) * visc_decay
        wz_hat_new = (wz_hat + self.dt * nlz_hat) * visc_decay

        return wx_hat_new, wy_hat_new, wz_hat_new


def create_max_strained_ic(n, target_Z0=9.4):
    """Max strained initial condition."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    omega = np.zeros((n, n, n, 3), dtype=np.float32)
    omega[..., 0] = 10 * np.sin(X) * np.cos(Y) * np.cos(Z)
    omega[..., 1] = 10 * np.cos(X) * np.sin(Y) * np.cos(Z)
    omega[..., 2] = -20 * np.cos(X) * np.cos(Y) * np.sin(Z)

    Z0 = 0.5 * np.mean(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2)
    if Z0 > 0:
        omega = omega * np.sqrt(target_Z0 / Z0)

    return omega


def run_test(n, tmax=5.0):
    """Run adversarial test at resolution n."""
    print(f"\n{'='*60}")
    print(f"  n = {n} ({n**3:,} points)")
    print(f"{'='*60}")

    solver = H3NavierStokesOptimized(n=n)
    omega_init = create_max_strained_ic(n)

    wx_hat = mx.fft.fftn(mx.array(omega_init[..., 0]))
    wy_hat = mx.fft.fftn(mx.array(omega_init[..., 1]))
    wz_hat = mx.fft.fftn(mx.array(omega_init[..., 2]))
    mx.eval(wx_hat, wy_hat, wz_hat)

    nsteps = int(tmax / solver.dt)
    Z_current = solver.compute_enstrophy(wx_hat, wy_hat, wz_hat)
    Z_max = Z_current
    t_at_max = 0

    print(f"  dt = {solver.dt:.6f}, steps = {nsteps:,}")
    print(f"  Z₀ = {Z_current:.2f}, Z_bound = {solver.Z_ref_bound:.1f}")

    start = time.time()
    report_interval = max(1, nsteps // 20)

    for step in range(nsteps):
        # Compute enstrophy every 20 steps (like original)
        if step % 20 == 0:
            Z_current = solver.compute_enstrophy(wx_hat, wy_hat, wz_hat)
            if Z_current > Z_max:
                Z_max = Z_current
                t_at_max = step * solver.dt

            if np.isnan(Z_current):
                print(f"  NaN at step {step}")
                break

        # Progress report
        if step % report_interval == 0:
            elapsed = time.time() - start
            rate = step / elapsed if elapsed > 0 else 0
            eta = (nsteps - step) / rate / 60 if rate > 0 else 0
            pct = Z_max / solver.Z_ref_bound * 100
            print(f"  [{step*100//nsteps:3d}%] t={step*solver.dt:.2f} Z={Z_current:.1f} "
                  f"Z_max={Z_max:.1f} ({pct:.1f}%) | {rate:.0f} steps/s, ETA {eta:.1f}m")

        wx_hat, wy_hat, wz_hat = solver.step(wx_hat, wy_hat, wz_hat, Z_current)
        mx.eval(wx_hat, wy_hat, wz_hat)

    elapsed = time.time() - start
    Z_final = solver.compute_enstrophy(wx_hat, wy_hat, wz_hat)

    ratio = Z_max / solver.Z_ref_bound * 100
    status = "✓ BOUNDED" if ratio <= 100 else f"⚠ {ratio-100:.1f}% OVER"

    print(f"\n  RESULT: Z_max = {Z_max:.2f} ({ratio:.1f}% of bound) {status}")
    print(f"          t_max = {t_at_max:.3f}, time = {elapsed:.1f}s")

    return {
        'n': n,
        'Z_max': float(Z_max),
        'Z_ratio_pct': float(ratio),
        't_at_max': float(t_at_max),
        'Z_final': float(Z_final),
        'elapsed_s': float(elapsed),
        'bounded': ratio <= 100
    }


def main():
    # Test resolutions - include 64 and 128 for comparison with known results
    resolutions = [64, 128, 256, 512]
    tmax = 5.0

    print(f"\nResolutions: {resolutions}")
    print(f"t_max = {tmax}")
    print(f"Reference bound: 547 (from validated n=256 runs)")

    results = []
    for n in resolutions:
        result = run_test(n, tmax=tmax)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("  CONVERGENCE SUMMARY")
    print("=" * 70)
    print(f"\n  {'n':<6} {'Z_max':<10} {'% of bound':<12} {'Status':<15} {'Time'}")
    print("  " + "-" * 55)

    for r in results:
        status = "✓ BOUNDED" if r['bounded'] else "⚠ EXCEEDED"
        print(f"  {r['n']:<6} {r['Z_max']:<10.2f} {r['Z_ratio_pct']:<12.1f} {status:<15} {r['elapsed_s']:.0f}s")

    # Convergence analysis
    if len(results) >= 2:
        ratios = [r['Z_ratio_pct'] for r in results]
        trend = ratios[-1] - ratios[0]
        print(f"\n  Trend: {trend:+.1f}% from n={results[0]['n']} to n={results[-1]['n']}")

        if trend < 0:
            print("  → DECREASING: Overshoot is converging toward bound")
        elif trend > 5:
            print("  → INCREASING: Overshoot is growing (concerning)")
        else:
            print("  → STABLE: Overshoot is roughly constant")

    # Save
    os.makedirs('results', exist_ok=True)
    output = {
        'test': 'max_strained_convergence',
        'ref_bound': 547,
        'results': results
    }
    with open('results/adversarial_convergence.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: results/adversarial_convergence.json")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
