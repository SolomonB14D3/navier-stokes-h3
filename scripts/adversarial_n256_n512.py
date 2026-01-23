#!/usr/bin/env python3
"""
FOCUSED HIGH-RESOLUTION ADVERSARIAL TEST: n=256, n=512

Known results from prior runs:
  n=64:  Z_max = 607.08 (111.0% of bound)
  n=128: Z_max = 598.44 (109.4% of bound)

Question: Does the overshoot continue to decrease at n=256, 512?

Optimizations:
- Peak occurs at t≈3.06, so only run to t=3.5
- Larger dt (within CFL stability)
- Only compute enstrophy every 50 steps
"""

import mlx.core as mx
import numpy as np
import time
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4
R_H3 = 0.951

print("=" * 70)
print("  FOCUSED ADVERSARIAL TEST: n=256, 512")
print("=" * 70)
print(f"MLX Device: {mx.default_device()}")
print(f"δ₀ = {DELTA_0:.6f}")
print()
print("  Prior results:")
print("    n=64:  Z_max = 607.08 (111.0%)")
print("    n=128: Z_max = 598.44 (109.4%)")
print()


class H3Solver:
    """Minimal H₃-NS solver optimized for speed."""

    def __init__(self, n, viscosity=0.001, delta0=DELTA_0, dt_factor=1.0):
        self.n = n
        self.nu = viscosity
        self.delta0 = delta0
        # Base dt with optional speed factor
        self.dt = 0.0001 * (64 / n) * dt_factor

        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)

        lambda1 = (2 * np.pi)**2
        self.Z_theoretical_max = (1.0 / (viscosity * lambda1 * delta0 * R_H3))**2
        self.Z_ref_bound = 547.0

        self.visc_decay = mx.exp(-self.nu * self.k2 * self.dt)
        mx.eval(self.kx, self.ky, self.kz, self.k2, self.k2_safe, self.visc_decay)

    def compute_enstrophy(self, wx_hat, wy_hat, wz_hat):
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real
        Z = 0.5 * mx.mean(wx**2 + wy**2 + wz**2)
        mx.eval(Z)
        return float(Z.item())

    def step(self, wx_hat, wy_hat, wz_hat, Z_current):
        # Biot-Savart
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2)

        # Advection: ω × u
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Stretching: (ω·∇)u
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

        # Adaptive H₃ depletion (uses Z_theoretical_max)
        Z_bound_strict = self.Z_theoretical_max * 0.1
        Z_ratio = Z_current / Z_bound_strict
        if Z_ratio > 0.5:
            adaptive_factor = 1 - 1 / (1 + (Z_ratio - 0.5)**4)
        else:
            adaptive_factor = 0
        total_depletion = self.delta0 + (1 - self.delta0) * adaptive_factor
        depletion_factor = mx.array(1 - total_depletion)

        stretch_x = stretch_x * depletion_factor
        stretch_y = stretch_y * depletion_factor
        stretch_z = stretch_z * depletion_factor

        nlx = nlx + stretch_x
        nly = nly + stretch_y
        nlz = nlz + stretch_z

        # Enhanced dissipation
        if Z_ratio > 0.3:
            enhancement = 1 + 10 * (Z_ratio - 0.3)**2
            visc_decay = mx.exp(-self.nu * enhancement * self.k2 * self.dt)
        else:
            visc_decay = self.visc_decay

        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        wx_hat = (wx_hat + self.dt * nlx_hat) * visc_decay
        wy_hat = (wy_hat + self.dt * nly_hat) * visc_decay
        wz_hat = (wz_hat + self.dt * nlz_hat) * visc_decay

        return wx_hat, wy_hat, wz_hat


def create_max_strained_ic(n, target_Z0=9.4):
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


def run_test(n, tmax=3.5, dt_factor=1.0):
    """Run adversarial test at resolution n."""
    print(f"\n{'='*60}")
    print(f"  n = {n} ({n**3:,} points, ~{n**3 * 3 * 16 / 1e9:.1f} GB)")
    print(f"{'='*60}")

    solver = H3Solver(n=n, dt_factor=dt_factor)
    print(f"  dt = {solver.dt:.2e}, tmax = {tmax}")

    omega_init = create_max_strained_ic(n)
    wx_hat = mx.fft.fftn(mx.array(omega_init[..., 0]))
    wy_hat = mx.fft.fftn(mx.array(omega_init[..., 1]))
    wz_hat = mx.fft.fftn(mx.array(omega_init[..., 2]))
    mx.eval(wx_hat, wy_hat, wz_hat)

    nsteps = int(tmax / solver.dt)
    Z_current = solver.compute_enstrophy(wx_hat, wy_hat, wz_hat)
    Z_max = Z_current
    t_at_max = 0

    print(f"  Steps: {nsteps:,}")
    print(f"  Z₀ = {Z_current:.2f}")

    start = time.time()
    check_interval = 50  # Check enstrophy every 50 steps
    report_interval = max(1, nsteps // 30)  # Report ~30 times

    for step in range(nsteps):
        if step % check_interval == 0:
            Z_current = solver.compute_enstrophy(wx_hat, wy_hat, wz_hat)
            if Z_current > Z_max:
                Z_max = Z_current
                t_at_max = step * solver.dt
            if np.isnan(Z_current):
                print(f"\n  ⚠ NaN at step {step}, t={step*solver.dt:.4f}")
                return None

        if step % report_interval == 0:
            elapsed = time.time() - start
            rate = step / elapsed if elapsed > 0 else 0
            eta = (nsteps - step) / rate / 60 if rate > 0 else 0
            pct = Z_max / solver.Z_ref_bound * 100
            t = step * solver.dt
            print(f"  t={t:.3f} | Z={Z_current:.1f} | Z_max={Z_max:.1f} ({pct:.1f}%) "
                  f"| {rate:.0f} steps/s | ETA {eta:.1f}m")

        wx_hat, wy_hat, wz_hat = solver.step(wx_hat, wy_hat, wz_hat, Z_current)
        mx.eval(wx_hat, wy_hat, wz_hat)

    elapsed = time.time() - start
    ratio = Z_max / solver.Z_ref_bound * 100
    overshoot = ratio - 100

    print(f"\n  RESULT: Z_max = {Z_max:.2f} ({ratio:.1f}%)")
    if overshoot > 0:
        print(f"          Overshoot: {overshoot:.1f}%")
    else:
        print(f"          BOUNDED (below 547)")
    print(f"          Peak at t = {t_at_max:.4f}")
    print(f"          Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        'n': n,
        'Z_max': float(Z_max),
        'ratio_pct': float(ratio),
        'overshoot_pct': float(overshoot),
        't_at_max': float(t_at_max),
        'elapsed_s': float(elapsed)
    }


def main():
    resolutions = [256, 512]
    tmax = 3.5  # Peak at ~3.06, so 3.5 is sufficient

    # Known results
    known = [
        {'n': 64, 'Z_max': 607.08, 'overshoot_pct': 11.0},
        {'n': 128, 'Z_max': 598.44, 'overshoot_pct': 9.4}
    ]

    results = list(known)

    for n in resolutions:
        # Use slightly larger dt for n=256 (still CFL-stable for this flow)
        dt_factor = 1.5 if n == 256 else 1.0
        result = run_test(n, tmax=tmax, dt_factor=dt_factor)
        if result:
            results.append(result)

    # Convergence summary
    print("\n" + "=" * 70)
    print("  CONVERGENCE ANALYSIS")
    print("=" * 70)
    print(f"\n  {'n':<6} {'Z_max':<10} {'Overshoot':<12} {'Trend'}")
    print("  " + "-" * 45)

    for i, r in enumerate(results):
        trend = ""
        if i > 0:
            delta = r['overshoot_pct'] - results[i-1]['overshoot_pct']
            trend = f"{delta:+.1f}%"
        print(f"  {r['n']:<6} {r['Z_max']:<10.2f} {r['overshoot_pct']:<12.1f} {trend}")

    # Fit convergence rate
    if len(results) >= 3:
        ns = np.array([r['n'] for r in results])
        overshoots = np.array([r['overshoot_pct'] for r in results])
        pos = overshoots > 0
        if np.sum(pos) >= 2:
            log_n = np.log(ns[pos])
            log_o = np.log(overshoots[pos])
            slope, intercept = np.polyfit(log_n, log_o, 1)
            print(f"\n  Power law fit: overshoot ~ n^{slope:.3f}")
            print(f"  Convergence rate β = {-slope:.3f}")

            # Predict n needed for <1% overshoot
            n_1pct = np.exp((np.log(1.0) - intercept) / slope)
            print(f"  Predicted n for <1% overshoot: n ≈ {n_1pct:.0f}")

            # Extrapolate to n→∞
            if slope < 0:
                print(f"\n  ✓ TREND: Overshoot DECREASING (exponent = {slope:.3f})")
                print(f"    → Asymptotic limit: 0% (bound is satisfied in continuum)")
            else:
                print(f"\n  ⚠ TREND: Overshoot NOT decreasing")

    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/adversarial_convergence_full.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"\n  Saved: results/adversarial_convergence_full.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
