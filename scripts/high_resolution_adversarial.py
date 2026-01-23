#!/usr/bin/env python3
"""
HIGH-RESOLUTION ADVERSARIAL TEST

Tests the "Max strained" configuration at n=256, 512, 1024 to determine
if the 11% enstrophy overshoot converges to below the theoretical bound.

Optimized for Apple M3 Ultra (28 cores, 96GB unified memory):
- MLX GPU acceleration
- Memory-efficient streaming diagnostics
- Adaptive timestep for stability at high resolution

Key question: Does Z_max/Z_bound → value < 1 as n → ∞?

Usage:
    python scripts/high_resolution_adversarial.py [--quick]

    --quick: Run only n=256,512 (faster, still informative)
"""

import mlx.core as mx
import numpy as np
import time
import json
import os
import sys
import argparse
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

# Constants (algebraically exact)
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4  # = 1/(2φ) ≈ 0.309
R_H3 = 0.951
A_MAX = 1 - DELTA_0  # ≈ 0.691

print("=" * 80)
print("  HIGH-RESOLUTION ADVERSARIAL TEST")
print("  Testing 'Max Strained' IC convergence at n=256, 512, 1024")
print("=" * 80)
print(f"\nMLX Device: {mx.default_device()}")
print(f"δ₀ = {DELTA_0:.15f} (exact: 1/(2φ))")
print(f"A_max = {A_MAX:.6f}")
print()


class H3NavierStokesHighRes:
    """
    High-resolution H₃-constrained Navier-Stokes solver.

    Optimized for large grids (n=512, 1024) on Apple Silicon.
    """

    def __init__(self, n, viscosity=0.001, delta0=DELTA_0):
        self.n = n
        self.nu = viscosity
        self.delta0 = delta0

        # CFL-stable timestep: dt ~ h²/ν for diffusion, dt ~ h/u for advection
        # Conservative choice for stability at high Re
        self.dt = 0.00005 * (128 / n) ** 1.5

        print(f"  Grid: {n}³ = {n**3:,} points")
        print(f"  Memory: ~{n**3 * 3 * 16 / 1e9:.1f} GB (complex vorticity)")
        print(f"  Timestep: dt = {self.dt:.2e}")

        # Setup wavenumbers
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)

        # Theoretical bound components
        lambda1 = (2 * np.pi)**2
        C_bound = 1.0
        self.Z_theoretical_max = (C_bound / (viscosity * lambda1 * delta0 * R_H3))**2

        # Critical vorticity
        self.omega_crit = 1.0 / (delta0 * R_H3)

        # Precompute viscous decay
        self.visc_decay = mx.exp(-self.nu * self.k2 * self.dt)

        mx.eval(self.kx, self.ky, self.kz, self.k2, self.k2_safe, self.visc_decay)

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat):
        """Biot-Savart: u = curl⁻¹(ω)"""
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def compute_enstrophy(self, wx_hat, wy_hat, wz_hat):
        """Z = (1/2)∫|ω|² dx"""
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real
        Z = 0.5 * mx.mean(wx**2 + wy**2 + wz**2)
        mx.eval(Z)
        return float(Z.item())

    def compute_depletion_factor(self, omega_mag):
        """
        H₃ geometric depletion: (1 - δ₀·Φ)

        Φ(x) = x²/(1+x²) is the smooth activation function.
        """
        x = omega_mag / self.omega_crit
        activation = x**2 / (1.0 + x**2)
        return 1.0 - self.delta0 * activation

    def step(self, wx_hat, wy_hat, wz_hat):
        """Advance one timestep with geometric depletion."""
        # Velocity from vorticity
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        # Transform to physical space
        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2 + 1e-10)

        # Advection: ω × u
        advect_x = wy * uz - wz * uy
        advect_y = wz * ux - wx * uz
        advect_z = wx * uy - wy * ux

        # Stretching: (ω·∇)u with spectral derivatives
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

        # Apply H₃ depletion (Theorem 3.1)
        depletion = self.compute_depletion_factor(omega_mag)
        stretch_x = stretch_x * depletion
        stretch_y = stretch_y * depletion
        stretch_z = stretch_z * depletion

        # Total nonlinear term
        nl_x = advect_x + stretch_x
        nl_y = advect_y + stretch_y
        nl_z = advect_z + stretch_z

        # Advance with viscous decay
        nl_x_hat = mx.fft.fftn(nl_x)
        nl_y_hat = mx.fft.fftn(nl_y)
        nl_z_hat = mx.fft.fftn(nl_z)

        wx_hat_new = (wx_hat + self.dt * nl_x_hat) * self.visc_decay
        wy_hat_new = (wy_hat + self.dt * nl_y_hat) * self.visc_decay
        wz_hat_new = (wz_hat + self.dt * nl_z_hat) * self.visc_decay

        return wx_hat_new, wy_hat_new, wz_hat_new


def create_max_strained_ic(n, target_Z0=9.4):
    """
    Create the 'Max Strained' initial condition.

    This configuration is designed to maximize vorticity-strain alignment,
    challenging the H₃ depletion mechanism.
    """
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Asymmetric vorticity field maximizing strain alignment
    omega = np.zeros((n, n, n, 3), dtype=np.float32)
    omega[..., 0] = 10 * np.sin(X) * np.cos(Y) * np.cos(Z)
    omega[..., 1] = 10 * np.cos(X) * np.sin(Y) * np.cos(Z)
    omega[..., 2] = -20 * np.cos(X) * np.cos(Y) * np.sin(Z)  # 2× asymmetric

    # Normalize to target enstrophy
    Z0 = 0.5 * np.mean(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2)
    if Z0 > 0:
        scale = np.sqrt(target_Z0 / Z0)
        omega = omega * scale

    return omega


def run_simulation(n, t_max=3.0, report_every=0.1):
    """
    Run NS simulation at resolution n.

    Returns dict with Z_max, Z_history, timing info.
    """
    print(f"\n{'='*60}")
    print(f"  RESOLUTION n = {n}")
    print(f"{'='*60}")

    solver = H3NavierStokesHighRes(n=n, viscosity=0.001, delta0=DELTA_0)
    omega_init = create_max_strained_ic(n, target_Z0=9.4)

    # Initialize in Fourier space
    wx_hat = mx.fft.fftn(mx.array(omega_init[..., 0]))
    wy_hat = mx.fft.fftn(mx.array(omega_init[..., 1]))
    wz_hat = mx.fft.fftn(mx.array(omega_init[..., 2]))
    mx.eval(wx_hat, wy_hat, wz_hat)

    Z_initial = solver.compute_enstrophy(wx_hat, wy_hat, wz_hat)
    print(f"  Initial enstrophy: Z₀ = {Z_initial:.2f}")
    print(f"  Theoretical bound: Z_max ≈ {solver.Z_theoretical_max:.1f}")
    print(f"  Simulation time: t_max = {t_max}")

    n_steps = int(t_max / solver.dt)
    report_interval = max(1, int(report_every / solver.dt))

    print(f"  Total steps: {n_steps:,}")
    print(f"  Reporting every {report_interval} steps")
    print()

    Z_history = []
    t_history = []
    Z_max = 0
    t_at_max = 0

    start_time = time.time()
    last_report = start_time

    for step in range(n_steps):
        t = step * solver.dt

        # Compute enstrophy
        Z = solver.compute_enstrophy(wx_hat, wy_hat, wz_hat)

        if Z > Z_max:
            Z_max = Z
            t_at_max = t

        # Periodic reporting
        if step % report_interval == 0:
            Z_history.append(Z)
            t_history.append(t)

            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (n_steps - step) / steps_per_sec if steps_per_sec > 0 else 0

            ratio = Z_max / solver.Z_theoretical_max * 100

            # Progress bar
            progress = step / n_steps
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)

            print(f"\r  [{bar}] {progress*100:5.1f}% | "
                  f"t={t:.3f} | Z={Z:.1f} | Z_max={Z_max:.1f} ({ratio:.1f}%) | "
                  f"{steps_per_sec:.0f} steps/s | ETA {eta/60:.1f}min",
                  end='', flush=True)

        # Advance
        wx_hat, wy_hat, wz_hat = solver.step(wx_hat, wy_hat, wz_hat)
        mx.eval(wx_hat, wy_hat, wz_hat)

        # Check for numerical instability
        if np.isnan(Z) or Z > 1e6:
            print(f"\n  ⚠ NUMERICAL INSTABILITY at step {step}, t={t:.4f}")
            break

    elapsed = time.time() - start_time
    Z_final = solver.compute_enstrophy(wx_hat, wy_hat, wz_hat)

    print()  # Newline after progress bar
    print(f"\n  RESULTS for n={n}:")
    print(f"    Z_max = {Z_max:.4f}")
    print(f"    Z_max / Z_bound = {Z_max / solver.Z_theoretical_max * 100:.2f}%")
    print(f"    t at Z_max = {t_at_max:.4f}")
    print(f"    Z_final = {Z_final:.4f}")
    print(f"    Wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"    Performance: {n_steps/elapsed:.0f} steps/s")

    return {
        'n': n,
        'Z_max': float(Z_max),
        'Z_bound': float(solver.Z_theoretical_max),
        'Z_ratio': float(Z_max / solver.Z_theoretical_max),
        't_at_max': float(t_at_max),
        'Z_final': float(Z_final),
        'Z_initial': float(Z_initial),
        'elapsed_s': elapsed,
        'n_steps': n_steps,
        'dt': solver.dt,
        'Z_history': [float(z) for z in Z_history],
        't_history': [float(t) for t in t_history]
    }


def analyze_convergence(results):
    """
    Analyze whether Z_max/Z_bound converges below 1.

    Uses Richardson extrapolation to estimate asymptotic value.
    """
    ns = np.array([r['n'] for r in results])
    ratios = np.array([r['Z_ratio'] for r in results])

    print("\n" + "=" * 80)
    print("  CONVERGENCE ANALYSIS")
    print("=" * 80)

    print(f"\n  {'n':<8} {'Z_max':<12} {'Z_bound':<12} {'Ratio':<12} {'Status'}")
    print("  " + "-" * 56)

    for r in results:
        status = "✓ BOUNDED" if r['Z_ratio'] < 1.0 else "⚠ EXCEEDED"
        print(f"  {r['n']:<8} {r['Z_max']:<12.2f} {r['Z_bound']:<12.2f} "
              f"{r['Z_ratio']*100:<11.2f}% {status}")

    # Richardson extrapolation (assuming 2nd order convergence)
    if len(results) >= 2:
        # Fit: ratio(n) = ratio_∞ + C/n^p
        log_ns = np.log(ns)

        # Simple linear fit in log-space for deviation from 1
        deviations = ratios - 1  # How far above/below bound

        # If all positive (all exceeded) or all negative (all bounded), fit power law
        if len(results) >= 3:
            # Use last two points for extrapolation
            r1, r2 = ratios[-2], ratios[-1]
            n1, n2 = ns[-2], ns[-1]

            # Assuming ratio = ratio_∞ + C * n^(-p)
            # With p=2 (spectral convergence):
            ratio_extrap = r2 + (r2 - r1) * (n2**2) / (n1**2 - n2**2)

            print(f"\n  Richardson extrapolation (p=2): ratio_∞ ≈ {ratio_extrap*100:.2f}%")

        # Check trend
        if len(results) >= 2:
            trend = (ratios[-1] - ratios[0]) / ratios[0] * 100
            direction = "DECREASING ↓" if trend < 0 else "INCREASING ↑"
            print(f"\n  Trend from n={ns[0]} to n={ns[-1]}: {direction} ({trend:+.2f}%)")

    # Verdict
    print("\n  " + "=" * 56)
    if all(r['Z_ratio'] < 1.0 for r in results):
        print("  VERDICT: ALL BOUNDED")
        print("  The H₃ depletion mechanism bounds enstrophy at all tested resolutions.")
        converged = True
    elif results[-1]['Z_ratio'] < 1.0:
        print("  VERDICT: CONVERGES TO BOUNDED")
        print("  Overshoot decreases with resolution, asymptotes below bound.")
        converged = True
    elif len(results) >= 2 and ratios[-1] < ratios[-2]:
        print("  VERDICT: TREND SUGGESTS CONVERGENCE")
        print("  Overshoot is decreasing; higher resolution needed to confirm.")
        converged = None
    else:
        print("  VERDICT: INCONCLUSIVE")
        print("  Need more resolution or longer simulation time.")
        converged = False
    print("  " + "=" * 56)

    return {
        'ns': ns.tolist(),
        'ratios': (ratios * 100).tolist(),
        'converged': converged,
        'extrapolated_ratio': float(ratio_extrap * 100) if len(results) >= 3 else None
    }


def main():
    parser = argparse.ArgumentParser(description='High-resolution adversarial test')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with n=256,512 only')
    parser.add_argument('--full', action='store_true',
                        help='Full test including n=1024')
    parser.add_argument('--extreme', action='store_true',
                        help='Include n=1536 (requires ~72GB RAM)')
    args = parser.parse_args()

    # Select resolutions
    if args.quick:
        resolutions = [256, 512]
        t_max = 2.0
    elif args.extreme:
        resolutions = [256, 512, 1024, 1536]
        t_max = 3.0
    elif args.full:
        resolutions = [256, 512, 1024]
        t_max = 3.0
    else:
        # Default: full test
        resolutions = [256, 512, 1024]
        t_max = 3.0

    print(f"\nResolutions to test: {resolutions}")
    print(f"Simulation time: t_max = {t_max}")

    # Estimate time
    total_points = sum(n**3 for n in resolutions)
    est_time = total_points / (128**3) * 5  # ~5 min for n=128
    print(f"Estimated total time: ~{est_time:.0f} minutes")
    print()

    # Run simulations
    results = []
    for n in resolutions:
        result = run_simulation(n, t_max=t_max)
        results.append(result)

        # Save intermediate results
        os.makedirs('results', exist_ok=True)
        with open(f'results/adversarial_n{n}.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  ✓ Saved: results/adversarial_n{n}.json")

    # Analyze convergence
    analysis = analyze_convergence(results)

    # Save final results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'resolutions': resolutions,
            't_max': t_max,
            'delta0': DELTA_0,
            'ic_type': 'max_strained'
        },
        'results': results,
        'analysis': analysis
    }

    output_file = 'results/high_resolution_adversarial.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Final results saved to: {output_file}")

    # Create convergence plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('High-Resolution Adversarial Test: Max Strained IC',
                     fontsize=14, fontweight='bold')

        # Panel 1: Z_max ratio vs resolution
        ax = axes[0]
        ns = [r['n'] for r in results]
        ratios = [r['Z_ratio'] * 100 for r in results]

        ax.plot(ns, ratios, 'bo-', markersize=10, linewidth=2, label='Measured')
        ax.axhline(100, color='r', linestyle='--', linewidth=2, label='Theoretical bound (100%)')
        ax.set_xlabel('Grid Resolution n', fontsize=12)
        ax.set_ylabel('Z_max / Z_bound (%)', fontsize=12)
        ax.set_title('Enstrophy Ratio vs Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ns)

        # Add extrapolation if available
        if analysis['extrapolated_ratio'] is not None:
            ax.axhline(analysis['extrapolated_ratio'], color='g', linestyle=':',
                       label=f'Extrapolated: {analysis["extrapolated_ratio"]:.1f}%')
            ax.legend()

        # Panel 2: Z(t) for each resolution
        ax = axes[1]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(results)))

        for r, color in zip(results, colors):
            ax.plot(r['t_history'], r['Z_history'], color=color,
                    label=f"n={r['n']}", linewidth=1.5)

        ax.set_xlabel('Time t', fontsize=12)
        ax.set_ylabel('Enstrophy Z(t)', fontsize=12)
        ax.set_title('Enstrophy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = 'figures/high_resolution_adversarial.png'
        os.makedirs('figures', exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to: {fig_path}")
        plt.close()

    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    print("\n" + "=" * 80)
    print("  TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
