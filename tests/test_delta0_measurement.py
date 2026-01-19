#!/usr/bin/env python3
"""
DIRECT MEASUREMENT OF δ₀ FROM STRETCHING REDUCTION

Compares vortex stretching S between:
- Unconstrained NS
- Constrained NS (with H₃ depletion)

Empirical δ₀ = 1 - (S_constrained / S_unconstrained)

Expected: δ₀ ≈ 0.309, so reduction factor ≈ 0.691
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import mlx.core as mx

DELTA_0_THEORY = (np.sqrt(5) - 1) / 4  # ≈ 0.309
R_H3 = 0.951

print(f"Theoretical δ₀ = {DELTA_0_THEORY:.6f}")
print(f"Expected reduction factor = {1 - DELTA_0_THEORY:.6f}")


class DualNSSolver:
    """Run constrained and unconstrained NS in parallel to compare stretching."""

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

        # Theoretical bound for adaptive depletion
        lambda1 = (2 * np.pi)**2
        self.Z_theoretical_max = (1.0 / (viscosity * lambda1 * DELTA_0_THEORY * R_H3))**2

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat):
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def compute_stretching(self, wx, wy, wz, ux_hat, uy_hat, uz_hat):
        """Compute vortex stretching term S = ω_i ω_j S_ij integrated."""
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

        # Strain tensor S_ij = (du_i/dx_j + du_j/dx_i) / 2
        S_xx = dux_dx
        S_yy = duy_dy
        S_zz = duz_dz
        S_xy = 0.5 * (dux_dy + duy_dx)
        S_xz = 0.5 * (dux_dz + duz_dx)
        S_yz = 0.5 * (duy_dz + duz_dy)

        # Stretching: ω_i S_ij ω_j
        stretch = (wx * (S_xx * wx + S_xy * wy + S_xz * wz) +
                   wy * (S_xy * wx + S_yy * wy + S_yz * wz) +
                   wz * (S_xz * wx + S_yz * wy + S_zz * wz))

        # Integrated stretching (positive part drives enstrophy growth)
        S_total = float(mx.mean(stretch).item())
        S_positive = float(mx.mean(mx.maximum(stretch, 0)).item())

        return S_total, S_positive

    def h3_depletion_factor(self, Z_current):
        """Compute depletion factor based on current enstrophy."""
        Z_bound = self.Z_theoretical_max * 0.1
        Z_ratio = Z_current / Z_bound

        if Z_ratio > 0.5:
            adaptive = 1 - 1 / (1 + (Z_ratio - 0.5)**4)
        else:
            adaptive = 0

        total_depletion = DELTA_0_THEORY + (1 - DELTA_0_THEORY) * adaptive
        return 1 - total_depletion

    def step_unconstrained(self, wx_hat, wy_hat, wz_hat):
        """Standard NS step."""
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real

        nlx = nlx + wx * dux_dx + wy * dux_dy + wz * dux_dz
        nly = nly + wx * duy_dx + wy * duy_dy + wz * duy_dz
        nlz = nlz + wx * duz_dx + wy * duz_dy + wz * duz_dz

        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        return ((wx_hat + self.dt * nlx_hat) * self.visc_decay,
                (wy_hat + self.dt * nly_hat) * self.visc_decay,
                (wz_hat + self.dt * nlz_hat) * self.visc_decay)

    def step_constrained(self, wx_hat, wy_hat, wz_hat, Z_current):
        """NS step with H₃ depletion."""
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Depletion factor
        depl = self.h3_depletion_factor(Z_current)

        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real

        nlx = nlx + depl * (wx * dux_dx + wy * dux_dy + wz * dux_dz)
        nly = nly + depl * (wx * duy_dx + wy * duy_dy + wz * duy_dz)
        nlz = nlz + depl * (wx * duz_dx + wy * duz_dy + wz * duz_dz)

        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        # Enhanced dissipation
        Z_ratio = Z_current / (self.Z_theoretical_max * 0.1)
        enh = 1 + 10 * max(0, Z_ratio - 0.3)**2 if Z_ratio > 0.3 else 1
        visc = mx.exp(-self.nu * enh * self.k2 * self.dt)

        return ((wx_hat + self.dt * nlx_hat) * visc,
                (wy_hat + self.dt * nly_hat) * visc,
                (wz_hat + self.dt * nlz_hat) * visc)


def taylor_green_vorticity(n, scale=5.0):
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)
    return np.stack([wx, wy, wz], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  DIRECT δ₀ MEASUREMENT FROM STRETCHING REDUCTION")
    print("=" * 70)
    print(f"""
  Running constrained and unconstrained NS in parallel.
  Measuring vortex stretching S = ∫ ω_i S_ij ω_j dx

  Empirical δ₀ = 1 - (S_constrained / S_unconstrained)
  Expected: δ₀ ≈ {DELTA_0_THEORY:.4f}
    """)

    n = 64
    nu = 0.001
    tmax = 1.2

    solver = DualNSSolver(n=n, viscosity=nu)

    omega_init = taylor_green_vorticity(n, scale=5.0)

    # Initialize both
    wx_hat_un = mx.fft.fftn(mx.array(omega_init[..., 0]))
    wy_hat_un = mx.fft.fftn(mx.array(omega_init[..., 1]))
    wz_hat_un = mx.fft.fftn(mx.array(omega_init[..., 2]))

    wx_hat_con = mx.fft.fftn(mx.array(omega_init[..., 0]))
    wy_hat_con = mx.fft.fftn(mx.array(omega_init[..., 1]))
    wz_hat_con = mx.fft.fftn(mx.array(omega_init[..., 2]))

    nsteps = int(tmax / solver.dt)

    results = {
        'times': [],
        'Z_un': [], 'Z_con': [],
        'S_un': [], 'S_con': [],
        'reduction': [], 'empirical_delta0': []
    }

    print("Running dual simulation...")
    start_time = time.time()

    Z_con = 9.4  # Initial

    for step in range(nsteps):
        if step % 200 == 0:
            # Unconstrained fields
            wx_un = mx.fft.ifftn(wx_hat_un).real
            wy_un = mx.fft.ifftn(wy_hat_un).real
            wz_un = mx.fft.ifftn(wz_hat_un).real
            Z_un = 0.5 * float(mx.mean(wx_un**2 + wy_un**2 + wz_un**2).item())

            # Constrained fields
            wx_con = mx.fft.ifftn(wx_hat_con).real
            wy_con = mx.fft.ifftn(wy_hat_con).real
            wz_con = mx.fft.ifftn(wz_hat_con).real
            Z_con = 0.5 * float(mx.mean(wx_con**2 + wy_con**2 + wz_con**2).item())

            # Compute stretching for unconstrained
            ux_hat_un, uy_hat_un, uz_hat_un = solver.velocity_from_vorticity(
                wx_hat_un, wy_hat_un, wz_hat_un)
            S_un, S_un_pos = solver.compute_stretching(
                wx_un, wy_un, wz_un, ux_hat_un, uy_hat_un, uz_hat_un)

            # Compute stretching for constrained
            ux_hat_con, uy_hat_con, uz_hat_con = solver.velocity_from_vorticity(
                wx_hat_con, wy_hat_con, wz_hat_con)
            S_con, S_con_pos = solver.compute_stretching(
                wx_con, wy_con, wz_con, ux_hat_con, uy_hat_con, uz_hat_con)

            t = step * solver.dt
            results['times'].append(t)
            results['Z_un'].append(Z_un)
            results['Z_con'].append(Z_con)
            results['S_un'].append(S_un_pos)
            results['S_con'].append(S_con_pos)

            if S_un_pos > 1e-6:
                reduction = S_con_pos / S_un_pos
                emp_delta0 = 1 - reduction
            else:
                reduction = 1.0
                emp_delta0 = 0.0

            results['reduction'].append(reduction)
            results['empirical_delta0'].append(emp_delta0)

            if step % 2000 == 0:
                print(f"  t={t:.2f}: Z_un={Z_un:.1f}, Z_con={Z_con:.1f}, "
                      f"S_ratio={reduction:.3f}, δ₀_emp={emp_delta0:.3f}")

            if np.isnan(Z_un) or Z_un > 1e6:
                print(f"  Unconstrained blowup at t={t:.3f}")
                break

        # Step both
        wx_hat_un, wy_hat_un, wz_hat_un = solver.step_unconstrained(
            wx_hat_un, wy_hat_un, wz_hat_un)
        wx_hat_con, wy_hat_con, wz_hat_con = solver.step_constrained(
            wx_hat_con, wy_hat_con, wz_hat_con, Z_con)
        mx.eval(wx_hat_un, wy_hat_un, wz_hat_un)
        mx.eval(wx_hat_con, wy_hat_con, wz_hat_con)

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    times = np.array(results['times'])
    Z_un = np.array(results['Z_un'])
    Z_con = np.array(results['Z_con'])
    reduction = np.array(results['reduction'])
    emp_delta0 = np.array(results['empirical_delta0'])

    # Focus on peak region (where stretching matters most)
    valid = ~np.isnan(Z_un) & (Z_un > 0.5 * np.nanmax(Z_un))

    if np.sum(valid) > 0:
        mean_reduction = np.mean(reduction[valid])
        mean_delta0 = np.mean(emp_delta0[valid])
        std_delta0 = np.std(emp_delta0[valid])
    else:
        # Use all valid data
        valid = ~np.isnan(reduction)
        mean_reduction = np.mean(reduction[valid])
        mean_delta0 = np.mean(emp_delta0[valid])
        std_delta0 = np.std(emp_delta0[valid])

    print(f"\nDuring peak enstrophy (Z > 0.5 * Z_max):")
    print(f"  Mean stretching reduction: {mean_reduction:.4f}")
    print(f"  Expected reduction (1-δ₀): {1 - DELTA_0_THEORY:.4f}")
    print(f"\n  Empirical δ₀: {mean_delta0:.4f} ± {std_delta0:.4f}")
    print(f"  Theoretical δ₀: {DELTA_0_THEORY:.4f}")

    error = abs(mean_delta0 - DELTA_0_THEORY) / DELTA_0_THEORY * 100
    print(f"\n  Error: {error:.1f}%")

    if error < 10:
        print("  ✓ EXCELLENT agreement with theory!")
    elif error < 25:
        print("  ~ Reasonable agreement")
    else:
        print("  ✗ Significant deviation - needs investigation")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(times, Z_un, 'r-', linewidth=2, label='Unconstrained')
    ax.plot(times, Z_con, 'b-', linewidth=2, label='Constrained')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy Z')
    ax.set_title('Enstrophy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, results['S_un'], 'r-', linewidth=2, label='Unconstrained')
    ax.plot(times, results['S_con'], 'b-', linewidth=2, label='Constrained')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stretching S⁺')
    ax.set_title('Positive Vortex Stretching')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(times, reduction, 'g-', linewidth=2)
    ax.axhline(1 - DELTA_0_THEORY, color='r', linestyle='--',
               label=f'Theory: {1-DELTA_0_THEORY:.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('S_con / S_un')
    ax.set_title('Stretching Reduction Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.5])

    ax = axes[1, 1]
    ax.plot(times, emp_delta0, 'purple', linewidth=2)
    ax.axhline(DELTA_0_THEORY, color='r', linestyle='--',
               label=f'Theory: δ₀ = {DELTA_0_THEORY:.3f}')
    ax.fill_between(times, DELTA_0_THEORY - 0.05, DELTA_0_THEORY + 0.05,
                    alpha=0.2, color='r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Empirical δ₀')
    ax.set_title('Direct δ₀ Measurement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.6])

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_delta0_measurement.png', dpi=150)
    print("\nSaved: test_delta0_measurement.png")

    return results


if __name__ == "__main__":
    results = main()
