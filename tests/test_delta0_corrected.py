#!/usr/bin/env python3
"""
CORRECTED δ₀ MEASUREMENT WITH PROPER EIGENVALUE COMPUTATION

The alignment factor must use |λ_max| (spectral norm), not ||S||_F (Frobenius).

A = (ω^T S ω) / (|ω|² |λ_max|)

where λ_max is the largest eigenvalue of the strain tensor S.

This test computes eigenvalues at each grid point for accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Use numpy for eigenvalue computation (MLX doesn't have eigh)
# We'll still use MLX for the main simulation

try:
    import mlx.core as mx
    HAS_MLX = True
    print(f"MLX Device: {mx.default_device()}")
except ImportError:
    HAS_MLX = False
    print("MLX not available, using NumPy only")

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0_THEORY = (np.sqrt(5) - 1) / 4  # ≈ 0.309
BOUND = 1 - DELTA_0_THEORY  # ≈ 0.691

print(f"Theoretical δ₀ = {DELTA_0_THEORY:.6f}")
print(f"Alignment bound (1-δ₀) = {BOUND:.6f}")


class CorrectedAlignmentSolver:
    """NS solver with proper eigenvalue-based alignment measurement."""

    def __init__(self, n=64, viscosity=0.001, dt=0.0001):
        self.n = n
        self.nu = viscosity
        self.dt = dt

        # Wavenumbers
        k = np.fft.fftfreq(n, d=1/n) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = np.where(self.k2 == 0, 1e-10, self.k2)

        # Viscous decay
        self.visc_decay = np.exp(-self.nu * self.k2 * self.dt)

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat):
        """Biot-Savart law."""
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def compute_strain_tensor(self, ux_hat, uy_hat, uz_hat):
        """Compute strain rate tensor components."""
        dux_dx = np.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = np.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = np.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = np.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = np.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = np.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = np.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = np.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = np.fft.ifftn(1j * self.kz * uz_hat).real

        # Strain tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2
        S = np.zeros((self.n, self.n, self.n, 3, 3))
        S[..., 0, 0] = dux_dx
        S[..., 1, 1] = duy_dy
        S[..., 2, 2] = duz_dz
        S[..., 0, 1] = S[..., 1, 0] = 0.5 * (dux_dy + duy_dx)
        S[..., 0, 2] = S[..., 2, 0] = 0.5 * (dux_dz + duz_dx)
        S[..., 1, 2] = S[..., 2, 1] = 0.5 * (duy_dz + duz_dy)

        return S

    def compute_alignment_with_eigenvalues(self, wx, wy, wz, S, sample_frac=0.1):
        """
        Compute CORRECT alignment factor using eigenvalues.

        A = (ω^T S ω) / (|ω|² |λ_max|)

        Since eigenvalue computation is expensive, we sample a fraction of points.
        """
        omega_mag = np.sqrt(wx**2 + wy**2 + wz**2)

        # Only compute at high-vorticity points
        threshold = 0.3 * np.max(omega_mag)
        mask = omega_mag > threshold

        # Get indices of significant vorticity points
        indices = np.argwhere(mask)
        n_points = len(indices)

        if n_points == 0:
            return {'mean_A': 0, 'max_A': 0, 'samples': 0, 'empirical_delta0': 0}

        # Sample a subset for efficiency
        n_sample = max(100, int(sample_frac * n_points))
        n_sample = min(n_sample, n_points, 5000)  # Cap at 5000 points
        sample_idx = np.random.choice(n_points, n_sample, replace=False)

        alignments = []

        for idx in sample_idx:
            i, j, k = indices[idx]

            # Get vorticity at this point
            omega = np.array([wx[i, j, k], wy[i, j, k], wz[i, j, k]])
            omega_norm = omega_mag[i, j, k]

            if omega_norm < 1e-10:
                continue

            # Get strain tensor at this point
            S_local = S[i, j, k]

            # Compute eigenvalues of S (symmetric, so use eigh)
            eigenvalues = np.linalg.eigvalsh(S_local)
            lambda_max = np.max(np.abs(eigenvalues))

            if lambda_max < 1e-10:
                continue

            # Compute ω^T S ω
            omega_S_omega = omega @ S_local @ omega

            # Alignment factor with CORRECT denominator
            A = omega_S_omega / (omega_norm**2 * lambda_max)
            alignments.append(abs(A))

        if len(alignments) == 0:
            return {'mean_A': 0, 'max_A': 0, 'samples': 0, 'empirical_delta0': 0}

        alignments = np.array(alignments)
        mean_A = np.mean(alignments)
        max_A = np.max(alignments)

        # Empirical δ₀ = 1 - mean_A (if NS naturally exhibits the bound)
        empirical_delta0 = 1 - mean_A

        return {
            'mean_A': mean_A,
            'max_A': max_A,
            'std_A': np.std(alignments),
            'p90_A': np.percentile(alignments, 90),
            'p99_A': np.percentile(alignments, 99),
            'samples': len(alignments),
            'empirical_delta0': empirical_delta0
        }

    def step_unconstrained(self, wx_hat, wy_hat, wz_hat):
        """Standard NS step - NO depletion imposed."""
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        ux = np.fft.ifftn(ux_hat).real
        uy = np.fft.ifftn(uy_hat).real
        uz = np.fft.ifftn(uz_hat).real
        wx = np.fft.ifftn(wx_hat).real
        wy = np.fft.ifftn(wy_hat).real
        wz = np.fft.ifftn(wz_hat).real

        # Advection: ω × u
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Stretching: (ω · ∇)u - FULL, NO DEPLETION
        dux_dx = np.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = np.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = np.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = np.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = np.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = np.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = np.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = np.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = np.fft.ifftn(1j * self.kz * uz_hat).real

        nlx += wx * dux_dx + wy * dux_dy + wz * dux_dz
        nly += wx * duy_dx + wy * duy_dy + wz * duy_dz
        nlz += wx * duz_dx + wy * duz_dy + wz * duz_dz

        nlx_hat = np.fft.fftn(nlx)
        nly_hat = np.fft.fftn(nly)
        nlz_hat = np.fft.fftn(nlz)

        wx_hat_new = (wx_hat + self.dt * nlx_hat) * self.visc_decay
        wy_hat_new = (wy_hat + self.dt * nly_hat) * self.visc_decay
        wz_hat_new = (wz_hat + self.dt * nlz_hat) * self.visc_decay

        return wx_hat_new, wy_hat_new, wz_hat_new


def taylor_green_vorticity(n, scale=5.0):
    """Taylor-Green vortex initial condition."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)
    return np.stack([wx, wy, wz], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  CORRECTED δ₀ MEASUREMENT (EIGENVALUE-BASED)")
    print("=" * 70)
    print(f"""
  Using CORRECT alignment factor:
    A = (ω^T S ω) / (|ω|² |λ_max|)

  where λ_max = max eigenvalue of strain tensor S.

  Previous tests used Frobenius norm ||S||_F which UNDERESTIMATES A.
  This should close the 14% gap.
    """)

    n = 64
    nu = 0.001
    dt = 0.0001
    tmax = 2.0
    measure_interval = 500

    solver = CorrectedAlignmentSolver(n=n, viscosity=nu, dt=dt)

    omega_init = taylor_green_vorticity(n, scale=5.0)

    wx_hat = np.fft.fftn(omega_init[..., 0])
    wy_hat = np.fft.fftn(omega_init[..., 1])
    wz_hat = np.fft.fftn(omega_init[..., 2])

    nsteps = int(tmax / dt)

    results = {
        'times': [],
        'enstrophies': [],
        'mean_A': [],
        'max_A': [],
        'p90_A': [],
        'empirical_delta0': []
    }

    print(f"Parameters: n={n}, ν={nu}, dt={dt}, tmax={tmax}")
    print(f"Theoretical δ₀ = {DELTA_0_THEORY:.4f}")
    print(f"Theoretical bound (1-δ₀) = {BOUND:.4f}")
    print("\nRunning UNCONSTRAINED NS with eigenvalue-based alignment...")
    print()

    start_time = time.time()

    for step in range(nsteps):
        if step % measure_interval == 0:
            wx = np.fft.ifftn(wx_hat).real
            wy = np.fft.ifftn(wy_hat).real
            wz = np.fft.ifftn(wz_hat).real

            Z = 0.5 * np.mean(wx**2 + wy**2 + wz**2)

            # Compute strain tensor
            ux_hat, uy_hat, uz_hat = solver.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)
            S = solver.compute_strain_tensor(ux_hat, uy_hat, uz_hat)

            # Corrected alignment with eigenvalues
            align = solver.compute_alignment_with_eigenvalues(wx, wy, wz, S)

            t = step * dt
            results['times'].append(t)
            results['enstrophies'].append(Z)
            results['mean_A'].append(align['mean_A'])
            results['max_A'].append(align['max_A'])
            results['p90_A'].append(align.get('p90_A', 0))
            results['empirical_delta0'].append(align['empirical_delta0'])

            elapsed = time.time() - start_time
            eta = elapsed / max(step, 1) * (nsteps - step) / 60

            print(f"  t={t:.3f}: Z={Z:.1f}, A_mean={align['mean_A']:.4f}, "
                  f"A_max={align['max_A']:.4f}, δ₀_emp={align['empirical_delta0']:.4f} "
                  f"(n={align['samples']}) [ETA {eta:.1f}m]")

            if np.isnan(Z) or Z > 1e6:
                print(f"  BLOWUP at t={t:.3f}")
                break

        wx_hat, wy_hat, wz_hat = solver.step_unconstrained(wx_hat, wy_hat, wz_hat)

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    times = np.array(results['times'])
    Z = np.array(results['enstrophies'])
    mean_A = np.array(results['mean_A'])
    max_A = np.array(results['max_A'])
    emp_delta0 = np.array(results['empirical_delta0'])

    # Focus on peak enstrophy region
    Z_max = np.max(Z)
    peak_mask = Z > 0.3 * Z_max

    if np.sum(peak_mask) > 0:
        mean_A_peak = np.mean(mean_A[peak_mask])
        max_A_peak = np.max(max_A[peak_mask])
        emp_delta0_peak = np.mean(emp_delta0[peak_mask])
    else:
        mean_A_peak = np.mean(mean_A)
        max_A_peak = np.max(max_A)
        emp_delta0_peak = np.mean(emp_delta0)

    print(f"\nAlignment (during peak enstrophy Z > 0.3·Z_max):")
    print(f"  Mean alignment A:     {mean_A_peak:.4f}")
    print(f"  Max alignment A:      {max_A_peak:.4f}")
    print(f"  Theoretical bound:    {BOUND:.4f} (1-δ₀)")
    print(f"\n  Empirical δ₀ = 1 - A_mean: {emp_delta0_peak:.4f}")
    print(f"  Theoretical δ₀:            {DELTA_0_THEORY:.4f}")

    error = abs(emp_delta0_peak - DELTA_0_THEORY) / DELTA_0_THEORY * 100
    print(f"\n  Error: {error:.1f}%")

    if error < 5:
        print("  ✓ EXCELLENT: <5% error - δ₀ confirmed!")
    elif error < 10:
        print("  ✓ GOOD: <10% error")
    elif error < 15:
        print("  ~ ACCEPTABLE: 10-15% error")
    else:
        print(f"  ✗ SIGNIFICANT GAP: {error:.1f}% error")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(times, Z, 'b-', linewidth=2)
    ax.axhline(0.3 * Z_max, color='gray', linestyle=':', label='30% of peak')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy Z')
    ax.set_title('Enstrophy Evolution (Unconstrained)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, mean_A, 'g-', linewidth=2, label='Mean A')
    ax.plot(times, max_A, 'r-', linewidth=1, alpha=0.5, label='Max A')
    ax.axhline(BOUND, color='k', linestyle='--', linewidth=2, label=f'Bound (1-δ₀)={BOUND:.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Alignment Factor')
    ax.set_title('CORRECTED Alignment (Eigenvalue-based)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.2])

    ax = axes[1, 0]
    ax.plot(times, emp_delta0, 'purple', linewidth=2)
    ax.axhline(DELTA_0_THEORY, color='r', linestyle='--', linewidth=2,
               label=f'Theory δ₀={DELTA_0_THEORY:.4f}')
    ax.fill_between(times, DELTA_0_THEORY - 0.02, DELTA_0_THEORY + 0.02,
                    alpha=0.2, color='r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Empirical δ₀ = 1 - A')
    ax.set_title('Direct δ₀ Measurement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.6])

    ax = axes[1, 1]
    ax.scatter(Z, mean_A, c=times, cmap='viridis', alpha=0.6, s=30)
    ax.axhline(BOUND, color='r', linestyle='--', linewidth=2, label=f'Bound={BOUND:.3f}')
    ax.set_xlabel('Enstrophy Z')
    ax.set_ylabel('Mean Alignment A')
    ax.set_title('Alignment vs Enstrophy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Time')

    plt.tight_layout()
    plt.savefig('test_delta0_corrected.png', dpi=150)
    print("\nSaved: test_delta0_corrected.png")

    # Summary box
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"""
  Previous measurement (Frobenius norm): δ₀ ≈ 0.267 (14% error)
  Corrected measurement (eigenvalues):   δ₀ ≈ {emp_delta0_peak:.3f} ({error:.1f}% error)
  Theoretical value:                     δ₀ = {DELTA_0_THEORY:.4f}
    """)

    return results


if __name__ == "__main__":
    results = main()
