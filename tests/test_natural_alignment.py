#!/usr/bin/env python3
"""
TEST: NATURAL VORTICITY-STRAIN ALIGNMENT IN UNCONSTRAINED NS

Critical question: Does standard Navier-Stokes NATURALLY exhibit
the H₃ geometric constraint, or must it be imposed?

Theory predicts:
- Alignment factor A = |ω·e_inter| / |ω| should be bounded
- On H₃ manifold: A ≤ 1 - δ₀ ≈ 0.691
- If NS naturally exhibits this, it's evidence for geometric regularity

This test runs UNCONSTRAINED NS and measures the alignment.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import time

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4  # ≈ 0.309
BOUND = 1 - DELTA_0  # ≈ 0.691

print(f"MLX Device: {mx.default_device()}")
print(f"δ₀ = {DELTA_0:.6f}")
print(f"Alignment bound (1-δ₀) = {BOUND:.6f}")


class NaturalAlignmentTest:
    """NS solver that measures natural vorticity-strain alignment."""

    def __init__(self, n=64, viscosity=0.001, dt=0.0001):
        self.n = n
        self.nu = viscosity
        self.dt = dt

        # Wavenumbers
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)

        # Viscous decay
        self.visc_decay = mx.exp(-self.nu * self.k2 * self.dt)

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat):
        """Biot-Savart law."""
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def compute_strain_tensor(self, ux_hat, uy_hat, uz_hat):
        """
        Compute strain rate tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2.
        Returns S as (n, n, n, 3, 3) array.
        """
        # Velocity gradients via spectral differentiation
        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real

        # Build strain tensor by stacking
        # S[i,j] = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_01 = 0.5 * (dux_dy + duy_dx)
        S_02 = 0.5 * (dux_dz + duz_dx)
        S_12 = 0.5 * (duy_dz + duz_dy)

        # Build rows then stack
        row0 = mx.stack([dux_dx, S_01, S_02], axis=-1)  # (n,n,n,3)
        row1 = mx.stack([S_01, duy_dy, S_12], axis=-1)
        row2 = mx.stack([S_02, S_12, duz_dz], axis=-1)

        S = mx.stack([row0, row1, row2], axis=-2)  # (n,n,n,3,3)
        return S

    def compute_alignment(self, wx, wy, wz, S):
        """
        Compute vorticity-strain alignment factor.

        The alignment factor is: A = (ω^T S ω) / (|ω|² |λ_max|)

        Returns statistics over the domain.
        """
        # Stack vorticity
        omega = mx.stack([wx, wy, wz], axis=-1)  # (n, n, n, 3)
        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2 + 1e-10)

        # Compute ω^T S ω at each point using explicit matrix multiplication
        # S @ ω: (n,n,n,3,3) @ (n,n,n,3) -> need to expand omega for matmul
        omega_expanded = omega[..., :, None]  # (n,n,n,3,1)
        S_omega = mx.squeeze(S @ omega_expanded, axis=-1)  # (n,n,n,3)

        # ω · (S @ ω) -> scalar at each point
        omega_S_omega = mx.sum(omega * S_omega, axis=-1)

        # Compute |λ_max| using Frobenius norm as upper bound
        # |λ_max| ≤ ||S||_F / sqrt(3) for 3x3 matrix
        S_frob = mx.sqrt(mx.sum(S**2, axis=(-2, -1)) + 1e-10)

        # Alignment factor: A = (ω^T S ω) / (|ω|² ||S||_F)
        alignment = omega_S_omega / (omega_mag**2 * S_frob + 1e-10)

        # Only consider points with significant vorticity
        threshold = 0.1 * mx.mean(omega_mag)
        mask = omega_mag > threshold

        # Compute statistics on masked data
        alignment_abs = mx.abs(alignment)
        alignment_masked = mx.where(mask, alignment_abs, mx.array(0.0))
        n_valid = mx.sum(mask.astype(mx.float32))

        # Mean over valid points
        mean_alignment = mx.sum(alignment_masked) / (n_valid + 1e-10)
        max_alignment = mx.max(alignment_masked)

        # Stretching term
        stretching = mx.mean(omega_S_omega)

        # Fraction exceeding bound
        exceeds_bound = mx.where(mask, alignment_abs > BOUND, mx.array(False))
        frac_exceeding = mx.sum(exceeds_bound.astype(mx.float32)) / (n_valid + 1e-10)

        # 90th and 99th percentile (approximate via sorting subset)
        # For efficiency, sample a subset
        alignment_flat = mx.reshape(alignment_masked, (-1,))
        # Sort and get percentiles
        sorted_a = mx.sort(alignment_flat)
        n_total = alignment_flat.shape[0]
        p90_idx = int(0.90 * n_total)
        p99_idx = int(0.99 * n_total)
        p90 = sorted_a[p90_idx]
        p99 = sorted_a[p99_idx]

        mx.eval(mean_alignment, max_alignment, stretching, n_valid, frac_exceeding, p90, p99)

        return {
            'mean_alignment': float(mean_alignment.item()),
            'max_alignment': float(max_alignment.item()),
            'stretching': float(stretching.item()),
            'n_valid': int(n_valid.item()),
            'frac_exceeding': float(frac_exceeding.item()),
            'p90': float(p90.item()),
            'p99': float(p99.item())
        }

    def step_unconstrained(self, wx_hat, wy_hat, wz_hat):
        """Advance one timestep - STANDARD NS, NO DEPLETION."""
        # Get velocity
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        # Transform to physical space
        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        # Advection: ω × u
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Stretching: (ω · ∇)u - NO DEPLETION APPLIED
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

        # Total nonlinear (UNCONSTRAINED)
        nlx = nlx + stretch_x
        nly = nly + stretch_y
        nlz = nlz + stretch_z

        # Transform to Fourier
        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        # Update with viscous decay
        wx_hat_new = (wx_hat + self.dt * nlx_hat) * self.visc_decay
        wy_hat_new = (wy_hat + self.dt * nly_hat) * self.visc_decay
        wz_hat_new = (wz_hat + self.dt * nlz_hat) * self.visc_decay

        return wx_hat_new, wy_hat_new, wz_hat_new

    def run(self, omega_init, tmax=1.0, measure_interval=100):
        """Run UNCONSTRAINED simulation and measure alignment."""
        wx = mx.array(omega_init[..., 0])
        wy = mx.array(omega_init[..., 1])
        wz = mx.array(omega_init[..., 2])

        wx_hat = mx.fft.fftn(wx)
        wy_hat = mx.fft.fftn(wy)
        wz_hat = mx.fft.fftn(wz)

        nsteps = int(tmax / self.dt)
        results = {
            'times': [],
            'enstrophies': [],
            'mean_alignments': [],
            'max_alignments': [],
            'stretchings': [],
            'frac_exceeding': [],
            'p90': [],
            'p99': []
        }

        start_time = time.time()

        for step in range(nsteps):
            if step % measure_interval == 0:
                # Get physical fields
                ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)
                wx = mx.fft.ifftn(wx_hat).real
                wy = mx.fft.ifftn(wy_hat).real
                wz = mx.fft.ifftn(wz_hat).real

                # Compute strain tensor
                S = self.compute_strain_tensor(ux_hat, uy_hat, uz_hat)

                # Compute alignment
                align = self.compute_alignment(wx, wy, wz, S)

                # Enstrophy
                Z = 0.5 * float(mx.mean(wx**2 + wy**2 + wz**2).item())

                results['times'].append(step * self.dt)
                results['enstrophies'].append(Z)
                results['mean_alignments'].append(align['mean_alignment'])
                results['max_alignments'].append(align['max_alignment'])
                results['stretchings'].append(align['stretching'])
                results['frac_exceeding'].append(align['frac_exceeding'])
                results['p90'].append(align['p90'])
                results['p99'].append(align['p99'])

                if step % (measure_interval * 10) == 0:
                    elapsed = time.time() - start_time
                    print(f"  t={step*self.dt:.3f}, Z={Z:.2f}, "
                          f"A_mean={align['mean_alignment']:.4f}, "
                          f"A_max={align['max_alignment']:.4f}, "
                          f"frac>{BOUND:.2f}: {align['frac_exceeding']*100:.1f}%")

                # Check for blowup
                if np.isnan(Z) or Z > 1e6:
                    print(f"  BLOWUP at t={step*self.dt:.3f}")
                    break

            wx_hat, wy_hat, wz_hat = self.step_unconstrained(wx_hat, wy_hat, wz_hat)
            mx.eval(wx_hat, wy_hat, wz_hat)

        return results


def taylor_green_vorticity(n, scale=1.0):
    """Taylor-Green initial vorticity with perturbation."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)

    # Perturbation
    theta = np.arctan2(Y - np.pi, X - np.pi)
    pert = 0.1 * scale * np.sin(8 * theta)
    wx += pert * np.cos(Z)
    wy += pert * np.sin(Z)

    return np.stack([wx, wy, wz], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  NATURAL ALIGNMENT TEST - UNCONSTRAINED NAVIER-STOKES")
    print("=" * 70)
    print("""
  QUESTION: Does standard NS naturally exhibit the H₃ geometric constraint?

  If alignment A naturally stays below (1-δ₀) ≈ 0.691, it suggests
  the geometric constraint is INTRINSIC to fluid dynamics, not imposed.
    """)

    n = 64
    nu = 0.001  # Lower viscosity for higher Re
    scale = 5.0  # Higher amplitude
    tmax = 2.0  # Run longer to see blowup approach

    omega_init = taylor_green_vorticity(n, scale=scale)

    print(f"Parameters: n={n}, ν={nu}, scale={scale}, tmax={tmax}")
    print(f"Initial max vorticity: {np.max(np.linalg.norm(omega_init, axis=-1)):.2f}")
    print(f"Theoretical bound (1-δ₀): {BOUND:.4f}")

    print("\n--- Running UNCONSTRAINED NS ---")
    solver = NaturalAlignmentTest(n=n, viscosity=nu)
    results = solver.run(omega_init, tmax=tmax, measure_interval=50)

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    times = np.array(results['times'])
    Z = np.array(results['enstrophies'])
    A_mean = np.array(results['mean_alignments'])
    A_max = np.array(results['max_alignments'])

    # Filter valid data
    valid = ~np.isnan(Z) & ~np.isnan(A_mean)
    times = times[valid]
    Z = Z[valid]
    A_mean = A_mean[valid]
    A_max = A_max[valid]

    if len(times) == 0:
        print("No valid data collected!")
        return

    frac_exc = np.array(results['frac_exceeding'])[valid]
    p90 = np.array(results['p90'])[valid]
    p99 = np.array(results['p99'])[valid]

    print(f"\nAlignment statistics (UNCONSTRAINED):")
    print(f"  Mean alignment:     {np.mean(A_mean):.4f} ± {np.std(A_mean):.4f}")
    print(f"  90th percentile:    {np.mean(p90):.4f}")
    print(f"  99th percentile:    {np.mean(p99):.4f}")
    print(f"  Max alignment:      {np.max(A_max):.4f}")
    print(f"  Theoretical bound:  {BOUND:.4f}")
    print(f"\n  Fraction of domain exceeding bound: {np.mean(frac_exc)*100:.2f}%")

    # Key test: Is alignment naturally below the bound?
    below_bound = np.mean(A_mean) < BOUND
    max_below_bound = np.max(A_max) < BOUND

    print(f"\n  Mean alignment < (1-δ₀)?  {'YES' if below_bound else 'NO'}")
    print(f"  Max alignment < (1-δ₀)?   {'YES' if max_below_bound else 'NO'}")

    # Fraction of time below bound
    frac_below = np.mean(A_mean < BOUND)
    print(f"  Fraction of time below bound: {frac_below*100:.1f}%")

    # Correlation with enstrophy
    if len(Z) > 3:
        corr = np.corrcoef(Z, A_mean)[0, 1]
        print(f"  Correlation(Z, A): {corr:.3f}")

    # Verdict
    print("\n" + "-" * 70)
    if below_bound and frac_below > 0.9:
        print("  RESULT: Alignment NATURALLY below bound!")
        print("  This suggests H₃ geometry may be INTRINSIC to NS.")
    elif frac_below > 0.5:
        print("  RESULT: Alignment MOSTLY below bound.")
        print("  Partial evidence for natural geometric constraint.")
    else:
        print("  RESULT: Alignment EXCEEDS bound frequently.")
        print("  The H₃ constraint must be IMPOSED, not natural.")
    print("-" * 70)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(times, Z, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy Z(t)')
    ax.set_title('Enstrophy Evolution (Unconstrained)')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, A_mean, 'g-', linewidth=2, label='Mean alignment')
    ax.axhline(BOUND, color='r', linestyle='--', linewidth=2, label=f'Bound (1-δ₀)={BOUND:.3f}')
    ax.fill_between(times, A_mean - np.std(A_mean), A_mean + np.std(A_mean), alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Alignment Factor A')
    ax.set_title('Vorticity-Strain Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(Z, A_mean, c=times, cmap='viridis', alpha=0.6)
    ax.axhline(BOUND, color='r', linestyle='--', linewidth=2, label='Bound')
    ax.set_xlabel('Enstrophy Z')
    ax.set_ylabel('Alignment A')
    ax.set_title('Alignment vs Enstrophy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Time')

    ax = axes[1, 1]
    ax.hist(A_mean, bins=30, density=True, alpha=0.7, color='green')
    ax.axvline(BOUND, color='r', linestyle='--', linewidth=2, label=f'Bound={BOUND:.3f}')
    ax.axvline(np.mean(A_mean), color='blue', linestyle='-', linewidth=2, label=f'Mean={np.mean(A_mean):.3f}')
    ax.set_xlabel('Alignment Factor')
    ax.set_ylabel('Density')
    ax.set_title('Alignment Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_natural_alignment.png', dpi=150)
    print("\nSaved: test_natural_alignment.png")

    return results


if __name__ == "__main__":
    results = main()
