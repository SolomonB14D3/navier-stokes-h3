#!/usr/bin/env python3
"""
TEST: NATURAL ALIGNMENT IN UNBOUNDED DOMAIN

Uses a sponge/damping layer near boundaries to simulate ℝ³ (unbounded domain).
This addresses the gap between periodic torus tests and the Clay problem on ℝ³.

Key technique:
- Center coordinates at origin
- Apply Gaussian damper for r > r_max
- Confine forcing to core region
- Effectively simulates localized vorticity in infinite domain
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Use MLX for GPU acceleration
import mlx.core as mx

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4  # ≈ 0.309
ALIGNMENT_BOUND = 1 - DELTA_0   # ≈ 0.691

print(f"δ₀ = {DELTA_0:.6f}")
print(f"Alignment bound (1-δ₀) = {ALIGNMENT_BOUND:.6f}")


class UnboundedNSSolver:
    """
    NS solver with sponge layer to simulate unbounded domain.

    Uses damping near boundaries to absorb outgoing waves,
    effectively simulating ℝ³ instead of periodic torus.
    """

    def __init__(self, n=64, viscosity=0.001, dt=0.0001):
        self.n = n
        self.nu = viscosity
        self.dt = dt
        self.L = 2 * np.pi  # Domain [-π, π]³

        # Centered coordinates
        x = np.linspace(-np.pi, np.pi, n, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)

        # Sponge layer parameters
        r_max = np.pi * 0.85  # Core radius
        sigma = np.pi * 0.15   # Damping width

        # Gaussian damper: 1 in core, exponential decay outside
        damper = np.exp(-((np.maximum(r - r_max, 0))**2) / (2 * sigma**2))
        self.damper = mx.array(damper)

        # Core mask for forcing
        self.core_mask = mx.array((r < r_max).astype(np.float32))

        # Store r for diagnostics
        self.r = mx.array(r)
        self.r_max = r_max

        # Wavenumbers (standard periodic, but damping handles boundaries)
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)

        # Viscous decay
        self.visc_decay = mx.exp(-self.nu * self.k2 * self.dt)

        print(f"Unbounded domain simulation:")
        print(f"  Core radius: {r_max:.3f} ({r_max/np.pi*100:.1f}% of half-domain)")
        print(f"  Damping width: {sigma:.3f}")
        print(f"  Effective volume: {4/3*np.pi*r_max**3:.1f} (vs {(2*np.pi)**3:.1f} full)")

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat):
        """Biot-Savart law in Fourier space."""
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def apply_damping(self, wx, wy, wz):
        """Apply sponge layer damping."""
        return wx * self.damper, wy * self.damper, wz * self.damper

    def compute_strain_tensor(self, ux_hat, uy_hat, uz_hat):
        """Compute strain rate tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2."""
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

        # Symmetric strain tensor
        S_01 = 0.5 * (dux_dy + duy_dx)
        S_02 = 0.5 * (dux_dz + duz_dx)
        S_12 = 0.5 * (duy_dz + duz_dy)

        # Build tensor using stack
        row0 = mx.stack([dux_dx, S_01, S_02], axis=-1)
        row1 = mx.stack([S_01, duy_dy, S_12], axis=-1)
        row2 = mx.stack([S_02, S_12, duz_dz], axis=-1)
        S = mx.stack([row0, row1, row2], axis=-2)

        return S

    def compute_alignment(self, omega, S):
        """
        Compute vorticity-strain alignment factor A = (ω^T S ω) / (|ω|² ||S||).
        """
        omega_mag_sq = mx.sum(omega**2, axis=-1)
        omega_mag = mx.sqrt(omega_mag_sq + 1e-10)

        # S norm (Frobenius)
        S_norm = mx.sqrt(mx.sum(S**2, axis=(-2, -1)) + 1e-10)

        # ω^T S ω
        omega_expanded = omega[..., :, None]  # (n,n,n,3,1)
        S_omega = mx.squeeze(S @ omega_expanded, axis=-1)  # (n,n,n,3)
        omega_S_omega = mx.sum(omega * S_omega, axis=-1)  # (n,n,n)

        # Alignment factor
        A = omega_S_omega / (omega_mag_sq * S_norm + 1e-10)

        return A, omega_mag

    def step(self, wx_hat, wy_hat, wz_hat):
        """Advance one timestep (unconstrained NS with damping)."""
        # Get velocity
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        # Transform to physical space
        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        # Apply damping to contain vorticity in core
        wx, wy, wz = self.apply_damping(wx, wy, wz)
        ux = ux * self.damper
        uy = uy * self.damper
        uz = uz * self.damper

        # Advection: ω × u
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Stretching: (ω · ∇)u
        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real * self.damper
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real * self.damper
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real * self.damper
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real * self.damper
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real * self.damper
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real * self.damper
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real * self.damper
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real * self.damper
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real * self.damper

        stretch_x = wx * dux_dx + wy * dux_dy + wz * dux_dz
        stretch_y = wx * duy_dx + wy * duy_dy + wz * duy_dz
        stretch_z = wx * duz_dx + wy * duz_dy + wz * duz_dz

        nlx = nlx + stretch_x
        nly = nly + stretch_y
        nlz = nlz + stretch_z

        # Apply core mask to nonlinear terms (confine dynamics to core)
        nlx = nlx * self.core_mask
        nly = nly * self.core_mask
        nlz = nlz * self.core_mask

        # Transform to Fourier
        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        # Update with viscous decay
        wx_hat_new = (wx_hat + self.dt * nlx_hat) * self.visc_decay
        wy_hat_new = (wy_hat + self.dt * nly_hat) * self.visc_decay
        wz_hat_new = (wz_hat + self.dt * nlz_hat) * self.visc_decay

        return wx_hat_new, wy_hat_new, wz_hat_new

    def run_and_measure(self, omega_init, tmax=1.5, measure_interval=100):
        """Run simulation and measure alignment statistics."""
        wx = mx.array(omega_init[..., 0])
        wy = mx.array(omega_init[..., 1])
        wz = mx.array(omega_init[..., 2])

        # Apply initial damping
        wx, wy, wz = self.apply_damping(wx, wy, wz)

        wx_hat = mx.fft.fftn(wx)
        wy_hat = mx.fft.fftn(wy)
        wz_hat = mx.fft.fftn(wz)

        nsteps = int(tmax / self.dt)
        results = {
            'times': [],
            'enstrophies': [],
            'mean_alignment': [],
            'max_alignment': [],
            'p99_alignment': [],
            'fraction_exceeding': [],
            'core_enstrophy': []
        }

        start_time = time.time()

        for step in range(nsteps):
            if step % measure_interval == 0:
                # Get physical fields
                wx = mx.fft.ifftn(wx_hat).real
                wy = mx.fft.ifftn(wy_hat).real
                wz = mx.fft.ifftn(wz_hat).real

                # Apply damping for measurement
                wx_d, wy_d, wz_d = self.apply_damping(wx, wy, wz)

                # Enstrophy (full and core)
                Z_full = 0.5 * float(mx.mean(wx_d**2 + wy_d**2 + wz_d**2).item())
                Z_core = 0.5 * float(mx.mean((wx_d**2 + wy_d**2 + wz_d**2) * self.core_mask).item())

                # Get velocity for strain computation
                ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

                # Compute strain tensor
                S = self.compute_strain_tensor(ux_hat, uy_hat, uz_hat)

                # Compute alignment
                omega = mx.stack([wx_d, wy_d, wz_d], axis=-1)
                A, omega_mag = self.compute_alignment(omega, S)

                # Only analyze core region with significant vorticity
                core_mask_np = np.array(self.core_mask)
                omega_mag_np = np.array(omega_mag)
                A_np = np.array(A)

                # High vorticity threshold (mean + 1 std in core)
                core_omega = omega_mag_np[core_mask_np > 0.5]
                if len(core_omega) > 0:
                    omega_thresh = np.mean(core_omega) + np.std(core_omega)
                    high_vort_mask = (omega_mag_np > omega_thresh) & (core_mask_np > 0.5)

                    if np.sum(high_vort_mask) > 10:
                        A_high = A_np[high_vort_mask]

                        mean_A = np.mean(A_high)
                        max_A = np.max(A_high)
                        p99_A = np.percentile(A_high, 99)
                        frac_exceed = np.mean(A_high > ALIGNMENT_BOUND)
                    else:
                        mean_A = max_A = p99_A = frac_exceed = np.nan
                else:
                    mean_A = max_A = p99_A = frac_exceed = np.nan

                results['times'].append(step * self.dt)
                results['enstrophies'].append(Z_full)
                results['core_enstrophy'].append(Z_core)
                results['mean_alignment'].append(mean_A)
                results['max_alignment'].append(max_A)
                results['p99_alignment'].append(p99_A)
                results['fraction_exceeding'].append(frac_exceed)

                if step % (measure_interval * 10) == 0:
                    elapsed = time.time() - start_time
                    print(f"  t={step*self.dt:.3f}, Z_core={Z_core:.2f}, "
                          f"A_mean={mean_A:.3f}, A_max={max_A:.3f}, "
                          f"frac>{1-DELTA_0:.2f}: {frac_exceed:.1%}")

                # Check for blowup
                if np.isnan(Z_full) or Z_full > 1e6:
                    print(f"  BLOWUP at t={step*self.dt:.3f}")
                    break

            wx_hat, wy_hat, wz_hat = self.step(wx_hat, wy_hat, wz_hat)
            mx.eval(wx_hat, wy_hat, wz_hat)

        return results


def localized_vorticity(n, scale=5.0, center_scale=0.5):
    """
    Create localized vorticity blob in center of domain.

    This represents a finite-energy configuration in unbounded domain.
    """
    x = np.linspace(-np.pi, np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Gaussian envelope
    envelope = np.exp(-r**2 / (2 * (center_scale * np.pi)**2))

    # Taylor-Green like structure inside
    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z) * envelope
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z) * envelope
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z) * envelope

    # Add some helical perturbation
    theta = np.arctan2(Y, X)
    wx += 0.2 * scale * np.sin(6 * theta) * np.cos(Z) * envelope
    wy += 0.2 * scale * np.cos(6 * theta) * np.sin(Z) * envelope

    return np.stack([wx, wy, wz], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  UNBOUNDED DOMAIN TEST - NATURAL ALIGNMENT IN ℝ³")
    print("=" * 70)
    print("""
  This test simulates NS in an effectively unbounded domain by:
  1. Centering coordinates at origin
  2. Using sponge layer (Gaussian damping) near boundaries
  3. Confining forcing/dynamics to core region

  This addresses the gap between periodic tests and the Clay problem on ℝ³.
    """)

    n = 64
    nu = 0.001
    scale = 5.0
    tmax = 1.5

    print(f"Parameters: n={n}, ν={nu}, scale={scale}, tmax={tmax}")
    print(f"Re ≈ {scale * np.pi / nu:.0f}")

    # Create localized initial vorticity
    omega_init = localized_vorticity(n, scale=scale)
    print(f"Initial max vorticity: {np.max(np.linalg.norm(omega_init, axis=-1)):.2f}")

    print("\n--- Running unbounded domain simulation ---")
    solver = UnboundedNSSolver(n=n, viscosity=nu)
    results = solver.run_and_measure(omega_init, tmax=tmax, measure_interval=50)

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS - UNBOUNDED DOMAIN")
    print("=" * 70)

    times = np.array(results['times'])
    Z = np.array(results['enstrophies'])
    Z_core = np.array(results['core_enstrophy'])
    mean_A = np.array(results['mean_alignment'])
    max_A = np.array(results['max_alignment'])
    p99_A = np.array(results['p99_alignment'])
    frac_exceed = np.array(results['fraction_exceeding'])

    # Filter valid
    valid = ~np.isnan(Z) & ~np.isnan(mean_A)

    if np.sum(valid) < 3:
        print("Insufficient valid data points!")
        return results

    times_v = times[valid]
    mean_A_v = mean_A[valid]
    max_A_v = max_A[valid]
    p99_A_v = p99_A[valid]
    frac_exceed_v = frac_exceed[valid]
    Z_v = Z[valid]

    print(f"\nAlignment statistics (high-vorticity core regions):")
    print(f"  Mean alignment:       {np.mean(mean_A_v):.4f}")
    print(f"  99th percentile:      {np.mean(p99_A_v):.4f}")
    print(f"  Max alignment:        {np.max(max_A_v):.4f}")
    print(f"  Fraction > (1-δ₀):    {np.mean(frac_exceed_v):.2%}")
    print(f"  Theoretical bound:    {ALIGNMENT_BOUND:.4f}")

    # Comparison with periodic domain
    print(f"\nComparison with periodic domain results:")
    print(f"  Periodic: ~6% exceeds bound, mean A ≈ 0.35")
    print(f"  Unbounded: {np.mean(frac_exceed_v):.1%} exceeds bound, mean A ≈ {np.mean(mean_A_v):.2f}")

    if np.mean(frac_exceed_v) < 0.10:
        print("\n  ✓ Similar behavior to periodic - gap NOT significant")
    else:
        print("\n  ✗ Different behavior - bounded vs unbounded matters")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(times_v, Z_v, 'b-', linewidth=2, label='Total')
    ax.plot(times[valid], np.array(results['core_enstrophy'])[valid], 'g--',
            linewidth=2, label='Core')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy')
    ax.set_title('Enstrophy Evolution (Unbounded Domain)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times_v, mean_A_v, 'b-', linewidth=2, label='Mean')
    ax.plot(times_v, p99_A_v, 'r--', linewidth=2, label='99th percentile')
    ax.axhline(ALIGNMENT_BOUND, color='k', linestyle=':',
               label=f'Bound (1-δ₀)={ALIGNMENT_BOUND:.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Alignment Factor A')
    ax.set_title('Vorticity-Strain Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    ax = axes[1, 0]
    ax.plot(times_v, frac_exceed_v * 100, 'r-', linewidth=2)
    ax.axhline(6, color='b', linestyle='--', label='Periodic result (~6%)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction Exceeding Bound (%)')
    ax.set_title('Fraction Violating Geometric Constraint')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(Z_v, frac_exceed_v * 100, c=times_v, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Enstrophy Z')
    ax.set_ylabel('Fraction Exceeding (%)')
    ax.set_title('Violation vs Enstrophy')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Time')

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_unbounded_alignment.png', dpi=150)
    print("\nSaved: test_unbounded_alignment.png")

    return results


if __name__ == "__main__":
    results = main()
