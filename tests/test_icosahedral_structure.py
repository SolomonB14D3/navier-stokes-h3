#!/usr/bin/env python3
"""
TEST: ICOSAHEDRAL STRUCTURE IN VORTEX POSITIONS

Uses Steinhardt order parameters (Q6) to detect whether vortex cores
naturally arrange in icosahedral patterns during NS evolution.

Q6 values:
- Random/liquid: ~0.0-0.1
- FCC crystal: ~0.57
- Icosahedral: ~0.66
- BCC crystal: ~0.51

If vortex cores show elevated Q6 (especially near 0.66), it suggests
icosahedral geometry is INTRINSIC to turbulent structure.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import freud
    FREUD_AVAILABLE = True
except ImportError:
    FREUD_AVAILABLE = False
    print("Warning: freud not available. Install with: pip install freud-analysis")

# Also use MLX for NS simulation
import mlx.core as mx
import time

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4

print(f"Freud available: {FREUD_AVAILABLE}")


class NSVortexTracker:
    """NS solver that tracks vortex core positions."""

    def __init__(self, n=64, viscosity=0.001, dt=0.0001):
        self.n = n
        self.nu = viscosity
        self.dt = dt
        self.box_size = 2 * np.pi

        # Wavenumbers
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)
        self.visc_decay = mx.exp(-self.nu * self.k2 * self.dt)

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat):
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def step(self, wx_hat, wy_hat, wz_hat):
        """Standard NS step (no constraint)."""
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

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

        nlx = nlx + stretch_x
        nly = nly + stretch_y
        nlz = nlz + stretch_z

        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        wx_hat_new = (wx_hat + self.dt * nlx_hat) * self.visc_decay
        wy_hat_new = (wy_hat + self.dt * nly_hat) * self.visc_decay
        wz_hat_new = (wz_hat + self.dt * nlz_hat) * self.visc_decay

        return wx_hat_new, wy_hat_new, wz_hat_new

    def extract_vortex_positions(self, wx, wy, wz, threshold_factor=2.0):
        """Extract positions of vortex cores (high vorticity regions)."""
        omega_mag = np.sqrt(np.array(wx)**2 + np.array(wy)**2 + np.array(wz)**2)
        threshold = threshold_factor * np.mean(omega_mag)

        # Find points above threshold
        mask = omega_mag > threshold
        indices = np.argwhere(mask)

        if len(indices) == 0:
            return np.array([]).reshape(0, 3), omega_mag

        # Convert to physical coordinates
        positions = indices.astype(np.float32) * (self.box_size / self.n)

        return positions, omega_mag

    def run_and_track(self, omega_init, tmax=1.0, measure_interval=100):
        """Run simulation and track vortex structure."""
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
            'n_vortices': [],
            'q6_values': [],
            'q4_values': []
        }

        start_time = time.time()

        for step in range(nsteps):
            if step % measure_interval == 0:
                wx = mx.fft.ifftn(wx_hat).real
                wy = mx.fft.ifftn(wy_hat).real
                wz = mx.fft.ifftn(wz_hat).real

                Z = 0.5 * float(mx.mean(wx**2 + wy**2 + wz**2).item())

                # Extract vortex positions
                positions, omega_mag = self.extract_vortex_positions(wx, wy, wz)
                n_vortices = len(positions)

                # Compute Steinhardt order parameters
                q6, q4 = self.compute_steinhardt(positions)

                results['times'].append(step * self.dt)
                results['enstrophies'].append(Z)
                results['n_vortices'].append(n_vortices)
                results['q6_values'].append(q6)
                results['q4_values'].append(q4)

                if step % (measure_interval * 10) == 0:
                    elapsed = time.time() - start_time
                    q6_str = f"{q6:.4f}" if q6 is not None else "N/A"
                    print(f"  t={step*self.dt:.3f}, Z={Z:.2f}, "
                          f"N_vortex={n_vortices}, Q6={q6_str}")

                if np.isnan(Z) or Z > 1e6:
                    print(f"  BLOWUP at t={step*self.dt:.3f}")
                    break

            wx_hat, wy_hat, wz_hat = self.step(wx_hat, wy_hat, wz_hat)
            mx.eval(wx_hat, wy_hat, wz_hat)

        return results

    def compute_steinhardt(self, positions, l_values=[4, 6]):
        """Compute Steinhardt order parameters Q_l."""
        if not FREUD_AVAILABLE:
            return None, None

        if len(positions) < 13:  # Need at least 12 neighbors + 1
            return None, None

        try:
            # Set up periodic box
            freud_box = freud.box.Box(
                Lx=self.box_size, Ly=self.box_size, Lz=self.box_size, is2D=False
            )

            # Neighbor query parameters
            # Use adaptive r_max based on typical vortex spacing
            n_vortices = len(positions)
            estimated_spacing = self.box_size / (n_vortices ** (1/3))
            r_max = min(2.0 * estimated_spacing, self.box_size / 2)

            # Q6 (icosahedral order)
            steinhardt6 = freud.order.Steinhardt(l=6)
            steinhardt6.compute(
                system=(freud_box, positions),
                neighbors={'r_max': r_max, 'num_neighbors': 12}
            )
            q6 = np.mean(steinhardt6.ql)

            # Q4 (cubic order for comparison)
            steinhardt4 = freud.order.Steinhardt(l=4)
            steinhardt4.compute(
                system=(freud_box, positions),
                neighbors={'r_max': r_max, 'num_neighbors': 12}
            )
            q4 = np.mean(steinhardt4.ql)

            return q6, q4

        except Exception as e:
            print(f"Steinhardt computation failed: {e}")
            return None, None


def taylor_green_vorticity(n, scale=1.0):
    """Taylor-Green initial vorticity."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)

    theta = np.arctan2(Y - np.pi, X - np.pi)
    pert = 0.1 * scale * np.sin(8 * theta)
    wx += pert * np.cos(Z)
    wy += pert * np.sin(Z)

    return np.stack([wx, wy, wz], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  ICOSAHEDRAL STRUCTURE TEST - VORTEX CORE ARRANGEMENT")
    print("=" * 70)
    print("""
  QUESTION: Do vortex cores naturally arrange in icosahedral patterns?

  Steinhardt Q6 reference values:
    Random/liquid:  ~0.0-0.1
    BCC crystal:    ~0.51
    FCC crystal:    ~0.57
    ICOSAHEDRAL:    ~0.66

  If vortex cores show Q6 > 0.4, especially near 0.66, it suggests
  icosahedral geometry is INTRINSIC to turbulent structure.
    """)

    if not FREUD_AVAILABLE:
        print("ERROR: freud library not available.")
        print("Install with: pip install freud-analysis")
        return None

    n = 64
    nu = 0.001
    scale = 5.0
    tmax = 1.5

    omega_init = taylor_green_vorticity(n, scale=scale)

    print(f"Parameters: n={n}, ν={nu}, scale={scale}, tmax={tmax}")
    print(f"Initial max vorticity: {np.max(np.linalg.norm(omega_init, axis=-1)):.2f}")

    print("\n--- Running NS and tracking vortex structure ---")
    solver = NSVortexTracker(n=n, viscosity=nu)
    results = solver.run_and_track(omega_init, tmax=tmax, measure_interval=50)

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    times = np.array(results['times'])
    Z = np.array(results['enstrophies'])
    n_vort = np.array(results['n_vortices'])
    q6 = np.array([x if x is not None else np.nan for x in results['q6_values']])
    q4 = np.array([x if x is not None else np.nan for x in results['q4_values']])

    # Filter valid
    valid = ~np.isnan(Z) & ~np.isnan(q6)

    if np.sum(valid) == 0:
        print("No valid Q6 measurements obtained!")
        return results

    times_v = times[valid]
    Z_v = Z[valid]
    q6_v = q6[valid]
    q4_v = q4[valid]
    n_vort_v = n_vort[valid]

    print(f"\nSteinhardt order parameter statistics:")
    print(f"  Q6 (icosahedral):  {np.mean(q6_v):.4f} ± {np.std(q6_v):.4f}")
    print(f"  Q4 (cubic):        {np.mean(q4_v):.4f} ± {np.std(q4_v):.4f}")
    print(f"  Q6/Q4 ratio:       {np.mean(q6_v)/np.mean(q4_v):.3f}")

    print(f"\nVortex core statistics:")
    print(f"  Mean N_vortex:     {np.mean(n_vort_v):.0f}")
    print(f"  Range:             {np.min(n_vort_v):.0f} - {np.max(n_vort_v):.0f}")

    # Reference values
    print(f"\nReference Q6 values:")
    print(f"  Random:       ~0.05")
    print(f"  Liquid:       ~0.1")
    print(f"  BCC:          ~0.51")
    print(f"  FCC:          ~0.57")
    print(f"  Icosahedral:  ~0.66")
    print(f"  Measured:     {np.mean(q6_v):.4f}")

    # Interpretation
    print("\n" + "-" * 70)
    mean_q6 = np.mean(q6_v)
    if mean_q6 > 0.5:
        print(f"  RESULT: Strong local order detected (Q6={mean_q6:.3f})")
        if mean_q6 > 0.6:
            print("  Near icosahedral! Strong evidence for H₃ geometry.")
        else:
            print("  Between BCC and icosahedral ordering.")
    elif mean_q6 > 0.3:
        print(f"  RESULT: Moderate local order (Q6={mean_q6:.3f})")
        print("  Some structure, but not strongly icosahedral.")
    elif mean_q6 > 0.15:
        print(f"  RESULT: Weak local order (Q6={mean_q6:.3f})")
        print("  Slightly above liquid, some correlations present.")
    else:
        print(f"  RESULT: No significant order (Q6={mean_q6:.3f})")
        print("  Vortex cores are randomly distributed.")
    print("-" * 70)

    # Correlation with enstrophy
    if len(Z_v) > 3:
        corr_q6_Z = np.corrcoef(Z_v, q6_v)[0, 1]
        print(f"\n  Correlation(Z, Q6): {corr_q6_Z:.3f}")
        if corr_q6_Z > 0.3:
            print("  Q6 increases with enstrophy - structure emerges during cascade!")
        elif corr_q6_Z < -0.3:
            print("  Q6 decreases with enstrophy - structure disrupted by cascade.")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(times_v, Z_v, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy Z(t)')
    ax.set_title('Enstrophy Evolution')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times_v, q6_v, 'g-', linewidth=2, label='Q6 (icosahedral)')
    ax.plot(times_v, q4_v, 'b--', linewidth=2, label='Q4 (cubic)')
    ax.axhline(0.66, color='r', linestyle=':', label='Icosahedral ref')
    ax.axhline(0.57, color='orange', linestyle=':', label='FCC ref')
    ax.axhline(0.1, color='gray', linestyle=':', label='Liquid ref')
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter')
    ax.set_title('Steinhardt Order Parameters')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(Z_v, q6_v, c=times_v, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Enstrophy Z')
    ax.set_ylabel('Q6 (icosahedral)')
    ax.set_title('Q6 vs Enstrophy')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Time')

    ax = axes[1, 1]
    ax.plot(times_v, n_vort_v, 'k-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Vortex Cores')
    ax.set_title('Vortex Core Count')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_icosahedral_structure.png', dpi=150)
    print("\nSaved: test_icosahedral_structure.png")

    return results


if __name__ == "__main__":
    results = main()
