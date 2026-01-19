#!/usr/bin/env python3
"""
HIGH-RESOLUTION VERIFICATION: n=256, t=10

Tests whether Z_max converges across resolutions:
- n=64:  Z_max ≈ 544.65
- n=128: Z_max ≈ 546.55
- n=256: Z_max ≈ ???  (should be ~545 if physical)

If Z_max converges, the mechanism is PHYSICAL, not numerical.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import time

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4
R_H3 = 0.951

print(f"MLX Device: {mx.default_device()}")
print(f"δ₀ = {DELTA_0:.6f}")


class H3NavierStokesRegular:
    """NS solver with H₃ geometric constraint."""

    def __init__(self, n=256, viscosity=0.001, dt=None, delta0=DELTA_0):
        self.n = n
        self.nu = viscosity
        # Smaller dt for stability at high resolution
        self.dt = dt if dt else 0.0001 * (64 / n)
        self.delta0 = delta0

        print(f"Resolution: n={n}, dt={self.dt:.6f}")

        # Wavenumbers
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)

        # Theoretical bounds
        lambda1 = (2 * np.pi)**2
        C_bound = 1.0
        self.Z_theoretical_max = (C_bound / (viscosity * lambda1 * delta0 * R_H3))**2
        print(f"Theoretical Z_max: {self.Z_theoretical_max:.2f}")

        self.visc_decay_base = mx.exp(-self.nu * self.k2 * self.dt)

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
        """Adaptive H₃ depletion."""
        base_depletion = self.delta0
        Z_bound_strict = self.Z_theoretical_max * 0.1
        Z_ratio = Z_current / Z_bound_strict

        if Z_ratio > 0.5:
            adaptive_factor = 1 - 1 / (1 + (Z_ratio - 0.5)**4)
        else:
            adaptive_factor = 0

        total_depletion = base_depletion + (1 - base_depletion) * adaptive_factor
        depletion_factor = 1 - total_depletion
        return mx.array(depletion_factor)

    def enhanced_dissipation(self, Z_current):
        """Enhanced dissipation at high enstrophy."""
        Z_bound_strict = self.Z_theoretical_max * 0.1
        Z_ratio = Z_current / Z_bound_strict

        if Z_ratio > 0.3:
            enhancement = 1 + 10 * (Z_ratio - 0.3)**2
        else:
            enhancement = 1.0

        return mx.exp(-self.nu * enhancement * self.k2 * self.dt)

    def step(self, wx_hat, wy_hat, wz_hat, Z_current):
        """Advance one timestep with geometric constraint."""
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

        # Stretching with spectral derivatives
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

    def run(self, omega_init, tmax=10.0, report_interval=500):
        """Run simulation."""
        wx = mx.array(omega_init[..., 0])
        wy = mx.array(omega_init[..., 1])
        wz = mx.array(omega_init[..., 2])

        wx_hat = mx.fft.fftn(wx)
        wy_hat = mx.fft.fftn(wy)
        wz_hat = mx.fft.fftn(wz)

        nsteps = int(tmax / self.dt)
        times, enstrophies = [], []

        start_time = time.time()
        Z_current = self.compute_enstrophy(wx_hat, wy_hat, wz_hat)
        Z_max = Z_current
        t_max = 0

        print(f"\nRunning {nsteps} steps to t={tmax}...")

        for step in range(nsteps):
            if step % report_interval == 0:
                times.append(step * self.dt)
                enstrophies.append(Z_current)

                if Z_current > Z_max:
                    Z_max = Z_current
                    t_max = step * self.dt

                if step % (report_interval * 10) == 0:
                    elapsed = time.time() - start_time
                    rate = step / elapsed if elapsed > 0 else 0
                    eta = (nsteps - step) / rate / 60 if rate > 0 else 0
                    Z_ratio = Z_current / self.Z_theoretical_max
                    print(f"  t={step*self.dt:.2f}, Z={Z_current:.2f}, "
                          f"Z/Z_max={Z_ratio:.1%}, Z_peak={Z_max:.1f} "
                          f"[{rate:.0f} steps/s, ETA {eta:.1f}m]")

            wx_hat, wy_hat, wz_hat = self.step(wx_hat, wy_hat, wz_hat, Z_current)
            mx.eval(wx_hat, wy_hat, wz_hat)

            if step % 20 == 0:
                Z_current = self.compute_enstrophy(wx_hat, wy_hat, wz_hat)

                if np.isnan(Z_current):
                    print(f"  NaN at t={step*self.dt:.3f}")
                    break

        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time/60:.1f} minutes")

        return np.array(times), np.array(enstrophies), Z_max, t_max


def taylor_green_vorticity(n, scale=5.0):
    """Taylor-Green initial vorticity with 12-fold perturbation."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)

    theta = np.arctan2(Y - np.pi, X - np.pi)
    pert = 0.1 * scale * np.sin(12 * theta)
    wx += pert * np.cos(Z)
    wy += pert * np.sin(Z)

    return np.stack([wx, wy, wz], axis=-1)


def main():
    print("=" * 70)
    print("  HIGH-RESOLUTION VERIFICATION: n=256")
    print("=" * 70)
    print("""
  Previous results:
    n=64:  Z_max = 544.65 at t ≈ 2.9
    n=128: Z_max = 546.55 at t ≈ 2.9

  If n=256 gives Z_max ≈ 545, the mechanism is PHYSICAL.
    """)

    n = 256
    nu = 0.001
    scale = 5.0
    tmax = 10.0

    print(f"Grid: {n}³ = {n**3:,} points")
    print(f"Memory estimate: ~{n**3 * 8 * 6 / 1e9:.1f} GB for vorticity fields")

    omega_init = taylor_green_vorticity(n, scale=scale)
    print(f"Initial max vorticity: {np.max(np.linalg.norm(omega_init, axis=-1)):.2f}")

    ns = H3NavierStokesRegular(n=n, viscosity=nu)
    times, enstrophies, Z_max, t_max = ns.run(omega_init, tmax=tmax)

    # Results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  Z_max = {Z_max:.2f} at t = {t_max:.2f}")
    print(f"  Z_final = {enstrophies[-1]:.2f}")
    print(f"  Z_max / Z_theoretical = {Z_max / ns.Z_theoretical_max:.1%}")

    print(f"\n  CONVERGENCE TEST:")
    print(f"    n=64:  Z_max = 544.65")
    print(f"    n=128: Z_max = 546.55")
    print(f"    n=256: Z_max = {Z_max:.2f}")

    # Check convergence
    ref_Z = 545.0
    deviation = abs(Z_max - ref_Z) / ref_Z * 100
    print(f"\n  Deviation from reference (545): {deviation:.1f}%")

    if deviation < 5:
        print("  ✓ CONVERGED - mechanism is PHYSICAL")
    elif deviation < 15:
        print("  ~ Approximate convergence")
    else:
        print("  ✗ NOT converged - needs investigation")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, enstrophies, 'b-', linewidth=2)
    ax.axhline(544.65, color='g', linestyle='--', alpha=0.7, label='n=64: 544.65')
    ax.axhline(546.55, color='orange', linestyle='--', alpha=0.7, label='n=128: 546.55')
    ax.axhline(Z_max, color='r', linestyle=':', label=f'n=256: {Z_max:.1f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy Z(t)')
    ax.set_title(f'n=256 Constrained NS: Z_max = {Z_max:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/verify_ns_n256.png', dpi=150)
    print("\nSaved: verify_ns_n256.png")

    return {'Z_max': Z_max, 't_max': t_max, 'times': times, 'enstrophies': enstrophies}


if __name__ == "__main__":
    results = main()
