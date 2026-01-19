#!/usr/bin/env python3
"""
GPU-ACCELERATED NAVIER-STOKES VERIFICATION

Uses MLX (Apple's GPU framework) for M3 Ultra acceleration.
This enables high-resolution (n=128+) and high-Re simulations.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import time

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4  # ≈ 0.309
R_H3 = 0.951

print(f"MLX Device: {mx.default_device()}")
print(f"δ₀ = {DELTA_0:.6f}")


class H3NavierStokesGPU:
    """GPU-accelerated NS solver using MLX."""

    def __init__(self, n=128, viscosity=0.001, dt=0.0001, delta0=DELTA_0):
        self.n = n
        self.nu = viscosity
        self.dt = dt
        self.delta0 = delta0

        # Wavenumbers on GPU
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx**2 + ky**2 + kz**2
        # Avoid division by zero
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)

        # Critical vorticity
        self.omega_crit = 1.0 / (delta0 * R_H3) if delta0 > 0 else 1e10

        # Precompute viscous decay
        self.visc_decay = mx.exp(-self.nu * self.k2 * self.dt)

    def curl_hat(self, ux_hat, uy_hat, uz_hat):
        """Compute curl in Fourier space."""
        wx_hat = 1j * (self.ky * uz_hat - self.kz * uy_hat)
        wy_hat = 1j * (self.kz * ux_hat - self.kx * uz_hat)
        wz_hat = 1j * (self.kx * uy_hat - self.ky * ux_hat)
        return wx_hat, wy_hat, wz_hat

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat):
        """Biot-Savart: u = curl^{-1}(ω)."""
        # u_hat = -i(k × ω_hat)/|k|²
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def step(self, wx_hat, wy_hat, wz_hat):
        """Advance one timestep with H3 depletion."""
        # Get velocity
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        # Transform to physical space
        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        # Vorticity magnitude
        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2)

        # Nonlinear term: ω × u (advection)
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Compute stretching (ω · ∇)u via spectral derivatives
        # ∂u/∂x_k in Fourier space
        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real

        # (ω · ∇)u
        stretch_x = wx * dux_dx + wy * dux_dy + wz * dux_dz
        stretch_y = wx * duy_dx + wy * duy_dy + wz * duy_dz
        stretch_z = wx * duz_dx + wy * duz_dy + wz * duz_dz

        # H3 depletion factor
        if self.delta0 > 0:
            x = omega_mag / self.omega_crit
            # Smooth activation
            activation = x**2 / (1 + x**2)
            depletion = 1 - self.delta0 * activation
            stretch_x = stretch_x * depletion
            stretch_y = stretch_y * depletion
            stretch_z = stretch_z * depletion

        # Total nonlinear = advection + stretching
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

    def compute_diagnostics(self, wx_hat, wy_hat, wz_hat):
        """Compute energy, enstrophy, max vorticity."""
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        E = 0.5 * mx.mean(ux**2 + uy**2 + uz**2)
        Z = 0.5 * mx.mean(wx**2 + wy**2 + wz**2)
        omega_max = mx.max(mx.sqrt(wx**2 + wy**2 + wz**2))

        # Force evaluation and convert to Python floats
        mx.eval(E, Z, omega_max)
        return float(E.item()), float(Z.item()), float(omega_max.item())

    def run(self, omega_init, tmax=1.0, report_interval=100):
        """Run simulation."""
        # Convert to MLX and FFT
        wx = mx.array(omega_init[..., 0])
        wy = mx.array(omega_init[..., 1])
        wz = mx.array(omega_init[..., 2])

        wx_hat = mx.fft.fftn(wx)
        wy_hat = mx.fft.fftn(wy)
        wz_hat = mx.fft.fftn(wz)

        nsteps = int(tmax / self.dt)
        times, energies, enstrophies, max_vorts = [], [], [], []

        start_time = time.time()

        for step in range(nsteps):
            if step % report_interval == 0:
                E, Z, omega_max = self.compute_diagnostics(wx_hat, wy_hat, wz_hat)
                times.append(step * self.dt)
                energies.append(E)
                enstrophies.append(Z)
                max_vorts.append(omega_max)

                if step % (report_interval * 10) == 0:
                    elapsed = time.time() - start_time
                    print(f"  t={step*self.dt:.3f}, Z={Z:.4f}, ω_max={omega_max:.2f}, "
                          f"({elapsed:.1f}s, {step/(elapsed+1e-6):.0f} steps/s)")

            wx_hat, wy_hat, wz_hat = self.step(wx_hat, wy_hat, wz_hat)
            mx.eval(wx_hat, wy_hat, wz_hat)

        return np.array(times), np.array(energies), np.array(enstrophies), np.array(max_vorts)


def taylor_green_vorticity(n, scale=1.0):
    """Taylor-Green initial vorticity."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)

    # Add 12-fold perturbation
    theta = np.arctan2(Y - np.pi, X - np.pi)
    pert = 0.1 * scale * np.sin(12 * theta)
    wx += pert * np.cos(Z)
    wy += pert * np.sin(Z)

    return np.stack([wx, wy, wz], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  GPU-ACCELERATED NAVIER-STOKES VERIFICATION")
    print("=" * 70)

    # High resolution, high Re simulation
    n = 128         # Higher resolution for M3 Ultra
    nu = 0.001      # Moderate viscosity for stability
    scale = 5.0     # High initial amplitude
    tmax = 3.0

    omega_init = taylor_green_vorticity(n, scale=scale)
    omega_crit = 1.0 / (DELTA_0 * R_H3)

    print(f"\nParameters: n={n}, ν={nu}, scale={scale}")
    print(f"Initial max vorticity: {np.max(np.linalg.norm(omega_init, axis=-1)):.2f}")
    print(f"Critical vorticity ω_c: {omega_crit:.2f}")
    print(f"Reynolds number Re ~ {scale / nu:.0f}")

    # Run unconstrained
    print("\n--- Unconstrained (δ₀=0) ---")
    ns_un = H3NavierStokesGPU(n=n, viscosity=nu, delta0=0)
    t_un, E_un, Z_un, w_un = ns_un.run(omega_init, tmax=tmax)

    # Run constrained
    print("\n--- Constrained (δ₀=0.309) ---")
    ns_con = H3NavierStokesGPU(n=n, viscosity=nu, delta0=DELTA_0)
    t_con, E_con, Z_con, w_con = ns_con.run(omega_init, tmax=tmax)

    # Results - filter out NaN
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Find valid (non-NaN) data
    valid_un = ~np.isnan(Z_un)
    valid_con = ~np.isnan(Z_con)

    Z_un_valid = Z_un[valid_un]
    Z_con_valid = Z_con[valid_con]
    t_un_valid = t_un[valid_un]
    t_con_valid = t_con[valid_con]
    w_un_valid = w_un[valid_un]
    w_con_valid = w_con[valid_con]

    Z_max_un = np.max(Z_un_valid) if len(Z_un_valid) > 0 else 0
    Z_max_con = np.max(Z_con_valid) if len(Z_con_valid) > 0 else 0
    w_max_un = np.max(w_un_valid) if len(w_un_valid) > 0 else 0
    w_max_con = np.max(w_con_valid) if len(w_con_valid) > 0 else 0

    blowup_time_un = t_un_valid[-1] if len(t_un_valid) > 0 else 0
    blowup_time_con = t_con_valid[-1] if len(t_con_valid) > 0 else 0

    # Compare at common time
    t_compare = min(blowup_time_un, blowup_time_con) * 0.8  # 80% of first blowup
    idx_un = np.argmin(np.abs(t_un_valid - t_compare)) if len(t_un_valid) > 0 else 0
    idx_con = np.argmin(np.abs(t_con_valid - t_compare)) if len(t_con_valid) > 0 else 0

    Z_at_compare_un = Z_un_valid[idx_un] if len(Z_un_valid) > 0 else 0
    Z_at_compare_con = Z_con_valid[idx_con] if len(Z_con_valid) > 0 else 0

    reduction_at_compare = (Z_at_compare_un - Z_at_compare_con) / Z_at_compare_un if Z_at_compare_un > 0 else 0
    survival_increase = (blowup_time_con - blowup_time_un) / blowup_time_un if blowup_time_un > 0 else 0

    print(f"\nBlowup times:")
    print(f"  Unconstrained: t = {blowup_time_un:.3f}")
    print(f"  Constrained:   t = {blowup_time_con:.3f}")
    print(f"  Survival increase: {survival_increase*100:.1f}%")

    print(f"\nAt t = {t_compare:.3f} (before first blowup):")
    print(f"  Unconstrained: Z = {Z_at_compare_un:.4f}")
    print(f"  Constrained:   Z = {Z_at_compare_con:.4f}")
    print(f"  Reduction: {reduction_at_compare*100:.1f}%")

    print(f"\nMax values before blowup:")
    print(f"  Unconstrained: Z_max = {Z_max_un:.4f}, ω_max = {w_max_un:.2f}")
    print(f"  Constrained:   Z_max = {Z_max_con:.4f}, ω_max = {w_max_con:.2f}")

    # Did vorticity exceed critical threshold?
    print(f"\nVorticity exceeded ω_c = {omega_crit:.2f}?")
    print(f"  Unconstrained: {'Yes' if w_max_un > omega_crit else 'No'} (max={w_max_un:.2f})")
    print(f"  Constrained:   {'Yes' if w_max_con > omega_crit else 'No'} (max={w_max_con:.2f})")

    # Update arrays for plotting (replace NaN with last valid)
    Z_un_plot = np.where(np.isnan(Z_un), Z_max_un, Z_un)
    Z_con_plot = np.where(np.isnan(Z_con), Z_max_con, Z_con)
    w_un_plot = np.where(np.isnan(w_un), w_max_un, w_un)
    w_con_plot = np.where(np.isnan(w_con), w_max_con, w_con)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.semilogy(t_un_valid, Z_un_valid, 'b-', label='Unconstrained', linewidth=2)
    ax.semilogy(t_con_valid, Z_con_valid, 'r-', label=f'H3 (δ₀={DELTA_0:.3f})', linewidth=2)
    ax.axvline(blowup_time_un, color='b', linestyle=':', alpha=0.5, label=f'Blowup (un): t={blowup_time_un:.2f}')
    ax.axvline(blowup_time_con, color='r', linestyle=':', alpha=0.5, label=f'Blowup (con): t={blowup_time_con:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy Z(t) [log scale]')
    ax.set_title(f'Enstrophy Evolution (n={n}, Re≈{scale/nu:.0f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(t_un_valid, w_un_valid, 'b-', label='Unconstrained', linewidth=2)
    ax.semilogy(t_con_valid, w_con_valid, 'r-', label='Constrained', linewidth=2)
    ax.axhline(omega_crit, color='k', linestyle='--', label=f'ω_c = {omega_crit:.1f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Vorticity [log scale]')
    ax.set_title('Maximum Vorticity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    E_un_valid = E_un[valid_un]
    E_con_valid = E_con[valid_con]
    ax.plot(t_un_valid, E_un_valid, 'b-', label='Unconstrained', linewidth=2)
    ax.plot(t_con_valid, E_con_valid, 'r-', label='Constrained', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy E(t)')
    ax.set_title('Energy Dissipation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/verify_ns_gpu.png', dpi=150)
    print("\nSaved: verify_ns_gpu.png")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if reduction_at_compare > 0.1:
        print(f"  ✓ H3 depletion reduced enstrophy by {reduction_at_compare*100:.1f}% at t={t_compare:.2f}")
    else:
        print(f"  ✗ H3 depletion effect small ({reduction_at_compare*100:.1f}%)")

    if survival_increase > 0.1:
        print(f"  ✓ Constrained survives {survival_increase*100:.1f}% longer before blowup")

    if w_max_un > omega_crit:
        print(f"  ✓ Reached high-vorticity regime (ω_max = {w_max_un:.1f} > ω_c = {omega_crit:.1f})")
    else:
        print(f"  ⚠ Moderate vorticity (ω_max = {w_max_un:.1f} vs ω_c = {omega_crit:.1f})")

    print("=" * 70)

    return {
        'Z_max_un': Z_max_un,
        'Z_max_con': Z_max_con,
        'reduction_at_compare': reduction_at_compare,
        'survival_increase': survival_increase,
        'blowup_time_un': blowup_time_un,
        'blowup_time_con': blowup_time_con
    }


if __name__ == "__main__":
    results = main()
