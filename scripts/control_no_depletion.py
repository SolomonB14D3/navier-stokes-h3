#!/usr/bin/env python3
"""
CONTROL EXPERIMENT: Standard NS (NO H₃ depletion)

Critical test: Does the solver blow up WITHOUT the depletion mechanism?

If δ₀=0 also stays bounded → the NUMERICS prevent blowup, not physics
If δ₀=0 blows up but δ₀>0 doesn't → the mechanism WORKS

This is the most important test for validating the proof.
"""

import numpy as np
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

print("=" * 70)
print("  CONTROL EXPERIMENT: NS WITHOUT H₃ DEPLETION")
print("  Does the solver blow up when δ₀ = 0?")
print("=" * 70)

# Try both MLX and pure NumPy to rule out tool issues
try:
    import mlx.core as mx
    HAS_MLX = True
    print(f"\n  MLX available: {mx.default_device()}")
except ImportError:
    HAS_MLX = False
    print("\n  MLX not available, using NumPy (slower but identical math)")


def run_ns_numpy(n, omega_init, viscosity, dt, tmax, use_depletion=False, delta0=0.309):
    """
    Pure NumPy spectral NS solver. No tricks, no safety valves.

    This is the simplest possible implementation to rule out
    framework-specific issues.
    """
    # Wavenumbers
    k = np.fft.fftfreq(n, d=1.0/n) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2_safe = np.where(k2 == 0, 1e-10, k2)

    # Viscous decay (integrating factor)
    visc_decay = np.exp(-viscosity * k2 * dt)

    # Initialize
    wx_hat = np.fft.fftn(omega_init[..., 0])
    wy_hat = np.fft.fftn(omega_init[..., 1])
    wz_hat = np.fft.fftn(omega_init[..., 2])

    nsteps = int(tmax / dt)
    Z_max = 0
    t_at_max = 0

    print(f"    n={n}, dt={dt:.2e}, steps={nsteps}, ν={viscosity}")
    print(f"    depletion: {'δ₀='+str(delta0) if use_depletion else 'NONE (standard NS)'}")

    start = time.time()
    report_interval = max(1, nsteps // 20)

    for step in range(nsteps):
        # Compute enstrophy every 20 steps
        if step % 20 == 0:
            wx = np.fft.ifftn(wx_hat).real
            wy = np.fft.ifftn(wy_hat).real
            wz = np.fft.ifftn(wz_hat).real
            Z = 0.5 * np.mean(wx**2 + wy**2 + wz**2)

            if Z > Z_max:
                Z_max = Z
                t_at_max = step * dt

            if np.isnan(Z) or Z > 1e8:
                print(f"\n    *** BLOWUP at step {step}, t={step*dt:.4f}, Z={Z:.2e} ***")
                return Z_max, t_at_max, True, step * dt

        # Report
        if step % report_interval == 0:
            elapsed = time.time() - start
            rate = step / elapsed if elapsed > 0 else 0
            eta = (nsteps - step) / rate / 60 if rate > 0 else 0
            t = step * dt
            print(f"    t={t:.3f} Z={Z:.1f} Z_max={Z_max:.1f} | {rate:.0f} steps/s ETA {eta:.1f}m")

        # Velocity from vorticity (Biot-Savart)
        ux_hat = -1j * (ky * wz_hat - kz * wy_hat) / k2_safe
        uy_hat = -1j * (kz * wx_hat - kx * wz_hat) / k2_safe
        uz_hat = -1j * (kx * wy_hat - ky * wx_hat) / k2_safe

        # Physical space
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

        # Stretching: (ω·∇)u
        dux_dx = np.fft.ifftn(1j * kx * ux_hat).real
        dux_dy = np.fft.ifftn(1j * ky * ux_hat).real
        dux_dz = np.fft.ifftn(1j * kz * ux_hat).real
        duy_dx = np.fft.ifftn(1j * kx * uy_hat).real
        duy_dy = np.fft.ifftn(1j * ky * uy_hat).real
        duy_dz = np.fft.ifftn(1j * kz * uy_hat).real
        duz_dx = np.fft.ifftn(1j * kx * uz_hat).real
        duz_dy = np.fft.ifftn(1j * ky * uz_hat).real
        duz_dz = np.fft.ifftn(1j * kz * uz_hat).real

        stretch_x = wx * dux_dx + wy * dux_dy + wz * dux_dz
        stretch_y = wx * duy_dx + wy * duy_dy + wz * duy_dz
        stretch_z = wx * duz_dx + wy * duz_dy + wz * duz_dz

        # Apply depletion if requested
        if use_depletion:
            omega_mag = np.sqrt(wx**2 + wy**2 + wz**2 + 1e-10)
            omega_crit = 1.0 / (delta0 * 0.951)
            x_ratio = omega_mag / omega_crit
            activation = x_ratio**2 / (1.0 + x_ratio**2)
            depletion_factor = 1.0 - delta0 * activation
            stretch_x = stretch_x * depletion_factor
            stretch_y = stretch_y * depletion_factor
            stretch_z = stretch_z * depletion_factor

        nlx = nlx + stretch_x
        nly = nly + stretch_y
        nlz = nlz + stretch_z

        # Advance (Euler + integrating factor)
        nlx_hat = np.fft.fftn(nlx)
        nly_hat = np.fft.fftn(nly)
        nlz_hat = np.fft.fftn(nlz)

        wx_hat = (wx_hat + dt * nlx_hat) * visc_decay
        wy_hat = (wy_hat + dt * nly_hat) * visc_decay
        wz_hat = (wz_hat + dt * nlz_hat) * visc_decay

    elapsed = time.time() - start
    print(f"    Completed in {elapsed:.1f}s. Z_max = {Z_max:.2f}")
    return Z_max, t_at_max, False, tmax


def create_max_strained_ic(n, amplitude=10.0):
    """Max strained IC without enstrophy normalization (use raw amplitude)."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    omega = np.zeros((n, n, n, 3), dtype=np.float64)
    omega[..., 0] = amplitude * np.sin(X) * np.cos(Y) * np.cos(Z)
    omega[..., 1] = amplitude * np.cos(X) * np.sin(Y) * np.cos(Z)
    omega[..., 2] = -2 * amplitude * np.cos(X) * np.cos(Y) * np.sin(Z)
    return omega


def create_high_energy_ic(n, amplitude=50.0):
    """High-energy IC designed to challenge regularity."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    omega = np.zeros((n, n, n, 3), dtype=np.float64)
    # Multi-scale structure
    for k in [1, 2, 3, 4]:
        omega[..., 0] += amplitude/k * np.sin(k*X) * np.cos(k*Y) * np.cos(k*Z)
        omega[..., 1] += amplitude/k * np.cos(k*X) * np.sin(k*Y) * np.cos(k*Z)
        omega[..., 2] -= 2*amplitude/k * np.cos(k*X) * np.cos(k*Y) * np.sin(k*Z)
    return omega


# ============================================================
# RUN EXPERIMENTS
# ============================================================

n = 64  # Start small for speed
dt = 0.0001
tmax = 5.0

print("\n" + "=" * 70)
print("  EXPERIMENT 1: Standard NS (no depletion) - low amplitude")
print("=" * 70)
omega = create_max_strained_ic(n, amplitude=10.0)
Z0 = 0.5 * np.mean(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2)
print(f"  Initial enstrophy: Z₀ = {Z0:.2f}")
Z_max_no, t_max_no, blew_up_no, t_end_no = run_ns_numpy(
    n, omega, viscosity=0.001, dt=dt, tmax=tmax, use_depletion=False)

print("\n" + "=" * 70)
print("  EXPERIMENT 2: H₃-NS (with depletion) - low amplitude")
print("=" * 70)
Z_max_yes, t_max_yes, blew_up_yes, t_end_yes = run_ns_numpy(
    n, omega, viscosity=0.001, dt=dt, tmax=tmax, use_depletion=True)

print("\n" + "=" * 70)
print("  EXPERIMENT 3: Standard NS (no depletion) - HIGH amplitude")
print("=" * 70)
omega_high = create_max_strained_ic(n, amplitude=100.0)
Z0_high = 0.5 * np.mean(omega_high[..., 0]**2 + omega_high[..., 1]**2 + omega_high[..., 2]**2)
print(f"  Initial enstrophy: Z₀ = {Z0_high:.2f}")
Z_max_high_no, t_max_high_no, blew_up_high_no, t_end_high_no = run_ns_numpy(
    n, omega_high, viscosity=0.001, dt=dt, tmax=tmax, use_depletion=False)

print("\n" + "=" * 70)
print("  EXPERIMENT 4: Standard NS - HIGH amplitude, LOWER viscosity")
print("=" * 70)
Z_max_lownu, t_max_lownu, blew_up_lownu, t_end_lownu = run_ns_numpy(
    n, omega_high, viscosity=0.0001, dt=dt/2, tmax=tmax, use_depletion=False)

print("\n" + "=" * 70)
print("  EXPERIMENT 5: Standard NS - EXTREME amplitude, near-zero viscosity")
print("=" * 70)
omega_extreme = create_high_energy_ic(n, amplitude=200.0)
Z0_extreme = 0.5 * np.mean(omega_extreme[..., 0]**2 + omega_extreme[..., 1]**2 + omega_extreme[..., 2]**2)
print(f"  Initial enstrophy: Z₀ = {Z0_extreme:.2f}")
Z_max_extreme, t_max_extreme, blew_up_extreme, t_end_extreme = run_ns_numpy(
    n, omega_extreme, viscosity=0.00001, dt=dt/4, tmax=2.0, use_depletion=False)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print(f"\n  {'Experiment':<45} {'Z₀':<8} {'Z_max':<10} {'Blowup?':<10} {'Depletion'}")
print(f"  {'-'*85}")
print(f"  {'1. Standard NS, amp=10, ν=0.001':<45} {Z0:<8.1f} {Z_max_no:<10.1f} {'YES' if blew_up_no else 'NO':<10} {'OFF'}")
print(f"  {'2. H₃-NS, amp=10, ν=0.001':<45} {Z0:<8.1f} {Z_max_yes:<10.1f} {'YES' if blew_up_yes else 'NO':<10} {'ON'}")
print(f"  {'3. Standard NS, amp=100, ν=0.001':<45} {Z0_high:<8.1f} {Z_max_high_no:<10.1f} {'YES' if blew_up_high_no else 'NO':<10} {'OFF'}")
print(f"  {'4. Standard NS, amp=100, ν=0.0001':<45} {Z0_high:<8.1f} {Z_max_lownu:<10.1f} {'YES' if blew_up_lownu else 'NO':<10} {'OFF'}")
print(f"  {'5. Standard NS, amp=200 multi, ν=0.00001':<45} {Z0_extreme:<8.1f} {Z_max_extreme:<10.1f} {'YES' if blew_up_extreme else 'NO':<10} {'OFF'}")

if not any([blew_up_no, blew_up_high_no, blew_up_lownu, blew_up_extreme]):
    print(f"\n  ⚠ NO BLOWUP IN ANY CONTROL EXPERIMENT")
    print(f"  This means the NUMERICAL METHOD prevents blowup, not the physics.")
    print(f"  The spectral solver with viscous integrating factor is inherently stable.")
    print(f"")
    print(f"  IMPLICATIONS:")
    print(f"  - The H₃ depletion mechanism cannot be validated by this solver")
    print(f"  - To test the mechanism, you need a solver that CAN blow up:")
    print(f"    a) Remove or reduce viscosity to near-zero")
    print(f"    b) Use a higher-order time integrator (RK4) without integrating factor")
    print(f"    c) Use much higher resolution to resolve small scales")
    print(f"    d) Test the inviscid (Euler) equations where blowup is expected")
elif any([blew_up_no, blew_up_high_no, blew_up_lownu, blew_up_extreme]) and not blew_up_yes:
    print(f"\n  ✓ STANDARD NS BLOWS UP, H₃-NS STAYS BOUNDED")
    print(f"  The depletion mechanism IS preventing blowup!")
    print(f"  The mechanism has genuine physical content.")
else:
    print(f"\n  Mixed results - needs further investigation")

print("\n" + "=" * 70)
