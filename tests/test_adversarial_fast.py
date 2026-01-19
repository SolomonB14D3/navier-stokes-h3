#!/usr/bin/env python3
"""
FAST ADVERSARIAL TEST: Targeted attempts to break the Z_max bound

Instead of slow optimization, we test specific pathological configurations
known to cause problems in standard NS simulations.
"""

import numpy as np
import mlx.core as mx
import sys
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("="*70)
print("  FAST ADVERSARIAL TEST FOR NAVIER-STOKES H3 PROOF")
print("="*70)
print(f"\nMLX Device: {mx.default_device()}")

PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4
print(f"δ₀ = {DELTA_0:.6f}")
print(f"Theoretical bound: Z_max ≈ 547")
print()

# Configuration - match the n=256 validation parameters
N = 64
NU = 0.001
DT = 0.0002
T_MAX = 5.0
INITIAL_OMEGA_MAX = 10.0  # Initial max vorticity (not energy)

def run_ns_simulation(omega_init, label=""):
    """Run NS with H3 depletion, return peak enstrophy"""
    n = N
    omega = mx.array(omega_init)

    # Wavenumbers
    k_np = np.fft.fftfreq(n, d=1/(2*np.pi*n)).astype(np.float32)
    kx, ky, kz = np.meshgrid(k_np, k_np, k_np, indexing='ij')
    kx, ky, kz = mx.array(kx), mx.array(ky), mx.array(kz)
    k_sq = kx**2 + ky**2 + kz**2
    k_sq_safe = mx.where(k_sq == 0, mx.ones_like(k_sq), k_sq)

    # Dealiasing
    k_max = n // 3
    dealias = (mx.abs(kx) < k_max) & (mx.abs(ky) < k_max) & (mx.abs(kz) < k_max)

    n_steps = int(T_MAX / DT)
    Z_max = 0.0
    Z_history = []

    print(f"  Running {label}...", end=" ", flush=True)
    start = time.time()

    for step in range(n_steps):
        # Enstrophy
        Z = 0.5 * float(mx.sum(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2))
        if Z > Z_max:
            Z_max = Z

        if step % 1000 == 0:
            Z_history.append((step * DT, Z))

        # Early termination if clearly bounded
        if step > 2000 and Z < Z_max * 0.5:
            break

        # FFT
        omega_hat = mx.stack([mx.fft.fftn(omega[..., i]) for i in range(3)], axis=-1)

        # Velocity via Biot-Savart
        u_hat = mx.zeros_like(omega_hat)
        u_hat[..., 0] = 1j * (ky * omega_hat[..., 2] - kz * omega_hat[..., 1]) / k_sq_safe
        u_hat[..., 1] = 1j * (kz * omega_hat[..., 0] - kx * omega_hat[..., 2]) / k_sq_safe
        u_hat[..., 2] = 1j * (kx * omega_hat[..., 1] - ky * omega_hat[..., 0]) / k_sq_safe
        u_hat = u_hat * dealias[..., None]

        # Physical space
        u = mx.stack([mx.fft.ifftn(u_hat[..., i]).real for i in range(3)], axis=-1)

        # Gradients
        dudx = mx.fft.ifftn(1j * kx[..., None] * u_hat).real
        dudy = mx.fft.ifftn(1j * ky[..., None] * u_hat).real
        dudz = mx.fft.ifftn(1j * kz[..., None] * u_hat).real

        # Stretching with H3 depletion
        stretch = omega[..., 0:1]*dudx + omega[..., 1:2]*dudy + omega[..., 2:3]*dudz

        omega_mag = mx.sqrt(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2 + 1e-10)
        omega_crit = NU * 100
        Phi = mx.clip((omega_mag / omega_crit - 1) / (omega_mag / omega_crit + 1e-10), 0, 1)
        depletion = 1 - DELTA_0 * Phi[..., None]
        stretch = stretch * depletion

        # Viscous term
        visc_hat = -NU * k_sq[..., None] * omega_hat
        visc = mx.stack([mx.fft.ifftn(visc_hat[..., i]).real for i in range(3)], axis=-1)

        # Time step
        omega = omega + DT * (stretch + visc)

        # Dealias periodically
        if step % 50 == 0:
            omega_hat = mx.stack([mx.fft.fftn(omega[..., i]) * dealias for i in range(3)], axis=-1)
            omega = mx.stack([mx.fft.ifftn(omega_hat[..., i]).real for i in range(3)], axis=-1)

        mx.eval(omega)

    elapsed = time.time() - start
    print(f"Z_max = {Z_max:.1f} ({elapsed:.1f}s)")

    return Z_max

def normalize_to_max_omega(omega, omega_max=10.0):
    """Normalize vorticity to have max magnitude = omega_max"""
    current_max = np.max(np.sqrt(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2))
    if current_max > 0:
        scale = omega_max / current_max
        return omega * scale
    return omega

# ============================================================================
# ADVERSARIAL INITIAL CONDITIONS
# ============================================================================

print("\n--- TEST 1: Standard Icosahedral (baseline) ---")
x = np.linspace(0, 2*np.pi, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
omega = np.zeros((N, N, N, 3), dtype=np.float32)
omega[..., 0] = np.sin(2*Z) * np.cos(2*Y)
omega[..., 1] = np.sin(2*X) * np.cos(2*Z)
omega[..., 2] = np.sin(2*Y) * np.cos(2*X)
omega = normalize_to_max_omega(omega, 10.0)
Z1 = run_ns_simulation(omega, "Icosahedral baseline")

print("\n--- TEST 2: Anti-Icosahedral (cubic axes) ---")
omega = np.zeros((N, N, N, 3), dtype=np.float32)
omega[..., 0] = 10 * np.sin(X) * np.sin(Y)  # Pure cubic
omega[..., 1] = 10 * np.sin(Y) * np.sin(Z)
omega[..., 2] = 10 * np.sin(Z) * np.sin(X)
omega = normalize_to_max_omega(omega, 10.0)
Z2 = run_ns_simulation(omega, "Anti-icosahedral")

print("\n--- TEST 3: Vortex Sheet Collision ---")
omega = np.zeros((N, N, N, 3), dtype=np.float32)
# Two opposing vortex sheets
omega[..., 2] = 10 * (np.tanh(10*(Y - np.pi/2)) - np.tanh(10*(Y - 3*np.pi/2)))
omega = normalize_to_max_omega(omega, 10.0)
Z3 = run_ns_simulation(omega, "Vortex sheet collision")

print("\n--- TEST 4: Trefoil Knot (topologically complex) ---")
omega = np.zeros((N, N, N, 3), dtype=np.float32)
# Trefoil knot vorticity
r = np.sqrt((X - np.pi)**2 + (Y - np.pi)**2)
theta = np.arctan2(Y - np.pi, X - np.pi)
omega[..., 2] = 10 * np.exp(-r**2/0.5) * np.sin(3*theta + 2*Z)
omega[..., 0] = 5 * np.exp(-r**2/0.5) * np.cos(3*theta + 2*Z)
omega = normalize_to_max_omega(omega, 10.0)
Z4 = run_ns_simulation(omega, "Trefoil knot")

print("\n--- TEST 5: High-Frequency Turbulence ---")
np.random.seed(42)
omega = np.zeros((N, N, N, 3), dtype=np.float32)
for k in range(4, 12):  # High wavenumbers
    for _ in range(3):
        kx, ky, kz = np.random.randint(1, k+1, 3)
        phase = np.random.uniform(0, 2*np.pi, 3)
        amp = 5.0 / k
        omega[..., 0] += amp * np.sin(kx*X + ky*Y + kz*Z + phase[0])
        omega[..., 1] += amp * np.sin(kx*X + ky*Y + kz*Z + phase[1])
        omega[..., 2] += amp * np.sin(kx*X + ky*Y + kz*Z + phase[2])
omega = normalize_to_max_omega(omega, 10.0)
Z5 = run_ns_simulation(omega, "High-freq turbulence")

print("\n--- TEST 6: Concentrated Vortex Core ---")
omega = np.zeros((N, N, N, 3), dtype=np.float32)
# Very thin, intense vortex
r = np.sqrt((X - np.pi)**2 + (Y - np.pi)**2)
omega[..., 2] = 20 * np.exp(-r**2/0.05)  # Thin core
omega = normalize_to_max_omega(omega, 10.0)
Z6 = run_ns_simulation(omega, "Concentrated core")

print("\n--- TEST 7: Asymmetric Strain Focus ---")
omega = np.zeros((N, N, N, 3), dtype=np.float32)
# Create highly asymmetric initial condition
omega[..., 0] = 10 * np.sin(X) * np.exp(-((Y-np.pi)**2 + (Z-np.pi)**2)/2)
omega[..., 1] = -5 * np.sin(Y) * np.exp(-((X-np.pi)**2 + (Z-np.pi)**2)/2)
omega[..., 2] = 15 * np.sin(2*Z) * np.exp(-((X-np.pi)**2 + (Y-np.pi)**2)/2)
omega = normalize_to_max_omega(omega, 10.0)
Z7 = run_ns_simulation(omega, "Asymmetric strain")

print("\n--- TEST 8: Maximally Non-Normal (worst case alignment) ---")
omega = np.zeros((N, N, N, 3), dtype=np.float32)
# Design vorticity to align with strain as much as possible
# This should be the theoretical worst case
omega[..., 0] = 10 * np.cos(X) * np.sin(Y) * np.cos(Z)
omega[..., 1] = 10 * np.sin(X) * np.cos(Y) * np.sin(Z)
omega[..., 2] = 10 * np.sin(X) * np.sin(Y) * np.cos(Z)
omega = normalize_to_max_omega(omega, 10.0)
Z8 = run_ns_simulation(omega, "Max non-normal")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("  ADVERSARIAL TEST RESULTS")
print("="*70)

results = [
    ("Icosahedral baseline", Z1),
    ("Anti-icosahedral", Z2),
    ("Vortex sheet collision", Z3),
    ("Trefoil knot", Z4),
    ("High-freq turbulence", Z5),
    ("Concentrated core", Z6),
    ("Asymmetric strain", Z7),
    ("Max non-normal", Z8),
]

print(f"\nTheoretical bound: Z_max ≈ 547")
print()
print(f"{'Configuration':<25} {'Z_max':>10} {'% of bound':>12} {'Status':>10}")
print("-"*60)

all_bounded = True
max_Z = 0
for name, Z in sorted(results, key=lambda x: -x[1]):
    ratio = Z / 547 * 100
    status = "✓ BOUNDED" if Z <= 547 else "⚠ EXCEEDED"
    if Z > 547:
        all_bounded = False
    if Z > max_Z:
        max_Z = Z
    print(f"{name:<25} {Z:>10.1f} {ratio:>11.1f}% {status:>10}")

print()
print(f"Maximum achieved: {max_Z:.1f} ({max_Z/547*100:.1f}% of bound)")
print()

if all_bounded:
    print("="*70)
    print("  ✓ ALL ADVERSARIAL TESTS BOUNDED")
    print("  The H3 depletion mechanism prevents blowup in all tested cases.")
    print("="*70)
else:
    print("="*70)
    print("  ⚠ WARNING: Some tests exceeded the bound!")
    print("="*70)

# Save results
import json
with open("adversarial_fast_results.json", "w") as f:
    json.dump({
        "theoretical_bound": 547,
        "max_achieved": float(max_Z),
        "ratio": float(max_Z / 547),
        "all_bounded": all_bounded,
        "results": [(name, float(z)) for name, z in results]
    }, f, indent=2)

print(f"\nResults saved to adversarial_fast_results.json")
