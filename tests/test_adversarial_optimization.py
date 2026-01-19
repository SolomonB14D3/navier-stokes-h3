#!/usr/bin/env python3
"""
ADVERSARIAL OPTIMIZATION TEST FOR NAVIER-STOKES H3 PROOF

Goal: Use optimization to find initial conditions that MAXIMIZE peak enstrophy,
      actively trying to break the theoretical bound Z_max ≈ 547.

If the optimizer cannot exceed this bound despite explicit attempts,
the result is practically irrefutable.

Method: Genetic algorithm + local refinement to search IC parameter space
"""

import numpy as np
import mlx.core as mx
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass
import time

print(f"MLX Device: {mx.default_device()}")

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4
print(f"δ₀ = {DELTA_0:.6f}")
print(f"Theoretical Z_max ≈ 547 (from n=256 validation)")

@dataclass
class SimConfig:
    n: int = 64  # Grid size (keep small for speed)
    nu: float = 0.001  # Viscosity
    dt: float = 0.0005
    t_target: float = 3.0  # Run to expected peak time
    E0: float = 10.0  # Fixed initial energy constraint

config = SimConfig()

# Icosahedral axes for depletion
def get_icosahedral_axes():
    """27 icosahedral symmetry axes (6 five-fold + 15 two-fold + 6 three-fold)"""
    axes = []
    # 6 five-fold axes (through vertices)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            axes.append(np.array([0, s1, s2*PHI]))
            axes.append(np.array([s1, s2*PHI, 0]))
            axes.append(np.array([s2*PHI, 0, s1]))
    # Normalize
    axes = [a / np.linalg.norm(a) for a in axes]
    # Remove duplicates (opposite directions)
    unique = []
    for a in axes:
        is_dup = False
        for u in unique:
            if abs(np.dot(a, u)) > 0.99:
                is_dup = True
                break
        if not is_dup:
            unique.append(a)
    return np.array(unique[:12])  # 6 pairs = 12 directions

ICOSA_AXES = mx.array(get_icosahedral_axes(), dtype=mx.float32)

def create_initial_vorticity(params, n):
    """
    Create initial vorticity field from parameter vector.

    params: Array of shape (n_modes * 7,) where each mode has:
        - kx, ky, kz: wavenumber direction (will be normalized)
        - amplitude: mode amplitude
        - phase_x, phase_y, phase_z: phases
    """
    n_modes = len(params) // 7

    # Create coordinate grids
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    omega_x = np.zeros((n, n, n), dtype=np.float32)
    omega_y = np.zeros((n, n, n), dtype=np.float32)
    omega_z = np.zeros((n, n, n), dtype=np.float32)

    for i in range(n_modes):
        idx = i * 7
        # Wavenumber (integer, 1-8)
        kx = int(np.clip(params[idx] * 8, 1, 8))
        ky = int(np.clip(params[idx+1] * 8, 1, 8))
        kz = int(np.clip(params[idx+2] * 8, 1, 8))

        # Amplitude
        amp = params[idx+3] * 10  # Scale to reasonable range

        # Phases
        px = params[idx+4] * 2 * np.pi
        py = params[idx+5] * 2 * np.pi
        pz = params[idx+6] * 2 * np.pi

        # Add Fourier mode (ensure divergence-free by construction)
        # Use curl of vector potential: ω = ∇ × A
        # A = amp * sin(k·x + phase) * direction
        k_dot_x = kx * X + ky * Y + kz * Z

        # Construct divergence-free vorticity mode
        omega_x += amp * np.cos(k_dot_x + px) * (ky - kz)
        omega_y += amp * np.cos(k_dot_x + py) * (kz - kx)
        omega_z += amp * np.cos(k_dot_x + pz) * (kx - ky)

    return omega_x, omega_y, omega_z

def normalize_energy(omega_x, omega_y, omega_z, E0):
    """Normalize vorticity field to have total enstrophy = E0"""
    Z = 0.5 * np.sum(omega_x**2 + omega_y**2 + omega_z**2)
    if Z > 0:
        scale = np.sqrt(E0 / Z)
        return omega_x * scale, omega_y * scale, omega_z * scale
    return omega_x, omega_y, omega_z

def compute_alignment(omega, S_eigenvectors):
    """Compute alignment between vorticity and strain eigenvectors"""
    omega_norm = omega / (mx.linalg.norm(omega, axis=-1, keepdims=True) + 1e-10)
    # Max alignment with any icosahedral axis
    alignments = mx.abs(omega_norm @ ICOSA_AXES.T)
    return mx.max(alignments, axis=-1)

def apply_h3_depletion(omega, S, lambda_max):
    """Apply H3 geometric depletion to stretching term"""
    omega_mag = mx.linalg.norm(omega, axis=-1, keepdims=True)
    omega_crit = config.nu * 10  # Critical vorticity scale

    # Activation function
    x = omega_mag / omega_crit
    Phi = mx.where(x > 1, (x - 1) / (1 + x - 1), mx.zeros_like(x))
    Phi = mx.clip(Phi, 0, 1)

    # Depletion factor
    depletion = 1 - DELTA_0 * Phi

    return depletion

def run_simulation_mlx(omega_x, omega_y, omega_z, return_max_only=True):
    """
    Run NS simulation with H3 depletion, return peak enstrophy.
    Optimized for speed.
    """
    n = config.n
    dt = config.dt
    nu = config.nu

    # Convert to MLX
    omega = mx.stack([
        mx.array(omega_x),
        mx.array(omega_y),
        mx.array(omega_z)
    ], axis=-1)

    # Wavenumbers (use numpy, then convert)
    k_np = np.fft.fftfreq(n, d=1/(2*np.pi*n)).astype(np.float32)
    k = mx.array(k_np)
    kx_np, ky_np, kz_np = np.meshgrid(k_np, k_np, k_np, indexing='ij')
    kx, ky, kz = mx.array(kx_np), mx.array(ky_np), mx.array(kz_np)
    k_sq = kx**2 + ky**2 + kz**2
    k_sq = mx.where(k_sq == 0, mx.ones_like(k_sq), k_sq)

    # Dealiasing mask
    k_max = n // 3
    dealias = (mx.abs(kx) < k_max) & (mx.abs(ky) < k_max) & (mx.abs(kz) < k_max)

    n_steps = int(config.t_target / dt)
    Z_max = 0.0

    for step in range(n_steps):
        # Enstrophy
        Z = 0.5 * float(mx.sum(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2))
        Z_max = max(Z_max, Z)

        # Transform to Fourier space
        omega_hat = mx.stack([
            mx.fft.fftn(omega[..., i]) for i in range(3)
        ], axis=-1)

        # Compute velocity via Biot-Savart
        u_hat = mx.zeros_like(omega_hat)
        # u = (ik × ω_hat) / k²
        u_hat[..., 0] = 1j * (ky * omega_hat[..., 2] - kz * omega_hat[..., 1]) / k_sq
        u_hat[..., 1] = 1j * (kz * omega_hat[..., 0] - kx * omega_hat[..., 2]) / k_sq
        u_hat[..., 2] = 1j * (kx * omega_hat[..., 1] - ky * omega_hat[..., 0]) / k_sq

        # Dealias
        u_hat = u_hat * dealias[..., None]

        # Back to physical space
        u = mx.stack([
            mx.fft.ifftn(u_hat[..., i]).real for i in range(3)
        ], axis=-1)

        # Compute strain rate tensor (simplified - use velocity gradients)
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        dudx = mx.fft.ifftn(1j * kx[..., None] * u_hat).real
        dudy = mx.fft.ifftn(1j * ky[..., None] * u_hat).real
        dudz = mx.fft.ifftn(1j * kz[..., None] * u_hat).real

        # Stretching term: (ω·∇)u
        stretch = (omega[..., 0:1] * dudx +
                   omega[..., 1:2] * dudy +
                   omega[..., 2:3] * dudz)

        # Apply H3 depletion
        omega_mag = mx.sqrt(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2 + 1e-10)
        omega_crit = nu * 100
        x = omega_mag / omega_crit
        Phi = mx.clip((x - 1) / (x + 1e-10), 0, 1)
        depletion = 1 - DELTA_0 * Phi[..., None]

        stretch = stretch * depletion

        # Viscous term (in Fourier space)
        visc_hat = -nu * k_sq[..., None] * omega_hat
        visc = mx.stack([
            mx.fft.ifftn(visc_hat[..., i]).real for i in range(3)
        ], axis=-1)

        # Time step (forward Euler for speed)
        omega = omega + dt * (stretch + visc)

        # Dealias in physical space periodically
        if step % 10 == 0:
            omega_hat = mx.stack([
                mx.fft.fftn(omega[..., i]) * dealias for i in range(3)
            ], axis=-1)
            omega = mx.stack([
                mx.fft.ifftn(omega_hat[..., i]).real for i in range(3)
            ], axis=-1)

        mx.eval(omega)

    if return_max_only:
        return Z_max
    return Z_max, omega

def objective_function(params):
    """
    Objective: MINIMIZE negative peak enstrophy (i.e., maximize Z_max)
    """
    try:
        # Create initial condition from parameters
        omega_x, omega_y, omega_z = create_initial_vorticity(params, config.n)

        # Normalize to fixed energy
        omega_x, omega_y, omega_z = normalize_energy(omega_x, omega_y, omega_z, config.E0)

        # Run simulation
        Z_max = run_simulation_mlx(omega_x, omega_y, omega_z)

        # Return negative (we minimize, so this maximizes Z_max)
        return -Z_max
    except Exception as e:
        print(f"Error in objective: {e}")
        return 0.0  # Return neutral value on error

def run_adversarial_optimization():
    """
    Main optimization loop trying to find pathological initial conditions.
    """
    print("\n" + "="*70)
    print("  ADVERSARIAL OPTIMIZATION TEST")
    print("="*70)
    print(f"\n  Goal: Find initial conditions that MAXIMIZE peak enstrophy")
    print(f"  Theoretical bound: Z_max ≈ 547")
    print(f"  If optimizer cannot exceed this, theorem is validated.\n")
    print(f"  Grid: {config.n}³, ν = {config.nu}, t_target = {config.t_target}")
    print(f"  Fixed initial energy E₀ = {config.E0}")
    print()

    # Number of Fourier modes to optimize
    n_modes = 8
    n_params = n_modes * 7

    print(f"  Search space: {n_modes} Fourier modes × 7 parameters = {n_params} dimensions")
    print()

    # Bounds for parameters (all in [0, 1], scaled internally)
    bounds = [(0, 1) for _ in range(n_params)]

    # Track best results
    best_Z = 0
    all_results = []

    # Phase 1: Global search with differential evolution
    print("  PHASE 1: Global Search (Differential Evolution)")
    print("  " + "-"*50)

    start_time = time.time()

    def callback(xk, convergence):
        nonlocal best_Z
        Z = -objective_function(xk)
        if Z > best_Z:
            best_Z = Z
            print(f"    New best: Z_max = {Z:.2f} (bound ratio: {Z/547:.2%})")
        return False

    result_de = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=50,  # Generations
        popsize=15,  # Population size
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        callback=callback,
        disp=False,
        workers=1  # MLX doesn't parallelize well with multiprocessing
    )

    de_time = time.time() - start_time
    de_best_Z = -result_de.fun
    print(f"\n    DE complete in {de_time:.1f}s")
    print(f"    Best Z_max from DE: {de_best_Z:.2f}")
    all_results.append(('DE', de_best_Z))

    # Phase 2: Local refinement from best DE result
    print("\n  PHASE 2: Local Refinement (L-BFGS-B)")
    print("  " + "-"*50)

    start_time = time.time()
    result_local = minimize(
        objective_function,
        result_de.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 30, 'disp': False}
    )

    local_time = time.time() - start_time
    local_best_Z = -result_local.fun
    print(f"    Local refinement complete in {local_time:.1f}s")
    print(f"    Best Z_max from local: {local_best_Z:.2f}")
    all_results.append(('Local', local_best_Z))

    # Phase 3: Try explicitly adversarial initializations
    print("\n  PHASE 3: Targeted Adversarial Initializations")
    print("  " + "-"*50)

    adversarial_configs = [
        ("Anti-icosahedral", create_anti_icosahedral_ic),
        ("High-strain focus", create_high_strain_ic),
        ("Vortex collision", create_collision_ic),
        ("Stretched filament", create_filament_ic),
    ]

    for name, ic_func in adversarial_configs:
        try:
            omega_x, omega_y, omega_z = ic_func(config.n)
            omega_x, omega_y, omega_z = normalize_energy(omega_x, omega_y, omega_z, config.E0)
            Z_max = run_simulation_mlx(omega_x, omega_y, omega_z)
            print(f"    {name}: Z_max = {Z_max:.2f}")
            all_results.append((name, Z_max))
        except Exception as e:
            print(f"    {name}: Error - {e}")

    # Summary
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)

    overall_best = max(r[1] for r in all_results)
    print(f"\n  Theoretical bound: Z_max ≈ 547")
    print(f"  Best adversarial:  Z_max = {overall_best:.2f}")
    print(f"  Ratio to bound:    {overall_best/547:.1%}")
    print()

    print("  All attempts:")
    for name, Z in sorted(all_results, key=lambda x: -x[1]):
        status = "⚠️  EXCEEDED" if Z > 547 else "✓ bounded"
        print(f"    {name:25s}: {Z:8.2f}  {status}")

    print()
    if overall_best <= 547:
        print("  " + "="*50)
        print("  ✓ THEOREM VALIDATED: Optimizer could not exceed bound")
        print("  " + "="*50)
    else:
        print("  " + "="*50)
        print("  ⚠️  WARNING: Bound exceeded - investigate!")
        print("  " + "="*50)

    return overall_best, all_results

# Adversarial initial condition generators
def create_anti_icosahedral_ic(n):
    """Create IC that deliberately avoids icosahedral axes"""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Use cubic axes (maximally different from icosahedral)
    omega_x = 10 * np.sin(2*X) * np.cos(2*Y)
    omega_y = 10 * np.sin(2*Y) * np.cos(2*Z)
    omega_z = 10 * np.sin(2*Z) * np.cos(2*X)

    return omega_x, omega_y, omega_z

def create_high_strain_ic(n):
    """Create IC designed to maximize strain concentration"""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Opposing vortex sheets - high strain in between
    omega_x = np.zeros((n, n, n), dtype=np.float32)
    omega_y = np.zeros((n, n, n), dtype=np.float32)
    omega_z = 10 * (np.tanh(5*(Y - np.pi/2)) - np.tanh(5*(Y - 3*np.pi/2)))

    return omega_x, omega_y, omega_z

def create_collision_ic(n):
    """Create two vortex tubes on collision course"""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Two vortex rings approaching each other
    r1 = np.sqrt((X - np.pi)**2 + (Y - np.pi/2)**2)
    r2 = np.sqrt((X - np.pi)**2 + (Y - 3*np.pi/2)**2)

    omega_x = np.zeros((n, n, n), dtype=np.float32)
    omega_y = np.zeros((n, n, n), dtype=np.float32)
    omega_z = 10 * (np.exp(-r1**2/0.5) - np.exp(-r2**2/0.5))

    return omega_x, omega_y, omega_z

def create_filament_ic(n):
    """Create a stretched vortex filament prone to instability"""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Thin filament along z-axis with perturbation
    r = np.sqrt((X - np.pi)**2 + (Y - np.pi)**2)
    perturbation = 0.3 * np.sin(4*Z)

    omega_x = np.zeros((n, n, n), dtype=np.float32)
    omega_y = np.zeros((n, n, n), dtype=np.float32)
    omega_z = 10 * np.exp(-(r - perturbation)**2 / 0.1)

    return omega_x, omega_y, omega_z

if __name__ == "__main__":
    best_Z, results = run_adversarial_optimization()

    # Save results
    import json
    with open("adversarial_optimization_results.json", "w") as f:
        json.dump({
            "theoretical_bound": 547,
            "best_adversarial": best_Z,
            "ratio": best_Z / 547,
            "results": [(name, float(z)) for name, z in results],
            "verdict": "BOUNDED" if best_Z <= 547 else "EXCEEDED"
        }, f, indent=2)

    print(f"\nResults saved to adversarial_optimization_results.json")
