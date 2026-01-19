#!/usr/bin/env python3
"""
TEST: Non-Icosahedral Initial Conditions

Question: Does icosahedral alignment emerge from dynamics, or is it
imposed by our choice of initial conditions?

Previous tests used Taylor-Green with 12-fold icosahedral perturbation.
This test uses explicitly NON-icosahedral ICs:

1. Pure Taylor-Green (cubic symmetry, no icosahedral)
2. Random smooth vorticity field
3. Axisymmetric vortex ring (cylindrical symmetry)
4. Octahedral perturbation (4-fold, not 5-fold)

If icosahedral alignment emerges in ALL cases, the geometry is intrinsic.
"""

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4

# 27 icosahedral axes (12 five-fold + 15 two-fold)
def get_icosahedral_axes():
    axes = []
    # 6 five-fold axes (through vertices)
    for signs in [(1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)]:
        s1, s2, s3 = signs
        axes.append(np.array([0, s1*1, s2*PHI]) / np.sqrt(1 + PHI**2))
        axes.append(np.array([s1*1, s2*PHI, 0]) / np.sqrt(1 + PHI**2))
        axes.append(np.array([s2*PHI, 0, s1*1]) / np.sqrt(1 + PHI**2))
    # 15 two-fold axes (through edge midpoints)
    for s in [1, -1]:
        axes.append(np.array([s*1, 0, 0]))
        axes.append(np.array([0, s*1, 0]))
        axes.append(np.array([0, 0, s*1]))
    for s1, s2 in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        axes.append(np.array([s1*PHI, s2*1, 0]) / np.sqrt(1 + PHI**2))
        axes.append(np.array([0, s1*PHI, s2*1]) / np.sqrt(1 + PHI**2))
        axes.append(np.array([s1*1, 0, s2*PHI]) / np.sqrt(1 + PHI**2))
    return np.array(axes[:27])

ICO_AXES = get_icosahedral_axes()


def pure_taylor_green(n, scale=5.0):
    """Pure Taylor-Green vortex - CUBIC symmetry, no icosahedral."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Standard TG - has cubic O_h symmetry, NOT icosahedral
    wx = scale * np.cos(X) * np.sin(Y) * np.cos(Z)
    wy = -scale * np.sin(X) * np.cos(Y) * np.cos(Z)
    wz = np.zeros_like(X)

    return np.stack([wx, wy, wz], axis=-1)


def random_smooth_vorticity(n, scale=5.0, k_max=4):
    """Random smooth vorticity field - NO symmetry."""
    np.random.seed(42)  # Reproducible

    # Random Fourier coefficients for low-k modes
    wx_hat = np.zeros((n, n, n), dtype=complex)
    wy_hat = np.zeros((n, n, n), dtype=complex)
    wz_hat = np.zeros((n, n, n), dtype=complex)

    k = np.fft.fftfreq(n, d=1/n) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    # Fill low-k modes with random complex numbers
    mask = (k_mag > 0) & (k_mag < k_max * 2 * np.pi / n * n)
    n_modes = np.sum(mask)

    wx_hat[mask] = (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))
    wy_hat[mask] = (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))
    wz_hat[mask] = (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))

    # Make divergence-free: project out k·ω component
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1  # Avoid division by zero

    k_dot_w = kx * wx_hat + ky * wy_hat + kz * wz_hat
    wx_hat = wx_hat - kx * k_dot_w / k2
    wy_hat = wy_hat - ky * k_dot_w / k2
    wz_hat = wz_hat - kz * k_dot_w / k2

    # Transform to physical space
    wx = np.real(np.fft.ifftn(wx_hat))
    wy = np.real(np.fft.ifftn(wy_hat))
    wz = np.real(np.fft.ifftn(wz_hat))

    # Normalize
    w_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    norm = scale / (np.max(w_mag) + 1e-10)

    return np.stack([wx * norm, wy * norm, wz * norm], axis=-1)


def axisymmetric_vortex_ring(n, scale=5.0):
    """Axisymmetric vortex ring - CYLINDRICAL symmetry."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Center at (π, π, π)
    X = X - np.pi
    Y = Y - np.pi
    Z = Z - np.pi

    # Cylindrical coords (r, θ, z) with axis along z
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Vortex ring: ω in θ direction, localized at r=R, z=0
    R = 1.5  # Ring radius
    a = 0.5  # Core size

    # Gaussian core profile
    dist_from_core = np.sqrt((r - R)**2 + Z**2)
    core_profile = np.exp(-dist_from_core**2 / (2 * a**2))

    # Vorticity in θ direction (tangent to ring)
    w_theta = scale * core_profile

    # Convert to Cartesian
    wx = -w_theta * np.sin(theta)
    wy = w_theta * np.cos(theta)
    wz = np.zeros_like(X)

    return np.stack([wx, wy, wz], axis=-1)


def octahedral_vortex(n, scale=5.0):
    """Vortex with OCTAHEDRAL (4-fold) symmetry, not icosahedral (5-fold)."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Octahedral axes: face centers of cube (6 axes)
    # This has 4-fold rotational symmetry, NOT 5-fold
    wx = scale * (np.sin(2*X) * np.cos(2*Y) - np.sin(2*Z) * np.cos(2*X))
    wy = scale * (np.sin(2*Y) * np.cos(2*Z) - np.sin(2*X) * np.cos(2*Y))
    wz = scale * (np.sin(2*Z) * np.cos(2*X) - np.sin(2*Y) * np.cos(2*Z))

    return np.stack([wx, wy, wz], axis=-1)


class SimpleSolver:
    """Minimal NS solver for testing IC dependence."""

    def __init__(self, n=64, nu=0.001, dt=0.0001):
        self.n, self.nu, self.dt = n, nu, dt
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)
        self.visc = mx.exp(-nu * self.k2 * dt)

    def step(self, wh):
        """One timestep - standard NS, NO depletion."""
        wx_hat, wy_hat, wz_hat = wh

        # Velocity from vorticity
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe

        # Physical space
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real
        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real

        # Nonlinear term: (ω·∇)u - (u·∇)ω
        # Simplified: just advection
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Update
        return ((wx_hat + self.dt * mx.fft.fftn(nlx)) * self.visc,
                (wy_hat + self.dt * mx.fft.fftn(nly)) * self.visc,
                (wz_hat + self.dt * mx.fft.fftn(nlz)) * self.visc)

    def compute_alignment(self, wh):
        """Compute alignment to icosahedral axes."""
        wx = mx.fft.ifftn(wh[0]).real
        wy = mx.fft.ifftn(wh[1]).real
        wz = mx.fft.ifftn(wh[2]).real

        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2)
        threshold = 2.0 * mx.mean(omega_mag)
        mask = omega_mag > threshold

        if float(mx.sum(mask).item()) < 100:
            return 0.0, 0.0

        # Sample high-vorticity points
        wx_np = np.array(wx)
        wy_np = np.array(wy)
        wz_np = np.array(wz)
        mask_np = np.array(mask)

        omega_vec = np.stack([wx_np[mask_np], wy_np[mask_np], wz_np[mask_np]], axis=1)
        omega_norm = omega_vec / (np.linalg.norm(omega_vec, axis=1, keepdims=True) + 1e-10)

        # Max alignment to any icosahedral axis
        alignments = np.abs(omega_norm @ ICO_AXES.T)
        max_align = np.max(alignments, axis=1)

        return float(np.mean(max_align)), float(np.std(max_align))

    def compute_enstrophy(self, wh):
        """Compute enstrophy."""
        Z = 0.5 * float(mx.mean(sum(mx.fft.ifftn(h).real**2 for h in wh)).item())
        return Z


def run_test(name, omega_init, n=64, t_max=2.0, dt=0.0001):
    """Run simulation and track alignment."""
    print(f"\n{'='*60}")
    print(f"  IC: {name}")
    print(f"{'='*60}")

    solver = SimpleSolver(n=n, nu=0.001, dt=dt)
    wh = tuple(mx.fft.fftn(mx.array(omega_init[..., i])) for i in range(3))

    nsteps = int(t_max / dt)
    results = {'t': [], 'Z': [], 'align_mean': [], 'align_std': []}

    # Initial alignment
    align_mean, align_std = solver.compute_alignment(wh)
    print(f"  t=0.00: Z={solver.compute_enstrophy(wh):.1f}, "
          f"align={align_mean:.3f}±{align_std:.3f}")

    results['t'].append(0)
    results['Z'].append(solver.compute_enstrophy(wh))
    results['align_mean'].append(align_mean)
    results['align_std'].append(align_std)

    for step in range(nsteps):
        wh = solver.step(wh)

        if (step + 1) % 2000 == 0:
            mx.eval(*wh)
            t = (step + 1) * dt
            Z = solver.compute_enstrophy(wh)
            align_mean, align_std = solver.compute_alignment(wh)

            results['t'].append(t)
            results['Z'].append(Z)
            results['align_mean'].append(align_mean)
            results['align_std'].append(align_std)

            print(f"  t={t:.2f}: Z={Z:.1f}, align={align_mean:.3f}±{align_std:.3f}")

            # Check for blowup
            if Z > 1e6 or np.isnan(Z):
                print(f"  BLOWUP at t={t:.2f}")
                break

    return results


def main():
    print("="*70)
    print("  NON-ICOSAHEDRAL INITIAL CONDITIONS TEST")
    print("="*70)
    print("""
  Question: Does icosahedral alignment EMERGE from dynamics?

  Testing 4 different ICs with NO icosahedral symmetry:
  1. Pure Taylor-Green (cubic O_h symmetry)
  2. Random smooth vorticity (no symmetry)
  3. Axisymmetric vortex ring (cylindrical symmetry)
  4. Octahedral vortex (4-fold, not 5-fold)

  If alignment emerges in all cases → geometry is INTRINSIC
    """)

    n = 64

    # Generate ICs
    ics = {
        'Pure Taylor-Green (cubic)': pure_taylor_green(n),
        'Random smooth': random_smooth_vorticity(n),
        'Axisymmetric ring': axisymmetric_vortex_ring(n),
        'Octahedral (4-fold)': octahedral_vortex(n),
    }

    all_results = {}

    for name, omega_init in ics.items():
        results = run_test(name, omega_init, n=n, t_max=1.5)
        all_results[name] = results

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    print("\n  Initial vs Final Icosahedral Alignment:")
    print("  " + "-"*50)
    print(f"  {'IC Type':<25} {'t=0':>10} {'t=final':>10} {'Change':>10}")
    print("  " + "-"*50)

    for name, results in all_results.items():
        init_align = results['align_mean'][0]
        final_align = results['align_mean'][-1]
        change = final_align - init_align
        sign = '+' if change > 0 else ''
        print(f"  {name:<25} {init_align:>10.3f} {final_align:>10.3f} {sign}{change:>9.3f}")

    # Interpretation
    print("\n  INTERPRETATION:")
    increases = sum(1 for r in all_results.values()
                   if r['align_mean'][-1] > r['align_mean'][0])

    if increases == len(all_results):
        print("  ✓ ALL ICs show INCREASED icosahedral alignment!")
        print("  → The geometry is INTRINSIC to NS dynamics")
    elif increases > len(all_results) // 2:
        print(f"  ◐ {increases}/{len(all_results)} ICs show increased alignment")
        print("  → Partial evidence for intrinsic geometry")
    else:
        print(f"  ✗ Only {increases}/{len(all_results)} ICs show increased alignment")
        print("  → Icosahedral structure may be IC-dependent")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for ax, (name, results) in zip(axes.flat, all_results.items()):
        ax.plot(results['t'], results['align_mean'], 'b-', linewidth=2, label='Mean alignment')
        ax.fill_between(results['t'],
                       np.array(results['align_mean']) - np.array(results['align_std']),
                       np.array(results['align_mean']) + np.array(results['align_std']),
                       alpha=0.3)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.axhline(results['align_mean'][0], color='r', linestyle=':', alpha=0.5, label='Initial')
        ax.set_xlabel('Time')
        ax.set_ylabel('Icosahedral Alignment')
        ax.set_title(name)
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_non_icosahedral_ic.png', dpi=150)
    print("\nSaved: test_non_icosahedral_ic.png")

    return all_results


if __name__ == "__main__":
    results = main()
