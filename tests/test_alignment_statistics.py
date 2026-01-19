#!/usr/bin/env python3
"""
TEST: Statistical Alignment to Icosahedral Axes

Problem with previous test: 27 icosahedral axes cover the sphere so well
that ANY direction has high "max alignment". We need a STATISTICAL test.

Approach:
1. Generate uniform random directions on sphere
2. Compute alignment distribution for random → this is the NULL hypothesis
3. Compare actual vorticity alignment to random baseline
4. Use Kolmogorov-Smirnov test for significance

If vorticity directions are MORE clustered around icosahedral axes than
random, the K-S test will show significant deviation.
"""

import numpy as np
import mlx.core as mx
from scipy import stats
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4

# 12 five-fold axes only (more stringent test)
def get_fivefold_axes():
    """Just the 12 five-fold icosahedral axes (6 pairs)."""
    axes = []
    # Vertices of icosahedron
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            axes.append(np.array([0, s1*1, s2*PHI]))
            axes.append(np.array([s1*1, s2*PHI, 0]))
            axes.append(np.array([s2*PHI, 0, s1*1]))
    axes = np.array(axes)
    return axes / np.linalg.norm(axes, axis=1, keepdims=True)

FIVEFOLD_AXES = get_fivefold_axes()
print(f"Using {len(FIVEFOLD_AXES)} five-fold axes")


def compute_alignment_distribution(directions, axes):
    """Compute max alignment of each direction to any axis."""
    # directions: (N, 3), axes: (M, 3)
    alignments = np.abs(directions @ axes.T)  # (N, M)
    max_align = np.max(alignments, axis=1)  # (N,)
    return max_align


def random_directions_on_sphere(n):
    """Generate n uniformly random directions on unit sphere."""
    # Marsaglia method
    z = np.random.uniform(-1, 1, n)
    phi = np.random.uniform(0, 2*np.pi, n)
    r = np.sqrt(1 - z**2)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)


def taylor_green_stretched(n, scale=10.0):
    """Taylor-Green with higher amplitude for enstrophy growth."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Add 12-fold icosahedral perturbation for growth
    perturb = np.zeros_like(X)
    for axis in FIVEFOLD_AXES[:6]:  # Just 6 axes (not duplicates)
        k = 2 * axis
        perturb += 0.3 * np.sin(k[0]*X + k[1]*Y + k[2]*Z)

    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z) * (1 + 0.2*perturb)
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z) * (1 + 0.2*perturb)
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z) * (1 + 0.2*perturb)

    return np.stack([wx, wy, wz], axis=-1)


def random_smooth_vorticity(n, scale=10.0, k_max=4):
    """Random smooth vorticity with higher amplitude."""
    np.random.seed(123)  # Different seed

    wx_hat = np.zeros((n, n, n), dtype=complex)
    wy_hat = np.zeros((n, n, n), dtype=complex)
    wz_hat = np.zeros((n, n, n), dtype=complex)

    k = np.fft.fftfreq(n, d=1/n) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    mask = (k_mag > 0) & (k_mag < k_max * 2 * np.pi / n * n)
    n_modes = np.sum(mask)

    wx_hat[mask] = (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))
    wy_hat[mask] = (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))
    wz_hat[mask] = (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))

    # Make divergence-free
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1
    k_dot_w = kx * wx_hat + ky * wy_hat + kz * wz_hat
    wx_hat = wx_hat - kx * k_dot_w / k2
    wy_hat = wy_hat - ky * k_dot_w / k2
    wz_hat = wz_hat - kz * k_dot_w / k2

    wx = np.real(np.fft.ifftn(wx_hat))
    wy = np.real(np.fft.ifftn(wy_hat))
    wz = np.real(np.fft.ifftn(wz_hat))

    w_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    norm = scale / (np.max(w_mag) + 1e-10)

    return np.stack([wx * norm, wy * norm, wz * norm], axis=-1)


class Solver:
    def __init__(self, n=64, nu=0.001, dt=0.0001):
        self.n, self.nu, self.dt = n, nu, dt
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)
        self.visc = mx.exp(-nu * self.k2 * dt)

    def step(self, wh):
        wx_hat, wy_hat, wz_hat = wh
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe

        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real
        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real

        # Full vorticity equation with stretching
        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real

        # Stretching: (ω·∇)u
        stretch_x = wx * dux_dx + wy * dux_dy + wz * dux_dz
        stretch_y = wx * duy_dx + wy * duy_dy + wz * duy_dz
        stretch_z = wx * duz_dx + wy * duz_dy + wz * duz_dz

        # Advection: -(u·∇)ω ≈ ω×u for incompressible
        nlx = wy * uz - wz * uy + stretch_x
        nly = wz * ux - wx * uz + stretch_y
        nlz = wx * uy - wy * ux + stretch_z

        return ((wx_hat + self.dt * mx.fft.fftn(nlx)) * self.visc,
                (wy_hat + self.dt * mx.fft.fftn(nly)) * self.visc,
                (wz_hat + self.dt * mx.fft.fftn(nlz)) * self.visc)

    def get_vorticity_directions(self, wh, threshold_factor=2.0):
        """Extract normalized vorticity directions in high-|ω| regions."""
        wx = np.array(mx.fft.ifftn(wh[0]).real)
        wy = np.array(mx.fft.ifftn(wh[1]).real)
        wz = np.array(mx.fft.ifftn(wh[2]).real)

        omega_mag = np.sqrt(wx**2 + wy**2 + wz**2)
        threshold = threshold_factor * np.mean(omega_mag)
        mask = omega_mag > threshold

        if np.sum(mask) < 100:
            return None

        omega_vec = np.stack([wx[mask], wy[mask], wz[mask]], axis=1)
        omega_norm = omega_vec / (np.linalg.norm(omega_vec, axis=1, keepdims=True) + 1e-10)

        return omega_norm

    def compute_enstrophy(self, wh):
        return 0.5 * float(mx.mean(sum(mx.fft.ifftn(h).real**2 for h in wh)).item())


def run_statistical_test(name, omega_init, n=64, t_target=1.0):
    """Run simulation and perform K-S test against random baseline."""
    print(f"\n{'='*60}")
    print(f"  IC: {name}")
    print(f"{'='*60}")

    solver = Solver(n=n, nu=0.001, dt=0.0001)
    wh = tuple(mx.fft.fftn(mx.array(omega_init[..., i])) for i in range(3))

    # Generate random baseline
    n_random = 10000
    random_dirs = random_directions_on_sphere(n_random)
    random_alignment = compute_alignment_distribution(random_dirs, FIVEFOLD_AXES)

    print(f"  Random baseline: mean={np.mean(random_alignment):.3f}, "
          f"std={np.std(random_alignment):.3f}")

    # Initial state
    dirs_init = solver.get_vorticity_directions(wh)
    if dirs_init is not None:
        align_init = compute_alignment_distribution(dirs_init, FIVEFOLD_AXES)
        ks_init, p_init = stats.ks_2samp(align_init, random_alignment)
        print(f"  t=0.00: Z={solver.compute_enstrophy(wh):.1f}, "
              f"mean_align={np.mean(align_init):.3f}, "
              f"KS={ks_init:.3f}, p={p_init:.2e}")
    else:
        align_init = None
        ks_init, p_init = 0, 1

    # Run to target time
    nsteps = int(t_target / solver.dt)
    for step in range(nsteps):
        wh = solver.step(wh)
        if (step + 1) % 5000 == 0:
            mx.eval(*wh)

    mx.eval(*wh)

    # Final state
    dirs_final = solver.get_vorticity_directions(wh)
    if dirs_final is not None:
        align_final = compute_alignment_distribution(dirs_final, FIVEFOLD_AXES)
        ks_final, p_final = stats.ks_2samp(align_final, random_alignment)
        print(f"  t={t_target:.2f}: Z={solver.compute_enstrophy(wh):.1f}, "
              f"mean_align={np.mean(align_final):.3f}, "
              f"KS={ks_final:.3f}, p={p_final:.2e}")
    else:
        align_final = None
        ks_final, p_final = 0, 1

    return {
        'random_alignment': random_alignment,
        'align_init': align_init,
        'align_final': align_final,
        'ks_init': ks_init, 'p_init': p_init,
        'ks_final': ks_final, 'p_final': p_final,
    }


def main():
    print("="*70)
    print("  STATISTICAL ALIGNMENT TEST")
    print("="*70)
    print("""
  Comparing vorticity alignment to RANDOM BASELINE using K-S test.

  Null hypothesis: vorticity directions are uniformly distributed
  Alternative: vorticity clusters around icosahedral axes

  If p < 0.05 → significant clustering (reject null)
  If p > 0.05 → consistent with random (cannot reject null)

  Using 12 FIVE-FOLD axes only (more stringent than 27 axes)
    """)

    n = 64

    # Test cases
    results = {}

    # 1. Taylor-Green with icosahedral perturbation (expected: significant)
    print("\n" + "="*70)
    print("  Test 1: Taylor-Green with icosahedral perturbation")
    omega_tg = taylor_green_stretched(n, scale=10.0)
    results['TG+Ico'] = run_statistical_test('TG+Icosahedral', omega_tg, t_target=1.0)

    # 2. Random smooth (expected: not significant initially, test if emerges)
    print("\n" + "="*70)
    print("  Test 2: Random smooth vorticity")
    omega_rand = random_smooth_vorticity(n, scale=10.0)
    results['Random'] = run_statistical_test('Random', omega_rand, t_target=1.0)

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY: K-S Test Results")
    print("="*70)
    print(f"\n  {'IC':<20} {'KS_init':>10} {'p_init':>12} {'KS_final':>10} {'p_final':>12}")
    print("  " + "-"*66)

    for name, r in results.items():
        p_init_str = f"{r['p_init']:.2e}" if r['p_init'] < 0.01 else f"{r['p_init']:.3f}"
        p_final_str = f"{r['p_final']:.2e}" if r['p_final'] < 0.01 else f"{r['p_final']:.3f}"
        print(f"  {name:<20} {r['ks_init']:>10.3f} {p_init_str:>12} "
              f"{r['ks_final']:>10.3f} {p_final_str:>12}")

    print("\n  INTERPRETATION:")
    for name, r in results.items():
        if r['p_final'] < 0.05:
            print(f"  {name}: ✓ Significant icosahedral clustering (p={r['p_final']:.2e})")
        else:
            print(f"  {name}: ✗ Consistent with random (p={r['p_final']:.3f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, r) in zip(axes, results.items()):
        # Histogram of alignments
        bins = np.linspace(0.5, 1.0, 30)
        ax.hist(r['random_alignment'], bins=bins, alpha=0.3, density=True,
                label='Random baseline', color='gray')
        if r['align_init'] is not None:
            ax.hist(r['align_init'], bins=bins, alpha=0.5, density=True,
                    label='t=0', color='blue')
        if r['align_final'] is not None:
            ax.hist(r['align_final'], bins=bins, alpha=0.5, density=True,
                    label='t=final', color='red')
        ax.set_xlabel('Max alignment to five-fold axis')
        ax.set_ylabel('Density')
        ax.set_title(f'{name}\n(KS={r["ks_final"]:.3f}, p={r["p_final"]:.2e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_alignment_statistics.png', dpi=150)
    print("\nSaved: test_alignment_statistics.png")

    return results


if __name__ == "__main__":
    results = main()
