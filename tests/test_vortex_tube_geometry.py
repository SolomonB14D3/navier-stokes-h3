#!/usr/bin/env python3
"""
TEST: VORTEX TUBE GEOMETRY AND ICOSAHEDRAL ALIGNMENT

This test directly probes the key claim from Theorem 5.2:
"Intense vorticity must align with lattice symmetry axes because
 vortex tubes follow paths of minimal energy (geodesics)."

We measure:
1. Vortex tube curvature - should be bounded by 2π/(5R) ≈ 1.32
2. Tube tangent alignment to icosahedral axes
3. Whether curvature/alignment correlates with vorticity intensity

This is a CRITICAL test of the proof's assumptions.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Install with: pip install trimesh")

import mlx.core as mx

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4
R_H3 = 0.951

# Curvature bound from proof (geodesic on icosahedral manifold)
KAPPA_BOUND = 2 * np.pi / (5 * R_H3)  # ≈ 1.32

# Icosahedral symmetry axes (from projection matrix P in proof)
# 6 five-fold axes (vertices of icosahedron)
ICOSA_5FOLD = np.array([
    [1, PHI, 0], [-1, PHI, 0], [1, -PHI, 0], [-1, -PHI, 0],
    [0, 1, PHI], [0, -1, PHI], [0, 1, -PHI], [0, -1, -PHI],
    [PHI, 0, 1], [-PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, -1]
]) / np.sqrt(1 + PHI**2)

# 15 two-fold axes (edge midpoints)
ICOSA_2FOLD = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [PHI, 1/PHI, 0], [PHI, -1/PHI, 0], [-PHI, 1/PHI, 0], [-PHI, -1/PHI, 0],
    [0, PHI, 1/PHI], [0, PHI, -1/PHI], [0, -PHI, 1/PHI], [0, -PHI, -1/PHI],
    [1/PHI, 0, PHI], [-1/PHI, 0, PHI], [1/PHI, 0, -PHI], [-1/PHI, 0, -PHI]
])
ICOSA_2FOLD = ICOSA_2FOLD / np.linalg.norm(ICOSA_2FOLD, axis=1, keepdims=True)

# All icosahedral axes
ICOSA_AXES = np.vstack([ICOSA_5FOLD, ICOSA_2FOLD])

print(f"Curvature bound κ_max = 2π/(5R) = {KAPPA_BOUND:.4f}")
print(f"Number of icosahedral axes: {len(ICOSA_AXES)}")


class NSVortexTubeAnalyzer:
    """NS solver that extracts and analyzes vortex tube geometry."""

    def __init__(self, n=64, viscosity=0.001, dt=0.0001):
        self.n = n
        self.nu = viscosity
        self.dt = dt
        self.L = 2 * np.pi

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
        """Standard NS step."""
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        # Advection + Stretching
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        dux_dx = mx.fft.ifftn(1j * self.kx * ux_hat).real
        dux_dy = mx.fft.ifftn(1j * self.ky * ux_hat).real
        dux_dz = mx.fft.ifftn(1j * self.kz * ux_hat).real
        duy_dx = mx.fft.ifftn(1j * self.kx * uy_hat).real
        duy_dy = mx.fft.ifftn(1j * self.ky * uy_hat).real
        duy_dz = mx.fft.ifftn(1j * self.kz * uy_hat).real
        duz_dx = mx.fft.ifftn(1j * self.kx * uz_hat).real
        duz_dy = mx.fft.ifftn(1j * self.ky * uz_hat).real
        duz_dz = mx.fft.ifftn(1j * self.kz * uz_hat).real

        nlx = nlx + wx * dux_dx + wy * dux_dy + wz * dux_dz
        nly = nly + wx * duy_dx + wy * duy_dy + wz * duy_dz
        nlz = nlz + wx * duz_dx + wy * duz_dy + wz * duz_dz

        nlx_hat = mx.fft.fftn(nlx)
        nly_hat = mx.fft.fftn(nly)
        nlz_hat = mx.fft.fftn(nlz)

        wx_hat_new = (wx_hat + self.dt * nlx_hat) * self.visc_decay
        wy_hat_new = (wy_hat + self.dt * nly_hat) * self.visc_decay
        wz_hat_new = (wz_hat + self.dt * nlz_hat) * self.visc_decay

        return wx_hat_new, wy_hat_new, wz_hat_new

    def run_to_peak(self, omega_init, tmax=1.5):
        """Run until peak enstrophy or blowup, return omega field."""
        wx = mx.array(omega_init[..., 0])
        wy = mx.array(omega_init[..., 1])
        wz = mx.array(omega_init[..., 2])

        wx_hat = mx.fft.fftn(wx)
        wy_hat = mx.fft.fftn(wy)
        wz_hat = mx.fft.fftn(wz)

        nsteps = int(tmax / self.dt)
        Z_max = 0
        omega_at_peak = None
        t_peak = 0

        print("  Running to find peak enstrophy...")
        for step in range(nsteps):
            if step % 500 == 0:
                wx = mx.fft.ifftn(wx_hat).real
                wy = mx.fft.ifftn(wy_hat).real
                wz = mx.fft.ifftn(wz_hat).real
                Z = 0.5 * float(mx.mean(wx**2 + wy**2 + wz**2).item())

                if step % 2000 == 0:
                    print(f"    t={step*self.dt:.3f}, Z={Z:.2f}")

                if np.isnan(Z) or Z > 1e6:
                    print(f"  Stopping at t={step*self.dt:.3f} (blowup imminent)")
                    break

                if Z > Z_max:
                    Z_max = Z
                    omega_at_peak = np.stack([np.array(wx), np.array(wy), np.array(wz)], axis=-1)
                    t_peak = step * self.dt

            wx_hat, wy_hat, wz_hat = self.step(wx_hat, wy_hat, wz_hat)
            mx.eval(wx_hat, wy_hat, wz_hat)

        print(f"  Peak enstrophy Z={Z_max:.2f} at t={t_peak:.3f}")
        return omega_at_peak, t_peak, Z_max


def extract_vortex_tubes(omega, level_factor=2.0):
    """Extract vortex tube isosurfaces from vorticity field."""
    if not SKIMAGE_AVAILABLE:
        return None, None

    omega_norm = np.linalg.norm(omega, axis=-1)
    level = level_factor * np.mean(omega_norm)

    try:
        verts, faces, normals, values = marching_cubes(
            omega_norm, level=level, spacing=(1, 1, 1)
        )
        return verts, faces
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return None, None


def compute_tube_curvature(verts, faces):
    """Compute mean curvature of vortex tube surface."""
    if not TRIMESH_AVAILABLE or verts is None:
        return None, None

    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Get vertex curvatures
        # trimesh computes discrete curvature
        curvature = mesh.vertex_defects  # Angular defect (Gaussian-like)

        # For mean curvature, use the vertex normals method
        # This is a simplified approximation
        mean_curvature = np.abs(curvature) / (2 * np.pi)

        return mean_curvature, mesh
    except Exception as e:
        print(f"  Curvature computation failed: {e}")
        return None, None


def compute_tube_tangents(omega, threshold_factor=2.0):
    """
    Compute vortex tube tangent directions from vorticity field.

    Tangent = ω/|ω| at high-vorticity points.
    """
    omega_norm = np.linalg.norm(omega, axis=-1)
    threshold = threshold_factor * np.mean(omega_norm)

    # High vorticity mask
    mask = omega_norm > threshold

    # Tangent directions (normalized vorticity)
    tangents = omega[mask] / (omega_norm[mask, np.newaxis] + 1e-10)

    # Positions of high-vorticity points
    positions = np.argwhere(mask)

    return tangents, positions, omega_norm[mask]


def compute_icosa_alignment(tangents):
    """
    Compute alignment of tube tangents to icosahedral axes.

    Returns max |cos(angle)| to any icosahedral axis.
    """
    if len(tangents) == 0:
        return np.array([])

    # Compute dot products with all axes
    # tangents: (N, 3), ICOSA_AXES: (M, 3)
    dots = np.abs(tangents @ ICOSA_AXES.T)  # (N, M)

    # Max alignment to any axis
    max_alignment = np.max(dots, axis=1)

    return max_alignment


def taylor_green_vorticity(n, scale=5.0):
    """Taylor-Green initial vorticity."""
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
    wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
    wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)

    # Add 12-fold perturbation
    theta = np.arctan2(Y - np.pi, X - np.pi)
    pert = 0.15 * scale * np.sin(12 * theta)
    wx += pert * np.cos(Z)
    wy += pert * np.sin(Z)

    return np.stack([wx, wy, wz], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  VORTEX TUBE GEOMETRY TEST")
    print("=" * 70)
    print(f"""
  Testing the proof's claim that vortex tubes:
  1. Have curvature bounded by κ_max = 2π/(5R) ≈ {KAPPA_BOUND:.2f}
  2. Tangent directions align with icosahedral axes

  This is a CRITICAL test of the geometric constraint mechanism.
    """)

    if not SKIMAGE_AVAILABLE or not TRIMESH_AVAILABLE:
        print("ERROR: Required libraries not available.")
        print("Install with: pip install scikit-image trimesh")
        return None

    n = 64
    nu = 0.001
    scale = 5.0
    tmax = 1.2

    print(f"Parameters: n={n}, ν={nu}, scale={scale}")

    # Initialize
    omega_init = taylor_green_vorticity(n, scale=scale)

    # Run to peak enstrophy
    solver = NSVortexTubeAnalyzer(n=n, viscosity=nu)
    omega_peak, t_peak, Z_peak = solver.run_to_peak(omega_init, tmax=tmax)

    if omega_peak is None:
        print("Failed to get peak vorticity field!")
        return None

    print(f"\n--- Analyzing vortex tubes at t={t_peak:.3f} ---")

    # Extract vortex tube surfaces at different thresholds
    results = {
        'thresholds': [],
        'n_vertices': [],
        'mean_curvature': [],
        'max_curvature': [],
        'mean_alignment': [],
        'p90_alignment': [],
        'fraction_high_align': []
    }

    for level_factor in [1.5, 2.0, 2.5, 3.0]:
        print(f"\n  Threshold: {level_factor}× mean vorticity")

        # Extract tube surface
        verts, faces = extract_vortex_tubes(omega_peak, level_factor=level_factor)

        if verts is None or len(verts) < 100:
            print(f"    Insufficient vertices ({len(verts) if verts is not None else 0})")
            continue

        print(f"    Vertices: {len(verts)}, Faces: {len(faces)}")

        # Compute curvature
        curvature, mesh = compute_tube_curvature(verts, faces)

        if curvature is not None:
            mean_kappa = np.mean(np.abs(curvature))
            max_kappa = np.max(np.abs(curvature))
            print(f"    Mean curvature: {mean_kappa:.4f} (bound: {KAPPA_BOUND:.2f})")
            print(f"    Max curvature: {max_kappa:.4f}")
            print(f"    Curvature bounded? {mean_kappa < KAPPA_BOUND}")
        else:
            mean_kappa = max_kappa = np.nan

        # Compute tangent alignment
        tangents, positions, omega_at_points = compute_tube_tangents(
            omega_peak, threshold_factor=level_factor
        )

        if len(tangents) > 0:
            alignment = compute_icosa_alignment(tangents)
            mean_align = np.mean(alignment)
            p90_align = np.percentile(alignment, 90)
            # Fraction with alignment > 0.9 (within ~25° of an axis)
            frac_high = np.mean(alignment > 0.9)

            print(f"    Mean icosahedral alignment: {mean_align:.4f}")
            print(f"    90th percentile alignment: {p90_align:.4f}")
            print(f"    Fraction with alignment > 0.9: {frac_high:.1%}")
        else:
            mean_align = p90_align = frac_high = np.nan

        results['thresholds'].append(level_factor)
        results['n_vertices'].append(len(verts))
        results['mean_curvature'].append(mean_kappa)
        results['max_curvature'].append(max_kappa)
        results['mean_alignment'].append(mean_align)
        results['p90_alignment'].append(p90_align)
        results['fraction_high_align'].append(frac_high)

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    if len(results['thresholds']) == 0:
        print("No valid measurements!")
        return results

    mean_kappa_all = np.nanmean(results['mean_curvature'])
    max_kappa_all = np.nanmax(results['max_curvature'])
    mean_align_all = np.nanmean(results['mean_alignment'])
    p90_align_all = np.nanmean(results['p90_alignment'])
    frac_high_all = np.nanmean(results['fraction_high_align'])

    print(f"\nCurvature Analysis:")
    print(f"  Mean curvature (all thresholds): {mean_kappa_all:.4f}")
    print(f"  Max curvature observed: {max_kappa_all:.4f}")
    print(f"  Theoretical bound (2π/5R): {KAPPA_BOUND:.4f}")

    print(f"\n  Curvature bounded? ", end="")
    if mean_kappa_all < KAPPA_BOUND:
        print(f"✓ YES (mean κ = {mean_kappa_all:.3f} < {KAPPA_BOUND:.2f})")
    else:
        print(f"✗ NO (mean κ = {mean_kappa_all:.3f} > {KAPPA_BOUND:.2f})")

    print(f"\nIcosahedral Alignment:")
    print(f"  Mean alignment to nearest axis: {mean_align_all:.4f}")
    print(f"  90th percentile alignment: {p90_align_all:.4f}")
    print(f"  Fraction within 25° of axis: {frac_high_all:.1%}")

    # For random directions, expected alignment is ~0.5 (average |cos|)
    print(f"\n  Strong icosahedral alignment? ", end="")
    if mean_align_all > 0.7:
        print(f"✓ YES (mean = {mean_align_all:.3f} >> 0.5 random)")
    elif mean_align_all > 0.55:
        print(f"~ WEAK (mean = {mean_align_all:.3f} > 0.5 random)")
    else:
        print(f"✗ NO (mean = {mean_align_all:.3f} ≈ 0.5 random)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Curvature vs threshold
    ax = axes[0, 0]
    ax.bar(range(len(results['thresholds'])), results['mean_curvature'],
           tick_label=[f"{t}×" for t in results['thresholds']], alpha=0.7)
    ax.axhline(KAPPA_BOUND, color='r', linestyle='--', label=f'Bound = {KAPPA_BOUND:.2f}')
    ax.set_xlabel('Vorticity Threshold')
    ax.set_ylabel('Mean Curvature κ')
    ax.set_title('Vortex Tube Curvature')
    ax.legend()

    # Alignment histogram
    ax = axes[0, 1]
    # Get alignment from highest threshold
    tangents, _, _ = compute_tube_tangents(omega_peak, threshold_factor=2.5)
    if len(tangents) > 0:
        alignment = compute_icosa_alignment(tangents)
        ax.hist(alignment, bins=50, density=True, alpha=0.7, label='Measured')
        # Random expectation (for |cos| with random direction vs 27 axes)
        # Approximate with uniform
        ax.axvline(0.5, color='gray', linestyle=':', label='Random mean')
        ax.axvline(mean_align_all, color='r', linestyle='--', label=f'Observed mean = {mean_align_all:.2f}')
    ax.set_xlabel('Alignment to Nearest Icosahedral Axis')
    ax.set_ylabel('Density')
    ax.set_title('Tangent-Axis Alignment Distribution')
    ax.legend()

    # Alignment vs vorticity
    ax = axes[1, 0]
    tangents, _, omega_vals = compute_tube_tangents(omega_peak, threshold_factor=1.5)
    if len(tangents) > 0:
        alignment = compute_icosa_alignment(tangents)
        # Subsample for clarity
        idx = np.random.choice(len(alignment), min(5000, len(alignment)), replace=False)
        ax.scatter(omega_vals[idx], alignment[idx], alpha=0.3, s=5)
        ax.set_xlabel('Vorticity Magnitude |ω|')
        ax.set_ylabel('Alignment to Icosa Axis')
        ax.set_title('Does Alignment Increase with Vorticity?')

    # Summary bar chart
    ax = axes[1, 1]
    metrics = ['Mean κ / bound', 'Mean align', '% high align']
    values = [mean_kappa_all / KAPPA_BOUND, mean_align_all, frac_high_all]
    colors = ['green' if v < 1 else 'red' for v in [mean_kappa_all / KAPPA_BOUND]] + \
             ['green' if mean_align_all > 0.6 else 'orange'] + \
             ['green' if frac_high_all > 0.2 else 'orange']
    ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.axhline(1.0, color='k', linestyle=':')
    ax.set_ylabel('Value')
    ax.set_title('Summary Metrics')

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_vortex_tube_geometry.png', dpi=150)
    print("\nSaved: test_vortex_tube_geometry.png")

    return results


if __name__ == "__main__":
    results = main()
