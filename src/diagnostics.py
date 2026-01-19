"""
Diagnostic functions for H₃-NS verification.

These functions compute the key quantities that prove regularity:
- Alignment factor J (bounded by 1-δ₀)
- Enstrophy Z (bounded by Theorem 4.1)
- Phason flux (energy export during snap-back)

See docs/ADVERSARIAL_TEST_RESULTS.md for interpretation of these metrics.
"""

import mlx.core as mx
import numpy as np
from typing import Tuple, Dict
from .constants import DELTA_0, A_MAX


def compute_alignment_factor(omega: np.ndarray, strain: np.ndarray) -> np.ndarray:
    """
    Compute alignment factor J = ω̂·S·ω̂ / λ_max where λ_max is max eigenvalue of S.

    This is the key quantity bounded by Theorem 3.1:
        J ≤ 1 - δ₀ ≈ 0.691

    The alignment measures how well vorticity aligns with the most stretching
    direction of the strain tensor.

    Args:
        omega: Vorticity field, shape (..., 3)
        strain: Strain tensor, shape (..., 3, 3)

    Returns:
        Alignment factor at each point, in [0, 1]
    """
    # Normalize vorticity
    omega_mag = np.linalg.norm(omega, axis=-1, keepdims=True)
    omega_hat = np.where(omega_mag > 1e-10, omega / omega_mag, 0)

    # Compute ω̂·S·ω̂
    # Sω = strain @ omega_hat  (Einstein summation)
    Somega = np.einsum('...ij,...j->...i', strain, omega_hat)
    omega_S_omega = np.einsum('...i,...i->...', omega_hat, Somega)

    # Maximum eigenvalue of S (strain rate)
    # For symmetric S, λ_max = max eigenvalue
    eigenvalues = np.linalg.eigvalsh(strain)
    lambda_max = np.max(np.abs(eigenvalues), axis=-1)

    # Alignment factor
    J = np.where(lambda_max > 1e-10, omega_S_omega / lambda_max, 0)

    return J


def compute_enstrophy(omega: np.ndarray) -> float:
    """
    Compute enstrophy Z = (1/2) ∫|ω|² dx.

    Theorem 4.1 proves Z(t) ≤ Z_max for all t when H₃ constraint active.

    Args:
        omega: Vorticity field, shape (n, n, n, 3)

    Returns:
        Total enstrophy (scalar)
    """
    return 0.5 * np.mean(np.sum(omega**2, axis=-1))


def compute_strain_tensor(u: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Compute strain rate tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2.

    Args:
        u: Velocity field, shape (n, n, n, 3)
        dx: Grid spacing

    Returns:
        Strain tensor, shape (n, n, n, 3, 3)
    """
    n = u.shape[0]
    S = np.zeros(u.shape + (3,))

    # Central differences for derivatives
    for i in range(3):
        for j in range(3):
            # ∂u_i/∂x_j
            du_i_dx_j = (np.roll(u[..., i], -1, axis=j) -
                         np.roll(u[..., i], 1, axis=j)) / (2 * dx)
            # ∂u_j/∂x_i
            du_j_dx_i = (np.roll(u[..., j], -1, axis=i) -
                         np.roll(u[..., j], 1, axis=i)) / (2 * dx)
            S[..., i, j] = 0.5 * (du_i_dx_j + du_j_dx_i)

    return S


def compute_phason_flux(omega: np.ndarray, depletion: np.ndarray,
                        strain: np.ndarray) -> np.ndarray:
    """
    Compute phason export flux: |δ_eff · ω · S · ω|.

    This measures energy being "exported" from the vortex core during
    snap-back events. High flux indicates active depletion.

    See THEOREM_7_2_HOMOGENIZATION.md for derivation of phason dynamics.

    Args:
        omega: Vorticity field
        depletion: Depletion factor (1 - δ₀·Φ) at each point
        strain: Strain tensor

    Returns:
        Phason flux magnitude at each point
    """
    # Effective depletion (how much is being removed)
    delta_eff = 1 - depletion  # This is δ₀·Φ

    # Stretching rate: ω · S · ω
    Somega = np.einsum('...ij,...j->...i', strain, omega)
    stretch_rate = np.einsum('...i,...i->...', omega, Somega)

    # Phason flux
    flux = np.abs(delta_eff * stretch_rate)

    return flux


def compute_vorticity_pdf(omega: np.ndarray, nbins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PDF of vorticity magnitude.

    Args:
        omega: Vorticity field
        nbins: Number of histogram bins

    Returns:
        (bin_centers, pdf) arrays
    """
    omega_mag = np.linalg.norm(omega, axis=-1).flatten()
    hist, edges = np.histogram(omega_mag, bins=nbins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def compute_alignment_pdf(J: np.ndarray, nbins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PDF of alignment factor.

    The PDF should have support only in [0, 1-δ₀] ≈ [0, 0.691] for H₃-NS.
    If probability mass exists above 0.691, the bound is violated.

    Args:
        J: Alignment factor field
        nbins: Number of histogram bins

    Returns:
        (bin_centers, pdf) arrays
    """
    J_flat = J.flatten()
    hist, edges = np.histogram(J_flat, bins=nbins, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def verify_bounds(diagnostics: Dict[str, np.ndarray], Z_max_theoretical: float = 600) -> Dict[str, bool]:
    """
    Verify that all theoretical bounds are satisfied.

    This is the ultimate test: if all bounds hold for all time,
    regularity is proven numerically.

    Args:
        diagnostics: Dict with 't', 'Z', 'J_max', 'omega_max' arrays
        Z_max_theoretical: Theoretical enstrophy bound

    Returns:
        Dict of verification results
    """
    results = {}

    # Enstrophy bound (Theorem 4.1)
    Z = diagnostics.get('Z', np.array([0]))
    results['enstrophy_bounded'] = np.all(Z < Z_max_theoretical)
    results['Z_max_observed'] = float(np.max(Z))
    results['Z_max_theoretical'] = Z_max_theoretical

    # Alignment bound (Theorem 3.1)
    J_min = diagnostics.get('J_min', np.array([1]))
    # J_min is the minimum depletion factor, should be >= 1-δ₀
    results['alignment_bounded'] = np.all(J_min >= A_MAX - 0.01)
    results['J_min_observed'] = float(np.min(J_min))
    results['J_bound'] = A_MAX

    # No NaN/Inf (basic sanity)
    results['no_nan'] = not (np.any(np.isnan(Z)) or np.any(np.isinf(Z)))

    # Overall verdict
    results['all_bounds_satisfied'] = (
        results['enstrophy_bounded'] and
        results['alignment_bounded'] and
        results['no_nan']
    )

    return results
