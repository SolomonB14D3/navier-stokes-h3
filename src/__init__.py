# H3 Navier-Stokes Solver Package
"""
H3-regularized Navier-Stokes solver with geometric depletion.

The depletion constant δ₀ = (√5-1)/4 ≈ 0.309 bounds vortex stretching
via icosahedral geometry. See docs/THEOREM_5_2_FORMALIZATION.md for derivation.
"""

from .constants import DELTA_0, PHI, R_H3, OMEGA_CRIT
from .solver import H3NavierStokesSolver
from .diagnostics import compute_alignment_factor, compute_enstrophy

__version__ = "1.0.0"
__all__ = [
    "DELTA_0", "PHI", "R_H3", "OMEGA_CRIT",
    "H3NavierStokesSolver",
    "compute_alignment_factor", "compute_enstrophy"
]
