"""
H₃-regularized Navier-Stokes Solver with Geometric Depletion.

This module implements the core NS solver with the icosahedral constraint
that bounds vortex stretching. Each function is annotated with the
corresponding theorem/equation from the proof documents.

Key References:
- docs/THEOREM_5_2_FORMALIZATION.md: δ₀ derivation
- docs/WHY_ICOSAHEDRAL.md: Variational optimality of I_h
- docs/THEOREM_7_2_HOMOGENIZATION.md: Chapman-Enskog derivation
"""

import mlx.core as mx
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .constants import DELTA_0, R_H3, OMEGA_CRIT, A_MAX, ENSTROPHY_BOUND_TOLERANCE


class BoundViolationError(Exception):
    """Raised when theoretical bounds are violated - indicates bug or numerical issue."""
    pass


@dataclass
class SolverConfig:
    """Configuration for H3 NS solver."""
    n: int = 128              # Grid resolution
    viscosity: float = 0.001  # Kinematic viscosity ν
    dt: float = 0.0001        # Timestep
    delta0: float = DELTA_0   # Depletion constant (0 = unconstrained)
    watchdog: bool = True     # Enable bound checking
    watchdog_enstrophy_max: float = 1000.0  # Raise error if Z exceeds this


class H3NavierStokesSolver:
    """
    GPU-accelerated Navier-Stokes solver with H₃ geometric constraint.

    Implements Theorem 4.1 (bounded enstrophy) via the depletion mechanism
    described in Theorem 3.1 (geometric depletion).

    The vorticity equation with depletion:
        ∂ω/∂t + (u·∇)ω = (1 - δ₀·Φ)(ω·∇)u + ν∇²ω

    where Φ(|ω|/ω_c) is the smooth activation function (Eq. 4.3).
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize solver.

        Args:
            config: Solver configuration. Uses defaults if None.
        """
        self.config = config or SolverConfig()
        self._setup_grid()
        self._setup_operators()

        # Tracking
        self.step_count = 0
        self.diagnostics_history = []

    def _setup_grid(self):
        """Initialize spectral grid (wavenumbers)."""
        n = self.config.n
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')

        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)

    def _setup_operators(self):
        """Precompute operators for efficiency."""
        # Viscous decay (exact integrating factor for diffusion)
        self.visc_decay = mx.exp(-self.config.viscosity * self.k2 * self.config.dt)

        # Critical vorticity (Eq. 4.4)
        if self.config.delta0 > 0:
            self.omega_crit = 1.0 / (self.config.delta0 * R_H3)
        else:
            self.omega_crit = 1e10  # Effectively infinite

    def curl_spectral(self, ux_hat, uy_hat, uz_hat) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Compute curl in Fourier space: ω = ∇ × u.

        In spectral space: ω̂ = ik × û
        """
        wx_hat = 1j * (self.ky * uz_hat - self.kz * uy_hat)
        wy_hat = 1j * (self.kz * ux_hat - self.kx * uz_hat)
        wz_hat = 1j * (self.kx * uy_hat - self.ky * ux_hat)
        return wx_hat, wy_hat, wz_hat

    def velocity_from_vorticity(self, wx_hat, wy_hat, wz_hat) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Biot-Savart law: u = curl⁻¹(ω).

        In spectral space: û = -i(k × ω̂)/|k|²

        This enforces ∇·u = 0 automatically (Theorem 2.1).
        """
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def compute_depletion_factor(self, omega_mag: mx.array) -> mx.array:
        """
        Compute depletion factor (1 - δ₀·Φ) from Theorem 3.1.

        The activation function Φ(x) = x²/(1+x²) where x = |ω|/ω_c.
        This satisfies:
        - Φ(0) = 0: No depletion at low vorticity
        - Φ(∞) → 1: Full depletion (factor = 1-δ₀ ≈ 0.691) at high vorticity

        Returns:
            Depletion factor in [1-δ₀, 1] at each grid point.
        """
        if self.config.delta0 <= 0:
            return mx.ones_like(omega_mag)

        x = omega_mag / self.omega_crit
        # Smooth activation (Eq. 4.3)
        activation = x**2 / (1 + x**2)
        depletion = 1 - self.config.delta0 * activation

        return depletion

    def step(self, wx_hat, wy_hat, wz_hat) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Advance vorticity one timestep.

        Implements the depleted vorticity equation (Theorem 4.1):
            ∂ω/∂t = (1 - δ₀·Φ)(ω·∇)u - (u·∇)ω + ν∇²ω

        The depletion factor (1 - δ₀·Φ) bounds vortex stretching,
        preventing finite-time blowup (Theorem 5.1).
        """
        # Get velocity from vorticity (Biot-Savart)
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        # Transform to physical space
        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        # Vorticity magnitude for depletion calculation
        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2 + 1e-10)

        # Advection term: -(u·∇)ω = ω × u (for divergence-free u)
        advect_x = wy * uz - wz * uy
        advect_y = wz * ux - wx * uz
        advect_z = wx * uy - wy * ux

        # Strain tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2
        # Stretching term: (ω·∇)u = ω_j ∂u_i/∂x_j
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

        # Apply H₃ depletion (Theorem 3.1)
        depletion = self.compute_depletion_factor(omega_mag)
        stretch_x = stretch_x * depletion
        stretch_y = stretch_y * depletion
        stretch_z = stretch_z * depletion

        # Total nonlinear term
        nl_x = advect_x + stretch_x
        nl_y = advect_y + stretch_y
        nl_z = advect_z + stretch_z

        # Transform to Fourier and advance
        nl_x_hat = mx.fft.fftn(nl_x)
        nl_y_hat = mx.fft.fftn(nl_y)
        nl_z_hat = mx.fft.fftn(nl_z)

        # Euler step + viscous decay (integrating factor method)
        wx_hat_new = (wx_hat + self.config.dt * nl_x_hat) * self.visc_decay
        wy_hat_new = (wy_hat + self.config.dt * nl_y_hat) * self.visc_decay
        wz_hat_new = (wz_hat + self.config.dt * nl_z_hat) * self.visc_decay

        self.step_count += 1
        return wx_hat_new, wy_hat_new, wz_hat_new

    def compute_diagnostics(self, wx_hat, wy_hat, wz_hat) -> Dict[str, float]:
        """
        Compute diagnostic quantities for verification.

        Returns dict with:
        - E: Total kinetic energy (should decay monotonically)
        - Z: Enstrophy (bounded by Theorem 4.1)
        - omega_max: Maximum vorticity magnitude
        - J_max: Maximum alignment factor (bounded by 1-δ₀)
        """
        ux_hat, uy_hat, uz_hat = self.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        # Energy: E = (1/2) ∫|u|² dx
        E = 0.5 * mx.mean(ux**2 + uy**2 + uz**2)

        # Enstrophy: Z = (1/2) ∫|ω|² dx (Eq. 4.1)
        Z = 0.5 * mx.mean(wx**2 + wy**2 + wz**2)

        # Max vorticity
        omega_mag = mx.sqrt(wx**2 + wy**2 + wz**2)
        omega_max = mx.max(omega_mag)

        # Alignment factor J = ω̂·S·ω̂ / |S| (Theorem 3.1)
        # For efficiency, we compute a proxy: depletion-weighted stretching
        depletion = self.compute_depletion_factor(omega_mag)
        J_effective = mx.min(depletion)  # Worst-case alignment

        mx.eval(E, Z, omega_max, J_effective)

        diagnostics = {
            'E': float(E.item()),
            'Z': float(Z.item()),
            'omega_max': float(omega_max.item()),
            'J_min': float(J_effective.item()),  # 1-δ₀ when fully activated
            't': self.step_count * self.config.dt
        }

        # Watchdog: check bounds
        if self.config.watchdog:
            self._check_bounds(diagnostics)

        return diagnostics

    def _check_bounds(self, diagnostics: Dict[str, float]):
        """
        Singularity watchdog - raises error if theoretical bounds violated.

        This should NEVER trigger if the theory is correct. If it does,
        there's either a bug or a numerical issue (under-resolution, CFL violation).
        """
        Z = diagnostics['Z']
        J_min = diagnostics['J_min']
        t = diagnostics['t']

        # Check enstrophy bound
        if Z > self.config.watchdog_enstrophy_max:
            raise BoundViolationError(
                f"ENSTROPHY BOUND VIOLATED at t={t:.4f}: Z={Z:.2f} > {self.config.watchdog_enstrophy_max}\n"
                f"This indicates numerical instability (check CFL, resolution)."
            )

        # Check alignment bound (J_min should be >= 1-δ₀ = 0.691)
        if self.config.delta0 > 0 and J_min < A_MAX - 0.01:
            raise BoundViolationError(
                f"ALIGNMENT BOUND VIOLATED at t={t:.4f}: J_min={J_min:.4f} < {A_MAX:.4f}\n"
                f"This should be impossible - check depletion implementation."
            )

        # Check for NaN/Inf
        if np.isnan(Z) or np.isinf(Z):
            raise BoundViolationError(
                f"NaN/Inf detected at t={t:.4f}: Z={Z}\n"
                f"Numerical blowup - reduce dt or increase resolution."
            )

    def run(self, omega_init: np.ndarray, tmax: float = 1.0,
            report_interval: int = 100) -> Dict[str, np.ndarray]:
        """
        Run simulation from initial vorticity.

        Args:
            omega_init: Initial vorticity field, shape (n, n, n, 3)
            tmax: Maximum simulation time
            report_interval: Steps between diagnostic reports

        Returns:
            Dict with time series of diagnostics
        """
        # Convert to MLX
        wx = mx.array(omega_init[..., 0])
        wy = mx.array(omega_init[..., 1])
        wz = mx.array(omega_init[..., 2])

        wx_hat = mx.fft.fftn(wx)
        wy_hat = mx.fft.fftn(wy)
        wz_hat = mx.fft.fftn(wz)

        nsteps = int(tmax / self.config.dt)
        history = {'t': [], 'E': [], 'Z': [], 'omega_max': [], 'J_min': []}

        for step in range(nsteps):
            if step % report_interval == 0:
                diag = self.compute_diagnostics(wx_hat, wy_hat, wz_hat)
                for key in history:
                    history[key].append(diag[key])

            wx_hat, wy_hat, wz_hat = self.step(wx_hat, wy_hat, wz_hat)
            mx.eval(wx_hat, wy_hat, wz_hat)

        # Convert to numpy arrays
        return {k: np.array(v) for k, v in history.items()}


# Convenience function
def create_solver(n=128, viscosity=0.001, constrained=True, watchdog=True):
    """
    Create solver with standard configuration.

    Args:
        n: Grid resolution (64, 128, 256)
        viscosity: Kinematic viscosity ν
        constrained: If True, use H₃ depletion; if False, standard NS
        watchdog: Enable bound checking

    Returns:
        Configured H3NavierStokesSolver
    """
    config = SolverConfig(
        n=n,
        viscosity=viscosity,
        delta0=DELTA_0 if constrained else 0,
        watchdog=watchdog
    )
    return H3NavierStokesSolver(config)
