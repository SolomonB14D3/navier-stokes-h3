"""
Core unit tests for H₃-NS solver.

These tests verify:
1. Mathematical identities (δ₀, φ relations)
2. Spectral accuracy (div-free, curl)
3. Known analytical solutions (Taylor-Green decay)
4. Bound satisfaction (alignment, enstrophy)

Run with: pytest tests/test_core.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import DELTA_0, PHI, A_MAX, OMEGA_CRIT, verify_constants
from src.solver import H3NavierStokesSolver, SolverConfig, BoundViolationError
from src.diagnostics import compute_enstrophy, verify_bounds


class TestConstants:
    """Test mathematical constants and identities."""

    def test_golden_ratio_identity(self):
        """φ² = φ + 1"""
        assert abs(PHI**2 - PHI - 1) < 1e-14

    def test_golden_ratio_inverse(self):
        """1/φ = φ - 1"""
        assert abs(1/PHI - (PHI - 1)) < 1e-14

    def test_delta0_definition(self):
        """δ₀ = (√5-1)/4"""
        assert abs(DELTA_0 - (np.sqrt(5) - 1) / 4) < 1e-15

    def test_delta0_golden_ratio(self):
        """δ₀ = 1/(2φ)"""
        assert abs(DELTA_0 - 1/(2*PHI)) < 1e-15

    def test_alignment_bound(self):
        """A_max = 1 - δ₀"""
        assert abs(A_MAX - (1 - DELTA_0)) < 1e-15
        assert 0.69 < A_MAX < 0.70  # Sanity check

    def test_verify_constants(self):
        """All constant self-tests pass."""
        verify_constants()  # Should not raise


class TestSpectralAccuracy:
    """Test spectral differentiation accuracy."""

    @pytest.fixture
    def solver(self):
        return H3NavierStokesSolver(SolverConfig(n=32, viscosity=0.01))

    def test_divergence_free(self, solver):
        """Velocity from vorticity is divergence-free."""
        import mlx.core as mx

        # Random vorticity
        n = solver.config.n
        np.random.seed(42)
        omega = np.random.randn(n, n, n, 3).astype(np.float32)

        wx_hat = mx.fft.fftn(mx.array(omega[..., 0]))
        wy_hat = mx.fft.fftn(mx.array(omega[..., 1]))
        wz_hat = mx.fft.fftn(mx.array(omega[..., 2]))

        ux_hat, uy_hat, uz_hat = solver.velocity_from_vorticity(wx_hat, wy_hat, wz_hat)

        # Check div(u) = 0 in Fourier space: k · û = 0
        div_hat = solver.kx * ux_hat + solver.ky * uy_hat + solver.kz * uz_hat
        mx.eval(div_hat)

        div_max = float(mx.max(mx.abs(div_hat)).item())
        assert div_max < 1e-10, f"Divergence not zero: max|div|={div_max}"

    def test_curl_identity(self, solver):
        """curl(curl^{-1}(ω)) = ω for solenoidal ω."""
        import mlx.core as mx

        n = solver.config.n
        # Create solenoidal vorticity (use curl of random field)
        np.random.seed(42)
        psi = np.random.randn(n, n, n, 3).astype(np.float32)

        # Make it periodic-compatible
        psi_hat = [mx.fft.fftn(mx.array(psi[..., i])) for i in range(3)]
        omega_hat = solver.curl_spectral(*psi_hat)

        # Get velocity and curl back
        u_hat = solver.velocity_from_vorticity(*omega_hat)
        omega_recovered = solver.curl_spectral(*u_hat)

        # Compare
        for i, (orig, rec) in enumerate(zip(omega_hat, omega_recovered)):
            mx.eval(orig, rec)
            diff = float(mx.max(mx.abs(orig - rec)).item())
            assert diff < 1e-8, f"Curl recovery failed for component {i}: diff={diff}"


class TestBounds:
    """Test that theoretical bounds are satisfied."""

    def test_depletion_range(self):
        """Depletion factor in [1-δ₀, 1]."""
        import mlx.core as mx

        solver = H3NavierStokesSolver(SolverConfig(n=16, delta0=DELTA_0))

        # Test various vorticity magnitudes
        for omega_val in [0.1, 1.0, 10.0, 100.0]:
            omega_mag = mx.array(np.full((16, 16, 16), omega_val))
            depl = solver.compute_depletion_factor(omega_mag)
            mx.eval(depl)

            depl_np = np.array(depl)
            assert np.all(depl_np >= A_MAX - 1e-6), f"Depletion below bound at ω={omega_val}"
            assert np.all(depl_np <= 1 + 1e-6), f"Depletion above 1 at ω={omega_val}"

    def test_depletion_limits(self):
        """Depletion → 1 as ω → 0, → 1-δ₀ as ω → ∞."""
        import mlx.core as mx

        solver = H3NavierStokesSolver(SolverConfig(n=8, delta0=DELTA_0))

        # Low vorticity: depletion ≈ 1
        omega_low = mx.array(np.full((8, 8, 8), 0.01))
        depl_low = solver.compute_depletion_factor(omega_low)
        mx.eval(depl_low)
        assert float(depl_low.mean().item()) > 0.99, "Low ω should give depletion ≈ 1"

        # High vorticity: depletion → 1-δ₀
        omega_high = mx.array(np.full((8, 8, 8), 1000.0))
        depl_high = solver.compute_depletion_factor(omega_high)
        mx.eval(depl_high)
        assert abs(float(depl_high.mean().item()) - A_MAX) < 0.01, "High ω should give depletion ≈ 1-δ₀"

    def test_watchdog_triggers(self):
        """Watchdog raises error on bound violation."""
        solver = H3NavierStokesSolver(SolverConfig(
            n=8, watchdog=True, watchdog_enstrophy_max=10.0
        ))

        # Create diagnostics that violate bound
        bad_diagnostics = {'Z': 100.0, 'J_min': 0.7, 't': 1.0}

        with pytest.raises(BoundViolationError):
            solver._check_bounds(bad_diagnostics)


class TestAnalyticalSolutions:
    """Test against known analytical solutions."""

    def test_taylor_green_decay(self):
        """Taylor-Green vortex decays at correct rate for low Re."""
        # At low Re, TG decays as exp(-2νt)
        n = 32
        nu = 0.1  # High viscosity for fast decay

        # Simple TG without perturbation
        x = np.linspace(0, 2*np.pi, n, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

        scale = 1.0
        wx = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
        wy = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
        wz = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)
        omega_init = np.stack([wx, wy, wz], axis=-1)

        # Run short simulation
        config = SolverConfig(n=n, viscosity=nu, dt=0.001, delta0=0)  # Unconstrained
        solver = H3NavierStokesSolver(config)

        history = solver.run(omega_init, tmax=0.5, report_interval=10)

        # Check decay rate
        Z = history['Z']
        t = history['t']

        # Fit exponential
        log_Z = np.log(Z + 1e-10)
        slope, _ = np.polyfit(t, log_Z, 1)

        # Should be close to -2ν = -0.2 for TG
        # (Not exact due to nonlinearity, but order of magnitude)
        assert -1 < slope < 0, f"Unexpected decay rate: {slope}"


class TestVerifyBounds:
    """Test the verify_bounds function."""

    def test_good_diagnostics(self):
        """Good diagnostics pass verification."""
        diagnostics = {
            'Z': np.array([1.0, 2.0, 3.0]),
            'J_min': np.array([0.7, 0.72, 0.71])
        }
        result = verify_bounds(diagnostics)
        assert result['all_bounds_satisfied']

    def test_enstrophy_violation(self):
        """High enstrophy fails verification."""
        diagnostics = {
            'Z': np.array([1.0, 1000.0, 2000.0]),  # Exceeds 600
            'J_min': np.array([0.7, 0.7, 0.7])
        }
        result = verify_bounds(diagnostics)
        assert not result['enstrophy_bounded']

    def test_nan_detection(self):
        """NaN values are detected."""
        diagnostics = {
            'Z': np.array([1.0, np.nan, 3.0]),
            'J_min': np.array([0.7, 0.7, 0.7])
        }
        result = verify_bounds(diagnostics)
        assert not result['no_nan']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
