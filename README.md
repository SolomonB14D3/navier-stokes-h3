# Navier-Stokes H₃ Geometric Depletion: Investigation

An investigation into whether **icosahedral (H₃) geometric constraints** on vortex stretching can bound enstrophy in Navier-Stokes equations.

## Status: REVISED (January 2026)

Rigorous analysis shows the H₃ depletion mechanism **cannot prove regularity** of the standard Navier-Stokes equations. The fundamental obstruction is mathematical: a bounded multiplicative reduction of the stretching term cannot change the supercritical Z^(3/2) exponent. See [`analytical_proof_attempt.md`](analytical_proof_attempt.md) for the full analysis.

## The Mechanism

The **icosahedral depletion constant**:

$$\delta_0 = \frac{\sqrt{5}-1}{4} = \frac{1}{2\varphi} \approx 0.309$$

reduces vortex stretching by 30.9% when vorticity exceeds a critical threshold.

## Why It Cannot Prove NS Regularity

The enstrophy evolution for 3D incompressible NS:

$$\frac{dZ}{dt} \leq C \cdot Z^{3/2} - \nu \lambda_1 Z$$

With H₃ depletion, the stretching coefficient is reduced:

$$\frac{dZ}{dt} \leq (1-\delta_0) \cdot C \cdot Z^{3/2} - \nu \lambda_1 Z$$

**The problem**: The ODE $y' = cy^{3/2} - ay$ admits finite-time blowup for large initial data regardless of the value of $c > 0$. A constant reduction of the coefficient **cannot change the exponent**. The supercritical nature of NS is in the 3/2 power, not the coefficient.

### All Approaches Fail

| Approach | Why It Fails |
|----------|--------------|
| Constant factor (1-δ₀) | Z^(3/2) exponent preserved — still supercritical |
| Nonlinear activation Φ(x) = x²/(1+x²) | Saturates at (1-δ₀) for large |ω| — no help |
| Constantin-Fefferman direction | No mechanism constrains generic vorticity directions |
| Modified PDE (H₃-NS) | Different equations — regularity of H₃-NS ≠ regularity of NS |
| Numerical validation | Spectral solver inherently stable — can't blow up regardless |

### What Would Actually Be Needed

To prove NS regularity, one would need the stretching integral to grow **strictly slower than Z^(3/2)**. Possible routes (none delivered by H₃):

1. A new functional inequality with subcritical exponent
2. A structural result preventing persistent stretching-vorticity coincidence
3. A new controlled quantity (weighted enstrophy, curvature-based, etc.)

---

## What Remains Valid

- δ₀ = 1/(2φ) = 0.309 matches measured alignment depletion in simulations
- The H₃-NS modified PDE is a legitimate regularization for computational use
- Vorticity-strain alignment IS observed to be sub-maximal in real flows
- The icosahedral geometry connection is aesthetically interesting
- The code correctly implements a spectral NS solver with depletion

## Numerical Results (Inconclusive)

### Adversarial Tests

| Resolution | Z_max | vs Bound (547) | Overshoot |
|-----------|-------|----------------|-----------|
| n=64 | 607.08 | 111.0% | 11.0% |
| n=128 | 598.44 | 109.4% | 9.4% |
| n=256 | 597.56 | 109.2% | 9.2% |

Overshoot convergence stalls at ~9.2% — the bound is not satisfied even with depletion active.

### Control Experiment (Critical Finding)

The spectral solver with exponential integrating factor exp(-ν|k|²dt) is **inherently stable**. Control runs with δ₀ = 0 (standard NS, no depletion) also do not blow up. This means:

- The **numerics** prevent blowup, not the physics
- All "bounded enstrophy" results are artifacts of solver stability
- The mechanism cannot be validated by this type of solver

### What Would Be Needed for Proper Numerical Validation

To numerically test whether depletion prevents blowup, one would need:
- A solver that CAN blow up (e.g., inviscid Euler, or NS with explicit time-stepping and no integrating factor)
- Show that δ₀ > 0 stays bounded while δ₀ = 0 blows up
- This has NOT been demonstrated

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/SolomonB14D3/navier-stokes-h3.git
cd navier-stokes-h3
pip install -r requirements.txt

# Run main solver (~10 min on M3)
python scripts/verify_ns_gpu.py

# Run adversarial test
python scripts/adversarial_highres_optimized.py

# Run control experiment (no depletion)
python scripts/control_no_depletion.py
```

## Directory Structure

```
├── src/                            # Core solver package
│   ├── constants.py                # δ₀, φ constants
│   ├── solver.py                   # H3NavierStokesSolver (MLX)
│   └── diagnostics.py              # Verification functions
├── docs/
│   ├── NAVIER_STOKES_H3_PROOF.tex  # Original proof attempt (claims not sustained)
│   └── ...
├── scripts/
│   ├── verify_ns_gpu.py            # Main solver
│   ├── adversarial_highres_optimized.py  # Adversarial tests
│   ├── control_no_depletion.py     # Control experiment (δ₀=0)
│   └── ...
├── analytical_proof_attempt.md     # Why the mechanism cannot work (rigorous)
├── tests/                          # Unit tests
├── figures/                        # Visualization outputs
├── results/                        # JSON test data
└── requirements.txt                # Dependencies
```

## The Modified PDE (H₃-NS)

What IS proven is that this code correctly solves a **modified** vorticity equation:

$$\partial_t \omega + (\mathbf{u}\cdot\nabla)\omega = (1 - \delta_0 \Phi(|\omega|/\omega_c)) \cdot (\omega\cdot\nabla)\mathbf{u} + \nu\Delta\omega$$

This is a different PDE from standard Navier-Stokes. Its solutions are bounded in our solver, but this is likely due to solver stability rather than the depletion mechanism.

## References

- Constantin & Fefferman (1993): Vorticity direction criterion
- Grujić (2009): Geometric depletion localization
- Beale, Kato & Majda (1984): Blowup criterion

## License

MIT License
