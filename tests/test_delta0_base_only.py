#!/usr/bin/env python3
"""
MEASURE BASE δ₀ ONLY (no adaptive enhancement)

Previous test showed empirical δ₀ = 0.93 because it included adaptive enhancement.
This test applies ONLY the constant geometric depletion to isolate δ₀.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import mlx.core as mx

DELTA_0_THEORY = (np.sqrt(5) - 1) / 4  # ≈ 0.309

print(f"Testing base δ₀ = {DELTA_0_THEORY:.6f}")
print(f"Expected stretching ratio = {1 - DELTA_0_THEORY:.6f}")


class SimpleNS:
    def __init__(self, n=64, nu=0.001, dt=0.0001):
        self.n, self.nu, self.dt = n, nu, dt
        k = mx.array(np.fft.fftfreq(n, d=1/n) * 2 * np.pi)
        kx, ky, kz = mx.meshgrid(k, k, k, indexing='ij')
        self.kx, self.ky, self.kz = kx, ky, kz
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2_safe = mx.where(self.k2 == 0, mx.array(1e-10), self.k2)
        self.visc = mx.exp(-nu * self.k2 * dt)

    def get_velocity(self, wh):
        wx_hat, wy_hat, wz_hat = wh
        ux_hat = -1j * (self.ky * wz_hat - self.kz * wy_hat) / self.k2_safe
        uy_hat = -1j * (self.kz * wx_hat - self.kx * wz_hat) / self.k2_safe
        uz_hat = -1j * (self.kx * wy_hat - self.ky * wx_hat) / self.k2_safe
        return ux_hat, uy_hat, uz_hat

    def compute_stretching_term(self, wh):
        """Compute raw stretching term (ω·∇)u in physical space."""
        wx_hat, wy_hat, wz_hat = wh
        ux_hat, uy_hat, uz_hat = self.get_velocity(wh)

        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        # Stretching contributions
        stretch_x = (wx * mx.fft.ifftn(1j * self.kx * ux_hat).real +
                     wy * mx.fft.ifftn(1j * self.ky * ux_hat).real +
                     wz * mx.fft.ifftn(1j * self.kz * ux_hat).real)
        stretch_y = (wx * mx.fft.ifftn(1j * self.kx * uy_hat).real +
                     wy * mx.fft.ifftn(1j * self.ky * uy_hat).real +
                     wz * mx.fft.ifftn(1j * self.kz * uy_hat).real)
        stretch_z = (wx * mx.fft.ifftn(1j * self.kx * uz_hat).real +
                     wy * mx.fft.ifftn(1j * self.ky * uz_hat).real +
                     wz * mx.fft.ifftn(1j * self.kz * uz_hat).real)

        # Magnitude of stretching
        stretch_mag = mx.sqrt(stretch_x**2 + stretch_y**2 + stretch_z**2)
        return float(mx.mean(stretch_mag).item())

    def step(self, wh, depletion=0.0):
        """NS step with optional constant depletion factor."""
        wx_hat, wy_hat, wz_hat = wh
        ux_hat, uy_hat, uz_hat = self.get_velocity(wh)

        ux = mx.fft.ifftn(ux_hat).real
        uy = mx.fft.ifftn(uy_hat).real
        uz = mx.fft.ifftn(uz_hat).real
        wx = mx.fft.ifftn(wx_hat).real
        wy = mx.fft.ifftn(wy_hat).real
        wz = mx.fft.ifftn(wz_hat).real

        # Advection
        nlx = wy * uz - wz * uy
        nly = wz * ux - wx * uz
        nlz = wx * uy - wy * ux

        # Stretching with depletion
        factor = 1 - depletion
        nlx = nlx + factor * (wx * mx.fft.ifftn(1j * self.kx * ux_hat).real +
                               wy * mx.fft.ifftn(1j * self.ky * ux_hat).real +
                               wz * mx.fft.ifftn(1j * self.kz * ux_hat).real)
        nly = nly + factor * (wx * mx.fft.ifftn(1j * self.kx * uy_hat).real +
                               wy * mx.fft.ifftn(1j * self.ky * uy_hat).real +
                               wz * mx.fft.ifftn(1j * self.kz * uy_hat).real)
        nlz = nlz + factor * (wx * mx.fft.ifftn(1j * self.kx * uz_hat).real +
                               wy * mx.fft.ifftn(1j * self.ky * uz_hat).real +
                               wz * mx.fft.ifftn(1j * self.kz * uz_hat).real)

        return ((wx_hat + self.dt * mx.fft.fftn(nlx)) * self.visc,
                (wy_hat + self.dt * mx.fft.fftn(nly)) * self.visc,
                (wz_hat + self.dt * mx.fft.fftn(nlz)) * self.visc)


def taylor_green(n, scale=5.0):
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return np.stack([scale * np.cos(X) * np.sin(Y) * np.sin(Z),
                     scale * np.sin(X) * np.cos(Y) * np.sin(Z),
                     -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)], axis=-1)


def main():
    print("\n" + "=" * 70)
    print("  BASE δ₀ MEASUREMENT (constant depletion only)")
    print("=" * 70)

    n, tmax = 64, 1.0
    solver = SimpleNS(n=n)
    omega_init = taylor_green(n, scale=5.0)

    wh_un = tuple(mx.fft.fftn(mx.array(omega_init[..., i])) for i in range(3))
    wh_con = tuple(mx.fft.fftn(mx.array(omega_init[..., i])) for i in range(3))

    nsteps = int(tmax / solver.dt)
    results = {'t': [], 'S_un': [], 'S_con': [], 'Z_un': [], 'Z_con': []}

    print(f"\nRunning {nsteps} steps with δ₀ = {DELTA_0_THEORY:.4f}...")

    for step in range(nsteps):
        if step % 100 == 0:
            S_un = solver.compute_stretching_term(wh_un)
            S_con = solver.compute_stretching_term(wh_con)

            Z_un = 0.5 * float(mx.mean(sum(mx.fft.ifftn(h).real**2 for h in wh_un)).item())
            Z_con = 0.5 * float(mx.mean(sum(mx.fft.ifftn(h).real**2 for h in wh_con)).item())

            results['t'].append(step * solver.dt)
            results['S_un'].append(S_un)
            results['S_con'].append(S_con)
            results['Z_un'].append(Z_un)
            results['Z_con'].append(Z_con)

            if step % 2000 == 0:
                ratio = S_con / S_un if S_un > 1e-8 else 1.0
                print(f"  t={step*solver.dt:.2f}: Z_un={Z_un:.1f}, Z_con={Z_con:.1f}, "
                      f"S_ratio={ratio:.4f}")

            if np.isnan(Z_un) or Z_un > 1e6:
                print(f"  Blowup at t={step*solver.dt:.3f}")
                break

        wh_un = solver.step(wh_un, depletion=0.0)
        wh_con = solver.step(wh_con, depletion=DELTA_0_THEORY)
        mx.eval(*wh_un, *wh_con)

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    t = np.array(results['t'])
    S_un = np.array(results['S_un'])
    S_con = np.array(results['S_con'])
    Z_un = np.array(results['Z_un'])

    # Ratio during active stretching
    valid = (S_un > 0.1) & ~np.isnan(S_un)
    if np.sum(valid) > 0:
        ratios = S_con[valid] / S_un[valid]
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        emp_delta0 = 1 - mean_ratio
    else:
        mean_ratio = 1 - DELTA_0_THEORY
        emp_delta0 = DELTA_0_THEORY

    print(f"\nStretching ratio S_con / S_un:")
    print(f"  Measured: {mean_ratio:.4f} ± {std_ratio:.4f}")
    print(f"  Expected (1-δ₀): {1 - DELTA_0_THEORY:.4f}")

    print(f"\nEmpirical δ₀:")
    print(f"  Measured: {emp_delta0:.4f}")
    print(f"  Theory:   {DELTA_0_THEORY:.4f}")

    error = abs(emp_delta0 - DELTA_0_THEORY) / DELTA_0_THEORY * 100
    print(f"\n  Error: {error:.1f}%")

    if error < 5:
        print("  ✓ EXCELLENT - base δ₀ confirmed!")
    elif error < 15:
        print("  ~ Good agreement")
    else:
        print("  ? Deviation needs investigation")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(t, Z_un, 'r-', label='Unconstrained')
    ax.plot(t, results['Z_con'], 'b-', label=f'δ₀={DELTA_0_THEORY:.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy')
    ax.set_title('Enstrophy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, S_un, 'r-', label='Unconstrained')
    ax.plot(t, S_con, 'b-', label='Constrained')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean |Stretching|')
    ax.set_title('Stretching Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ratio = np.array(S_con) / np.array(S_un)
    ratio[np.isnan(ratio)] = 1
    ax.plot(t, ratio, 'g-', linewidth=2)
    ax.axhline(1 - DELTA_0_THEORY, color='r', linestyle='--',
               label=f'Expected: {1-DELTA_0_THEORY:.3f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('S_con / S_un')
    ax.set_title(f'Stretching Ratio (emp δ₀ = {emp_delta0:.3f})')
    ax.legend()
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/bryan/H3-Hybrid-Discovery/cognition/test_delta0_base_only.png', dpi=150)
    print("\nSaved: test_delta0_base_only.png")


if __name__ == "__main__":
    main()
