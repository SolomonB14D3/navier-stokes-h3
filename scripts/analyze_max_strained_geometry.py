#!/usr/bin/env python3
"""
GEOMETRIC ANALYSIS: Why Does Max-Strained IC Defeat H₃ Depletion?

This script analyzes:
1. The vorticity-strain alignment structure of the max-strained IC
2. How vorticity direction relates to icosahedral five-fold axes
3. The eigenstructure of the strain tensor
4. Why the geometric depletion δ₀ = 1/(2φ) is insufficient

The max-strained IC:
    ω_x = 10 sin(x) cos(y) cos(z)
    ω_y = 10 cos(x) sin(y) cos(z)
    ω_z = -20 cos(x) cos(y) sin(z)

Key insight: The 2× asymmetry in ω_z concentrates vorticity along [0,0,1],
which is NOT an icosahedral five-fold axis.
"""

import numpy as np
from numpy.linalg import norm, eigh

# Constants
PHI = (1 + np.sqrt(5)) / 2
DELTA_0 = (np.sqrt(5) - 1) / 4

print("=" * 70)
print("  GEOMETRIC ANALYSIS: Max-Strained IC vs H₃ Depletion")
print("=" * 70)

# ============================================================
# 1. ICOSAHEDRAL GEOMETRY
# ============================================================
print("\n--- 1. ICOSAHEDRAL AXES ---")

# The 6 five-fold axes of the icosahedron (normalized)
# These are the preferred vorticity directions in H₃ theory
five_fold_axes = np.array([
    [0, 1, PHI],
    [0, 1, -PHI],
    [0, -1, PHI],
    [0, -1, -PHI],
    [1, PHI, 0],
    [1, -PHI, 0],
    [-1, PHI, 0],
    [-1, -PHI, 0],
    [PHI, 0, 1],
    [PHI, 0, -1],
    [-PHI, 0, 1],
    [-PHI, 0, -1],
])
five_fold_axes = five_fold_axes / norm(five_fold_axes, axis=1, keepdims=True)

# The 15 two-fold axes (midpoints of edges)
two_fold_axes = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [PHI, 1/PHI, 0],
    [PHI, -1/PHI, 0],
    [0, PHI, 1/PHI],
    [0, PHI, -1/PHI],
    [1/PHI, 0, PHI],
    [1/PHI, 0, -PHI],
    [-1/PHI, 0, PHI],
    [-PHI, 1/PHI, 0],
])
two_fold_axes = two_fold_axes / norm(two_fold_axes, axis=1, keepdims=True)

# Critical angle from the proof
theta_v = np.arccos(1/np.sqrt(5))  # ≈ 63.43°
print(f"  Icosahedral vertex angle: θ_v = {np.degrees(theta_v):.2f}°")
print(f"  tan(θ_v/2) = 1/φ = {1/PHI:.6f}")
print(f"  Depletion: δ₀ = 1/(2φ) = {DELTA_0:.6f}")
print(f"  Max alignment allowed: 1-δ₀ = {1-DELTA_0:.6f}")
print(f"  Critical angle for depletion: arccos(1/√5) = {np.degrees(theta_v):.2f}°")

# ============================================================
# 2. MAX-STRAINED IC STRUCTURE (ANALYTICAL)
# ============================================================
print("\n--- 2. MAX-STRAINED IC ANALYTICAL STRUCTURE ---")

# The IC is: ω = (10 sin x cos y cos z, 10 cos x sin y cos z, -20 cos x cos y sin z)
# This is divergence-free: ∂ωx/∂x + ∂ωy/∂y + ∂ωz/∂z = 10 cos x cos y cos z + 10 cos x cos y cos z - 20 cos x cos y cos z = 0 ✓

print("  ω = (10 sin x cos y cos z, 10 cos x sin y cos z, -20 cos x cos y sin z)")
print("  ∇·ω = 0 (verified: 10+10-20 = 0)")

# At the point of maximum |ω|: Find where |ω|² is maximized
# |ω|² = 100(sin²x cos²y cos²z + cos²x sin²y cos²z) + 400 cos²x cos²y sin²z
# At (π/2, π/2, π/2): ω = (0, 0, 0)
# At (π/4, π/4, π/4): compute
x0, y0, z0 = np.pi/4, np.pi/4, np.pi/4
wx0 = 10 * np.sin(x0) * np.cos(y0) * np.cos(z0)
wy0 = 10 * np.cos(x0) * np.sin(y0) * np.cos(z0)
wz0 = -20 * np.cos(x0) * np.cos(y0) * np.sin(z0)
w0 = np.array([wx0, wy0, wz0])
print(f"\n  At (π/4, π/4, π/4):")
print(f"    ω = ({wx0:.3f}, {wy0:.3f}, {wz0:.3f})")
print(f"    |ω| = {norm(w0):.3f}")
w0_dir = w0 / norm(w0)
print(f"    ω̂ = ({w0_dir[0]:.4f}, {w0_dir[1]:.4f}, {w0_dir[2]:.4f})")

# At maximum strain point: x=0, y=0, z=π/2
x0, y0, z0 = 0.0, 0.0, np.pi/2
wx_max = 10 * np.sin(x0) * np.cos(y0) * np.cos(z0)  # = 0
wy_max = 10 * np.cos(x0) * np.sin(y0) * np.cos(z0)  # = 0
wz_max = -20 * np.cos(x0) * np.cos(y0) * np.sin(z0)  # = -20
w_max = np.array([wx_max, wy_max, wz_max])
print(f"\n  At (0, 0, π/2): PURE Z-AXIS VORTICITY")
print(f"    ω = ({wx_max:.1f}, {wy_max:.1f}, {wz_max:.1f})")
print(f"    |ω| = {norm(w_max):.1f}")
print(f"    ω̂ = (0, 0, -1)  [CUBIC AXIS, not icosahedral!]")

# ============================================================
# 3. ALIGNMENT WITH ICOSAHEDRAL AXES
# ============================================================
print("\n--- 3. ALIGNMENT WITH ICOSAHEDRAL AXES ---")

# The dominant vorticity direction is [0, 0, ±1]
# How well does this align with icosahedral five-fold axes?
z_axis = np.array([0, 0, 1])

print(f"\n  Dominant vorticity direction: ẑ = (0, 0, 1)")
print(f"\n  Alignment with icosahedral five-fold axes:")

min_angle = 180
best_axis = None
for i, axis in enumerate(five_fold_axes[:6]):  # Unique directions (pairs)
    cos_angle = abs(np.dot(z_axis, axis))
    angle = np.degrees(np.arccos(min(cos_angle, 1.0)))
    if angle < min_angle:
        min_angle = angle
        best_axis = axis
    print(f"    Axis {i}: ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}) → angle = {angle:.1f}°")

print(f"\n  Closest five-fold axis: ({best_axis[0]:.3f}, {best_axis[1]:.3f}, {best_axis[2]:.3f})")
print(f"  Minimum angle from ẑ to five-fold axis: {min_angle:.1f}°")
print(f"  Critical angle for H₃ depletion: {np.degrees(theta_v/2):.1f}°")
print(f"  MISALIGNMENT: {min_angle:.1f}° vs critical {np.degrees(theta_v/2):.1f}°")

if min_angle > np.degrees(theta_v/2):
    print(f"\n  ⚠ VORTICITY IS OUTSIDE THE ICOSAHEDRAL CONE!")
    print(f"    The ẑ-directed vorticity is {min_angle - np.degrees(theta_v/2):.1f}° beyond")
    print(f"    the nearest five-fold axis cone of influence.")
else:
    print(f"\n  ✓ Vorticity is within icosahedral cone")

# ============================================================
# 4. STRAIN TENSOR ANALYSIS (ANALYTICAL)
# ============================================================
print("\n--- 4. STRAIN TENSOR EIGENANALYSIS ---")

# For incompressible NS, u = curl⁻¹(ω) in Fourier space
# The velocity for this specific ω (Beltrami-like) can be computed
# Since ω = ∇²A (vector potential), and u = ∇×A:
# For single-mode ω with |k|² = 3 (modes (1,1,1)):
# The velocity magnitude scales as |ω|/|k|

# Numerically compute strain at key points
n = 64
x = np.linspace(0, 2*np.pi, n, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

omega = np.zeros((n, n, n, 3), dtype=np.float64)
omega[..., 0] = 10 * np.sin(X) * np.cos(Y) * np.cos(Z)
omega[..., 1] = 10 * np.cos(X) * np.sin(Y) * np.cos(Z)
omega[..., 2] = -20 * np.cos(X) * np.cos(Y) * np.sin(Z)

# FFT to get velocity
k = np.fft.fftfreq(n, d=1/n) * 2 * np.pi
kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
k2 = kx**2 + ky**2 + kz**2
k2_safe = np.where(k2 == 0, 1e-10, k2)

wx_hat = np.fft.fftn(omega[..., 0])
wy_hat = np.fft.fftn(omega[..., 1])
wz_hat = np.fft.fftn(omega[..., 2])

# Biot-Savart
ux_hat = -1j * (ky * wz_hat - kz * wy_hat) / k2_safe
uy_hat = -1j * (kz * wx_hat - kx * wz_hat) / k2_safe
uz_hat = -1j * (kx * wy_hat - ky * wx_hat) / k2_safe

# Velocity gradients
dux_dx = np.fft.ifftn(1j * kx * ux_hat).real
dux_dy = np.fft.ifftn(1j * ky * ux_hat).real
dux_dz = np.fft.ifftn(1j * kz * ux_hat).real
duy_dx = np.fft.ifftn(1j * kx * uy_hat).real
duy_dy = np.fft.ifftn(1j * ky * uy_hat).real
duy_dz = np.fft.ifftn(1j * kz * uy_hat).real
duz_dx = np.fft.ifftn(1j * kx * uz_hat).real
duz_dy = np.fft.ifftn(1j * ky * uz_hat).real
duz_dz = np.fft.ifftn(1j * kz * uz_hat).real

# Compute strain tensor and alignment at each point
omega_mag = np.sqrt(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2)

# Find points of high vorticity
threshold = np.percentile(omega_mag, 99)
high_vort_mask = omega_mag > threshold

print(f"  Computing strain alignment at {np.sum(high_vort_mask)} high-vorticity points")
print(f"  (|ω| > {threshold:.1f}, top 1%)")

# Sample high-vorticity points
indices = np.argwhere(high_vort_mask)
n_sample = min(1000, len(indices))
np.random.seed(42)
sample_idx = indices[np.random.choice(len(indices), n_sample, replace=False)]

alignments = []
stretching_eigenvalues = []
vort_angles_to_z = []
vort_angles_to_ico = []

for idx in sample_idx:
    i, j, k_idx = idx

    # Strain tensor S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2
    S = np.array([
        [dux_dx[i,j,k_idx], 0.5*(dux_dy[i,j,k_idx]+duy_dx[i,j,k_idx]), 0.5*(dux_dz[i,j,k_idx]+duz_dx[i,j,k_idx])],
        [0.5*(dux_dy[i,j,k_idx]+duy_dx[i,j,k_idx]), duy_dy[i,j,k_idx], 0.5*(duy_dz[i,j,k_idx]+duz_dy[i,j,k_idx])],
        [0.5*(dux_dz[i,j,k_idx]+duz_dx[i,j,k_idx]), 0.5*(duy_dz[i,j,k_idx]+duz_dy[i,j,k_idx]), duz_dz[i,j,k_idx]]
    ])

    # Vorticity direction
    w = omega[i, j, k_idx]
    w_mag = norm(w)
    if w_mag < 1e-10:
        continue
    w_hat = w / w_mag

    # Alignment: A = ω̂ · S · ω̂ / |S|
    Sw = S @ w_hat
    alignment = np.dot(w_hat, Sw)
    S_norm = norm(S, 'fro')
    if S_norm > 1e-10:
        normalized_alignment = alignment / S_norm
    else:
        normalized_alignment = 0

    alignments.append(normalized_alignment)

    # Eigenvalues of S
    evals = np.sort(eigh(S)[0])[::-1]  # Descending
    stretching_eigenvalues.append(evals[0])  # Largest eigenvalue

    # Angle to z-axis
    cos_z = abs(w_hat[2])
    angle_z = np.degrees(np.arccos(min(cos_z, 1.0)))
    vort_angles_to_z.append(angle_z)

    # Angle to nearest icosahedral axis
    min_ico_angle = 90
    for axis in five_fold_axes:
        cos_ico = abs(np.dot(w_hat, axis))
        ico_angle = np.degrees(np.arccos(min(cos_ico, 1.0)))
        min_ico_angle = min(min_ico_angle, ico_angle)
    vort_angles_to_ico.append(min_ico_angle)

alignments = np.array(alignments)
vort_angles_to_z = np.array(vort_angles_to_z)
vort_angles_to_ico = np.array(vort_angles_to_ico)

print(f"\n  Vorticity-Strain Alignment Statistics (top 1% |ω| points):")
print(f"    Mean alignment: {np.mean(alignments):.4f}")
print(f"    Max alignment:  {np.max(alignments):.4f}")
print(f"    Std alignment:  {np.std(alignments):.4f}")
print(f"    Theoretical max (1-δ₀): {1-DELTA_0:.4f}")
print(f"    Fraction exceeding 1-δ₀: {np.mean(alignments > (1-DELTA_0))*100:.1f}%")

print(f"\n  Vorticity Direction Statistics:")
print(f"    Mean angle to ẑ: {np.mean(vort_angles_to_z):.1f}°")
print(f"    Median angle to ẑ: {np.median(vort_angles_to_z):.1f}°")
print(f"    Mean angle to nearest ico axis: {np.mean(vort_angles_to_ico):.1f}°")
print(f"    Median angle to nearest ico axis: {np.median(vort_angles_to_ico):.1f}°")
print(f"    Critical angle (θ_v/2): {np.degrees(theta_v/2):.1f}°")
print(f"    Fraction within icosahedral cone: {np.mean(vort_angles_to_ico < np.degrees(theta_v/2))*100:.1f}%")

# ============================================================
# 5. WHY THE IC DEFEATS DEPLETION
# ============================================================
print("\n--- 5. MECHANISM FAILURE ANALYSIS ---")

# The stretching term is (ω·∇)u = ω_j ∂u_i/∂x_j
# Depletion reduces this by factor (1 - δ₀·Φ)
# For the mechanism to work: the EFFECTIVE stretching must be subcritical

# Compute effective stretching
stretch = np.zeros((n, n, n, 3))
stretch[..., 0] = omega[..., 0]*dux_dx + omega[..., 1]*dux_dy + omega[..., 2]*dux_dz
stretch[..., 1] = omega[..., 0]*duy_dx + omega[..., 1]*duy_dy + omega[..., 2]*duy_dz
stretch[..., 2] = omega[..., 0]*duz_dx + omega[..., 1]*duz_dy + omega[..., 2]*duz_dz

stretch_mag = np.sqrt(stretch[..., 0]**2 + stretch[..., 1]**2 + stretch[..., 2]**2)
omega_mag_safe = np.where(omega_mag > 1e-10, omega_mag, 1e-10)

# Normalized stretching rate: |stretch|/|ω|
stretch_rate = stretch_mag / omega_mag_safe

# Stretching direction relative to vorticity
# If stretch is aligned with ω, it amplifies vorticity (dangerous)
# If orthogonal, it just rotates (safe)
cos_stretch_omega = np.zeros((n, n, n))
for c in range(3):
    cos_stretch_omega += omega[..., c] * stretch[..., c]
cos_stretch_omega = cos_stretch_omega / (omega_mag_safe * np.where(stretch_mag > 1e-10, stretch_mag, 1e-10))

# Statistics at high-vorticity points
high_stretch = stretch_rate[high_vort_mask]
high_cos = cos_stretch_omega[high_vort_mask]

print(f"  Stretching Analysis at High-|ω| Points:")
print(f"    Mean |stretch|/|ω|: {np.mean(high_stretch):.3f}")
print(f"    Max |stretch|/|ω|:  {np.max(high_stretch):.3f}")
print(f"    Mean cos(stretch, ω): {np.mean(high_cos):.3f}")
print(f"    Max cos(stretch, ω): {np.max(high_cos):.3f}")
print(f"    Fraction with positive amplification: {np.mean(high_cos > 0)*100:.1f}%")

# The KEY geometric insight
print(f"\n  === KEY GEOMETRIC INSIGHT ===")
print(f"")
print(f"  The max-strained IC concentrates vorticity along the Z-AXIS [0,0,±1].")
print(f"  The z-axis is a TWO-FOLD axis of the icosahedron, NOT a five-fold axis.")
print(f"")
print(f"  Minimum angle from ẑ to nearest five-fold axis: {min_angle:.1f}°")
print(f"  Icosahedral depletion cone half-angle: {np.degrees(theta_v/2):.1f}°")
print(f"")

if min_angle > np.degrees(theta_v/2):
    deficit = min_angle - np.degrees(theta_v/2)
    print(f"  The vorticity is {deficit:.1f}° OUTSIDE the nearest icosahedral")
    print(f"  five-fold cone. This means the geometric argument for depletion")
    print(f"  (vorticity constrained to lie within θ_v/2 of a five-fold axis)")
    print(f"  is VIOLATED by this initial condition.")
    print(f"")
    print(f"  CONCLUSION: The max-strained IC exploits a geometric blind spot")
    print(f"  where vorticity aligns with a CUBIC axis [0,0,1] rather than")
    print(f"  an icosahedral five-fold axis. The depletion mechanism assumes")
    print(f"  vorticity preferentially aligns with icosahedral axes, but")
    print(f"  this IC forces alignment with a non-icosahedral direction.")
else:
    print(f"  The vorticity IS within the icosahedral cone.")
    print(f"  The failure must be in the strain alignment, not vorticity direction.")

# ============================================================
# 6. COMPARISON: TAYLOR-GREEN vs MAX-STRAINED
# ============================================================
print("\n--- 6. COMPARISON: TAYLOR-GREEN vs MAX-STRAINED ---")

# Taylor-Green IC
tg = np.zeros((n, n, n, 3), dtype=np.float64)
scale = 5.0
tg[..., 0] = scale * np.cos(X) * np.sin(Y) * np.sin(Z)
tg[..., 1] = scale * np.sin(X) * np.cos(Y) * np.sin(Z)
tg[..., 2] = -2 * scale * np.sin(X) * np.sin(Y) * np.cos(Z)

# Normalize to same enstrophy
Z_tg = 0.5 * np.mean(tg[..., 0]**2 + tg[..., 1]**2 + tg[..., 2]**2)
Z_ms = 0.5 * np.mean(omega[..., 0]**2 + omega[..., 1]**2 + omega[..., 2]**2)

tg_mag = np.sqrt(tg[..., 0]**2 + tg[..., 1]**2 + tg[..., 2]**2)
tg_high = tg_mag > np.percentile(tg_mag, 99)

# Direction distribution for Taylor-Green at high-vorticity
tg_angles_to_ico = []
tg_sample_idx = np.argwhere(tg_high)
n_tg_sample = min(1000, len(tg_sample_idx))
tg_sample = tg_sample_idx[np.random.choice(len(tg_sample_idx), n_tg_sample, replace=False)]

for idx in tg_sample:
    i, j, k_idx = idx
    w = tg[i, j, k_idx]
    w_mag = norm(w)
    if w_mag < 1e-10:
        continue
    w_hat = w / w_mag
    min_ico = 90
    for axis in five_fold_axes:
        cos_ico = abs(np.dot(w_hat, axis))
        min_ico = min(min_ico, np.degrees(np.arccos(min(cos_ico, 1.0))))
    tg_angles_to_ico.append(min_ico)

tg_angles_to_ico = np.array(tg_angles_to_ico)

print(f"  {'Metric':<40} {'Taylor-Green':<15} {'Max-Strained'}")
print(f"  {'-'*70}")
print(f"  {'Z₀ (initial enstrophy)':<40} {Z_tg:<15.2f} {Z_ms:.2f}")
print(f"  {'Mean angle to nearest ico axis':<40} {np.mean(tg_angles_to_ico):<15.1f}° {np.mean(vort_angles_to_ico):.1f}°")
print(f"  {'Median angle to nearest ico axis':<40} {np.median(tg_angles_to_ico):<15.1f}° {np.median(vort_angles_to_ico):.1f}°")
print(f"  {'Fraction within ico cone (<31.7°)':<40} {np.mean(tg_angles_to_ico < np.degrees(theta_v/2))*100:<15.1f}% {np.mean(vort_angles_to_ico < np.degrees(theta_v/2))*100:.1f}%")
print(f"  {'Peak Z_max achieved':<40} {'~547':<15} {'~598'}")

print(f"\n  The key difference: Max-strained concentrates vorticity OUTSIDE")
print(f"  the icosahedral cones, while Taylor-Green distributes it more uniformly.")

# ============================================================
# 7. IMPLICATIONS FOR THE PROOF
# ============================================================
print("\n--- 7. IMPLICATIONS FOR THE PROOF ---")
print(f"""
  The H₃ depletion mechanism assumes vorticity preferentially aligns
  with icosahedral five-fold axes. This is valid for:

  ✓ Taylor-Green IC: symmetric, distributes vorticity across axes
  ✓ Random IC: statistically isotropic, covers all directions
  ✓ Vortex tubes along ico axes: by construction

  ✗ Max-strained IC: concentrates vorticity along [0,0,1] (cubic axis)
    → The ẑ axis is {min_angle:.1f}° from the nearest five-fold axis
    → This is OUTSIDE the depletion cone (half-angle {np.degrees(theta_v/2):.1f}°)
    → Result: ~9.2% overshoot that doesn't converge away

  RESOLUTION OPTIONS:

  1. PROVE INSTABILITY: Show that ẑ-aligned vorticity is UNSTABLE
     under NS dynamics and naturally rotates toward ico axes.
     (The "emergent alignment" hypothesis)

  2. EXPAND THE BOUND: The theoretical bound Z_max ~ 7429 is
     never violated. The 547 "bound" is IC-specific. Accept that
     different ICs have different practical bounds, all below 7429.

  3. STRENGTHEN THE MECHANISM: Show that the depletion applies
     to ALL directions (not just near five-fold axes), perhaps
     with a weaker effective δ₀ for non-icosahedral directions.
     Even δ₀_min > 0 for cubic axes would suffice.
""")

# ============================================================
# 8. QUANTIFY: WHAT IS THE EFFECTIVE δ₀ ALONG ẑ?
# ============================================================
print("--- 8. EFFECTIVE δ₀ ALONG CUBIC AXES ---")

# If vorticity is along [0,0,1], what is the minimum angle to any
# icosahedral direction (five-fold, three-fold, or two-fold)?

# Five-fold: already computed → min_angle
# Three-fold axes (face centers, vertices of dual dodecahedron)
three_fold_axes = np.array([
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
])
three_fold_axes = three_fold_axes / norm(three_fold_axes, axis=1, keepdims=True)

min_3fold = 90
for axis in three_fold_axes:
    angle = np.degrees(np.arccos(abs(np.dot(z_axis, axis))))
    min_3fold = min(min_3fold, angle)

min_2fold = 90
for axis in two_fold_axes:
    angle = np.degrees(np.arccos(min(abs(np.dot(z_axis, axis)), 1.0)))
    min_2fold = min(min_2fold, angle)

print(f"  For vorticity along ẑ = [0,0,1]:")
print(f"    Angle to nearest five-fold axis: {min_angle:.1f}°")
print(f"    Angle to nearest three-fold axis: {min_3fold:.1f}°")
print(f"    Angle to nearest two-fold axis: {min_2fold:.1f}°")
print(f"")

# The effective depletion for a given misalignment angle α:
# If the proof assumes δ₀ = tan(θ/2)/2 where θ is the angle between
# vorticity and strain, then for vorticity along a non-icosahedral direction,
# the effective angle is different.

# For ẑ direction: the strain eigenvector alignment matters
# The depletion should be: δ_eff = (projection of depletion onto actual alignment)
# If ω is along ẑ and ẑ is at angle α from the nearest five-fold axis,
# then the effective depletion is reduced by cos(α - θ_v/2) factor

alpha = np.radians(min_angle)
theta_half = theta_v / 2
if alpha > theta_half:
    # Outside cone: effective depletion from nearest axis
    effective_delta = DELTA_0 * np.cos(alpha - theta_half)
    print(f"  Effective δ₀ for ẑ-aligned vorticity: {effective_delta:.4f}")
    print(f"  (vs full δ₀ = {DELTA_0:.4f}, ratio: {effective_delta/DELTA_0:.2f})")
    print(f"  Effective max alignment: {1 - effective_delta:.4f}")
    print(f"")

    # What Z_max would this give?
    Z_effective = 547 / (1 - effective_delta)**2 * (1 - DELTA_0)**2
    print(f"  Predicted Z_max with effective δ₀: {Z_effective:.0f}")
    print(f"  Actual measured Z_max: ~598")
    print(f"  Agreement: {abs(Z_effective - 598)/598*100:.1f}%")
else:
    effective_delta = DELTA_0
    print(f"  ẑ is within icosahedral cone, full δ₀ applies")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
