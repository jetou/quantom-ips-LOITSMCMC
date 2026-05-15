#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml


def lin_or_log(minv, maxv, n, log=False):
    if n <= 1:
        return np.array([minv], dtype=float)
    if log:
        return np.logspace(np.log10(minv), np.log10(maxv), n)
    return np.linspace(minv, maxv, n)


def qt_over_Q_bound(x, Q2, z, M, Mh):
    """
    Implements the same inequality structure as SIDIS.kinematic_mask():
      qT/Q < sqrt(R2_clamped)
    where R2 = termA - termB with rho2 = 1 + 4 M^2 x^2 / Q2.
    Returns qTmax_over_Q (>=0).
    """
    rho2 = 1.0 + 4.0 * (M * M) * (x * x) / Q2
    denom = (rho2 - 1.0)

    # Avoid division by zero if denom -> 0 (can happen at very small x or large Q2)
    if denom <= 0:
        return 0.0

    termA = (1.0 / denom) * (1.0 - denom * (M * M) / (Q2 * (z * z)))

    bracket = (
        (1.0 - x - z) / x
        + (Mh * Mh) / Q2
        - 2.0 * z / denom
    )
    termB = (denom / (4.0 * rho2 * (z * z))) * (bracket * bracket)

    R2 = termA - termB
    if not np.isfinite(R2) or R2 <= 0:
        return 0.0
    return math.sqrt(R2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Path to mother kinematics YAML")
    ap.add_argument("--format", choices=["hydra"], default="hydra",
                    help="Output format (currently only hydra CLI overrides)")
    args = ap.parse_args()

    cfg_path = Path(args.yaml)
    if not cfg_path.exists():
        print(f"ERROR: YAML not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())
    sids = cfg["sidis"]

    # Independent knobs
    Ebeam = float(sids["Ebeam_GeV"])
    M     = float(sids["target_mass_GeV"])
    Mh    = float(sids["hadron_mass_GeV"])

    Q2_min = float(sids["Q2_min_GeV2"])
    W2_min = float(sids["W2_min_GeV2"])
    z_min  = float(sids["z_min"])
    z_max  = float(sids["z_max"])

    bt_min = float(sids["bt_min"])
    bt_max = float(sids["bt_max"])
    phi_min = float(sids["phi_min"])
    phi_max = float(sids["phi_max"])

    qT_over_Q_lim = float(sids.get("qT_over_Q_lim", 1.0e9))

    # Grid resolution + spacing
    grid = sids["grid"]
    nx   = int(grid["nx"])
    nQ2  = int(grid["nQ2"])
    nz   = int(grid["nz"])
    nbt  = int(grid["nbt"])
    nqt  = int(grid["nqt"])
    nphi = int(grid["nphi"])

    spacing = sids["spacing"]
    log_x  = bool(spacing["log_x"])
    log_Q2 = bool(spacing["log_Q2"])
    log_z  = bool(spacing["log_z"])

    # Theory identity knobs
    th = sids["theory"]
    order   = int(th["order"])
    tar     = str(th["tar"])
    had     = str(th["had"])
    average = bool(th["average"])

    # --- Derived: s, Q2_max, x_min
    s_val = 2.0 * Ebeam * M + M * M
    s_minus_M2 = s_val - (M * M)

    # Q2_max policy
    Q2_pol = sids["Q2_policy"]
    if Q2_pol.get("derive_Q2_max_from_s", True):
        Q2_max = s_minus_M2
    else:
        Q2_max = float(Q2_pol["Q2_max_override_GeV2"])

    # x_min policy (matches your notes): x_min = Q2_min / (s - M^2)
    x_pol = sids["x_policy"]
    if x_pol.get("derive_x_min_from_Q2min", True):
        x_min = Q2_min / s_minus_M2
    else:
        x_min = float(x_pol["x_min_override"])

    x_max = float(x_pol["x_max"])

    # --- Derived: qt_max by scanning the *same* inequality used in kinematic_mask
    qT_pol = sids["qT_policy"]
    if qT_pol.get("derive_qt_max_from_bound", True):
        qt_scan_min = float(qT_pol.get("qt_scan_min", 0.0))
        qt_cap      = float(qT_pol.get("qt_scan_max_cap", 1.0e9))

        # Build scan grids consistent with your eventual theory grid ranges.
        # Note: this is a *planning* scan; it’s okay if it’s coarse.
        xs  = lin_or_log(x_min, x_max, nx, log=log_x)
        Q2s = lin_or_log(Q2_min, Q2_max, nQ2, log=log_Q2)
        zs  = lin_or_log(z_min, z_max, nz, log=log_z)

        qt_max_star = 0.0
        for x in xs:
            for Q2 in Q2s:
                Q = math.sqrt(Q2)
                for z in zs:
                    bound = qt_over_Q_bound(x, Q2, z, M, Mh)
                    bound = min(bound, qT_over_Q_lim)  # match your mask’s extra cap
                    qt_here = bound * Q
                    if qt_here > qt_max_star:
                        qt_max_star = qt_here

        # Apply cap and ensure >= scan_min
        qt_max = max(qt_scan_min, min(qt_max_star, qt_cap))
        qt_min = qt_scan_min
    else:
        qt_min = float(qT_pol["qt_min_override"])
        qt_max = float(qT_pol["qt_max_override"])

    # Emit Hydra overrides (one line you can splice into CMD)
    # Keep names matching env.theory.* arguments used in your scripts.
    overrides = [
        f"env.theory.order={order}",
        f"env.theory.tar={tar}",
        f"env.theory.had={had}",
        f"env.theory.Ebeam={Ebeam}",
        f"env.theory.average={str(average)}",
        f"env.theory.nx={nx}",
        f"env.theory.nQ2={nQ2}",
        f"env.theory.nz={nz}",
        f"env.theory.nbt={nbt}",
        f"env.theory.nqt={nqt}",
        f"env.theory.nphi={nphi}",
        f"env.theory.x_min={x_min}",
        f"env.theory.x_max={x_max}",
        f"env.theory.Q2_min={Q2_min}",
        f"env.theory.Q2_max={Q2_max}",
        f"env.theory.z_min={z_min}",
        f"env.theory.z_max={z_max}",
        f"env.theory.bt_min={bt_min}",
        f"env.theory.bt_max={bt_max}",
        f"env.theory.qt_min={qt_min}",
        f"env.theory.qt_max={qt_max}",
        f"env.theory.phi_min={phi_min}",
        f"env.theory.phi_max={phi_max}",
        f"env.theory.W2_min={W2_min}",
        f"env.theory.log_x={str(log_x)}",
        f"env.theory.log_Q2={str(log_Q2)}",
        f"env.theory.log_z={str(log_z)}",
        # keep qT_lim consistent with mask (you already have qT_lim in theory init)
        f"env.theory.qT_lim={qT_over_Q_lim}",
    ]

    # Print as a single space-separated string so bash can capture it
    print(" ".join(overrides))


if __name__ == "__main__":
    main()
