import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantom_ips.envs.samplers.nf_mcmc_nd import NFMCMCND, _RealNVP
from quantom_ips.envs.samplers.mcmc_loits_nd import MCMCLOITSND


R_MIN = -2.0
R_MAX = 2.0
EPS = 1e-8


def ackley_np(x, y):
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20.0
    )


def compute_ackley_max(n=400):
    axis = np.linspace(R_MIN, R_MAX, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return float(ackley_np(xx, yy).max())


def target_weight_np(points, ackley_max):
    points = np.asarray(points)
    x = points[..., 0]
    y = points[..., 1]
    inside = (x >= R_MIN) & (x <= R_MAX) & (y >= R_MIN) & (y <= R_MAX)
    weight = np.maximum(ackley_max - ackley_np(x, y), EPS)
    return np.where(inside, weight, 0.0)


def make_ackley_grid(grid_n, device, dtype, ackley_max):
    x_axis = torch.linspace(R_MIN, R_MAX, grid_n, device=device, dtype=dtype)
    y_axis = torch.linspace(R_MIN, R_MAX, grid_n, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x_axis, y_axis, indexing="ij")
    z = (
        -20.0 * torch.exp(-0.2 * torch.sqrt(0.5 * (xx**2 + yy**2)))
        - torch.exp(0.5 * (torch.cos(2 * torch.pi * xx) + torch.cos(2 * torch.pi * yy)))
        + torch.e
        + 20.0
    )
    density = torch.clamp(torch.tensor(ackley_max, device=device, dtype=dtype) - z, min=EPS)
    return density, [x_axis, y_axis]


def random_walk_mcmc(n_samples, burn_in, step_size, ackley_max, seed):
    rng = np.random.default_rng(seed)
    total = int(n_samples) + int(burn_in)
    states = np.zeros((n_samples, 2), dtype=np.float64)
    current = np.array([0.0, 0.0], dtype=np.float64)
    current_w = float(target_weight_np(current, ackley_max))
    accepted = 0
    out_i = 0

    for i in range(total):
        proposal = current + rng.normal(0.0, step_size, size=2)
        proposal_w = float(target_weight_np(proposal, ackley_max))
        ratio = proposal_w / max(current_w, EPS)
        if rng.random() < min(1.0, ratio):
            current = proposal
            current_w = proposal_w
            accepted += 1
        if i >= burn_in:
            states[out_i] = current
            out_i += 1

    return states, accepted / max(total, 1)


def hist_prob(samples, bins=120):
    hist, _, _ = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=bins,
        range=[[R_MIN, R_MAX], [R_MIN, R_MAX]],
    )
    total = hist.sum()
    return hist / total if total > 0 else hist


def js_divergence(p, q, eps=1e-12):
    p = p.reshape(-1).astype(np.float64)
    q = q.reshape(-1).astype(np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def plot_2d_comparison(truth, corrected, raw_samples, acceptance_rate, outpath, method_name="NF", bins=120):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    panels = [
        ("MCMC reference", truth),
        (f"{method_name}-MCMC correction\nacceptance={acceptance_rate:.3f}", corrected),
        (f"Raw {method_name} proposal", raw_samples),
    ]
    for ax, (title, samples) in zip(axes, panels):
        ax.hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=bins,
            range=[[R_MIN, R_MAX], [R_MIN, R_MAX]],
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(R_MIN, R_MAX)
        ax.set_ylim(R_MIN, R_MAX)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_marginals(truth, corrected, raw_samples, outpath, method_name="NF", bins=80):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    labels = ["x", "y"]
    for dim in range(2):
        axs[0, dim].hist(
            truth[:, dim],
            bins=bins,
            range=(R_MIN, R_MAX),
            histtype="step",
            label="MCMC reference",
            color="tab:blue",
            linewidth=1.8,
        )
        axs[0, dim].hist(
            corrected[:, dim],
            bins=bins,
            range=(R_MIN, R_MAX),
            histtype="step",
            label=f"{method_name}-MCMC correction",
            color="tab:orange",
            linewidth=1.5,
        )
        axs[0, dim].set_title(f"{method_name}-MCMC correction vs truth - {labels[dim]} marginal")
        axs[0, dim].legend()

        axs[1, dim].hist(
            truth[:, dim],
            bins=bins,
            range=(R_MIN, R_MAX),
            histtype="step",
            label="MCMC reference",
            color="tab:blue",
            linewidth=1.8,
        )
        axs[1, dim].hist(
            raw_samples[:, dim],
            bins=bins,
            range=(R_MIN, R_MAX),
            histtype="step",
            label=f"Raw {method_name}",
            color="tab:orange",
            linewidth=1.5,
        )
        axs[1, dim].set_title(f"Raw {method_name} vs truth - {labels[dim]} marginal")
        axs[1, dim].legend()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_corrected_sampler_marginals(truth, nf_corrected, loits_corrected, outpath, bins=80):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for dim, (ax, label) in enumerate(zip(axes, ["x", "y"])):
        for samples, color, name in [
            (truth, "tab:blue", "MCMC reference"),
            (nf_corrected, "tab:orange", "NF-MCMC correction"),
            (loits_corrected, "tab:green", "LOITS-MCMC correction"),
        ]:
            ax.hist(
                samples[:, dim],
                bins=bins,
                range=(R_MIN, R_MAX),
                histtype="step",
                label=name,
                color=color,
                linewidth=1.8,
            )
        ax.set_title(f"Corrected samplers vs truth - {label} marginal")
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_nf_loits_2d_comparison(
    truth,
    raw_nf,
    nf_corrected,
    nf_acceptance,
    raw_loits,
    loits_corrected,
    loits_acceptance,
    outpath,
    bins=120,
):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), constrained_layout=True)
    rows = [
        (
            "NF proposal and correction",
            [
                ("MCMC reference", truth),
                ("Raw NF proposal", raw_nf),
                (f"NF-MCMC correction\nacceptance={nf_acceptance:.3f}", nf_corrected),
            ],
        ),
        (
            "LOITS proposal and correction",
            [
                ("MCMC reference", truth),
                ("Raw LOITS proposal", raw_loits),
                (f"LOITS-MCMC correction\nacceptance={loits_acceptance:.3f}", loits_corrected),
            ],
        ),
    ]
    for row_index, (row_title, panels) in enumerate(rows):
        for ax, (title, samples) in zip(axes[row_index], panels):
            ax.hist2d(
                samples[:, 0],
                samples[:, 1],
                bins=bins,
                range=[[R_MIN, R_MAX], [R_MIN, R_MAX]],
                cmap="viridis",
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        axes[row_index, 0].annotate(
            row_title,
            xy=(-0.18, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_acceptance_sweep(summary, outpath):
    summary = [
        row
        for row in summary
        if row.get("acceptance_rate") is not None
        or row.get("nf_mcmc_acceptance_rate") is not None
        or row.get("loits_mcmc_acceptance_rate") is not None
    ]
    if not summary:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    dims = [row["dim"] for row in summary]

    if any("nf_mcmc_acceptance_rate" in row for row in summary):
        nf_dims = [row["dim"] for row in summary if row.get("nf_mcmc_acceptance_rate") is not None]
        nf_rates = [row["nf_mcmc_acceptance_rate"] for row in summary if row.get("nf_mcmc_acceptance_rate") is not None]
        ax.plot(nf_dims, nf_rates, marker="o", label="NF-MCMC")

    if any("loits_mcmc_acceptance_rate" in row for row in summary):
        loits_dims = [row["dim"] for row in summary if row.get("loits_mcmc_acceptance_rate") is not None]
        loits_rates = [row["loits_mcmc_acceptance_rate"] for row in summary if row.get("loits_mcmc_acceptance_rate") is not None]
        ax.plot(loits_dims, loits_rates, marker="s", label="LOITS-MCMC")

    if not any("nf_mcmc_acceptance_rate" in row or "loits_mcmc_acceptance_rate" in row for row in summary):
        rates = [row["acceptance_rate"] for row in summary]
        ax.plot(dims, rates, marker="o", label="NF-MCMC")

    ax.set_xlabel("Dimensionality (D)")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("MCMC acceptance rate vs dimensionality")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def make_ackley_nd_grid(dim, grid_n, device, dtype):
    axes = [torch.linspace(R_MIN, R_MAX, grid_n, device=device, dtype=dtype) for _ in range(dim)]
    grids = torch.meshgrid(*axes, indexing="ij")
    r2 = sum(g**2 for g in grids) / dim
    c = sum(torch.cos(2 * torch.pi * g) for g in grids) / dim
    z = -20.0 * torch.exp(-0.2 * torch.sqrt(r2)) - torch.exp(c) + torch.e + 20.0
    density = torch.clamp(z.max() - z, min=EPS)
    return density, axes


def make_paired_ackley_nd_grid(dim, grid_n, device, dtype, ackley_max):
    axes = [torch.linspace(R_MIN, R_MAX, grid_n, device=device, dtype=dtype) for _ in range(dim)]
    grids = torch.meshgrid(*axes, indexing="ij")
    density = torch.ones_like(grids[0])

    for first in range(0, dim, 2):
        x = grids[first]
        y = grids[first + 1] if first + 1 < dim else torch.zeros_like(x)
        z = (
            -20.0 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2)))
            - torch.exp(0.5 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y)))
            + torch.e
            + 20.0
        )
        pair_density = torch.clamp(
            torch.tensor(ackley_max, device=device, dtype=dtype) - z,
            min=EPS,
        )
        density = density * pair_density

    return density, axes


def two_moons_weight_torch(x, y, dx=0.1, dy=0.1, w=0.1, radius=0.2, sigma=0.1):
    x = (x - R_MIN) / (R_MAX - R_MIN)
    y = (y - R_MIN) / (R_MAX - R_MIN)
    x1 = 0.5 - dx
    y1 = 0.5 + dy
    x2 = 0.5 + dx
    y2 = 0.5 - dy
    r1 = torch.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    r2 = torch.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    gate1 = 0.5 * (1 + torch.tanh((y - y1) / w))
    gate2 = 0.5 * (1 + torch.tanh((y2 - y) / w))
    moon1 = torch.exp(-((r1 - radius) ** 2) / (2 * sigma**2)) * gate1
    moon2 = torch.exp(-((r2 - radius) ** 2) / (2 * sigma**2)) * gate2
    return moon1 + moon2 + EPS


def make_paired_two_moons_nd_grid(dim, grid_n, device, dtype):
    axes = [torch.linspace(R_MIN, R_MAX, grid_n, device=device, dtype=dtype) for _ in range(dim)]
    grids = torch.meshgrid(*axes, indexing="ij")
    density = torch.ones_like(grids[0])

    for first in range(0, dim, 2):
        x = grids[first]
        y = grids[first + 1] if first + 1 < dim else torch.zeros_like(x)
        density = density * two_moons_weight_torch(x, y)

    return density, axes


def make_chain_two_moons_nd_grid(dim, grid_n, device, dtype):
    axes = [torch.linspace(R_MIN, R_MAX, grid_n, device=device, dtype=dtype) for _ in range(dim)]
    grids = torch.meshgrid(*axes, indexing="ij")
    density = torch.ones_like(grids[0])

    for first in range(dim - 1):
        density = density * two_moons_weight_torch(grids[first], grids[first + 1])

    return density, axes


def make_sweep_grid(args, dim, device, dtype, ackley_max):
    if args.sweep_target == "chain-two-moons":
        return make_chain_two_moons_nd_grid(dim, args.sweep_grid_n, device, dtype)
    if args.sweep_target == "paired-two-moons":
        return make_paired_two_moons_nd_grid(dim, args.sweep_grid_n, device, dtype)
    if args.sweep_target == "paired-ackley":
        return make_paired_ackley_nd_grid(dim, args.sweep_grid_n, device, dtype, ackley_max)
    return make_ackley_nd_grid(dim, args.sweep_grid_n, device, dtype)


def _coarse_slices(n_points, grid_size):
    edges = torch.linspace(0, n_points, grid_size + 1, dtype=torch.long).tolist()
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(grid_size)]


def notebook_style_loits_acceptance(density, n_events, grid_size=2, burn_in=0, thin=1):
    """Approximate the high-dimensional LOITS notebook acceptance experiment.

    The high-dim notebook builds a coarse grid, proposes from per-cell 1D
    conditional/slice inverse CDFs, then applies an independence MH correction.
    This routine mirrors that structure on the tensor density used by the demo.
    """
    device, dtype = density.device, density.dtype
    dim = density.ndim
    eps = torch.finfo(dtype).tiny
    grid_size = max(1, min(int(grid_size), min(density.shape)))
    axis_slices = [_coarse_slices(density.shape[d], grid_size) for d in range(dim)]
    cell_indices = list(itertools.product(range(grid_size), repeat=dim))

    weights = []
    cell_slices = []
    for idx in cell_indices:
        slices = tuple(axis_slices[d][idx[d]] for d in range(dim))
        cell_slices.append(slices)
        weights.append(density[slices].clamp_min(0).sum())
    weights = torch.stack(weights)
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()

    assignments = torch.multinomial(weights, n_events, replacement=True)
    counts = torch.bincount(assignments, minlength=len(cell_indices))

    accepted = torch.zeros((), device=device, dtype=dtype)
    proposed = torch.zeros((), device=device, dtype=dtype)
    n_steps = max(0, int(burn_in)) + max(1, int(thin))

    with torch.no_grad():
        for cell_id, n_cell_t in enumerate(counts):
            n_cell = int(n_cell_t.item())
            if n_cell <= 0:
                continue

            slices = cell_slices[cell_id]
            starts = [s.start for s in slices]
            stops = [s.stop for s in slices]
            sizes = [max(1, stop - start) for start, stop in zip(starts, stops)]
            centers = [start + size // 2 for start, size in zip(starts, sizes)]

            q_lines = []
            for d in range(dim):
                indexer = []
                for axis in range(dim):
                    if axis == d:
                        indexer.append(slice(starts[axis], stops[axis]))
                    else:
                        indexer.append(centers[axis])
                line = density[tuple(indexer)].clamp_min(eps)
                line = line / line.sum().clamp_min(eps)
                q_lines.append(line)

            def notebook_log_q(indices):
                log_q = torch.zeros(indices.shape[0], device=device, dtype=dtype)
                for d in range(dim):
                    center_indices = indices.clone()
                    center_indices[:, d] = centers[d]
                    log_q = log_q + torch.log(
                        density[
                            tuple(center_indices[:, axis] for axis in range(dim))
                        ].clamp_min(eps)
                    )
                return log_q

            def draw_from_q():
                global_indices = []
                for d, q_line in enumerate(q_lines):
                    local = torch.multinomial(q_line, n_cell, replacement=True)
                    global_indices.append(local + starts[d])
                indices = torch.stack(global_indices, dim=-1)
                log_p = torch.log(
                    density[tuple(indices[:, d] for d in range(dim))].clamp_min(eps)
                )
                log_q = notebook_log_q(indices)
                return indices, log_p, log_q

            current, current_log_p, current_log_q = draw_from_q()
            for _ in range(n_steps):
                proposal, proposal_log_p, proposal_log_q = draw_from_q()
                log_alpha = proposal_log_p + current_log_q - current_log_p - proposal_log_q
                accept = torch.log(torch.rand(n_cell, device=device, dtype=dtype)) < log_alpha.clamp_max(0)
                current = torch.where(accept.unsqueeze(-1), proposal, current)
                current_log_p = torch.where(accept, proposal_log_p, current_log_p)
                current_log_q = torch.where(accept, proposal_log_q, current_log_q)
                accepted = accepted + accept.to(dtype).sum()
                proposed = proposed + n_cell

    return (accepted / proposed.clamp_min(1)).detach().cpu().item()


def draw_notebook_style_loits_unit_samples(density, n_samples, grid_size=2):
    device, dtype = density.device, density.dtype
    dim = density.ndim
    eps = torch.finfo(dtype).tiny
    grid_size = max(1, min(int(grid_size), min(density.shape)))
    axis_slices = [_coarse_slices(density.shape[d], grid_size) for d in range(dim)]
    cell_indices = list(itertools.product(range(grid_size), repeat=dim))

    weights = []
    cell_slices = []
    for idx in cell_indices:
        slices = tuple(axis_slices[d][idx[d]] for d in range(dim))
        cell_slices.append(slices)
        weights.append(density[slices].clamp_min(0).sum())
    weights = torch.stack(weights)
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()

    assignments = torch.multinomial(weights, n_samples, replacement=True)
    samples = torch.empty(n_samples, dim, device=device, dtype=dtype)

    for cell_id, slices in enumerate(cell_slices):
        mask = assignments == cell_id
        n_cell = int(mask.sum().item())
        if n_cell <= 0:
            continue
        starts = [s.start for s in slices]
        stops = [s.stop for s in slices]
        sizes = [max(1, stop - start) for start, stop in zip(starts, stops)]
        centers = [start + size // 2 for start, size in zip(starts, sizes)]
        local_samples = []

        for d in range(dim):
            indexer = []
            for axis in range(dim):
                if axis == d:
                    indexer.append(slice(starts[axis], stops[axis]))
                else:
                    indexer.append(centers[axis])
            line = density[tuple(indexer)].clamp_min(eps)
            line = line / line.sum().clamp_min(eps)
            local = torch.multinomial(line, n_cell, replacement=True)
            jitter = torch.rand(n_cell, device=device, dtype=dtype)
            # Map grid-point proposals to unit coordinates and jitter inside the local bin.
            unit = (local.to(dtype) + starts[d] + jitter) / max(density.shape[d], 1)
            local_samples.append(unit.clamp(1e-6, 1.0 - 1e-6))

        samples[mask] = torch.stack(local_samples, dim=-1)

    return samples


def nf_mcmc_acceptance_from_training_units(
    density,
    axes,
    n_events,
    train_unit,
    train_steps,
    batch_size,
    flow_layers,
    hidden_dim,
    burn_in,
    thin,
):
    device, dtype = density.device, density.dtype
    dim = density.ndim
    helper = NFMCMCND(burn_in=burn_in, thin=thin)
    flow = _RealNVP(dim, hidden_dim=int(hidden_dim), n_layers=int(flow_layers)).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    n_train = train_unit.shape[0]
    batch_size = min(int(batch_size), n_train)

    for _ in range(max(0, int(train_steps))):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        batch = train_unit[idx]
        loss = -flow.log_prob(batch).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    flow.eval()
    flat_log_density = helper._cell_log_densities(density, axes).reshape(-1)
    cell_shape = tuple(max(grid.numel() - 1, 1) for grid in axes)

    with torch.no_grad():
        current_unit, current_log_q = flow.sample(n_events, device=device, dtype=dtype)
        current = helper._from_unit(current_unit, axes)
        current_log_p = helper._target_log_prob(current, axes, flat_log_density, cell_shape)

        accepted = torch.zeros((), device=device, dtype=dtype)
        proposed = torch.zeros((), device=device, dtype=dtype)
        n_steps = max(0, int(burn_in)) + max(1, int(thin))

        for _ in range(n_steps):
            proposal_unit, proposal_log_q = flow.sample(n_events, device=device, dtype=dtype)
            proposal = helper._from_unit(proposal_unit, axes)
            proposal_log_p = helper._target_log_prob(proposal, axes, flat_log_density, cell_shape)
            log_alpha = proposal_log_p + current_log_q - current_log_p - proposal_log_q
            accept = torch.log(torch.rand(n_events, device=device, dtype=dtype)) < log_alpha.clamp_max(0)

            current_unit = torch.where(accept.unsqueeze(-1), proposal_unit, current_unit)
            current_log_p = torch.where(accept, proposal_log_p, current_log_p)
            current_log_q = torch.where(accept, proposal_log_q, current_log_q)
            accepted = accepted + accept.to(dtype).sum()
            proposed = proposed + n_events

    return (accepted / proposed.clamp_min(1)).detach().cpu().item()


# ======================================================================
# Faithful port of the high-dimensional LOITS notebook
# (Loits-gridHighdim-Copy1.ipynb :: LOITS_ND.mcmc_correction_by_grid).
#
# The original notebook_style_loits_acceptance() above is a coarse, fully
# discrete stand-in: it proposes from a multinomial over only the few grid
# points inside each coarse cell, which severely under-estimates the LOITS
# acceptance and collapses much faster with dimension than the real notebook.
#
# This port reproduces the notebook algorithm step-for-step on a *continuous*
# density: per-cell proposals come from 100-point inverse CDFs of a
# reconstructed (smoothed-histogram) density, and the Metropolis ratio uses the
# true continuous target for p and the product-of-cell-center marginals for q.
# pdf evaluations are vectorized (identical numbers, faster); the per-proposal
# accept step stays sequential exactly like the notebook.
# ======================================================================
def _mean_ackley_np(points, a=20.0, b=0.2, c=2 * np.pi):
    sq = np.mean(points ** 2, axis=1)
    co = np.mean(np.cos(c * points), axis=1)
    return -a * np.exp(-b * np.sqrt(sq)) - np.exp(co) + a + np.e


def _ackley2d_np(x, y):
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + 20.0
        + np.e
    )


def _two_moons_np(x, y, dx=0.1, dy=0.1, w=0.1, radius=0.2, sigma=0.1):
    x = (x - R_MIN) / (R_MAX - R_MIN)
    y = (y - R_MIN) / (R_MAX - R_MIN)
    x1, y1, x2, y2 = 0.5 - dx, 0.5 + dy, 0.5 + dx, 0.5 - dy
    r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    g1 = 0.5 * (1 + np.tanh((y - y1) / w))
    g2 = 0.5 * (1 + np.tanh((y2 - y) / w))
    return (
        np.exp(-((r1 - radius) ** 2) / (2 * sigma ** 2)) * g1
        + np.exp(-((r2 - radius) ** 2) / (2 * sigma ** 2)) * g2
        + EPS
    )


def _make_target_density_fn(target, dim, ackley_max):
    """Continuous non-negative density f(points (N,dim)) -> (N,) for each sweep target."""
    eps = 1e-8
    if target == "mean-ackley":
        # density = amax - ackley, so amax MUST be >= max(ackley) over the box or
        # the true density gets truncated. The Ackley maximum sits near half-integer
        # coordinates (cos(2*pi*x) = -1), NOT the integer corners (cos = +1), so a
        # corners-only estimate is too small. Scan the diagonal (which passes through
        # those half-integer points) plus the corners for a safe estimate.
        t = np.linspace(R_MIN, R_MAX, 1001)
        diag = np.repeat(t[:, None], dim, axis=1)
        corners = np.array(
            [[R_MAX] * dim, [R_MIN] * dim, [R_MIN if i % 2 else R_MAX for i in range(dim)]]
        )
        amax = float(max(_mean_ackley_np(diag).max(), _mean_ackley_np(corners).max())) + 1.0

        def fn(p):
            return np.clip(amax - _mean_ackley_np(p), eps, None)

        return fn
    if target == "paired-ackley":
        def fn(p):
            d = np.ones(p.shape[0])
            for f in range(0, dim, 2):
                x = p[:, f]
                y = p[:, f + 1] if f + 1 < dim else np.zeros_like(x)
                d = d * np.clip(ackley_max - _ackley2d_np(x, y), eps, None)
            return d

        return fn
    if target == "paired-two-moons":
        def fn(p):
            d = np.ones(p.shape[0])
            for f in range(0, dim, 2):
                x = p[:, f]
                y = p[:, f + 1] if f + 1 < dim else np.zeros_like(x)
                d = d * _two_moons_np(x, y)
            return d

        return fn
    if target == "chain-two-moons":
        def fn(p):
            d = np.ones(p.shape[0])
            for f in range(dim - 1):
                d = d * _two_moons_np(p[:, f], p[:, f + 1])
            return d

        return fn
    raise ValueError(f"Unknown target for faithful LOITS: {target}")


class _NotebookFaithfulLOITS:
    """Step-for-step port of the notebook LOITS_ND high-dim algorithm."""

    def __init__(self, dim, bounds, density_fn, grid_size=2):
        self.dims = dim
        self.grid_size = grid_size
        self.bounds = bounds
        self.density_fn = density_fn
        # notebook point/bin schedule (LOITS_ND.__init__)
        self.grid_points = 10
        self.bins = 20
        if dim < 5:
            self.pdf_points = 100
        elif dim == 5:
            self.pdf_points = 50
        else:
            self.pdf_points = max(5, int(100 / (2 ** (dim - 4))))
            self.grid_points = max(2, 10 - (dim - 1))
            self.bins = max(5, 20 - 2 * (dim - 1))
        lo, hi = bounds
        self.grid_edges = [np.linspace(lo, hi, grid_size + 1) for _ in range(dim)]

    def pdf(self, points):
        return self.density_fn(np.atleast_2d(points))

    def initialize_distribution(self):
        lo, hi = self.bounds
        n = self.pdf_points
        axis = np.linspace(lo, hi, n)
        Z = np.empty([n] * self.dims, dtype=np.float32)
        if self.dims == 1:
            Z[:] = self.density_fn(axis.reshape(-1, 1))
        else:
            # evaluate slice-by-slice over axis 0 to cap peak memory in high dim
            sub_mesh = np.meshgrid(*([axis] * (self.dims - 1)), indexing="ij")
            sub_flat = np.column_stack([m.ravel() for m in sub_mesh])
            block = np.empty((sub_flat.shape[0], self.dims), dtype=np.float64)
            block[:, 1:] = sub_flat
            for i in range(n):
                block[:, 0] = axis[i]
                Z[i] = self.density_fn(block).reshape([n] * (self.dims - 1))
        self.normal_c = float(np.sum(Z))
        self.probabilities = (Z / self.normal_c).astype(np.float64)

    def compute_initial_cdfs(self):
        ppd = self.probabilities.shape[0]
        step = ppd // self.grid_size
        self.cell_cdfs = [
            np.zeros([self.grid_size] * self.dims, dtype=object) for _ in range(self.dims)
        ]
        for idx in itertools.product(*[range(self.grid_size) for _ in range(self.dims)]):
            bounds = [
                (self.grid_edges[d][idx[d]], self.grid_edges[d][idx[d] + 1]) for d in range(self.dims)
            ]
            slices = tuple(slice(i * step, (i + 1) * step) for i in idx)
            cell_probs = self.probabilities[slices]
            for d in range(self.dims):
                axes = tuple(i for i in range(self.dims) if i != d)
                p = np.sum(cell_probs, axis=axes)
                if np.sum(p) > 0:
                    p = p / np.sum(p)
                    cdf = np.cumsum(p)
                    cdf /= cdf[-1]
                    coords = np.linspace(bounds[d][0], bounds[d][1], len(cdf))
                    self.cell_cdfs[d][idx] = interp1d(
                        cdf, coords, kind="linear", bounds_error=False,
                        fill_value=(bounds[d][0], bounds[d][1]),
                    )
                else:
                    s, e = bounds[d]
                    self.cell_cdfs[d][idx] = (lambda u, s=s, e=e: s + u * (e - s))

    def calculate_grid_weights(self):
        gw = np.zeros([self.grid_size] * self.dims)
        for idx in itertools.product(*[range(self.grid_size) for _ in range(self.dims)]):
            bounds = [
                (self.grid_edges[d][idx[d]], self.grid_edges[d][idx[d] + 1]) for d in range(self.dims)
            ]
            pts = [np.linspace(b[0], b[1], self.grid_points) for b in bounds]
            mesh = np.meshgrid(*pts, indexing="ij")
            flat = np.column_stack([m.ravel() for m in mesh])
            dens = self.pdf(flat)
            dx = [(b[1] - b[0]) / (self.grid_points - 1) for b in bounds]
            gw[idx] = np.sum(dens) * np.prod(dx)
        tot = np.sum(gw)
        if tot > 0:
            gw = gw / tot
        return gw

    def generate_initial_samples(self, n_samples):
        gw = self.calculate_grid_weights()
        cnt = np.round(gw * n_samples).astype(int)
        diff = n_samples - int(np.sum(cnt))
        if diff > 0:
            cnt[np.unravel_index(np.argmax(gw), gw.shape)] += diff
        samples = []
        for idx in itertools.product(*[range(self.grid_size) for _ in range(self.dims)]):
            n = cnt[idx]
            if n > 0:
                cs = np.zeros((n, self.dims))
                for d in range(self.dims):
                    cs[:, d] = self.cell_cdfs[d][idx](np.random.random(n))
                samples.extend(cs)
        return np.array(samples)

    def get_reconstructed_density(self, samples):
        lo, hi = self.bounds
        bins = [np.linspace(lo, hi, self.bins + 1) for _ in range(self.dims)]
        H, edges = np.histogramdd(samples, bins=bins)
        sigma = max(0.5, 1.0 / np.sqrt(self.dims))
        H = gaussian_filter(H, sigma=sigma)
        bin_vol = np.prod([(edges[i][1] - edges[i][0]) for i in range(self.dims)])
        density = H / (np.sum(H) * bin_vol)
        centers = [0.5 * (edges[i][1:] + edges[i][:-1]) for i in range(self.dims)]
        f = RegularGridInterpolator(
            centers, density, method="linear", bounds_error=False, fill_value=0
        )
        return lambda pts: f(np.atleast_2d(pts))

    def _marginal_invcdf(self, recon, centers, target_dim, bounds, n_points=100):
        sp = [
            np.linspace(bounds[d][0], bounds[d][1], n_points)
            if d == target_dim
            else np.array([centers[d]])
            for d in range(self.dims)
        ]
        mesh = np.meshgrid(*sp, indexing="ij")
        pts = np.column_stack([m.ravel() for m in mesh])
        dv = recon(pts)
        if np.all(dv == 0):
            lo, hi = bounds[target_dim]
            return (lambda u, lo=lo, hi=hi: lo + u * (hi - lo))
        cdf = np.cumsum(dv)
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return interp1d(
            cdf, sp[target_dim], kind="linear", bounds_error=False,
            fill_value=(bounds[target_dim][0], bounds[target_dim][1]),
        )

    def mcmc_correction_by_grid(self, recon, n_samples):
        eps = 1e-10
        gw = self.calculate_grid_weights()
        cnt = np.round(gw * n_samples).astype(int)
        diff = n_samples - int(np.sum(cnt))
        if diff > 0:
            cnt[np.unravel_index(np.argmax(gw), gw.shape)] += diff
        total_acc = 0
        total_att = 0
        for idx in itertools.product(*[range(self.grid_size) for _ in range(self.dims)]):
            spg = int(cnt[idx])
            if spg == 0:
                continue
            bounds = [
                (self.grid_edges[d][idx[d]], self.grid_edges[d][idx[d] + 1]) for d in range(self.dims)
            ]
            centers = [(b[0] + b[1]) / 2 for b in bounds]
            margin = [self._marginal_invcdf(recon, centers, d, bounds, 100) for d in range(self.dims)]

            current = None
            for _ in range(100):
                cand = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
                if float(self.pdf(cand)[0]) > 0:
                    current = cand
                    break
            if current is None:
                continue
            curr_p = float(self.pdf(current)[0])
            curr_q = 1.0
            for d in range(self.dims):
                cp = current.copy()
                cp[d] = centers[d]
                curr_q *= (float(self.pdf(cp)[0]) + eps)

            props = np.zeros((spg, self.dims))
            for d in range(self.dims):
                props[:, d] = margin[d](np.random.random(spg))
            prop_p = self.pdf(props) + eps
            prop_q = np.ones(spg)
            for d in range(self.dims):
                sub = props.copy()
                sub[:, d] = centers[d]
                prop_q *= (self.pdf(sub) + eps)
            rnd = np.random.random(spg)
            accepts = 0
            for k in range(spg):
                ratio = min(1.0, (prop_p[k] * curr_q) / (curr_p * prop_q[k]))
                if rnd[k] < ratio:
                    if all(bounds[d][0] <= props[k, d] <= bounds[d][1] for d in range(self.dims)):
                        curr_p = prop_p[k]
                        curr_q = prop_q[k]
                        accepts += 1
            total_acc += accepts
            total_att += spg
        return total_acc / max(total_att, 1)

    def run(self, n_init, mcmc_samples):
        self.initialize_distribution()
        self.compute_initial_cdfs()
        init = self.generate_initial_samples(n_init)
        recon = self.get_reconstructed_density(init)
        return self.mcmc_correction_by_grid(recon, mcmc_samples)


def notebook_faithful_loits_acceptance(
    target, dim, ackley_max, n_events, grid_size=2, n_init=200000
):
    """LOITS acceptance via the faithful notebook port (continuous proposals)."""
    density_fn = _make_target_density_fn(target, dim, ackley_max)
    loits = _NotebookFaithfulLOITS(dim, (R_MIN, R_MAX), density_fn, grid_size=grid_size)
    return float(loits.run(n_init=n_init, mcmc_samples=n_events))


def run_dimension_sweep(args, device, dtype):
    rows = []
    ackley_max = compute_ackley_max()
    for dim in range(args.sweep_min_dim, args.sweep_max_dim + 1):
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            density, axes = make_sweep_grid(args, dim, device, dtype, ackley_max)
            if args.nf_train_source == "notebook-loits":
                train_unit = draw_notebook_style_loits_unit_samples(
                    density,
                    args.sweep_train_samples,
                    grid_size=args.loits_grid_size,
                )
                nf_rate = nf_mcmc_acceptance_from_training_units(
                    density,
                    axes,
                    args.sweep_events,
                    train_unit,
                    args.sweep_train_steps,
                    args.batch_size,
                    args.flow_layers,
                    args.hidden_dim,
                    args.nf_burn_in,
                    args.nf_thin,
                )
            else:
                nf_sampler = NFMCMCND(
                    train_steps=args.sweep_train_steps,
                    train_samples=args.sweep_train_samples,
                    batch_size=args.batch_size,
                    flow_layers=args.flow_layers,
                    hidden_dim=args.hidden_dim,
                    burn_in=args.nf_burn_in,
                    thin=args.nf_thin,
                )
                _, nf_rate = nf_sampler.sample(
                    density,
                    axes,
                    args.sweep_events,
                    cache_key=("ackley-sweep", dim),
                )

            if args.loits_impl == "faithful":
                loits_rate = notebook_faithful_loits_acceptance(
                    args.sweep_target,
                    dim,
                    ackley_max,
                    args.sweep_events,
                    grid_size=args.loits_grid_size,
                    n_init=args.loits_init_samples,
                )
            else:
                loits_rate = notebook_style_loits_acceptance(
                    density,
                    args.sweep_events,
                    grid_size=args.loits_grid_size,
                    burn_in=args.loits_burn_in,
                    thin=args.loits_thin,
                )

            row = {
                "dim": dim,
                "acceptance_rate": float(nf_rate),
                "nf_mcmc_acceptance_rate": float(nf_rate),
                "loits_mcmc_acceptance_rate": float(loits_rate),
                "grid_n": args.sweep_grid_n,
                "target": args.sweep_target,
                "nf_train_source": args.nf_train_source,
            }
            rows.append(row)
            print(
                f"D={dim}: NF-MCMC acceptance={nf_rate:.3f} | "
                f"LOITS-MCMC acceptance={loits_rate:.3f}"
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row = {
                "dim": dim,
                "acceptance_rate": None,
                "nf_mcmc_acceptance_rate": None,
                "loits_mcmc_acceptance_rate": None,
                "grid_n": args.sweep_grid_n,
                "target": args.sweep_target,
                "nf_train_source": args.nf_train_source,
                "error": "CUDA out of memory",
            }
            rows.append(row)
            print(f"D={dim}: skipped because CUDA ran out of memory at grid_n={args.sweep_grid_n}.")
            break
        finally:
            with (Path(args.outdir) / "dimension_sweep_partial.json").open("w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            plot_acceptance_sweep(rows, Path(args.outdir) / "ackley_nf_mcmc_acceptance_vs_dim_partial.png")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Ackley NF-MCMC correction demo")
    parser.add_argument("--outdir", default="examples/results/nf_mcmc_ackley_demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-n", type=int, default=80)
    parser.add_argument("--n-events", type=int, default=60000)
    parser.add_argument("--truth-burn-in", type=int, default=5000)
    parser.add_argument("--truth-step-size", type=float, default=0.16)
    parser.add_argument("--nf-train-steps", type=int, default=1000)
    parser.add_argument("--nf-train-samples", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--flow-layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--nf-burn-in", type=int, default=20)
    parser.add_argument("--nf-thin", type=int, default=5)
    parser.add_argument("--loits-burn-in", type=int, default=16)
    parser.add_argument("--loits-thin", type=int, default=4)
    parser.add_argument("--loits-step-radius", type=int, default=1)
    parser.add_argument("--loits-grid-size", type=int, default=2)
    parser.add_argument(
        "--loits-impl",
        choices=["faithful", "discrete"],
        default="faithful",
        help=(
            "LOITS acceptance implementation for the sweep. 'faithful' ports the "
            "high-dimensional LOITS notebook (continuous per-cell inverse-CDF proposals); "
            "'discrete' is the old coarse grid-index stand-in that collapses too fast."
        ),
    )
    parser.add_argument(
        "--loits-init-samples",
        type=int,
        default=200000,
        help="Initial samples used to reconstruct the density for the faithful LOITS port.",
    )
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--sweep-min-dim", type=int, default=2)
    parser.add_argument("--sweep-max-dim", type=int, default=8)
    parser.add_argument("--sweep-grid-n", type=int, default=7)
    parser.add_argument("--sweep-events", type=int, default=3000)
    parser.add_argument("--sweep-train-steps", type=int, default=250)
    parser.add_argument("--sweep-train-samples", type=int, default=4096)
    parser.add_argument(
        "--nf-train-source",
        choices=["exact-its", "notebook-loits"],
        default="exact-its",
        help=(
            "Training samples for the NF dimension sweep. exact-its uses direct samples from the "
            "grid target; notebook-loits trains NF from the same approximate grid/cell LOITS "
            "proposal family used by the high-dimensional LOITS notebook."
        ),
    )
    parser.add_argument(
        "--sweep-target",
        choices=["mean-ackley", "paired-ackley", "paired-two-moons", "chain-two-moons"],
        default="mean-ackley",
        help=(
            "Target used only for dimension sweeps. mean-ackley is the standard D-dimensional "
            "Ackley average; paired-ackley repeats the 2D Ackley structure across dimension pairs; "
            "paired-two-moons repeats the two-moon target from the LOITS notebook across pairs; "
            "chain-two-moons links adjacent dimensions with two-moon factors."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32

    if device.type == "cuda":
        _ = (torch.ones(1, device=device) + 1).item()

    print(f"Using device: {device}")
    ackley_max = compute_ackley_max()
    density, axes = make_ackley_grid(args.grid_n, device, dtype, ackley_max)

    print("Generating reference random-walk MCMC samples...")
    truth, truth_acc = random_walk_mcmc(
        args.n_events,
        args.truth_burn_in,
        args.truth_step_size,
        ackley_max,
        args.seed,
    )
    print(f"Reference random-walk MCMC acceptance: {truth_acc:.3f}")

    print("Training NF proposal and running NF-MCMC correction...")
    sampler = NFMCMCND(
        train_steps=args.nf_train_steps,
        train_samples=args.nf_train_samples,
        batch_size=args.batch_size,
        flow_layers=args.flow_layers,
        hidden_dim=args.hidden_dim,
        burn_in=args.nf_burn_in,
        thin=args.nf_thin,
    )
    cache_key = ("ackley-2d", args.nf_train_steps, args.nf_train_samples, args.grid_n)
    flow = sampler._get_flow(density, axes, cache_key=cache_key)

    with torch.no_grad():
        raw_unit, _ = flow.sample(args.n_events, device=device, dtype=dtype)
        raw_nf = sampler._from_unit(raw_unit, axes).detach().cpu().numpy()

    corrected_t, nf_acc = sampler.sample(density, axes, args.n_events, cache_key=cache_key)
    corrected = corrected_t.detach().cpu().numpy()
    print(f"NF-MCMC correction acceptance: {nf_acc:.3f}")

    print("Drawing LOITS proposals and running LOITS-MCMC correction...")
    loits_sampler = MCMCLOITSND(
        burn_in=args.loits_burn_in,
        thin=args.loits_thin,
        step_radius=args.loits_step_radius,
    )
    with torch.no_grad():
        raw_loits = loits_sampler.propose(density, axes, args.n_events).detach().cpu().numpy()
    loits_corrected_t, loits_acc = loits_sampler.sample(density, axes, args.n_events)
    loits_corrected = loits_corrected_t.detach().cpu().numpy()
    print(f"LOITS-MCMC correction acceptance: {loits_acc:.3f}")

    truth_prob = hist_prob(truth)
    raw_prob = hist_prob(raw_nf)
    corrected_prob = hist_prob(corrected)
    raw_loits_prob = hist_prob(raw_loits)
    loits_corrected_prob = hist_prob(loits_corrected)
    metrics = {
        "device": str(device),
        "n_events": args.n_events,
        "truth_acceptance_rate": float(truth_acc),
        "nf_mcmc_acceptance_rate": float(nf_acc),
        "loits_mcmc_acceptance_rate": float(loits_acc),
        "raw_nf_js_to_truth": js_divergence(truth_prob, raw_prob),
        "nf_mcmc_js_to_truth": js_divergence(truth_prob, corrected_prob),
        "raw_loits_js_to_truth": js_divergence(truth_prob, raw_loits_prob),
        "loits_mcmc_js_to_truth": js_divergence(truth_prob, loits_corrected_prob),
        "nf_train_steps": args.nf_train_steps,
        "nf_train_samples": args.nf_train_samples,
    }

    np.save(outdir / "mcmc_reference_samples.npy", truth)
    np.save(outdir / "raw_nf_samples.npy", raw_nf)
    np.save(outdir / "nf_mcmc_samples.npy", corrected)
    np.save(outdir / "raw_loits_samples.npy", raw_loits)
    np.save(outdir / "loits_mcmc_samples.npy", loits_corrected)
    plot_2d_comparison(truth, corrected, raw_nf, nf_acc, outdir / "ackley_nf_mcmc_2d_comparison.png")
    plot_marginals(truth, corrected, raw_nf, outdir / "ackley_nf_mcmc_marginals.png")
    plot_2d_comparison(
        truth,
        loits_corrected,
        raw_loits,
        loits_acc,
        outdir / "ackley_loits_mcmc_2d_comparison.png",
        method_name="LOITS",
    )
    plot_marginals(
        truth,
        loits_corrected,
        raw_loits,
        outdir / "ackley_loits_mcmc_marginals.png",
        method_name="LOITS",
    )
    plot_corrected_sampler_marginals(
        truth,
        corrected,
        loits_corrected,
        outdir / "ackley_nf_loits_corrected_marginals.png",
    )
    plot_nf_loits_2d_comparison(
        truth,
        raw_nf,
        corrected,
        nf_acc,
        raw_loits,
        loits_corrected,
        loits_acc,
        outdir / "ackley_nf_loits_before_after_2d.png",
    )

    if args.run_sweep:
        sweep = run_dimension_sweep(args, device, dtype)
        metrics["dimension_sweep"] = sweep
        plot_acceptance_sweep(sweep, outdir / "ackley_nf_mcmc_acceptance_vs_dim.png")

    with (outdir / "ackley_nf_mcmc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {outdir}")


if __name__ == "__main__":
    main()
