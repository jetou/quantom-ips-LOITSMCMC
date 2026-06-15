import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantom_ips.envs.samplers.nf_mcmc_nd import NFMCMCND
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


def plot_2d_comparison(truth, corrected, raw_nf, acceptance_rate, outpath, bins=120):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    panels = [
        ("MCMC reference", truth),
        (f"NF-MCMC correction\nacceptance={acceptance_rate:.3f}", corrected),
        ("Raw NF proposal", raw_nf),
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


def plot_marginals(truth, corrected, raw_nf, outpath, bins=80):
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
            label="NF-MCMC correction",
            color="tab:orange",
            linewidth=1.5,
        )
        axs[0, dim].set_title(f"Correction vs truth - {labels[dim]} marginal")
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
            raw_nf[:, dim],
            bins=bins,
            range=(R_MIN, R_MAX),
            histtype="step",
            label="Raw NF",
            color="tab:orange",
            linewidth=1.5,
        )
        axs[1, dim].set_title(f"Raw NF vs truth - {labels[dim]} marginal")
        axs[1, dim].legend()
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


def run_dimension_sweep(args, device, dtype):
    rows = []
    for dim in range(args.sweep_min_dim, args.sweep_max_dim + 1):
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            density, axes = make_ackley_nd_grid(dim, args.sweep_grid_n, device, dtype)
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

            loits_sampler = MCMCLOITSND(
                burn_in=args.loits_burn_in,
                thin=args.loits_thin,
                step_radius=args.loits_step_radius,
            )
            _, loits_rate = loits_sampler.sample(density, axes, args.sweep_events)

            row = {
                "dim": dim,
                "acceptance_rate": float(nf_rate),
                "nf_mcmc_acceptance_rate": float(nf_rate),
                "loits_mcmc_acceptance_rate": float(loits_rate),
                "grid_n": args.sweep_grid_n,
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
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--sweep-min-dim", type=int, default=2)
    parser.add_argument("--sweep-max-dim", type=int, default=8)
    parser.add_argument("--sweep-grid-n", type=int, default=7)
    parser.add_argument("--sweep-events", type=int, default=3000)
    parser.add_argument("--sweep-train-steps", type=int, default=250)
    parser.add_argument("--sweep-train-samples", type=int, default=4096)
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

    truth_prob = hist_prob(truth)
    raw_prob = hist_prob(raw_nf)
    corrected_prob = hist_prob(corrected)
    metrics = {
        "device": str(device),
        "n_events": args.n_events,
        "truth_acceptance_rate": float(truth_acc),
        "nf_mcmc_acceptance_rate": float(nf_acc),
        "raw_nf_js_to_truth": js_divergence(truth_prob, raw_prob),
        "nf_mcmc_js_to_truth": js_divergence(truth_prob, corrected_prob),
        "nf_train_steps": args.nf_train_steps,
        "nf_train_samples": args.nf_train_samples,
    }

    np.save(outdir / "mcmc_reference_samples.npy", truth)
    np.save(outdir / "raw_nf_samples.npy", raw_nf)
    np.save(outdir / "nf_mcmc_samples.npy", corrected)
    plot_2d_comparison(truth, corrected, raw_nf, nf_acc, outdir / "ackley_nf_mcmc_2d_comparison.png")
    plot_marginals(truth, corrected, raw_nf, outdir / "ackley_nf_mcmc_marginals.png")

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
