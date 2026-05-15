import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quantom_ips import make
from quantom_ips.utils.registration import registry
from quantom_ips.utils.torch_nn_registry import get_dtype


AXIS_NAMES = ["x", "Q2", "z", "qT", "phi"]
AXIS_INDEX = {name.lower(): idx for idx, name in enumerate(AXIS_NAMES)}
DEFAULT_PROJECTIONS = ["x,qT", "x,Q2", "z,qT", "qT,phi"]


def load_events(path, event_dim=5, n_events=None, seed=0):
    data = np.load(path)
    if data.ndim == 1:
        if data.size % event_dim != 0:
            raise ValueError(f"Flat event array length {data.size} is not divisible by {event_dim}")
        data = data.reshape(-1, event_dim)
    elif data.ndim == 3 and data.shape[-1] == event_dim:
        data = data.reshape(-1, event_dim)
    if not (data.ndim == 2 and data.shape[1] == event_dim):
        raise ValueError(f"Expected events with shape (N,{event_dim}), got {data.shape}")
    if n_events is not None and data.shape[0] > n_events:
        rng = np.random.default_rng(seed)
        data = data[rng.choice(data.shape[0], size=n_events, replace=False)]
    return data.astype(np.float64)


def histogram_probability(events, bins):
    hist, _ = np.histogramdd(events, bins=bins)
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist, dtype=np.float64)
    return hist.astype(np.float64) / total


def js_divergence(p, q, eps=1e-12):
    p = p.reshape(-1).astype(np.float64)
    q = q.reshape(-1).astype(np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def metric_block(real_hist, sample_hist):
    return {
        "js_divergence": js_divergence(real_hist, sample_hist),
        "l1_error": float(np.abs(real_hist - sample_hist).sum()),
        "density_mae": float(np.abs(real_hist - sample_hist).mean()),
    }


def projection_metric_block(real_hist, sample_hist, x_edges, y_edges):
    x_width = np.diff(x_edges).astype(np.float64)
    y_width = np.diff(y_edges).astype(np.float64)
    area = np.outer(x_width, y_width)
    real_density = real_hist / np.maximum(area, 1e-300)
    sample_density = sample_hist / np.maximum(area, 1e-300)
    out = metric_block(real_hist, sample_hist)
    out["density_mae"] = float(np.abs(real_density - sample_density).mean())
    return out


def parse_projection_specs(specs):
    projections = []
    for spec in specs:
        parts = [part.strip() for part in spec.replace(":", ",").replace("-", ",").split(",")]
        parts = [part for part in parts if part]
        if len(parts) != 2:
            raise ValueError(f"Projection '{spec}' must contain exactly two axis names")
        dims = []
        labels = []
        for part in parts:
            key = part.lower()
            if key not in AXIS_INDEX:
                raise ValueError(f"Unknown axis '{part}'. Valid axes: {', '.join(AXIS_NAMES)}")
            dims.append(AXIS_INDEX[key])
            labels.append(AXIS_NAMES[AXIS_INDEX[key]])
        projections.append((tuple(dims), f"{labels[0]}_{labels[1]}"))
    return projections


def histogram2d_probability(events, bins, dims):
    hist, _, _ = np.histogram2d(
        events[:, dims[0]],
        events[:, dims[1]],
        bins=[bins[dims[0]], bins[dims[1]]],
    )
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist, dtype=np.float64)
    return hist.astype(np.float64) / total


def plot_marginals(real_events, samples_by_name, bins, outpath):
    n_dims = len(bins)
    fig, axes = plt.subplots(n_dims, 1, figsize=(8, 2.2 * n_dims), constrained_layout=True)
    if n_dims == 1:
        axes = [axes]
    for dim, ax in enumerate(axes):
        centers = 0.5 * (bins[dim][1:] + bins[dim][:-1])
        real_hist, _ = np.histogram(real_events[:, dim], bins=bins[dim], density=True)
        ax.step(centers, real_hist, where="mid", label="real", linewidth=2)
        for name, events in samples_by_name.items():
            hist, _ = np.histogram(events[:, dim], bins=bins[dim], density=True)
            ax.step(centers, hist, where="mid", label=name, alpha=0.85)
        ax.set_xlabel(AXIS_NAMES[dim] if dim < len(AXIS_NAMES) else f"dim{dim}")
        ax.set_ylabel("density")
        ax.legend()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _hist_density(events, bin_edges):
    hist, _ = np.histogram(events, bins=bin_edges, density=True)
    return hist.astype(np.float64)


def plot_event_distributions_with_uncertainty(real_events, sample_reps_by_name, bins, outpath):
    n_dims = len(bins)
    n_cols = 3
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.9 * n_rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    colors = {"ITS": "tab:orange", "MCMCLOITSND": "tab:green"}

    for dim in range(n_dims):
        ax = axes[dim]
        edges = bins[dim]
        centers = 0.5 * (edges[1:] + edges[:-1])

        real_hist = _hist_density(real_events[:, dim], edges)
        ax.step(centers, real_hist, where="mid", label="Real", linewidth=2.2, color="tab:blue")

        for name, reps in sample_reps_by_name.items():
            rep_hists = np.stack([_hist_density(rep[:, dim], edges) for rep in reps], axis=0)
            mean = rep_hists.mean(axis=0)
            std = rep_hists.std(axis=0)
            color = colors.get(name)
            ax.step(centers, mean, where="mid", label=f"{name} mean", linewidth=2.0, color=color)
            ax.errorbar(
                centers,
                mean,
                yerr=std,
                fmt="none",
                color=color,
                alpha=0.65,
                capsize=2.0,
                linewidth=1.0,
            )

        ax.set_xlabel(axis_label(AXIS_NAMES[dim]))
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        if AXIS_NAMES[dim] == "x":
            ax.set_xscale("log")

    for ax in axes[n_dims:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="center", frameon=True)
        if axes[-1].has_data():
            axes[-1].legend(loc="best", frameon=True)

    fig.suptitle("Event Distributions with Statistical Uncertainty (Real vs ITS vs MCMCLOITSND)")
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def axis_label(name):
    if name == "Q2":
        return r"$Q^2$ [GeV$^2$]"
    if name == "qT":
        return r"$q_T$ [GeV]"
    if name == "phi":
        return r"$\phi$"
    return name


def plot_pair_projection(real_events, samples_by_name, bins, outpath, dims=(0, 3)):
    names = ["true"] + list(samples_by_name.keys())
    events_list = [real_events] + list(samples_by_name.values())
    hists = [
        np.histogram2d(
            events[:, dims[0]], events[:, dims[1]], bins=[bins[dims[0]], bins[dims[1]]]
        )[0]
        for events in events_list
    ]
    vmax = max(float(hist.max()) for hist in hists) if hists else 1.0

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), constrained_layout=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name, hist in zip(axes, names, hists):
        im = ax.pcolormesh(
            bins[dims[0]],
            bins[dims[1]],
            hist.T,
            shading="auto",
            vmin=0.0,
            vmax=max(vmax, 1e-12),
        )
        ax.set_title(name)
        ax.set_xlabel(AXIS_NAMES[dims[0]])
        ax.set_ylabel(AXIS_NAMES[dims[1]])
        if AXIS_NAMES[dims[0]] == "x":
            ax.set_xscale("log")
    fig.colorbar(im, ax=axes, shrink=0.82, label="counts")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_difference_projection(real_events, samples_by_name, bins, outpath, dims=(0, 3)):
    real_hist = histogram2d_probability(real_events, bins, dims)
    comparisons = []
    for name, events in samples_by_name.items():
        comparisons.append((f"{name} - true", histogram2d_probability(events, bins, dims) - real_hist))
    if "ITS" in samples_by_name and "MCMCLOITSND" in samples_by_name:
        its_hist = histogram2d_probability(samples_by_name["ITS"], bins, dims)
        mcmc_hist = histogram2d_probability(samples_by_name["MCMCLOITSND"], bins, dims)
        comparisons.append(("MCMCLOITSND - ITS", mcmc_hist - its_hist))
    if not comparisons:
        return

    vmax = max(float(np.abs(diff).max()) for _, diff in comparisons)
    fig, axes = plt.subplots(1, len(comparisons), figsize=(4.4 * len(comparisons), 3.8), constrained_layout=True)
    if len(comparisons) == 1:
        axes = [axes]
    for ax, (title, diff) in zip(axes, comparisons):
        im = ax.pcolormesh(
            bins[dims[0]],
            bins[dims[1]],
            diff.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-max(vmax, 1e-12),
            vmax=max(vmax, 1e-12),
        )
        ax.set_title(title)
        ax.set_xlabel(AXIS_NAMES[dims[0]])
        ax.set_ylabel(AXIS_NAMES[dims[1]])
        if AXIS_NAMES[dims[0]] == "x":
            ax.set_xscale("log")
    fig.colorbar(im, ax=axes, shrink=0.82, label="probability difference")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _as_numpy_2d(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    else:
        value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D array for TMD plotting, got shape {value.shape}")
    return value


def _select_evolved_slice(value, q_index=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    else:
        value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError(f"Expected evolved TMD with shape (x,Q2,bT), got {value.shape}")
    idx = value.shape[1] // 2 if q_index is None else int(q_index)
    idx = max(0, min(idx, value.shape[1] - 1))
    return value[:, idx, :], idx


def _load_tmd_axes(sample_dir, fallback_shape):
    density_path = Path(sample_dir) / "density.npy"
    if density_path.exists():
        data = np.load(density_path, allow_pickle=True).item()
        if "x" in data and "bt" in data:
            return np.asarray(data["x"], dtype=np.float64), np.asarray(data["bt"], dtype=np.float64)
    nx, nbt = fallback_shape
    return np.arange(nx, dtype=np.float64), np.arange(nbt, dtype=np.float64)


def plot_tmd_model_check(rundir, sample_dir, outpath, q_index=None):
    losses_path = Path(rundir) / "losses.pt"
    if not losses_path.exists():
        print(f"Skipping TMD model check plot because losses.pt was not found: {losses_path}")
        return None

    try:
        losses = torch.load(losses_path, map_location="cpu", weights_only=False)
    except TypeError:
        losses = torch.load(losses_path, map_location="cpu")

    real_input = _as_numpy_2d(losses.get("real_tmd_input"))
    model_input = _as_numpy_2d(losses.get("last_gen_tmd_input"))
    real_evolved, selected_q = _select_evolved_slice(losses.get("real_tmd_evolved"), q_index=q_index)
    model_evolved, _ = _select_evolved_slice(losses.get("last_gen_tmd"), q_index=selected_q)

    x_axis, bt_axis = _load_tmd_axes(sample_dir, real_input.shape)
    panels = [
        ("Theory input f_th(x,b_T)", real_input),
        ("Model input (last epoch)", model_input),
        (f"Theory evolved f_th(x,Q2[{selected_q}],b_T)", real_evolved),
        (f"Model evolved (last epoch, Q2[{selected_q}])", model_evolved),
    ]
    vmax = max(float(np.nanmax(panel)) for _, panel in panels if panel is not None)
    vmin = min(0.0, min(float(np.nanmin(panel)) for _, panel in panels if panel is not None))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    for ax, (title, values) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(
            x_axis,
            bt_axis,
            values.T,
            shading="auto",
            vmin=vmin,
            vmax=max(vmax, 1e-12),
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("b_T")
        if np.all(x_axis > 0):
            ax.set_xscale("log")
    fig.suptitle("TMD: Theory vs Model (last epoch)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="f(x,b_T)")
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return outpath


def ensure_flavor_dim(env, params):
    if hasattr(env.theory, "e2") and hasattr(env.theory, "tar") and params.ndim == 4:
        n_flav = int(env.theory.e2[env.theory.tar].numel())
        if params.shape[1] == 1 and n_flav > 1:
            return params.repeat(1, n_flav, 1, 1)
    return params


def make_sampler(name):
    if name not in registry:
        raise KeyError(f"Sampler {name} is not registered")
    return make(registry[name].kwargs)


def main():
    parser = argparse.ArgumentParser(description="Compare TMD event sampler filters")
    parser.add_argument("--rundir", required=True, help="TMD-GAN Hydra run directory")
    parser.add_argument("--data", required=True, help="Real SIDIS events .npy file")
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--n-samples", type=int, default=1, help="Generator batch size for evaluation")
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Repeat each sampler this many times to estimate histogram uncertainty",
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--projections",
        nargs="*",
        default=DEFAULT_PROJECTIONS,
        help="Two-axis event projections, e.g. x,qT x,Q2 z,qT qT,phi",
    )
    parser.add_argument("--q-index", type=int, default=None, help="Q2 index for the TMD evolved plot")
    parser.add_argument(
        "--skip-tmd-plot",
        action="store_true",
        help="Only run sampler comparison; do not plot losses.pt TMD diagnostics",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rundir = Path(args.rundir)
    outdir = Path(args.outdir or rundir / "sampler_filter_eval")
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = rundir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {config_path}")
    config = OmegaConf.create(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    OmegaConf.set_struct(config, False)
    if "opt" in config:
        config.opt.logdir = str(rundir)
    if "output_dir" in config:
        config.output_dir = str(rundir)

    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    env = make(config.env).to(device=device, dtype=dtype)
    if getattr(env, "sample_dir", None) is None:
        env.sample_dir = str(Path(args.data).parent)

    nx = getattr(env.theory, "nx", None)
    nbt = getattr(env.theory, "nbt", None)
    if nx is not None and nbt is not None:
        if "generator" in config.opt and "image_size" in config.opt.generator:
            config.opt.generator.image_size = [nx, nbt]
    if "discriminator" in config.opt:
        config.opt.discriminator.input_dim = 5

    opt = make(config.opt).to(device=device, dtype=dtype)
    state = torch.load(rundir / "optimizer.pt", map_location=device, weights_only=True)
    opt.load_state_dict(state, strict=False)
    opt.generator.eval()
    opt.discriminator.eval()

    with torch.no_grad():
        noise = torch.normal(
            0.0,
            1.0,
            size=(args.n_samples, int(config.opt.noise_dim)),
            device=device,
            dtype=dtype,
        )
        params = ensure_flavor_dim(env, opt.generator(noise))
        density, grid_axes, _ = env.theory.forward(params)

        samplers = {"ITS": make_sampler("ITS"), "MCMCLOITSND": make_sampler("MCMCLOITSND")}
        samples_by_name = {}
        sample_reps_by_name = {}
        metrics = {}
        bins = [axis.detach().cpu().numpy().astype(np.float64) for axis in grid_axes]
        real_events = load_events(args.data, event_dim=len(bins), n_events=args.n_events, seed=args.seed)
        real_hist = histogram_probability(real_events, bins)
        projections = parse_projection_specs(args.projections)

        np.save(outdir / "real_events.npy", real_events)
        for name, sampler in samplers.items():
            reps = []
            acceptance_rates = []
            for repeat in range(max(1, int(args.n_repeats))):
                torch.manual_seed(args.seed + 1009 * repeat + 17 * (name == "MCMCLOITSND"))
                events = sampler.forward(density, grid_axes, args.n_events)
                events = events.reshape(-1, events.shape[-1]).detach().cpu().numpy()
                events = events[: args.n_events].astype(np.float64)
                reps.append(events)
                rate = getattr(sampler, "last_acceptance_rate", None)
                if rate is not None:
                    acceptance_rates.append(float(rate))

            events = reps[-1]
            np.save(outdir / f"{name.lower()}_events.npy", events)
            np.save(outdir / f"{name.lower()}_events_reps.npy", np.stack(reps, axis=0))
            samples_by_name[name] = events
            sample_reps_by_name[name] = reps

            sample_hist = histogram_probability(events, bins)
            metrics[name] = metric_block(real_hist, sample_hist)
            metrics[name]["acceptance_rate"] = (
                float(np.mean(acceptance_rates)) if acceptance_rates else None
            )
            metrics[name]["n_repeats"] = int(max(1, args.n_repeats))

        projection_metrics = {}
        for dims, label in projections:
            real_proj_hist = histogram2d_probability(real_events, bins, dims)
            projection_metrics[label] = {}
            for name, events in samples_by_name.items():
                sample_proj_hist = histogram2d_probability(events, bins, dims)
                projection_metrics[label][name] = projection_metric_block(
                    real_proj_hist,
                    sample_proj_hist,
                    bins[dims[0]],
                    bins[dims[1]],
                )
                projection_metrics[label][name]["acceptance_rate"] = metrics[name][
                    "acceptance_rate"
                ]
        metrics["projections"] = projection_metrics

    with (outdir / "sampler_filter_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if not args.skip_tmd_plot:
        plot_tmd_model_check(
            rundir,
            sample_dir=getattr(env, "sample_dir", Path(args.data).parent),
            outpath=outdir / "tmd_true_vs_model_last_epoch.png",
            q_index=args.q_index,
        )

    plot_marginals(real_events, samples_by_name, bins, outdir / "sampler_marginals.png")
    plot_event_distributions_with_uncertainty(
        real_events,
        sample_reps_by_name,
        bins,
        outdir / "sampler_event_distributions_uncertainty.png",
    )
    for dims, label in projections:
        plot_pair_projection(
            real_events,
            samples_by_name,
            bins,
            outdir / f"sampler_projection_{label}_true_its_mcmc.png",
            dims=dims,
        )
        plot_difference_projection(
            real_events,
            samples_by_name,
            bins,
            outdir / f"sampler_projection_{label}_differences.png",
            dims=dims,
        )

    print(f"Saved sampler comparison outputs to {outdir}")


if __name__ == "__main__":
    main()
