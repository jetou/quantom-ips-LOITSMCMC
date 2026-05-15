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


def load_events(path, event_dim=5, n_events=None, seed=0):
    data = np.load(path)
    if data.ndim == 1:
        if data.size % event_dim != 0:
            raise ValueError(f"Flat event array length {data.size} is not divisible by {event_dim}")
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


def plot_pair_projection(real_events, samples_by_name, bins, outpath, dims=(0, 3)):
    names = ["real"] + list(samples_by_name.keys())
    events_list = [real_events] + list(samples_by_name.values())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8), constrained_layout=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name, events in zip(axes, names, events_list):
        hist, _, _ = np.histogram2d(
            events[:, dims[0]], events[:, dims[1]], bins=[bins[dims[0]], bins[dims[1]]]
        )
        im = ax.imshow(hist.T, origin="lower", aspect="auto")
        ax.set_title(name)
        ax.set_xlabel(AXIS_NAMES[dims[0]])
        ax.set_ylabel(AXIS_NAMES[dims[1]])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


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
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
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
    opt.eval()

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
        metrics = {}
        bins = [axis.detach().cpu().numpy().astype(np.float64) for axis in grid_axes]
        real_events = load_events(args.data, event_dim=len(bins), n_events=args.n_events, seed=args.seed)
        real_hist = histogram_probability(real_events, bins)

        np.save(outdir / "real_events.npy", real_events)
        for name, sampler in samplers.items():
            events = sampler.forward(density, grid_axes, args.n_events)
            events = events.reshape(-1, events.shape[-1]).detach().cpu().numpy()
            events = events[: args.n_events].astype(np.float64)
            np.save(outdir / f"{name.lower()}_events.npy", events)
            samples_by_name[name] = events

            sample_hist = histogram_probability(events, bins)
            metrics[name] = metric_block(real_hist, sample_hist)
            metrics[name]["acceptance_rate"] = getattr(sampler, "last_acceptance_rate", None)

    with (outdir / "sampler_filter_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_marginals(real_events, samples_by_name, bins, outdir / "sampler_marginals.png")
    if len(bins) >= 4:
        plot_pair_projection(real_events, samples_by_name, bins, outdir / "sampler_x_qt_projection.png")

    print(f"Saved sampler comparison outputs to {outdir}")


if __name__ == "__main__":
    main()
