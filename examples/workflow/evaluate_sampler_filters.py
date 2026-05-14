import json
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from quantom_ips import make
from quantom_ips.utils.torch_nn_registry import get_dtype
from pytorch_workflow import get_device


logger = logging.getLogger("sampler_filter_eval")


def flatten_events(events):
    if isinstance(events, torch.Tensor):
        events = events.detach().cpu().numpy()
    events = np.asarray(events)
    if events.shape[-1] != 2:
        raise ValueError(f"Expected events with last dimension 2, got {events.shape}")
    return events.reshape(-1, 2)


def histogram_density(samples, x_range, y_range, bins):
    hist, x_edges, y_edges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=bins,
        range=[x_range, y_range],
    )
    prob = hist.astype(float)
    if prob.sum() > 0:
        prob = prob / prob.sum()
    bin_area = ((x_range[1] - x_range[0]) / bins) * (
        (y_range[1] - y_range[0]) / bins
    )
    density = prob / bin_area
    return prob, density, x_edges, y_edges


def compare_samples(real_samples, generated_samples, x_range, y_range, bins, eps=1e-12):
    real_prob, real_density, x_edges, y_edges = histogram_density(
        real_samples, x_range, y_range, bins
    )
    gen_prob, gen_density, _, _ = histogram_density(
        generated_samples, x_range, y_range, bins
    )

    p = gen_prob.ravel() + eps
    q = real_prob.ravel() + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return {
        "js_divergence": float(0.5 * kl_pm + 0.5 * kl_qm),
        "l1_error": float(np.sum(np.abs(p - q))),
        "density_mae": float(np.mean(np.abs(gen_density - real_density))),
        "generated_density": gen_density,
        "real_density": real_density,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def plot_comparison(real_samples, sampler_samples, metrics, x_range, y_range, out_path):
    sampler_names = list(sampler_samples.keys())
    n_cols = 1 + len(sampler_names) * 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
    axes = axes[0]

    axes[0].hist2d(real_samples[:, 0], real_samples[:, 1], bins=100, range=[x_range, y_range])
    axes[0].set_title("Real samples")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    for offset, name in enumerate(sampler_names):
        sample_ax = axes[1 + offset]
        sample_ax.hist2d(
            sampler_samples[name][:, 0],
            sampler_samples[name][:, 1],
            bins=100,
            range=[x_range, y_range],
        )
        sample_ax.set_title(
            f"{name}\nJS={metrics[name]['js_divergence']:.4f}"
        )
        sample_ax.set_xlabel("x")
        sample_ax.set_ylabel("y")

    diff_start = 1 + len(sampler_names)
    for offset, name in enumerate(sampler_names):
        diff_ax = axes[diff_start + offset]
        diff = metrics[name]["generated_density"] - metrics[name]["real_density"]
        max_abs = np.max(np.abs(diff)) if np.max(np.abs(diff)) > 0 else 1.0
        im = diff_ax.imshow(
            diff.T,
            origin="lower",
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            aspect="auto",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        diff_ax.set_title(f"{name} - real")
        diff_ax.set_xlabel("x")
        diff_ax.set_ylabel("y")
        fig.colorbar(im, ax=diff_ax)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def get_ranges_from_grid(grid_axes):
    x_axis = grid_axes[0].detach().cpu()
    y_axis = grid_axes[1].detach().cpu()
    return (float(x_axis.min()), float(x_axis.max())), (
        float(y_axis.min()),
        float(y_axis.max()),
    )


def summarize_metrics(metrics):
    result = {}
    for name, values in metrics.items():
        result[name] = {
            key: value
            for key, value in values.items()
            if key not in {"generated_density", "real_density", "x_edges", "y_edges"}
        }
    return result


@hydra.main(version_base=None, config_name=None, config_path="./conf")
def run(config) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.checkpoint:
        raise ValueError("Set checkpoint=/path/to/checkpoint.pt")

    dtype = get_dtype(config.dtype)
    device = get_device(config.device) if config.autodetect_device else config.device
    if device is None:
        device = "cpu"
    logger.info(f"Using device: {device}")

    opt = make(config.opt)
    opt.load(config.checkpoint)
    opt = opt.to(device=device, dtype=dtype)
    opt.eval()

    theory = make(config.eval.theory)
    dataloader = make(config.dataloader)
    samplers = {
        name: make(sampler_cfg) for name, sampler_cfg in config.eval.samplers.items()
    }

    real_batches = []
    sampler_batches = {name: [] for name in samplers}
    sampler_stats = {name: [] for name in samplers}
    n_batches = int(config.eval.n_batches)

    with torch.no_grad():
        for batch_idx, (_, targets) in enumerate(dataloader):
            if n_batches >= 0 and batch_idx >= n_batches:
                break

            targets = targets.to(device=device, dtype=dtype)
            opt.targets = targets
            params = opt.predict()
            probabilities, grid_axes, _ = theory.forward(params)
            x_range, y_range = get_ranges_from_grid(grid_axes)
            n_events = (
                int(config.eval.n_events)
                if config.eval.n_events is not None
                else int(targets.shape[-2])
            )

            real_batches.append(flatten_events(targets))
            for name, sampler in samplers.items():
                samples = sampler.forward(probabilities, grid_axes, n_events)
                sampler_batches[name].append(flatten_events(samples))
                sampler_stats[name].append(getattr(sampler, "last_stats", {}))

    real_samples = np.concatenate(real_batches, axis=0)
    sampler_samples = {
        name: np.concatenate(batches, axis=0)
        for name, batches in sampler_batches.items()
        if batches
    }

    metrics = {}
    for name, samples in sampler_samples.items():
        metrics[name] = compare_samples(
            real_samples,
            samples,
            x_range=x_range,
            y_range=y_range,
            bins=int(config.eval.histogram_bins),
        )
        acceptance_rates = [
            stats.get("acceptance_rate")
            for stats in sampler_stats[name]
            if "acceptance_rate" in stats
        ]
        if acceptance_rates:
            metrics[name]["acceptance_rate"] = float(np.mean(acceptance_rates))

    np.save(out_dir / "real_samples.npy", real_samples)
    for name, samples in sampler_samples.items():
        np.save(out_dir / f"{name}_samples.npy", samples)

    json_metrics = summarize_metrics(metrics)
    with open(out_dir / "sampler_filter_metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=2)

    plot_comparison(
        real_samples,
        sampler_samples,
        metrics,
        x_range=x_range,
        y_range=y_range,
        out_path=out_dir / "sampler_filter_comparison.png",
    )

    logger.info("Sampler filter metrics:")
    logger.info(OmegaConf.to_yaml(OmegaConf.create(json_metrics)))


if __name__ == "__main__":
    run()
