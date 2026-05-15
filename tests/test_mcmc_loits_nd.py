import json
import importlib.util

import pytest
import torch

from quantom_ips import make
from quantom_ips.envs.samplers.mcmc_loits_nd import MCMCLOITSND
from quantom_ips.utils.registration import registry


def test_mcmc_loits_nd_2d_shape_and_bounds():
    x = torch.linspace(-1.0, 1.0, 12)
    y = torch.linspace(0.0, 2.0, 9)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    density = torch.exp(-((xx - 0.2) ** 2 + (yy - 1.1) ** 2) / 0.2)
    A = density.unsqueeze(0)

    sampler = MCMCLOITSND(burn_in=4, thin=2)
    samples = sampler.forward(A, [x, y], 64)

    assert samples.shape == (1, 64, 2)
    assert torch.all(samples[..., 0] >= x.min())
    assert torch.all(samples[..., 0] <= x.max())
    assert torch.all(samples[..., 1] >= y.min())
    assert torch.all(samples[..., 1] <= y.max())
    assert 0.0 <= sampler.last_acceptance_rate <= 1.0


def test_mcmc_loits_nd_5d_shape_and_bounds():
    axes = [torch.linspace(0.0, 1.0, n) for n in [4, 5, 4, 5, 3]]
    grids = torch.meshgrid(*axes, indexing="ij")
    density = torch.ones(*(axis.numel() for axis in axes))
    density = density + 0.25 * sum(grid for grid in grids)
    A = density.unsqueeze(0)

    sampler = MCMCLOITSND(burn_in=3, thin=1)
    samples = sampler.forward(A, axes, 32)

    assert samples.shape == (1, 32, 5)
    for dim, axis in enumerate(axes):
        assert torch.all(samples[..., dim] >= axis.min())
        assert torch.all(samples[..., dim] <= axis.max())


def test_mcmc_loits_nd_registration():
    assert "MCMCLOITSND" in registry
    sampler = make(registry["MCMCLOITSND"].kwargs)
    assert isinstance(sampler, MCMCLOITSND)


def test_its_and_mcmc_loits_nd_synthetic_smoke(tmp_path):
    axes = [torch.linspace(0.0, 1.0, n) for n in [4, 4, 4, 4, 3]]
    density = torch.rand(*(axis.numel() for axis in axes)).unsqueeze(0) + 0.1
    its = make(registry["ITS"].kwargs)
    mcmc = MCMCLOITSND(burn_in=2, thin=1)

    its_samples = its.forward(density, axes, 16)
    mcmc_samples = mcmc.forward(density, axes, 16)

    metrics = {
        "ITS": {"n_events": int(its_samples.reshape(-1, 5).shape[0])},
        "MCMCLOITSND": {
            "n_events": int(mcmc_samples.reshape(-1, 5).shape[0]),
            "acceptance_rate": mcmc.last_acceptance_rate,
        },
    }
    out = tmp_path / "sampler_filter_metrics.json"
    out.write_text(json.dumps(metrics), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "ITS" in loaded
    assert "MCMCLOITSND" in loaded
    assert loaded["MCMCLOITSND"]["acceptance_rate"] is not None


def test_sidis_registration_when_lhapdf_available():
    if importlib.util.find_spec("lhapdf") is None:
        pytest.skip("lhapdf is not installed in this environment")

    assert "SIDIS_masked_cc" in registry
