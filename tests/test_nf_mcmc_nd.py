import torch

from quantom_ips import make
from quantom_ips.envs.samplers.nf_mcmc_nd import NFMCMCND
from quantom_ips.utils.registration import registry


def test_nf_mcmc_nd_2d_shape_bounds_and_acceptance():
    x = torch.linspace(-1.0, 1.0, 8)
    y = torch.linspace(0.0, 2.0, 7)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    density = torch.exp(-((xx - 0.2) ** 2 + (yy - 1.1) ** 2) / 0.2).unsqueeze(0)

    sampler = NFMCMCND(train_steps=2, train_samples=128, batch_size=32, burn_in=1, thin=1)
    samples = sampler.forward(density, [x, y], 16)

    assert samples.shape == (1, 16, 2)
    assert torch.all(samples[..., 0] >= x.min())
    assert torch.all(samples[..., 0] <= x.max())
    assert torch.all(samples[..., 1] >= y.min())
    assert torch.all(samples[..., 1] <= y.max())
    assert 0.0 <= sampler.last_acceptance_rate <= 1.0


def test_nf_mcmc_nd_5d_shape_bounds_and_acceptance():
    axes = [torch.linspace(0.0, 1.0, n) for n in [4, 5, 4, 5, 3]]
    grids = torch.meshgrid(*axes, indexing="ij")
    density = torch.ones(*(axis.numel() for axis in axes))
    density = density + 0.25 * sum(grid for grid in grids)
    density = density.unsqueeze(0)

    sampler = NFMCMCND(train_steps=2, train_samples=128, batch_size=32, burn_in=1, thin=1)
    samples = sampler.forward(density, axes, 12)

    assert samples.shape == (1, 12, 5)
    for dim, axis in enumerate(axes):
        assert torch.all(samples[..., dim] >= axis.min())
        assert torch.all(samples[..., dim] <= axis.max())
    assert 0.0 <= sampler.last_acceptance_rate <= 1.0


def test_nf_mcmc_nd_registration():
    assert "NFMCMCND" in registry
    sampler = make(registry["NFMCMCND"].kwargs)
    assert isinstance(sampler, NFMCMCND)
