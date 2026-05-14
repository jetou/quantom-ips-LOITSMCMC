import torch

from quantom_ips import make
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.envs.samplers.mcmc_loits_2d import MCMCLOITS2D
from quantom_ips.utils.registration import registry


def two_moons_density(x, y):
    x1, y1 = 0.4, 0.6
    x2, y2 = 0.6, 0.4
    radius = 0.2
    width = 0.1
    sigma = 0.1

    r1 = torch.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    r2 = torch.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    gate1 = 0.5 * (1 + torch.tanh((y - y1) / width))
    gate2 = 0.5 * (1 + torch.tanh((y2 - y) / width))
    moon1 = torch.exp(-((r1 - radius) ** 2) / (2 * sigma**2)) * gate1
    moon2 = torch.exp(-((r2 - radius) ** 2) / (2 * sigma**2)) * gate2
    return moon1 + moon2


def make_density(batch_size=2, n_targets=2, grid_size=8):
    x_axis = torch.linspace(0, 1, grid_size)
    y_axis = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(x_axis, y_axis, indexing="ij")
    density = two_moons_density(X, Y)
    density = density[None, None, :, :].repeat(batch_size, n_targets, 1, 1)
    return density, [x_axis, y_axis]


def test_mcmc_loits_2d_shape_bounds_and_tensor_output():
    density, grid_axes = make_density()
    sampler = MCMCLOITS2D(n_marginal_points=8)

    events = sampler.forward(density, grid_axes, n_events=40)

    assert isinstance(events, torch.Tensor)
    assert events.shape == (2, 2, 40, 2)
    assert events.dtype == density.dtype
    assert events.device == density.device
    assert torch.all(events[..., 0] >= grid_axes[0].min())
    assert torch.all(events[..., 0] <= grid_axes[0].max())
    assert torch.all(events[..., 1] >= grid_axes[1].min())
    assert torch.all(events[..., 1] <= grid_axes[1].max())
    assert "acceptance_rate" in sampler.last_stats


def test_mcmc_loits_2d_registered_with_hydra_registry():
    assert "MCMCLOITS2D" in registry
    sampler = make(registry["MCMCLOITS2D"].kwargs)
    assert isinstance(sampler, MCMCLOITS2D)


def test_loits_and_mcmc_loits_run_on_same_two_moons_density():
    density, grid_axes = make_density(batch_size=1, n_targets=1, grid_size=10)
    loits = LOInverseTransformSampler2D(
        average=False,
        use_threading=False,
        n_interpolations_x=4,
        n_interpolations_y=4,
    )
    mcmc_loits = MCMCLOITS2D(n_marginal_points=8)

    loits_events = loits.forward(density, grid_axes, n_events=50)
    mcmc_events = mcmc_loits.forward(density, grid_axes, n_events=50)

    assert loits_events.shape[0] == 1
    assert loits_events.shape[1] == 1
    assert loits_events.shape[-1] == 2
    assert mcmc_events.shape == (1, 1, 50, 2)
    assert "acceptance_rate" in mcmc_loits.last_stats
