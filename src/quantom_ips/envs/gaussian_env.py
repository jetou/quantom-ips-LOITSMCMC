import torch
import logging
from quantom_ips.utils.stateful_module import StatefulModule

logger = logging.getLogger(__name__)


class GaussianEnv(StatefulModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pipeline(self, params, coeffs, nevs_per_batch, real_sample_sizes):
        u = torch.randn(
            size=(real_sample_sizes[0], nevs_per_batch, real_sample_sizes[2]),
            device=params.device,
            dtype=params.dtype,
        )
        means = torch.einsum("bnd,bd->bn", coeffs, params.view(1, -1))
        return u + means[:, :, None, None]

    def forward(self, params, real_samples):
        coefficients = real_samples[1]
        data = real_samples[0]
        batch_size = params.size()[0]
        nevs_per_batch = int(data.size()[1] / batch_size)

        u = torch.vmap(
            lambda p: self.pipeline(p, coefficients, nevs_per_batch, data.size()),
            in_dims=0,
            randomness="different",
        )(params).flatten(0, 1)

        observation = {"fake": u, "real": data}
        reward = None
        terminated = True  # Every "episode" is a single action/reward pair
        truncated = False
        info = {"coefficients": coefficients}

        return observation, reward, terminated, truncated, info
