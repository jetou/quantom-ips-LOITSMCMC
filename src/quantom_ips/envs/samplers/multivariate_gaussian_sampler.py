import torch
import logging

logger = logging.getLogger(__name__)


class MVGSampler:
    def __init__(self):
        pass

    def particle_forward(self, A, grid_list, n):
        output = torch.stack(
            [self.sample(A_particle, grid_list, n) for A_particle in A]
        )
        return output

    def forward(self, A, grid_list, n):
        logger.debug(f"Sampler forward() received {A.shape} tensor")
        output = self.sample(A, grid_list, n)
        # output = torch.stack(
        #     [self.particle_forward(A_batch, grid_list, n) for A_batch in A]
        # )
        return output

    def sample(self, A, grid_list, n):
        # Grid list will be ignored
        # A is a (batch_size, n_dims * 2) tensor of means and variances
        assert int(A.shape[-1]) % 2 == 0
        d = A.shape[-1] // 2
        batch_size = A.shape[0]
        samples = torch.randn(size=(batch_size, n, d), device=A.device, dtype=A.dtype)
        output = []
        for batch, params in zip(samples, A):
            output.append(params[d:] * batch + params[:d])

        return torch.stack(output, dim=0)
