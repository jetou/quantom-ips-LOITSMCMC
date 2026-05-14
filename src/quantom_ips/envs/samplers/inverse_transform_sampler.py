import torch
import logging

logger = logging.getLogger(__name__)


class InverseTransformSampler:
    def __init__(self):
        pass

    def get_densities_from_probabilities(self, probs):
        eps = torch.finfo(probs.dtype).eps
        result = probs.clone() + (eps / probs.numel())
        for axis in range(probs.ndim):
            slice1 = [slice(None)] * probs.ndim
            slice2 = [slice(None)] * probs.ndim
            slice1[axis] = slice(1, None)
            slice2[axis] = slice(None, -1)
            result = (result[tuple(slice1)] + result[tuple(slice2)]) / 2

        # Since we smooth the input probabilities with eps,
        # the sum here should always be non-zero
        result = result / result.sum()

        return result

    def particle_forward(self, A, grid_list, n):
        output = torch.stack(
            [self.sample(A_particle, grid_list, n) for A_particle in A]
        )
        return output

    def forward(self, A, grid_list, n):
        output = torch.stack(
            [self.particle_forward(A_batch, grid_list, n) for A_batch in A]
        )
        return output

    def sample(self, A, grid_list, n):
        A = torch.clamp(A, min=0)  # Ensure input is positive
        all_probs = self.get_densities_from_probabilities(A)
        eps = torch.finfo(all_probs.dtype).eps

        val_list = []
        inds_list = []

        for i, pts in enumerate(grid_list):
            sum_range = tuple(range(i + 1, all_probs.ndim))
            probs = all_probs.sum(dim=sum_range) if sum_range else all_probs

            cdf = probs.cumsum(dim=i)
            zero_shape = list(cdf.shape)
            zero_shape[i] = 1

            cdf = torch.concat([torch.zeros(zero_shape), cdf], dim=i)
            cdf = (
                cdf / cdf[..., -1:]
            )  # normalize CDF (might cause NaNs, it would be nice to check for 0s)
            trunc_cdf = cdf[..., 1:-1]  # remove 0 and 1 from CDF for safer searching

            u = torch.rand(n, device=cdf.device, dtype=cdf.dtype)
            if i == 0:
                inds = torch.searchsorted(trunc_cdf, u, right=True)
            else:
                inds = torch.searchsorted(
                    trunc_cdf[tuple(inds_list)], u.unsqueeze(-1), right=True
                ).squeeze()

            samples = (u - cdf[(*inds_list, inds)]) / (
                cdf[(*inds_list, inds + 1)] - cdf[(*inds_list, inds)]
            )
            vals = pts[inds] + samples * (pts[inds + 1] - pts[inds])
            inds_list.append(inds)
            val_list.append(vals)

        samples = torch.stack(val_list, dim=-1)
        return samples
