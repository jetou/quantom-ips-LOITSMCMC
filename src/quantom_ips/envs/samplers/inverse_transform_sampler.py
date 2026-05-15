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

        result = result / result.sum()
        return result

    def particle_forward(self, A, grid_list, n):
        """
        A: either
          - [*grid_shape]               (single particle)
          - [n_particles, *grid_shape]  (multiple particles)
        grid_list: list of 1D grids, len(grid_list) = len(grid_shape)
        """
        grid_ndim = len(grid_list)

        if A.ndim == grid_ndim:
            # Single particle case: just sample once
            return self.sample(A, grid_list, n)

        elif A.ndim == grid_ndim + 1:
            # Multiple particles: iterate over first axis
            return torch.stack(
                [self.sample(A_particle, grid_list, n) for A_particle in A]
            )

        else:
            raise ValueError(
                f"InverseTransformSampler.particle_forward: "
                f"unexpected A.ndim={A.ndim} for grid_ndim={grid_ndim}"
            )

    def forward(self, A, grid_list, n):
        # A is expected to have leading batch dimension(s); we iterate over the first one
        output = torch.stack(
            [self.particle_forward(A_batch, grid_list, n) for A_batch in A]
        )
        return output

    def sample(self, A, grid_list, n):
        A = torch.clamp(A, min=0)  # Ensure input is positive
        device, dtype = A.device, A.dtype

        # OPTIONAL: ensure grids are on same device (helps avoid future device mismatches)
        grid_list = [pts.to(device=device, dtype=dtype) for pts in grid_list]

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

            # Put zeros on the same device and dtype as cdf.
            zeros = torch.zeros(zero_shape, device=cdf.device, dtype=cdf.dtype)
            cdf = torch.concat([zeros, cdf], dim=i)

            cdf = cdf / cdf[..., -1:]
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
