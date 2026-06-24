import itertools

import torch

from quantom_ips.utils.stateful_module import StatefulModule


class MCMCLOITSND(StatefulModule):
    def __init__(self, burn_in: int = 16, thin: int = 4, step_radius: int = 1):
        super().__init__()
        self.burn_in = burn_in
        self.thin = thin
        self.step_radius = step_radius
        self.last_acceptance_rate = None

    def forward(self, A, grid_list, n_events):
        if len(grid_list) < 1:
            raise ValueError("MCMCLOITSND expects at least one grid axis")

        grid_ndim = len(grid_list)
        outputs = []
        rates = []
        for A_batch in A:
            out, rate = self.particle_forward(A_batch, grid_list, n_events, grid_ndim)
            outputs.append(out)
            rates.append(rate)

        self.last_acceptance_rate = float(sum(rates) / max(len(rates), 1))
        return torch.stack(outputs)

    def particle_forward(self, A, grid_list, n_events, grid_ndim):
        if A.ndim == grid_ndim:
            samples, rate = self.sample(A, grid_list, n_events)
            return samples, rate

        if A.ndim == grid_ndim + 1:
            samples = []
            rates = []
            for A_particle in A:
                particle_samples, rate = self.sample(A_particle, grid_list, n_events)
                samples.append(particle_samples)
                rates.append(rate)
            return torch.stack(samples), float(sum(rates) / max(len(rates), 1))

        raise ValueError(
            f"MCMCLOITSND got A.ndim={A.ndim}, expected {grid_ndim} or {grid_ndim + 1}"
        )

    def _cell_probabilities(self, A):
        probs = torch.clamp(A, min=0)
        eps = torch.finfo(probs.dtype).eps
        probs = probs + eps / probs.numel()

        cell_probs = torch.zeros(
            tuple(max(s - 1, 1) for s in probs.shape),
            device=probs.device,
            dtype=probs.dtype,
        )
        for corner in itertools.product([0, 1], repeat=probs.ndim):
            slices = tuple(
                slice(offset, probs.shape[axis] - 1 + offset)
                for axis, offset in enumerate(corner)
            )
            cell_probs = cell_probs + probs[slices]

        cell_probs = cell_probs / (2 ** probs.ndim)
        total = cell_probs.sum()
        if not torch.isfinite(total) or total <= 0:
            cell_probs = torch.ones_like(cell_probs)
            total = cell_probs.sum()
        return cell_probs / total

    def _cell_marginals(self, cell_probs):
        marginals = []
        for dim in range(cell_probs.ndim):
            reduce_dims = tuple(axis for axis in range(cell_probs.ndim) if axis != dim)
            marginal = cell_probs.sum(dim=reduce_dims)
            marginal = marginal / marginal.sum().clamp_min(torch.finfo(cell_probs.dtype).tiny)
            marginals.append(marginal)
        return marginals

    def _draw_from_product_marginals(self, marginals, n_events):
        indices = []
        q_prob = None
        for marginal in marginals:
            dim_idx = torch.multinomial(marginal, n_events, replacement=True)
            indices.append(dim_idx)
            dim_q = marginal[dim_idx].clamp_min(torch.finfo(marginal.dtype).tiny)
            q_prob = dim_q if q_prob is None else q_prob * dim_q
        return torch.stack(indices, dim=-1), q_prob

    def _flatten_indices(self, indices, shape):
        flat = torch.zeros(indices.shape[0], device=indices.device, dtype=torch.long)
        stride = 1
        for dim in reversed(range(len(shape))):
            flat = flat + indices[:, dim] * stride
            stride *= shape[dim]
        return flat

    def _unflatten_indices(self, flat, shape):
        coords = []
        rem = flat
        strides = []
        for i in range(len(shape)):
            stride = 1
            for size in shape[i + 1 :]:
                stride *= size
            strides.append(stride)
        for stride in torch.tensor(strides, device=flat.device, dtype=torch.long):
            coords.append(rem // stride)
            rem = rem % stride
        return torch.stack(coords, dim=-1)

    def _indices_to_coordinates(self, indices, grid_list):
        n_events = indices.shape[0]
        device, dtype = grid_list[0].device, grid_list[0].dtype
        coords = []
        for dim, grid in enumerate(grid_list):
            idx = indices[:, dim]
            lo = grid[idx]
            hi = grid[idx + 1]
            u = torch.rand(n_events, device=device, dtype=dtype)
            coords.append(lo + u * (hi - lo))
        return torch.stack(coords, dim=-1)

    def propose(self, A, grid_list, n_events):
        """Draw uncorrected LOITS proposal samples from the product-marginal proposal."""
        device, dtype = A.device, A.dtype
        grid_list = [grid.to(device=device, dtype=dtype) for grid in grid_list]
        if A.ndim != len(grid_list):
            raise ValueError(
                f"MCMCLOITSND density ndim {A.ndim} does not match {len(grid_list)} grid axes"
            )
        for axis, grid in enumerate(grid_list):
            if grid.numel() != A.shape[axis] or grid.numel() < 2:
                raise ValueError("MCMCLOITSND proposal requires grid axes matching the density")

        cell_probs = self._cell_probabilities(A)
        marginals = self._cell_marginals(cell_probs)
        indices, _ = self._draw_from_product_marginals(marginals, n_events)
        return self._indices_to_coordinates(indices, grid_list)

    def sample(self, A, grid_list, n_events):
        device, dtype = A.device, A.dtype
        grid_list = [grid.to(device=device, dtype=dtype) for grid in grid_list]
        if A.ndim != len(grid_list):
            raise ValueError(
                f"MCMCLOITSND density ndim {A.ndim} does not match {len(grid_list)} grid axes"
            )
        for axis, grid in enumerate(grid_list):
            if grid.numel() != A.shape[axis]:
                raise ValueError(
                    f"Grid axis {axis} length {grid.numel()} does not match density shape {A.shape[axis]}"
                )
            if grid.numel() < 2:
                raise ValueError("MCMCLOITSND requires at least two points per axis")

        cell_probs = self._cell_probabilities(A)
        cell_shape = tuple(cell_probs.shape)
        flat_probs = cell_probs.reshape(-1)
        marginals = self._cell_marginals(cell_probs)

        current, current_q = self._draw_from_product_marginals(marginals, n_events)
        current_flat = self._flatten_indices(current, cell_shape)
        current_prob = flat_probs[current_flat].clamp_min(torch.finfo(dtype).tiny)
        current_q = current_q.clamp_min(torch.finfo(dtype).tiny)

        n_steps = max(0, int(self.burn_in)) + max(1, int(self.thin))
        accepted = torch.zeros((), device=device, dtype=dtype)
        proposed = torch.zeros((), device=device, dtype=dtype)

        for _ in range(n_steps):
            candidate, candidate_q = self._draw_from_product_marginals(marginals, n_events)
            candidate_flat = self._flatten_indices(candidate, cell_shape)
            candidate_prob = flat_probs[candidate_flat].clamp_min(
                torch.finfo(dtype).tiny
            )
            candidate_q = candidate_q.clamp_min(torch.finfo(dtype).tiny)

            ratio = ((candidate_prob * current_q) / (current_prob * candidate_q)).clamp_max(1)
            u = torch.rand(n_events, device=device, dtype=dtype)
            accept = u < ratio
            current = torch.where(accept.unsqueeze(-1), candidate, current)
            current_prob = torch.where(accept, candidate_prob, current_prob)
            current_q = torch.where(accept, candidate_q, current_q)

            proposed = proposed + n_events
            accepted = accepted + accept.to(dtype).sum()

        rate = (accepted / proposed.clamp_min(1)).detach().cpu().item()
        return self._indices_to_coordinates(current, grid_list), rate
