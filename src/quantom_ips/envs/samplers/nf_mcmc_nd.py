import torch
import torch.nn as nn

from quantom_ips.envs.samplers.inverse_transform_sampler import InverseTransformSampler
from quantom_ips.envs.samplers.mcmc_loits_nd import MCMCLOITSND
from quantom_ips.utils.stateful_module import StatefulModule


class _CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * dim),
        )

    def _shift_scale(self, x):
        shift, log_scale = self.net(x * self.mask).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 2.0
        shift = shift * (1.0 - self.mask)
        log_scale = log_scale * (1.0 - self.mask)
        return shift, log_scale

    def forward(self, x):
        shift, log_scale = self._shift_scale(x)
        y = x * self.mask + (1.0 - self.mask) * (x * torch.exp(log_scale) + shift)
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        shift, log_scale = self._shift_scale(y)
        x = y * self.mask + (1.0 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        log_det = -log_scale.sum(dim=-1)
        return x, log_det


class _RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim=64, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            mask = torch.tensor([(j + i) % 2 for j in range(dim)], dtype=torch.float32)
            layers.append(_CouplingLayer(dim, hidden_dim, mask))
        self.layers = nn.ModuleList(layers)
        self.dim = dim

    def _base_log_prob(self, z):
        return -0.5 * (z.pow(2) + torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))).sum(dim=-1)

    def _unit_to_unconstrained(self, y, eps=1e-6):
        y = y.clamp(eps, 1.0 - eps)
        return torch.log(y) - torch.log1p(-y), y

    def _unconstrained_to_unit(self, u):
        return torch.sigmoid(u).clamp(1e-6, 1.0 - 1e-6)

    def forward(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        x = z
        for layer in self.layers:
            x, det = layer(x)
            log_det = log_det + det
        return x, log_det

    def inverse(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for layer in reversed(self.layers):
            z, det = layer.inverse(z)
            log_det = log_det + det
        return z, log_det

    def log_prob(self, y):
        u, y = self._unit_to_unconstrained(y)
        z, log_det_inv = self.inverse(u)
        log_sigmoid_jac = torch.log(y) + torch.log1p(-y)
        return self._base_log_prob(z) + log_det_inv - log_sigmoid_jac.sum(dim=-1)

    def sample(self, n, device, dtype):
        z = torch.randn(n, self.dim, device=device, dtype=dtype)
        u, log_det = self.forward(z)
        y = self._unconstrained_to_unit(u)
        log_sigmoid_jac = torch.log(y) + torch.log1p(-y)
        log_prob = self._base_log_prob(z) - log_det - log_sigmoid_jac.sum(dim=-1)
        return y, log_prob


class NFMCMCND(StatefulModule):
    """
    Evaluation-only normalizing-flow proposal sampler with independence-MH correction.

    The flow is trained on target-density samples from ITS, then proposals from
    the flow are corrected with Metropolis-Hastings using the original grid
    density. It is not differentiable and is intended for offline comparison.
    """

    def __init__(
        self,
        burn_in: int = 16,
        thin: int = 4,
        flow_layers: int = 4,
        hidden_dim: int = 64,
        train_steps: int = 300,
        train_samples: int = 4096,
        batch_size: int = 512,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.burn_in = burn_in
        self.thin = thin
        self.flow_layers = flow_layers
        self.hidden_dim = hidden_dim
        self.train_steps = train_steps
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.lr = lr
        self.last_acceptance_rate = None
        self._its = InverseTransformSampler()
        self._cache = {}

    def forward(self, A, grid_list, n_events):
        grid_ndim = len(grid_list)
        outputs = []
        rates = []
        for batch_index, A_batch in enumerate(A):
            out, rate = self.particle_forward(
                A_batch, grid_list, n_events, grid_ndim, cache_prefix=(A.data_ptr(), batch_index)
            )
            outputs.append(out)
            rates.append(rate)
        self.last_acceptance_rate = float(sum(rates) / max(len(rates), 1))
        return torch.stack(outputs)

    def particle_forward(self, A, grid_list, n_events, grid_ndim, cache_prefix):
        if A.ndim == grid_ndim:
            return self.sample(A, grid_list, n_events, cache_key=cache_prefix)

        if A.ndim == grid_ndim + 1:
            outputs = []
            rates = []
            for particle_index, A_particle in enumerate(A):
                out, rate = self.sample(
                    A_particle,
                    grid_list,
                    n_events,
                    cache_key=(*cache_prefix, particle_index),
                )
                outputs.append(out)
                rates.append(rate)
            return torch.stack(outputs), float(sum(rates) / max(len(rates), 1))

        raise ValueError(f"NFMCMCND got A.ndim={A.ndim}, expected {grid_ndim} or {grid_ndim + 1}")

    def _grid_to_device(self, grid_list, device, dtype):
        return [grid.to(device=device, dtype=dtype) for grid in grid_list]

    def _to_unit(self, samples, grid_list):
        vals = []
        for dim, grid in enumerate(grid_list):
            denom = (grid[-1] - grid[0]).clamp_min(torch.finfo(samples.dtype).tiny)
            vals.append((samples[..., dim] - grid[0]) / denom)
        return torch.stack(vals, dim=-1).clamp(1e-6, 1.0 - 1e-6)

    def _from_unit(self, unit_samples, grid_list):
        vals = []
        for dim, grid in enumerate(grid_list):
            vals.append(grid[0] + unit_samples[..., dim] * (grid[-1] - grid[0]))
        return torch.stack(vals, dim=-1)

    def _cell_log_densities(self, A, grid_list):
        helper = MCMCLOITSND()
        cell_probs = helper._cell_probabilities(A)
        cell_volume = torch.ones_like(cell_probs)
        for dim, grid in enumerate(grid_list):
            shape = [1] * cell_probs.ndim
            shape[dim] = grid.numel() - 1
            widths = torch.diff(grid).reshape(shape).clamp_min(torch.finfo(A.dtype).tiny)
            cell_volume = cell_volume * widths
        return torch.log(cell_probs.clamp_min(torch.finfo(A.dtype).tiny)) - torch.log(cell_volume)

    def _flatten_indices(self, indices, shape):
        flat = torch.zeros(indices.shape[0], device=indices.device, dtype=torch.long)
        stride = 1
        for dim in reversed(range(len(shape))):
            flat = flat + indices[:, dim] * stride
            stride *= shape[dim]
        return flat

    def _target_log_prob(self, samples, grid_list, flat_log_density, cell_shape):
        indices = []
        for dim, grid in enumerate(grid_list):
            idx = torch.searchsorted(grid, samples[:, dim].contiguous(), right=True) - 1
            idx = idx.clamp(0, cell_shape[dim] - 1)
            indices.append(idx)
        indices = torch.stack(indices, dim=-1)
        return flat_log_density[self._flatten_indices(indices, cell_shape)]

    def _train_flow(self, A, grid_list):
        device, dtype = A.device, A.dtype
        dim = len(grid_list)
        flow = _RealNVP(dim, hidden_dim=int(self.hidden_dim), n_layers=int(self.flow_layers)).to(device=device, dtype=dtype)

        n_train = max(int(self.train_samples), int(self.batch_size))
        with torch.no_grad():
            target_samples = self._its.sample(A, grid_list, n_train)
            target_unit = self._to_unit(target_samples, grid_list)

        optimizer = torch.optim.Adam(flow.parameters(), lr=float(self.lr))
        batch_size = min(int(self.batch_size), n_train)
        for _ in range(max(0, int(self.train_steps))):
            idx = torch.randint(0, n_train, (batch_size,), device=device)
            batch = target_unit[idx]
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        flow.eval()
        return flow

    def _get_flow(self, A, grid_list, cache_key):
        key = (cache_key, tuple(A.shape), A.device.type, str(A.dtype), int(self.train_steps), int(self.train_samples))
        if key not in self._cache:
            self._cache[key] = self._train_flow(A, grid_list)
        return self._cache[key]

    def sample(self, A, grid_list, n_events, cache_key=None):
        if A.ndim != len(grid_list):
            raise ValueError(f"NFMCMCND density ndim {A.ndim} does not match {len(grid_list)} grid axes")

        device, dtype = A.device, A.dtype
        grid_list = self._grid_to_device(grid_list, device, dtype)
        flow = self._get_flow(A, grid_list, cache_key or A.data_ptr())
        flat_log_density = self._cell_log_densities(A, grid_list).reshape(-1)
        cell_shape = tuple(max(grid.numel() - 1, 1) for grid in grid_list)

        with torch.no_grad():
            current_unit, current_log_q = flow.sample(n_events, device=device, dtype=dtype)
            current = self._from_unit(current_unit, grid_list)
            current_log_p = self._target_log_prob(current, grid_list, flat_log_density, cell_shape)

            accepted = torch.zeros((), device=device, dtype=dtype)
            proposed = torch.zeros((), device=device, dtype=dtype)
            n_steps = max(0, int(self.burn_in)) + max(1, int(self.thin))

            for _ in range(n_steps):
                proposal_unit, proposal_log_q = flow.sample(n_events, device=device, dtype=dtype)
                proposal = self._from_unit(proposal_unit, grid_list)
                proposal_log_p = self._target_log_prob(proposal, grid_list, flat_log_density, cell_shape)
                log_alpha = proposal_log_p + current_log_q - current_log_p - proposal_log_q
                accept = torch.log(torch.rand(n_events, device=device, dtype=dtype)) < log_alpha.clamp_max(0.0)

                current = torch.where(accept.unsqueeze(-1), proposal, current)
                current_unit = torch.where(accept.unsqueeze(-1), proposal_unit, current_unit)
                current_log_p = torch.where(accept, proposal_log_p, current_log_p)
                current_log_q = torch.where(accept, proposal_log_q, current_log_q)
                accepted = accepted + accept.to(dtype).sum()
                proposed = proposed + n_events

        rate = (accepted / proposed.clamp_min(1)).detach().cpu().item()
        return current, rate
