import torch
from quantom_ips.utils.stateful_module import StatefulModule


class MCMCLOITS2D(StatefulModule):
    """
    Evaluation-only 2D MCMC-LOITS sampler.

    This sampler matches the standard sampler contract used by QuantomEnv:
    input densities have shape (batch, targets, nx, ny), and the output has
    shape (batch, targets, n_events, 2). It intentionally does not promise a
    differentiable path through the MCMC accept/reject step.
    """

    def __init__(
        self,
        n_marginal_points: int = 32,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.n_marginal_points = n_marginal_points
        self.eps = eps
        self.last_stats = {}

    def forward(self, A, grid_axes, n_events, chosen_targets=None):
        if A.ndim != 4:
            raise ValueError("MCMCLOITS2D expects A with shape (batch, targets, nx, ny)")
        if len(grid_axes) != 2:
            raise ValueError("MCMCLOITS2D expects two grid axes")

        with torch.no_grad():
            A = torch.clamp(A.detach(), min=0)
            x_axis = grid_axes[0].to(device=A.device, dtype=A.dtype)
            y_axis = grid_axes[1].to(device=A.device, dtype=A.dtype)

            if x_axis.numel() < 2 or y_axis.numel() < 2:
                raise ValueError("MCMCLOITS2D requires at least two grid points per axis")

            n_targets = A.size(1)
            if chosen_targets is None:
                chosen_targets = range(n_targets)

            all_events = []
            total_accepted = 0
            total_attempted = 0

            for target_idx in chosen_targets:
                target_events = []
                for batch_idx in range(A.size(0)):
                    samples, accepted, attempted = self.sample_single_density(
                        A[batch_idx, target_idx], x_axis, y_axis, n_events
                    )
                    target_events.append(samples)
                    total_accepted += accepted
                    total_attempted += attempted
                all_events.append(torch.stack(target_events, dim=0))

            acceptance_rate = (
                float(total_accepted / total_attempted) if total_attempted > 0 else 0.0
            )
            self.last_stats = {
                "accepted": int(total_accepted),
                "attempted": int(total_attempted),
                "acceptance_rate": acceptance_rate,
            }

            return torch.stack(all_events, dim=1)

    def sample_single_density(self, density, x_axis, y_axis, n_events):
        cell_weights = self.compute_cell_weights(density, x_axis, y_axis)
        cell_counts = torch.multinomial(
            cell_weights.flatten(), n_events, replacement=True
        ).bincount(minlength=cell_weights.numel())
        cell_counts = cell_counts.reshape_as(cell_weights)

        samples = []
        accepted_total = 0
        attempted_total = 0

        for ix in range(cell_weights.size(0)):
            for iy in range(cell_weights.size(1)):
                count = int(cell_counts[ix, iy].item())
                if count == 0:
                    continue

                cell_samples, accepted, attempted = self.sample_cell(
                    density, x_axis, y_axis, ix, iy, count
                )
                samples.append(cell_samples)
                accepted_total += accepted
                attempted_total += attempted

        if not samples:
            empty = torch.empty((0, 2), device=density.device, dtype=density.dtype)
            return empty, accepted_total, attempted_total

        samples = torch.cat(samples, dim=0)
        if samples.size(0) > n_events:
            samples = samples[:n_events]
        return samples, accepted_total, attempted_total

    def compute_cell_weights(self, density, x_axis, y_axis):
        dx = x_axis[1:] - x_axis[:-1]
        dy = y_axis[1:] - y_axis[:-1]
        avg_density = 0.25 * (
            density[:-1, :-1]
            + density[1:, :-1]
            + density[:-1, 1:]
            + density[1:, 1:]
        )
        weights = torch.clamp(avg_density, min=0) * dx[:, None] * dy[None, :]
        total = weights.sum()
        if total <= self.eps:
            weights = torch.ones_like(weights)
            total = weights.sum()
        return weights / total

    def sample_cell(self, density, x_axis, y_axis, ix, iy, count):
        x0 = x_axis[ix]
        x1 = x_axis[ix + 1]
        y0 = y_axis[iy]
        y1 = y_axis[iy + 1]

        x_points = torch.linspace(
            x0, x1, self.n_marginal_points, device=density.device, dtype=density.dtype
        )
        y_points = torch.linspace(
            y0, y1, self.n_marginal_points, device=density.device, dtype=density.dtype
        )
        X, Y = torch.meshgrid(x_points, y_points, indexing="ij")
        local_density = self.bilinear_density(density, x_axis, y_axis, ix, iy, X, Y)

        marginal_x = torch.trapezoid(local_density, y_points, dim=1)
        marginal_y = torch.trapezoid(local_density, x_points, dim=0)

        current_x = self.inverse_cdf_sample(x_points, marginal_x, 1)[0]
        current_y = self.inverse_cdf_sample(y_points, marginal_y, 1)[0]
        current_p = self.bilinear_density(
            density, x_axis, y_axis, ix, iy, current_x, current_y
        )
        current_q = self.proposal_density(
            x_points, marginal_x, current_x
        ) * self.proposal_density(y_points, marginal_y, current_y)

        samples = torch.empty((count, 2), device=density.device, dtype=density.dtype)
        accepted = 0

        for sample_idx in range(count):
            proposal_x = self.inverse_cdf_sample(x_points, marginal_x, 1)[0]
            proposal_y = self.inverse_cdf_sample(y_points, marginal_y, 1)[0]
            proposal_p = self.bilinear_density(
                density, x_axis, y_axis, ix, iy, proposal_x, proposal_y
            )
            proposal_q = self.proposal_density(
                x_points, marginal_x, proposal_x
            ) * self.proposal_density(y_points, marginal_y, proposal_y)

            denom = torch.clamp(current_p * proposal_q, min=self.eps)
            ratio = torch.clamp((proposal_p * current_q) / denom, min=0.0, max=1.0)

            if torch.rand((), device=density.device, dtype=density.dtype) < ratio:
                current_x = proposal_x
                current_y = proposal_y
                current_p = proposal_p
                current_q = proposal_q
                accepted += 1

            samples[sample_idx, 0] = current_x
            samples[sample_idx, 1] = current_y

        return samples, accepted, count

    def bilinear_density(self, density, x_axis, y_axis, ix, iy, x, y):
        x0 = x_axis[ix]
        x1 = x_axis[ix + 1]
        y0 = y_axis[iy]
        y1 = y_axis[iy + 1]
        tx = torch.clamp((x - x0) / torch.clamp(x1 - x0, min=self.eps), 0.0, 1.0)
        ty = torch.clamp((y - y0) / torch.clamp(y1 - y0, min=self.eps), 0.0, 1.0)

        f00 = density[ix, iy]
        f10 = density[ix + 1, iy]
        f01 = density[ix, iy + 1]
        f11 = density[ix + 1, iy + 1]

        return torch.clamp(
            (1 - tx) * (1 - ty) * f00
            + tx * (1 - ty) * f10
            + (1 - tx) * ty * f01
            + tx * ty * f11,
            min=0,
        )

    def inverse_cdf_sample(self, points, pdf, n_samples):
        pdf = torch.clamp(pdf, min=0)
        segment_mass = 0.5 * (pdf[:-1] + pdf[1:]) * (points[1:] - points[:-1])
        total = segment_mass.sum()

        if total <= self.eps:
            u = torch.rand(n_samples, device=points.device, dtype=points.dtype)
            return points[0] + u * (points[-1] - points[0])

        cdf = torch.zeros_like(points)
        cdf[1:] = torch.cumsum(segment_mass, dim=0)
        cdf = cdf / cdf[-1]

        u = torch.rand(n_samples, device=points.device, dtype=points.dtype)
        idx = torch.searchsorted(cdf[1:-1], u, right=True)
        cdf_left = cdf[idx]
        cdf_right = cdf[idx + 1]
        denom = torch.clamp(cdf_right - cdf_left, min=self.eps)
        frac = (u - cdf_left) / denom
        return points[idx] + frac * (points[idx + 1] - points[idx])

    def proposal_density(self, points, pdf, value):
        pdf = torch.clamp(pdf, min=0)
        total = torch.trapezoid(pdf, points)
        if total <= self.eps:
            return torch.ones_like(value) / torch.clamp(points[-1] - points[0], min=self.eps)

        normalized_pdf = pdf / total
        idx = torch.searchsorted(points[1:-1], value, right=True)
        left = points[idx]
        right = points[idx + 1]
        frac = (value - left) / torch.clamp(right - left, min=self.eps)
        return torch.clamp(
            normalized_pdf[idx] + frac * (normalized_pdf[idx + 1] - normalized_pdf[idx]),
            min=self.eps,
        )
