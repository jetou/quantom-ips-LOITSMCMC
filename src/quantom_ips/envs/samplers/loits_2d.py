import torch
import concurrent.futures
from quantom_ips.utils.stateful_module import StatefulModule
from quantom_ips.utils.loits_utils import (
    empty_cache,
    calculate_norm,
    calculate_weight_tensor,
    calculate_grid_indices,
    linear_interpolation,
)


# Define LOITS for 2D:
class TorchLOITS2D:
    """
    Stand-alone PyTorch implementation of the Local inverse transform sampler (LOITS),
    i.e. no pre-processing of the theory-outputs.
    """

    # Initialize:
    # ***************************
    def __init__(
        self,
        use_threading=False,
        vmap_randomness="different",
        static_weight_tensor=False,
    ):
        self.vmap_randomness = vmap_randomness
        self.static_weight_tensor = static_weight_tensor
        self.use_threading = use_threading
        self.gen_settings = {"x": [1, 0, 0], "y": [0, 1, 1]}

    # ***************************

    # Components to calculate the CDF in x and Q2 coordinates:
    # ***************************
    # Determine the density: rho = xsec / norm
    def calc_rho(self, bins, xsec):
        # First, we need the norm:
        norm = torch.vmap(lambda s: calculate_norm(bins, s), in_dims=0)(xsec)
        # and get rho:
        rho = xsec / norm[:, :, None]

        return rho

    # --------------------------

    # Now compute the CDF itself:
    def calc_cdf(self, bins, xsec):
        rho = self.calc_rho(bins, xsec)

        # Define an empty CDF tensor and overwrite it with the cumulative sum
        cdf = torch.zeros_like(rho)
        cdf[:, :, 1:] = torch.cumulative_trapezoid(y=rho, x=bins)

        return cdf

    # ***************************

    # Generate events, based on the provided x / Q2 grid, the acceptance matrix,
    # weight tensors and grid index tensors:
    # ***************************
    # Define a helper function that generates one variable: (e.g. x or Q2)
    def generate_single_observable(
        self,
        obs_bins,
        obs_xsec,
        weight_tensor,
        grid_index_tensor,
        n_max,
        obs_type,
    ):
        settings = self.gen_settings[obs_type]

        # Compute the CDF:
        cdf_obs = self.calc_cdf(obs_bins, obs_xsec)

        # First generate u:
        u_obs = torch.rand(
            (grid_index_tensor.size()[0], n_max),
            device=weight_tensor.device,
            dtype=weight_tensor.dtype,
        )

        # Use the cdf in obs that we just computed and
        # combine them with the index tensor,
        # i.e. assign the proper values to the grid position:
        cdf_obs_flat = cdf_obs[
            grid_index_tensor[:, settings[0]], grid_index_tensor[:, settings[1]]
        ]

        # Do the same for the bins in x:
        bin_obs_flat = obs_bins[grid_index_tensor[:, settings[2]]]

        # Now utilize the linear interpolation function and compute x:
        # Also, we can now leverage that the indices are flat and run a vectorization:
        obs_gen = (
            torch.vmap(
                linear_interpolation, in_dims=0, randomness=self.vmap_randomness
            )(u_obs, cdf_obs_flat, bin_obs_flat)
            * weight_tensor
        )
        return obs_gen.flatten()[:, None]

    # --------------------------------------------

    def gen_events(
        self,
        x_bins,
        xsec_x,
        y_bins,
        xsec_y,
        weight_tensor,
        grid_index_tensor,
        n_max,
    ):

        x_gen = self.generate_single_observable(
            x_bins,
            xsec_x,
            weight_tensor,
            grid_index_tensor,
            n_max,
            "x",
        )

        y_gen = self.generate_single_observable(
            y_bins,
            xsec_y,
            weight_tensor,
            grid_index_tensor,
            n_max,
            "y",
        )

        return torch.cat([x_gen, y_gen], dim=1)

    # ***************************

    # Sample data:
    # ***************************
    # Sample from single prediction:
    def forward_single_sample(
        self, x_bins, xsec_x, y_bins, xsec_y, weight_tensor, grid_idx, max_n
    ):
        # Generate events:
        evts = self.gen_events(
            x_bins, xsec_x, y_bins, xsec_y, weight_tensor, grid_idx, max_n
        )

        cond = evts[:, 0] * evts[:, 1] != 0
        events = evts[cond]

        cond2 = (~torch.isnan(events[:, 0])) & (~torch.isnan(events[:, 1]))
        events = events[cond2]

        # Detach everything that we do not need --> Free the GPU memory:
        del cond
        del cond2
        del evts

        # Free cache on the device we are using
        empty_cache(str(weight_tensor.device))

        return events

    # --------------------------------------------

    # Formulate the entire forward pass:
    def forward(self, theory_outputs, n_events):
        # Get the information from theory: We are expecting 5 items:
        # i) bins and crossections in x and Q^2 coordinates = 4 variables
        # ii) cross section weights = 1 variable
        x_bins, xsec_x, y_bins, xsec_y, weights = theory_outputs[:5]

        # Compute flat grid-indices which help to avoid
        # another for-loop over the grid itself:
        npy_grid_idx = calculate_grid_indices(
            weights[0].size()[0], weights[0].size()[1]
        )
        torch_grid_idx = torch.as_tensor(
            npy_grid_idx, device=weights.device, dtype=torch.int
        )

        # Get batch size from theory outputs:
        batch_size = xsec_x.size()[0]
        data_list = []
        stats_list = []

        # Generate data, using one thread per prediction,
        # where #events_generated = n_events
        if self.use_threading:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_size
            ) as executor:
                threads = []

                # +++++++++++++++++++++++++++++++
                for i in range(batch_size):
                    n_low = i * n_events
                    n_high = (i + 1) * n_events

                    # Run this on CPU:
                    detached_weights = weights[i].detach().cpu()

                    if self.static_weight_tensor:
                        n = torch.abs(detached_weights * n_high).to(torch.int).reshape(
                            -1, 1
                        ) - torch.abs(detached_weights * n_low).to(torch.int).reshape(
                            -1, 1
                        )
                    else:
                        w_sum = torch.zeros(
                            detached_weights.flatten().shape[0] + 1,
                            device="cpu",
                            dtype=self.dtype,
                        )
                        w_sum[1:] = torch.cumsum(detached_weights.flatten(), 0)
                        u = torch.rand(n_events, device="cpu", dtype=weights.dtype)
                        n = (
                            torch.histogram(u, bins=w_sum)
                            .hist.to(torch.int)
                            .reshape(-1, 1)
                        )
                    max_n = torch.max(n)

                    weight_tensor_cpu = calculate_weight_tensor(n)
                    weight_tensor_gpu = torch.as_tensor(
                        weight_tensor_cpu, device=weights.device, dtype=weights.dtype
                    )

                    threads.append(
                        executor.submit(
                            self.forward_single_sample,
                            x_bins[i],
                            xsec_x[i],
                            y_bins[i],
                            xsec_y[i],
                            weight_tensor_gpu,
                            torch_grid_idx,
                            max_n,
                        )
                    )
                # +++++++++++++++++++++++++++++++

                # +++++++++++++++++++++++++++++++
                for thread in concurrent.futures.as_completed(threads):
                    evts = thread.result()
                    data_list.append(evts)
                    stats_list.append(evts.size()[0])
                # +++++++++++++++++++++++++++++++

        else:
            # Or run everything sequentially and generate n_events for each prediction,
            # where #events_generated = batch_size * n_events
            # +++++++++++++++++++++++++++++++
            for i in range(batch_size):
                # Run this on CPU:
                detached_weights = weights[i].detach().cpu()

                if self.static_weight_tensor:
                    n = (
                        torch.abs(detached_weights * n_events)
                        .to(torch.int)
                        .reshape(-1, 1)
                    )
                else:
                    w_sum = torch.zeros(
                        detached_weights.flatten().shape[0] + 1,
                        device="cpu",
                        dtype=weights.dtype,
                    )
                    w_sum[1:] = torch.cumsum(detached_weights.flatten(), 0)
                    u = torch.rand(n_events, device="cpu", dtype=weights.dtype)
                    n = torch.histogram(u, bins=w_sum).hist.to(torch.int).reshape(-1, 1)

                max_n = torch.max(n)

                weight_tensor_cpu = calculate_weight_tensor(n)
                weight_tensor_gpu = torch.as_tensor(
                    weight_tensor_cpu, device=weights.device, dtype=weights.dtype
                )

                current_evs = self.forward_single_sample(
                    x_bins[i],
                    xsec_x[i],
                    y_bins[i],
                    xsec_y[i],
                    weight_tensor_gpu,
                    torch_grid_idx,
                    max_n,
                )

                data_list.append(current_evs)
                stats_list.append(current_evs.size()[0])
            # +++++++++++++++++++++++++++++++

        # Determine smallest number of events collected:
        n_evts_min = min(stats_list)
        # Make sure that all batches have the same number of events:
        data_list = [d[:n_evts_min] for d in data_list]
        return torch.stack(data_list, dim=0)

    # ***************************


class LOInverseTransformSampler2D(StatefulModule):
    """
    Base implementation of 2D LOITS, including a pre-processing of the theory outputs:
    - Inputs: density, grid_axes
    - Outputs: samples
    """

    # Initialize:
    # *************************
    def __init__(
        self,
        average: bool = True,
        a_min: float = 0.0,
        a_max: float = 1.0,
        n_interpolations_x: int = 10,
        n_interpolations_y: int = 10,
        use_threading: bool = False,
        vmap_randomness: str = "different",
        log_space: bool = False,
        static_weight_tensor: bool = False,
    ):
        super().__init__()

        # Important settings:
        self.average = average
        self.a_min = a_min
        self.a_max = a_max
        self.n_interpolations_x = n_interpolations_x
        self.n_interpolations_y = n_interpolations_y
        self.log_space = log_space

        # Get LOITS:
        self.loits = TorchLOITS2D(
            use_threading,
            vmap_randomness,
            static_weight_tensor,
        )

    # *************************

    # Compute the weights that are passed to LOITS:
    # *************************
    # First, we need a linear interpolation of density A in one dimension:
    def linear_interpolation_1d(self, A, u):
        A_min = A[:-1]
        dA = A[1:] - A_min
        return A_min.reshape(-1, 1) + u * dA.reshape(-1, 1)

    # ----------------------------

    # Run interpolation on density A in both dimensions:
    def compute_weights(self, A):
        # Interpolating A w.r.t x or y:
        u_x = torch.linspace(
            0, 1, self.n_interpolations_x, device=A.device, dtype=A.dtype
        )
        u_y = torch.linspace(
            0, 1, self.n_interpolations_y, device=A.device, dtype=A.dtype
        )
        # Scale A to the desired range:
        A_scaled = (self.a_max - self.a_min) * A + self.a_min

        # Interpolate A w.r.t x:
        avg_A_y = 0.5 * (A_scaled[:, :-1] + A_scaled[:, 1:])
        A_interpol_x = torch.vmap(
            lambda a: self.linear_interpolation_1d(a, u_x), in_dims=1
        )(avg_A_y)

        # Interpolate A w.r.t y:
        avg_A_x = 0.5 * (A_scaled[:-1, :] + A_scaled[1:, :])
        A_interpol_y = torch.vmap(
            lambda a: self.linear_interpolation_1d(a, u_y), in_dims=0
        )(avg_A_x)

        # Now compute the weights:
        weights = torch.nn.functional.conv2d(
            A_scaled[None, None, :, :],
            torch.ones((1, 1, 2, 2), device=A.device, dtype=A.dtype) / 4,
        )
        sum_weights = weights.sum()
        weights = weights.squeeze() / sum_weights

        # Delete what we do not need:
        del avg_A_x, avg_A_y, u_x, u_y

        return (
            weights,
            sum_weights,
            A_interpol_x,
            A_interpol_y,
        )

    # *************************

    # Define the forward pass:
    # *************************
    # Single target:
    def forward_single_target(self, A, grid_axes, n_events):
        # Interpolating A w.r.t x or y:
        u_x = torch.linspace(
            0, 1, self.n_interpolations_x, device=A.device, dtype=A.dtype
        )
        u_y = torch.linspace(
            0, 1, self.n_interpolations_y, device=A.device, dtype=A.dtype
        )
        # Compute bins that match the dimensions of A:

        # x-Bins:
        xv = grid_axes[0]
        xv_min = xv[:-1]
        dxv = xv[1:] - xv_min
        x_bins = xv_min.reshape(-1, 1) + u_x * dxv.reshape(-1, 1)
        if self.log_space:
            x_bins = torch.exp(x_bins)
        x_bins = x_bins[None, :]

        # y-Bins:
        yv = grid_axes[1]
        yv_min = yv[:-1]
        dyv = yv[1:] - yv_min
        y_bins = yv_min.reshape(-1, 1) + u_y * dyv.reshape(-1, 1)
        if self.log_space:
            y_bins = torch.exp(y_bins)
        y_bins = y_bins[None, :]

        # Delete what we do not need:
        del xv, xv_min, dxv, u_x
        del yv, yv_min, dyv, u_y

        # Make sure that the dimensions of A make sense:
        if len(A.size()) == 2:
            A = A[None, :, :]
        # Get the weights and interpolated densities:
        weights, sum_weights, A_interpol_x, A_interpol_y = torch.vmap(
            lambda a: self.compute_weights(a), in_dims=0, randomness="same"
        )(A)

        # Average everything if requested by the user:
        if self.average:
            A_interpol_x = torch.mean(A_interpol_x, 0)[None, :]
            A_interpol_y = torch.mean(A_interpol_y, 0)[None, :]
            weights = torch.mean(weights, 0)[None, :]
        else:
            x_bins = torch.repeat_interleave(x_bins, A.size()[0], dim=0)
            y_bins = torch.repeat_interleave(y_bins, A.size()[0], dim=0)

        # Now generate samples using LOITS:
        theory_outputs = (
            x_bins,
            A_interpol_x,
            y_bins,
            A_interpol_y,
            weights,
            sum_weights,
        )
        return self.loits.forward(theory_outputs, n_events)

    # ----------------------------

    # All targets:
    def forward(self, A, grid_axes, n_events, chosen_targets=None):
        n_targets = A.size()[1]
        events = []
        stats = []

        if chosen_targets is None or len(chosen_targets) == 0:
            chosen_targets = [p for p in range(n_targets)]

        # +++++++++++++++++++++++
        for p in chosen_targets:
            samples = self.forward_single_target(A[:, p, :, :], grid_axes, n_events)
            events.append(samples)
            stats.append(samples.size()[1])
        # +++++++++++++++++++++++
        min_n_events = min(stats)

        # +++++++++++++++++++++++
        for i in range(len(events)):
            ev = events[i]
            idx = torch.randperm(ev.size()[1], device=A.device)[:min_n_events]
            events[i] = ev[:, idx, :]
            del idx
        # +++++++++++++++++++++++

        return torch.stack(events, dim=1)

    # *************************
