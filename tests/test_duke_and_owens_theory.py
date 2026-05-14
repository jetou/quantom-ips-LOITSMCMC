import pytest
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from quantom_ips.utils.pixelated_densities import ProxyApplication2DDensities
from quantom_ips.envs.theories.duke_and_owens_theory import DukeAndOwensTheory
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.utils import make


class TestDukeAndOwensTheory:
    # Create known density:
    def create_density(self, n_points_x, n_points_Q2, batch_size, devices, dtype):
        return torch.rand(
            size=(batch_size, 2, n_points_x, n_points_Q2), device=devices, dtype=dtype
        )

    # Plot events:
    def plot_events(
        self, evts, target, output_loc, grid_size, n_bins=100, ref_data=None
    ):
        idx = -1
        if target.lower() == "proton" or target.lower() == "p":
            idx = 0
        if target.lower() == "neutron" or target.lower() == "n":
            idx = 1

        evts_flat = np.log(
            torch.flatten(evts[:, idx, :, :], start_dim=0, end_dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        fig, ax = plt.subplots(1, 3, figsize=(18, 8))
        fig.subplots_adjust(wspace=0.5)
        fig.suptitle(
            f"{evts_flat.shape[0]} Events sampled from {target} X-Section, with Grid Size: {grid_size}"
        )

        ax[0].hist(evts_flat[:, 0], n_bins)
        ax[0].set_xlabel("Sampled " + r"$\log(x)$")
        ax[0].set_ylabel("Entries")
        ax[0].grid(True)

        ax[1].hist(evts_flat[:, 1], n_bins)
        ax[1].set_xlabel("Sampled " + r"$\log(Q^2)$")
        ax[1].set_ylabel("Entries")
        ax[1].grid(True)

        if ref_data is not None:
            ax[0].hist(
                np.log(ref_data[:, 0]),
                n_bins,
                histtype="step",
                color="red",
                linewidth=3.0,
                label="Reference",
            )
            ax[1].hist(
                np.log(ref_data[:, 1]),
                n_bins,
                histtype="step",
                color="red",
                linewidth=3.0,
                label="Reference",
            )
            ax[0].legend(fontsize=15)
            ax[1].legend(fontsize=15)

        ax[2].hist2d(evts_flat[:, 0], evts_flat[:, 1], n_bins, norm=LogNorm())
        ax[2].set_xlabel("Sampled " + r"$\log(x)$")
        ax[2].set_ylabel("Sampled " + r"$\log(Q^2)$")

        fig.savefig(f"{output_loc}/q2_vs_x_{target}.png")
        plt.close(fig)

    # Test theory together with a sampler:
    def test_theory_with_loits(self, n_events, batch_size, grid_size):
        torch_device = "cpu"
        if torch.cuda.is_available():
            torch_device = "cuda"
        if torch.backends.mps.is_available():
            torch_device = "mps"

        print("Load Duke and Owens theory")
        # Load Duke and Owens theory:
        t_cfg = {
            "n_points_x": grid_size,
            "n_points_Q2": grid_size,
            "a_min": 0.0,
            "a_max": 1.0,
            "fixed_density_index": 2,
            "average": False,
        }
        t_cfg = OmegaConf.create(t_cfg)
        theory = make("DukeAndOwensTheory", config=t_cfg, devices=torch_device)

        print("Get density and pass it through the theory module")
        # Get density:
        density = self.create_density(
            grid_size, grid_size, batch_size, torch_device, torch.float32
        )

        # Get response from theory module:
        A, grid_axes, _ = theory.forward(density)

        # Get LOITS:
        print("Load LOITS")
        s_cfg = {
            "use_threading": False,
            "vmap_randomness": "different",
            "average": False,
            "a_min": 0.0,
            "a_max": 1.0,
            "log_space": True,
            "n_interpolations_x": 10,
            "n_interpolations_y": 10,
        }
        s_cfg = OmegaConf.create(s_cfg)
        sampler = make(
            "LOInverseTransformSampler2D", config=s_cfg, devices=torch_device
        )
        # Sample events:
        print("Sample events with 2D lOITS")
        events = sampler.forward(A, grid_axes, n_events)

        # Run a consistency check:
        print("Run consistency check")
        assert events.size()[0] == batch_size
        assert events.size()[1] == 2
        assert events.size()[2] <= n_events
        assert events.size()[3] == 2

        # Plot results for visual inspection:
        print("Provide visual feedback")
        result_loc = "results_test_duke_and_owens"
        os.makedirs(result_loc, exist_ok=True)

        plt.rcParams.update({"font.size": 20})
        self.plot_events(events, "proton", result_loc, grid_size)
        self.plot_events(events, "neutron", result_loc, grid_size)
