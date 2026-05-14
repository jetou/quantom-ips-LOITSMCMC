import pytest
import torch
import numpy
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import os
from quantom_ips.utils.pixelated_densities import ProxyApplication2DDensities
from quantom_ips.envs.theories.identity import IdentityTheory
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.utils import make


class TestLOITS2D:

    # Create density:
    def create_density(self, n_points_x, n_points_y, batch_size, n_targets, devices):
        density_generator = ProxyApplication2DDensities(
            devices=devices, batch_size=batch_size, n_targets=n_targets
        )
        return density_generator.get_pixelated_density(n_points_x, n_points_y)

    def test_loits(
        self, n_events, batch_size, n_targets, grid_size, result_loc_loits_2d
    ):
        torch_device = "cpu"
        if torch.cuda.is_available():
            torch_device = "cuda"
        if torch.backends.mps.is_available():
            torch_device = "mps"

        os.makedirs(result_loc_loits_2d, exist_ok=True)

        print("Load identity theory")
        # Load identity theory:
        t_cfg = {
            "grid_minimums": [0.0, 0.0],
            "grid_maximums": [1.0, 1.0],
            "average": False,
        }
        t_cfg = OmegaConf.create(t_cfg)
        theory = make("IdentityTheory", config=t_cfg, devices=torch_device)

        print("Get density and pass it through the theory module")
        # Get density:
        density = self.create_density(
            grid_size, grid_size, batch_size, n_targets, torch_device
        )
        # Get response from theory module:
        A, grid_axes, _ = theory.forward(density)

        # Get LOITS:
        print("Load LOITS")
        s_cfg = {
            "use_threading": False,
            "vmap_randomness": "different",
            "static_weight_tensor": False,
            "average": False,
            "a_min": 0.0,
            "a_max": 1.0,
            "log_space": False,
            "n_interpolations_x": 10,
            "n_interpolations_y": 10,
        }
        s_cfg = OmegaConf.create(s_cfg)
        sampler = make(
            "LOInverseTransformSampler2D", config=s_cfg, devices=torch_device
        )

        # Sample events:
        print("Sample events")
        events = sampler.forward(A, grid_axes, n_events)

        # Run a consistency check:
        assert events.size()[0] == batch_size
        assert events.size()[1] == n_targets
        assert events.size()[2] <= n_events
        assert events.size()[3] == 2

        # Plot events:
        events = events.detach().cpu().flatten(0, 2).numpy()

        plt.rcParams.update({"font.size": 20})

        fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        fig.suptitle(f"{n_events*batch_size*n_targets} Events generated with 2D LOITS")

        ax[0].hist(events[:, 0], 100)
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("Entries")
        ax[0].grid(True)

        ax[1].hist(events[:, 1], 100)
        ax[1].set_xlabel("y")
        ax[1].grid(True)

        fig.savefig(result_loc_loits_2d + "/generated_events.png")
        plt.close(fig)
