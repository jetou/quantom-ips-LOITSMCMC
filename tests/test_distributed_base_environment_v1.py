import pytest

mpi4py = pytest.importorskip("mpi4py")
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import List, Any
from quantom_ips.envs.distributed_base_environment_v1 import (
    DistributedBaseEnvironmentV1,
)
from quantom_ips.envs.parsers.distributed_numpy_parser import DistributedNumpyParser
from quantom_ips.envs.theories.identity import IdentityTheory
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.envs.samplers.loits_1d import LOInverseTransformSampler1D
from quantom_ips.envs.sample_transformers.identity import IdentitySampleTransformer
from quantom_ips.envs.objectives.mlp_discriminator import MLPDiscriminator
import torch
from quantom_ips.utils.registration import make
from hydra import initialize, compose

# Defaults:
defaults = [
    {"environment": "distributed_base"},
    "_self_",
]


# Data class:
@dataclass
class DistributedEnvDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    environment: Any = MISSING


# Register with the config store:
cs = ConfigStore.instance()
cs.store(name="dist_env_defaults", node=DistributedEnvDefaults)


@pytest.fixture
def hydra_cfg_2d():
    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name="dist_env_defaults",
            overrides=[
                f"environment.parser.path={['../../../sample_data/events_2d_proxy_app_v0.npy']}",
                "environment.sampler.id=LOInverseTransformSampler2D",
                f"environment.theory.average={False}",
                f"environment.sampler.average={False}",
            ],
        )
    yield cfg


@pytest.fixture
def hydra_cfg_1d():
    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name="dist_env_defaults",
            overrides=[
                f"environment.parser.path={['../../../sample_data/events_1d_proxy_app_v0.npy']}",
                "environment.sampler.id=LOInverseTransformSampler1D",
                f"environment.theory.average={False}",
                f"environment.sampler.average={False}",
                f"environment.objective.layers.Linear_1.config.in_features={1}",
            ],
        )
    yield cfg


class TestDistributedBaseEnvironmentV1:

    # Test the 2D environment:
    def test_env_2d(
        self,
        hydra_cfg_2d,
        batch_size,
        n_targets,
        grid_size,
    ):

        torch_device = "cpu"
        if torch.cuda.is_available():
            torch_device = "cuda"
        if torch.backends.mps.is_available():
            torch_device = "mps"

        comm = MPI.COMM_WORLD

        env = make(
            "DistributedBaseEnvironmentV1",
            config=hydra_cfg_2d.environment,
            mpi_comm=comm,
            devices=torch_device,
        )

        # produce a single prediction:
        params = (
            torch.rand(
                size=(batch_size, n_targets, grid_size, grid_size),
                device=torch_device,
                dtype=torch.float32,
            )
            * 0.08
        )

        # Run a single step:
        response = env.step(params)

        # Make sure that the response has two entries:
        assert len(response) == 2
        # Now test that the generated (and filtered) events have the proper shape:
        assert batch_size == response[1].size()[0]
        assert n_targets == response[1].size()[1]
        assert response[1].size()[2] <= hydra_cfg_2d.environment.n_samples
        assert 2 == response[1].size()[3]

    # Test the 1D environment:
    def test_env_1d(self, hydra_cfg_1d, batch_size, n_targets, grid_size):

        torch_device = "cpu"
        if torch.cuda.is_available():
            torch_device = "cuda"
        if torch.backends.mps.is_available():
            torch_device = "mps"

        comm = MPI.COMM_WORLD

        env = make(
            "DistributedBaseEnvironmentV1",
            config=hydra_cfg_1d.environment,
            mpi_comm=comm,
            devices=torch_device,
        )

        # produce a single prediction:
        params = (
            torch.rand(
                size=(batch_size, n_targets, grid_size),
                device=torch_device,
                dtype=torch.float32,
            )
            * 0.08
        )

        # Run a single step:
        response = env.step(params)

        # Make sure that the response has two entries:
        assert len(response) == 2
        # Now test that the generated (and filtered) events have the proper shape:
        assert batch_size == response[1].size()[0]
        assert n_targets == response[1].size()[1]
        assert response[1].size()[2] <= hydra_cfg_1d.environment.n_samples
        assert 1 == response[1].size()[3]
