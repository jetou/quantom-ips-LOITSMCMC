import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI
import quantom_ips.envs.parsers.distributed_numpy_parser
from quantom_ips.utils import make
from omegaconf import OmegaConf
import numpy as np
import os


class TestDistributedNumpyParser:

    # Create test data:
    def create_test_data(self, n_events, dim, test_data_loc, current_rank):
        if current_rank == 0:
            test_data = np.random.normal(loc=0.0, scale=1.0, size=(n_events, dim))
            np.save(test_data_loc + ".npy", test_data)

    # Test the parser
    def test_distributed_numpy_paser(
        self,
        data_fraction=0.1,
        batch_size=10,
        n_targets=2,
        n_events=50000,
        dim=5,
        n_data_samples=1000,
        test_data_loc="example_data",
    ):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # Create the test data:
        self.create_test_data(n_events, dim, test_data_loc, comm.Get_rank())
        # Setup the parser:
        cfg = {
            "path": [test_data_loc + ".npy"],
            "event_axis": 0,
            "data_fraction": data_fraction,
            "data_size": None,
            "master_rank": 0,
        }
        cfg = OmegaConf.create(cfg)
        parser = make("DistributedNumpyParser", config=cfg, mpi_comm=comm)
        comm.Barrier()

        # Get the data:
        data = parser.data
        expected_data_size = int(data_fraction * n_events)
        # Get samples from the data:
        samples = parser.get_samples((batch_size, n_targets, n_data_samples, dim))

        print(
            f"Rank: {rank}, expected data size: {expected_data_size}, actual data size: {data.shape[0]}"
        )

        comm.Barrier()
        if rank == 0:
            os.remove(test_data_loc + ".npy")

        assert n_data_samples == samples.shape[2]
        assert batch_size == samples.shape[0]
        assert n_targets == samples.shape[1]
        assert dim == samples.shape[3]
