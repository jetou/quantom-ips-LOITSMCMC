import pytest

mpi4py = pytest.importorskip("mpi4py")
from quantom_ips.gradient_transport.torch_base_gradient_transport import (
    TorchBaseGradientTransport,
)
from tests.test_torch_gradient_transport import TestTorchGradientTransport


# Define a set up function:
@pytest.fixture
def setup_test():
    # Set up the transport module:
    grad_transport = TorchBaseGradientTransport()
    rank = grad_transport.rank
    yield grad_transport, rank, False


class TestTorchBaseGradientTransport(TestTorchGradientTransport):
    """
    This class tests the basic gradient allreduce.
    """

    def test_base_gradient_transport(self, setup_test):
        self.run_gradient_transport(setup_test)
