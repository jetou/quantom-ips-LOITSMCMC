import pytest

mpi4py = pytest.importorskip("mpi4py")
from quantom_ips.gradient_transport.torch_chunk_arar import (
    TorchChunkARAR,
)
from tests.test_torch_gradient_transport import TestTorchGradientTransport


# Define different test setups, that we can use for testing different settings:
# Settings for conventional ARAR:
@pytest.fixture
def setup_conv_arar_test():
    gradient_transport = TorchChunkARAR(gradient_sync_mode="conv_arar")
    rank = gradient_transport.rank
    yield gradient_transport, rank, False


# Settings for ARAR with grouping and weak coupling:
@pytest.fixture
def setup_arar_test():
    gradient_transport = TorchChunkARAR(gradient_sync_mode="arar", group_size=20)
    rank = gradient_transport.rank
    yield gradient_transport, rank, False


# Settings for ARAR with grouping and strong coupling:
@pytest.fixture
def setup_strong_arar_test():
    gradient_transport = TorchChunkARAR(gradient_sync_mode="strong_arar", group_size=20)
    rank = gradient_transport.rank
    yield gradient_transport, rank, False


# Settings for ARAR with RMA:
@pytest.fixture
def setup_rma_arar_test():
    gradient_transport = TorchChunkARAR(gradient_sync_mode="rma_arar", group_size=20)
    rank = gradient_transport.rank
    yield gradient_transport, rank, False


class TestTorchChunkARAR(TestTorchGradientTransport):

    # Test conv. ARAR
    def test_torch_conv_arar(self, setup_conv_arar_test):
        self.run_gradient_transport(setup_conv_arar_test)

    # Test ARAR (with weak coupling)
    def test_torch_arar(self, setup_arar_test):
        self.run_gradient_transport(setup_arar_test)

    # Test ARAR with strong coupling
    def test_torch_strong_arar(self, setup_strong_arar_test):
        self.run_gradient_transport(setup_strong_arar_test)

    # Test ARAR with RMA:
    def test_torch_rma_arar(self, setup_rma_arar_test):
        self.run_gradient_transport(setup_rma_arar_test)
