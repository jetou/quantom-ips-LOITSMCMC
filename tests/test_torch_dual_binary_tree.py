import pytest

mpi4py = pytest.importorskip("mpi4py")
from quantom_ips.gradient_transport.torch_dual_binary_tree import (
    TorchDualBinaryTree,
)
from tests.test_torch_gradient_transport import TestTorchGradientTransport


# Define different test setups, that we can use for testing different settings:
# Settings for single binary tree:
@pytest.fixture
def setup_single_binary_tree_test():
    gradient_transport = TorchDualBinaryTree(rank_threshold=20)
    rank = gradient_transport.rank
    yield gradient_transport, rank, False


# Settings for dual binary tree:
@pytest.fixture
def setup_dual_binary_tree_test():
    gradient_transport = TorchDualBinaryTree(rank_threshold=1)
    rank = gradient_transport.rank
    yield gradient_transport, rank, False


class TestTorchDualBinary(TestTorchGradientTransport):

    # Test single binary tree:
    def test_torch_single_tree(self, setup_single_binary_tree_test):
        self.run_gradient_transport(setup_single_binary_tree_test)

    # Test dual binary tree:
    def test_torch_dual_binary_tree(self, setup_dual_binary_tree_test):
        self.run_gradient_transport(setup_dual_binary_tree_test)
