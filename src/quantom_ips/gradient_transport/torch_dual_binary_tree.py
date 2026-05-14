from mpi4py import MPI
import numpy as np
import logging
from quantom_ips.gradient_transport.torch_base_gradient_transport import (
    TorchBaseGradientTransport,
)

logger = logging.getLogger(__name__)


class TorchDualBinaryTree(TorchBaseGradientTransport):
    """
    Dual binary tree implementation, following the approach presented here:
    https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/

    Parts of the code segments here where create with Claude
    """

    # Initialize:
    def __init__(
        self,
        master_rank: int = 0,
        second_master_rank: int = 1,
        rank_threshold: int = 4,
        dtype: str = "float32",
    ):
        super().__init__(master_rank, dtype)

        # Setup the trees:

        # Flag for using a dual / single tree:
        self.use_dual_tree = self.n_ranks >= rank_threshold
        self.tree_1 = self.build_binary_tree(master_rank)
        self.tree_2 = None
        if self.use_dual_tree:
            logging.info("Using dual binary trees")
            self.tree_2 = self.build_binary_tree(second_master_rank)
        else:
            logging.info("Using single binary tree")

    # Build the binary tree:
    def build_binary_tree(self, offset):
        tree = {"parent": None, "left_child": None, "right_child": None}
        # Shifft reference by offset:
        logical_rank = (self.rank - offset) % self.n_ranks

        # Calculate parent and children in binary tree
        if logical_rank > 0:
            parent_logical = (logical_rank - 1) // 2
            tree["parent"] = (parent_logical + offset) % self.n_ranks

        left_logical = 2 * logical_rank + 1
        if left_logical < self.n_ranks:
            tree["left_child"] = (left_logical + offset) % self.n_ranks

        right_logical = 2 * logical_rank + 2
        if right_logical < self.n_ranks:
            tree["right_child"] = (right_logical + offset) % self.n_ranks

        return tree

    # Reduce scatter gradients: Up-stream: Current tree node receives from children and sends to above parent:
    def reduce_scatter_tree(self, data_chunk, tree, tag_base):
        acc = np.copy(data_chunk)
        requests = []
        recv_buffers = []

        # Receive data from children:
        for child in [tree["left_child"], tree["right_child"]]:
            if child is not None:
                child_data = np.empty_like(data_chunk)
                recv_buffers.append(child_data)
                req = self.comm.Irecv(child_data, source=child, tag=tag_base)
                requests.append(req)

        # Wait for all receives and reduce:
        if requests:
            MPI.Request.Waitall(requests)
            for child_data in recv_buffers:
                acc += child_data

        # Send data to parent node:
        send_req = None
        if tree["parent"] is not None:
            send_req = self.comm.Isend(acc.copy(), dest=tree["parent"], tag=tag_base)

        return acc, send_req

    # Allgather gradients: Down-stream: Current tree node receives data from parent and sends data to children
    def allgather_tree(self, data_chunk, tree, tag_base):
        acc = np.empty_like(data_chunk)
        recv_req = None
        send_reqs = []

        # Non-blocking receive from parent:
        if tree["parent"] is not None:
            recv_req = self.comm.Irecv(acc, source=tree["parent"], tag=tag_base + 1000)
        else:
            # Root node already has the result
            acc[:] = data_chunk

        # Wait for receive if needed
        if recv_req is not None:
            recv_req.Wait()

        # Non-blocking sends to children
        for child in [tree["left_child"], tree["right_child"]]:
            if child is not None:
                req = self.comm.Isend(acc.copy(), dest=child, tag=tag_base + 1000)
                send_reqs.append(req)

        return acc, send_reqs

    # Allreduce on a single data chunk:
    def allreduce_chunk(self, chunk, tree_idx, chunk_idx):
        if self.use_dual_tree:
            tree = self.tree_1 if tree_idx == 0 else self.tree_2
        else:
            tree = self.tree_1

        tag_base = chunk_idx * 10000 + tree_idx * 1000

        # Reduce-scatter phase: Up-stream
        reduced, send_req = self.reduce_scatter_tree(chunk, tree, tag_base)
        if send_req is not None:
            send_req.Wait()

        # Allgather phase: Down-stream
        result, send_reqs = self.allgather_tree(reduced, tree, tag_base)
        if send_reqs:
            MPI.Request.Waitall(send_reqs)

        return result

    # Run a full allreduce for a single tensor, i.e. combine all data chunks:
    def allreduce_single_tensor(self, data, average=True):
        # Get original shape:
        orig_shape = data.shape
        # Get data chunks:
        data_chunks = np.array_split(np.copy(data).ravel(), self.n_ranks, 0)
        num_chunks = len(data_chunks)
        # Copy data so that the original chunks are not altered
        final_chunks = list(data_chunks)

        for i in range(num_chunks):
            data_chunk = data_chunks[i]
            tree_idx = (i % 2) if self.use_dual_tree else 0
            reduced_chunk = self.allreduce_chunk(data_chunk, tree_idx, i)

            if average:
                reduced_chunk = reduced_chunk / self.n_ranks
            final_chunks[i] = reduced_chunk

        return np.concatenate(final_chunks).reshape(orig_shape)

    # Full forward pass:
    def forward(self, model, **kwargs):
        # There is no need to distribute gradients if there is just one rank:
        if self.n_ranks == 1:
            return True, {}

        # Get the gradients:
        gradients = self.get_model_gradients(model)

        # Now share them via an allreduce for every layer
        new_gradients = {}
        for name, grads in gradients.items():
            new_gradients[name] = self.allreduce_single_tensor(grads)

        # Set model with new gradients:
        self.set_model_gradients(
            model, new_gradients, kwargs.get("gradient_scale", 1.0)
        )

        # Return a flag that gradient transport happened, as well as a dictionary that contains
        # whatever information the user finds helpful:
        return True, {}
