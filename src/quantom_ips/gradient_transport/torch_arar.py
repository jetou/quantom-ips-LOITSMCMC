from mpi4py import MPI
import numpy as np
import logging
from quantom_ips.gradient_transport.torch_base_gradient_transport import (
    TorchBaseGradientTransport,
)

logger = logging.getLogger(__name__)


class TorchARAR(TorchBaseGradientTransport):
    """
    Torch Asynchronous Ring All Reduce with or without grouping. The grouping mechansim can be varied between weak (default) and strong.
    Additionally, one might also run conventional asynchronous ring all reduce.
    """

    # Initialize:
    def __init__(
        self,
        master_rank: int = 0,
        group_size: int = 4,
        gradient_sync_mode: str = "arar",
        train_as_ensemble: bool = False,
        disable_rma_rank_synchronization: bool = False,
        dtype: str = "float32",
    ):
        super().__init__(master_rank, dtype)
        # Collect basic information:
        group_size = group_size
        gradient_sync_mode = gradient_sync_mode
        self.train_as_ensemble = train_as_ensemble
        self.disable_rma_rank_synchronization = disable_rma_rank_synchronization

        # Handle grouping:
        self.use_grouping = False
        self.use_rma = False
        self.rma_win_inner = None
        self.gradient_sync_mode = None
        self.group_size = -1
        self.neighbours = None

        # Inner group communication:
        self.inner_comm = None
        self.inner_rank = -1
        self.n_inner_ranks = -1
        self.inner_neighbours = None

        # Outer group communication:
        self.outer_comm = None
        self.outer_rank = -1
        self.n_outer_ranks = -1
        self.outer_neighbours = None

        # Bundle ranks into groups:
        self.group_ranks(gradient_sync_mode, group_size, 4)

        if self.use_grouping == True:
            logger.info(
                f"Inner Rank: {self.inner_rank} and Outer Rank {self.outer_rank} use torch device: {self.torch_device} and is on processor: {MPI.Get_processor_name()}"
            )

    # Handle the grouping:
    # Get the left and right "neighbour" of the current rank:
    def get_neighbours(self, current_rank, num_ranks):
        left = ((current_rank - 1) + num_ranks) % num_ranks
        right = (current_rank + 1) % num_ranks

        return left, right

    # ---------------------------

    # Group ranks:
    def group_ranks(self, gradient_sync_mode, group_size, default_group_size):
        # Determine the gradient synchronization mode and if grouping is active or not:
        # Modes with grouping:
        if (
            gradient_sync_mode.lower() == "arar"
            or gradient_sync_mode.lower() == "rma_arar"
            or gradient_sync_mode.lower() == "strong_arar"
        ):
            self.use_grouping = True
            self.gradient_sync_mode = gradient_sync_mode.lower()
            if self.gradient_sync_mode == "rma_arar":
                self.use_rma = True

            # Make sure that group size is properly set:
            if group_size < 1:
                logger.warning(
                    f"The group size you provided {group_size} is < 1. Going to set it to default: {default_group_size}"
                )
                self.group_size = default_group_size
            else:
                logger.info(f"Running with group size: {group_size}")
                self.group_size = group_size

            # Inner:
            self.inner_comm = self.comm.Split(color=self.rank / self.group_size)
            self.inner_rank = self.inner_comm.Get_rank()
            self.n_inner_ranks = self.inner_comm.Get_size()
            # Get inner group neighbours for (RMA) ring_allreduce:
            self.inner_neighbours = self.get_neighbours(
                self.inner_rank, self.n_inner_ranks
            )

            # Outer:

            # Just pick one GPU from each node for outer communication:
            if gradient_sync_mode.lower() != "strong_arar":
                logger.info("Using weak coupling for outer communication")
                outer_rank_list = [
                    0
                ]  # --> We just collect rank 0 from all inner groups:
                f = 1
                # ++++++++++++++++
                for r in range(self.n_ranks):
                    if r == f * self.group_size:
                        outer_rank_list.append(r)
                        f += 1
                # ++++++++++++++++

                outer_group = self.comm.group.Incl(outer_rank_list)
                self.outer_comm = self.comm.Create_group(outer_group)
                self.outer_rank = outer_group.Get_rank()
                self.n_outer_ranks = outer_group.Get_size()
                # Get outer group neighbours for ring_allreduce:
                self.outer_neighbours = self.get_neighbours(
                    self.outer_rank, self.n_outer_ranks
                )

            else:
                logger.info("Using strong coupling for outer communication")
                # Do a full ring-allreduce across all Nodes:
                self.outer_comm = self.comm
                self.outer_rank = self.rank
                self.n_outer_ranks = self.n_ranks
                # Get outer group neighbours for ring_allreduce:
                self.outer_neighbours = self.get_neighbours(
                    self.outer_rank, self.n_outer_ranks
                )

        # Mode without grouping:
        elif gradient_sync_mode.lower() == "conv_arar":
            self.use_grouping = False
            # Get overall neighbours for ring_allreduce:
            self.neighbours = self.get_neighbours(self.rank, self.n_ranks)

        else:
            logger.warning(
                f"The gradient synchronization mode you provided {gradient_sync_mode} is not implemented. Please check your settings"
            )

    # Overwrite the model sync, in case we have to deal with an ensemble:
    def sync_model(self, model, optimizer):
        # Distributed the optimizer / model states accross all ranks:

        # Training as ensemble on the outer group, i.e. ensemble accross nodes:
        # This means we need to synchronie the model/optimizer accross members of the inner group only.
        if self.train_as_ensemble:
            super().sync_state_dict(
                model, self.inner_rank, self.inner_comm, self.master_rank
            )
            super().sync_state_dict(
                optimizer, self.inner_rank, self.inner_comm, self.master_rank
            )
        else:  # Training with gradient transfer accross all ranks:
            super().sync_model(model, optimizer)

        # Now that we have the weights, we can determine the number of trainable model parameters, that we need for the RMA window:
        if self.use_rma == True and self.use_grouping == True:
            n_total_params = 0
            for name, params in model.named_parameters():
                if params.requires_grad:
                    n_total_params += np.prod(params.size())

            self.rma_win_inner = MPI.Win.Allocate(
                n_total_params * MPI.DOUBLE.Get_size(), comm=self.inner_comm
            )

    # Define 'conventional' ring-allreduce for a single tensor: Taken from: https://pytorch.org/tutorials/intermediate/dist_tuto.html#
    def async_ring_allreduce_single_tensor(
        self, data, mpi_comm, num_ranks, neighbours, op_is_sum=False
    ):
        send_data = np.copy(data)
        recv_data = np.empty_like(data)
        accum_data = np.copy(data)

        # +++++++++++++++++++++++
        for _ in range(num_ranks - 1):
            # Receive data from "left" neighbour:
            recv_req = mpi_comm.Irecv(recv_data, neighbours[0])
            # Send data to "right" neighbour:
            send_req = mpi_comm.Isend(send_data, neighbours[1])

            # Make sure we locally receive the incoming data and accumulate it:
            recv_req.wait()
            accum_data[:] += recv_data

            # Once data is sent away, we update the send data to what we received:
            send_req.wait()
            send_data[:] = recv_data
        # +++++++++++++++++++++++

        if op_is_sum:
            return accum_data

        return accum_data / num_ranks

    # Apply it to the entire model:
    def async_ring_allreduce(self, currrent_gradients, mpi_comm, num_ranks, neighbours):
        new_gradients = {}
        # +++++++++++++++++++++++
        for key in currrent_gradients:
            new_gradients[key] = self.async_ring_allreduce_single_tensor(
                currrent_gradients[key], mpi_comm, num_ranks, neighbours
            )
        # +++++++++++++++++++++++

        return new_gradients

    # Define ring all-reduce with RMA:
    def rma_ring_allreduce(
        self,
        current_gradients,
        mpi_comm,
        n_comm_cycles,
        current_rank,
        prev_rank,
        rma_win_dict,
    ):
        # Preparations for gradient transfer:
        send_grad_data = {}
        accum = {}
        for key in current_gradients:
            send_grad_data[key] = (
                current_gradients[key] / n_comm_cycles
            )  # --> Normalize the gradients here, so that we just need to add them later on
            accum[key] = current_gradients[key] / n_comm_cycles

        # Now transfer gradients:
        for _ in range(n_comm_cycles - 1):
            # Dump gradients into memory:
            rma_win_dict.Lock(rank=current_rank)
            for key in send_grad_data:
                rma_win_dict.Put(send_grad_data[key], target_rank=current_rank)

            rma_win_dict.Unlock(rank=current_rank)

            # Wait for everyone to finish:
            if not self.disable_rma_rank_synchronization:
                mpi_comm.Barrier()

            # Receive gradients and accumulate them:
            rma_win_dict.Lock(rank=prev_rank)
            for key in current_gradients:
                recv_grad = np.zeros(current_gradients[key].shape, dtype=np.float32)
                rma_win_dict.Get(recv_grad, target_rank=prev_rank)

                if np.isfinite(np.sum(recv_grad)) == True:
                    accum[key] += recv_grad
                    send_grad_data[key] = recv_grad

            rma_win_dict.Unlock(rank=prev_rank)

        return accum

    # Redefine the forward function:
    def forward(self, model, **kwargs):
        # There is no need to distribute gradients if there is just one rank:
        if self.n_ranks == 1:
            return True, {}

        # First, we need to get the gradients from the model:
        gradients = super().get_model_gradients(model)
        synced_grads = None

        # If the model is trained as ensemble on the outer ranks, then no outer grou communication will happen:
        outer_group_communication_active = kwargs.get(
            "use_outer_group_communication", False
        )
        if self.train_as_ensemble:
            outer_group_communication_active = False

        # Use grouping:
        if self.use_grouping == True:

            # Inner group updates, i.e. ranks on the same node:
            if outer_group_communication_active == False:

                # Check if RMA is requested:
                if self.use_rma == True:
                    try:
                        synced_grads = self.rma_ring_allreduce(
                            gradients,
                            self.inner_comm,
                            self.n_inner_ranks,
                            self.inner_rank,
                            self.inner_neighbours[0],
                            self.rma_win_inner,
                        )
                    except:
                        if self.rma_win_inner is None:
                            logging.error(
                                ">>> RMA window not defined! Please make sure that you run sync_model() first, before calling this function <<<"
                            )

                # "Regular" ring all reduce
                else:
                    synced_grads = self.async_ring_allreduce(
                        gradients,
                        self.inner_comm,
                        self.n_inner_ranks,
                        self.inner_neighbours,
                    )

            # Run outer group update, i.e. accross nodes:
            else:
                if self.n_outer_ranks > 1 and self.outer_rank >= 0:
                    synced_grads = self.async_ring_allreduce(
                        gradients,
                        self.outer_comm,
                        self.n_outer_ranks,
                        self.outer_neighbours,
                    )

        # No grouping, i.e. conventional ARAR:
        else:
            synced_grads = self.async_ring_allreduce(
                gradients, self.comm, self.n_ranks, self.neighbours
            )

        # Computing gradients is expensive, so if there are no updated gradients (because the current rank is NOT part of the sync)
        # we simply use the original ones
        if synced_grads is None:
            self.set_model_gradients(
                model, gradients, gradient_scale=kwargs.get("gradient_scale", 1.0)
            )
            return False, {}

        self.set_model_gradients(
            model, synced_grads, gradient_scale=kwargs.get("gradient_scale", 1.0)
        )
        return True, {}

    def clear(self):
        if self.rma_win_inner is not None:
            self.rma_win_inner.Free()
