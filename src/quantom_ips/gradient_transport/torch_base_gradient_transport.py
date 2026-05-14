from mpi4py import MPI
import numpy as np
import torch
import logging
from quantom_ips.utils.stateful_module import StatefulModule

logger = logging.getLogger(__name__)


class TorchBaseGradientTransport(StatefulModule):
    """
    Base class for implementing a gradient transport mechanism for torch models.
    This class uses a synchronous allreduce to share gradients across multiple ranks.
    Other gradient transport methods should inherit from this class and simply define their own forward function.
    """

    def __init__(self, master_rank: int = 0, dtype: str = "float32"):
        super().__init__()
        # Get dtype and master rank:
        self.master_rank = master_rank
        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        self.dtype = dtypes.get(dtype)
        self.np_dtype = dtype
        assert self.dtype is not None, logger.error(
            f"Provided dtype {dtype} is not supported"
        )

        # Set up communication among different ranks:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_ranks = self.comm.Get_size()

        # Assign device to rank:
        self.assign_device_to_rank()

    # Mechanisms to synchronize a torch model:

    # i) Share the state dictionary:
    def sync_state_dict(self, torch_object, current_rank, current_comm, master):
        if current_rank == master:
            ref_states = torch_object.state_dict()
        else:
            ref_states = None

        ref_states = current_comm.bcast(ref_states, root=master)
        torch_object.load_state_dict(ref_states)

    # ii) Syncronize model weights and optimizer states across all ranks:
    def sync_model(self, model, optimizer):
        self.sync_state_dict(model, self.rank, self.comm, self.master_rank)
        self.sync_state_dict(optimizer, self.rank, self.comm, self.master_rank)

    # Get and set the model gradients:

    # Get gradients:
    def get_model_gradients(self, model):
        model_grad_dict = {}
        # +++++++++++++++++++++++++++++++
        for name, params in model.named_parameters():
            if params.requires_grad and params.grad is not None:
                model_grad_dict[name] = (
                    params.grad.detach().cpu().numpy().astype(self.np_dtype)
                )
        # +++++++++++++++++++++++++++++++

        return model_grad_dict

    # Set gradients:
    def set_model_gradients(self, model, new_gradients, gradient_scale):
        gradients_set = False  # --> Make sure that we actually have new gradients...

        if bool(new_gradients) == True:
            for name, params in model.named_parameters():
                if name in new_gradients:
                    params.grad = torch.as_tensor(
                        new_gradients[name] * gradient_scale,
                        dtype=self.dtype,
                        device=self.torch_device,
                    )
                    gradients_set = True

        return gradients_set

    # Define the forward function:
    def forward(self, model, **kwargs):
        # There is no need to distribute gradients if there is just one rank:
        if self.n_ranks == 1:
            return True, {}

        # Get the gradients:
        gradients = self.get_model_gradients(model)

        # Now share them via an allreduce for every layer
        new_gradients = {}
        for name, grads in gradients.items():
            # Define data that should be send:
            send_data = np.copy(grads)
            # Define receiving data:
            recv_data = np.zeros_like(send_data)
            # Share them accross ranks:
            self.comm.Allreduce(send_data, recv_data, op=MPI.SUM)
            # And register them in the new dictionary:
            new_gradients[name] = recv_data / self.n_ranks

        # Set model with new gradients:
        self.set_model_gradients(
            model, new_gradients, kwargs.get("gradient_scale", 1.0)
        )

        # Return a flag that gradient transport happened, as well as a dictionary that contains
        # whatever information the user finds helpful:
        return True, {}

    # Clean up, if neccesary:
    def clear(self):
        pass

    # Assign the proper hardware to the current rank:
    def assign_device_to_rank(self):
        # CPU:
        dev = "cpu"
        self.device_is_cpu = True
        self.device_is_cuda = False
        self.device_is_mps = False

        self.torch_device_id = self.rank

        # CUDA:
        if torch.cuda.is_available():
            self.torch_device = "cuda"

            self.device_is_cpu = False
            self.device_is_cuda = True
            self.device_is_mps = False

            n_cudas = torch.cuda.device_count()
            accept_rank = False
            while accept_rank == False:

                if self.torch_device_id < n_cudas:
                    accept_rank = True
                else:
                    self.torch_device_id -= n_cudas

            dev = "cuda:" + str(self.torch_device_id)
        # MPS:
        elif torch.mps.is_available():
            dev = "mps"
            self.device_is_cpu = False
            self.device_is_cuda = False
            self.device_is_mps = True

        self.torch_device = torch.device(dev)
        self.device = dev
        logger.info(
            f"{self.rank} uses torch device: {self.torch_device} and is on processor: {MPI.Get_processor_name()}"
        )
