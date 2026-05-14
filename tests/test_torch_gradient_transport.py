import torch
from collections import OrderedDict


class TestTorchGradientTransport:
    """
    Base class to define a pytest for gradient transport methods, i.e. all tests regarding including a torch gradient
    transport should inherit from this class.

    Please note that this class is not a unit-test, it just covers a skeleton
    """

    # Define generic function that can be used by other tests that do the same thing:
    def run_gradient_transport(self, setup_fn):
        grad_transport, rank, skip_gradient_test = setup_fn
        # Now get the dummy model and register it on the device that the gradient transport module identified:
        model = TorchDummyModel().to(grad_transport.device, grad_transport.dtype)
        # Get a dummy optimizer:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Synchronize model and optimizer across ranks:
        grad_transport.sync_model(model, opt)
        # Now we are manually setting the model gradients. Usually ,this line would be replace by the loss computation
        # and backpropagation:
        for p in model.parameters():
            if p.requires_grad:
                new_grad = torch.ones_like(p) * rank
                p.grad = new_grad

        # Share gradients across ranks:
        grad_transport.forward(model)

        # Now we run a consistency check: We manually set the gradients such that their value is the current rank
        # If we have n_ranks: 0,1,2,..,N-1 then we expect, after transport, the gradients on each rank to be:
        # (0+1+2+...+N-1)/n_ranks:
        expected_grad_value = (
            sum([r for r in range(grad_transport.n_ranks)]) / grad_transport.n_ranks
        )
        # Make sure the gradients match the expectation:
        if not skip_gradient_test:
            for p in model.parameters():
                if p.requires_grad:
                    g = p.grad
                    g_expected = torch.ones_like(g) * expected_grad_value
                    assert torch.equal(g, g_expected)

        # Clear, just in case we need to free something:
        grad_transport.clear()


class TorchDummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        layers = OrderedDict()
        layers["layer_0"] = torch.nn.Linear(5, 4, bias=False)
        layers["layer_1"] = torch.nn.Linear(4, 3, bias=False)
        layers["layer_2"] = torch.nn.Linear(3, 2, bias=False)

        self.model = torch.nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)
