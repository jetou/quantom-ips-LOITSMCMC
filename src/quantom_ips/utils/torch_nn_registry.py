import torch
import quantom_ips.utils.torch_custom_layers as tcl

# TODO: Add logging and (possibly) a decorator function
#       to add custom layers/losses/initializers

torch_dtypes = {
    "half": torch.half,
    "float16": torch.float16,
    "float": torch.float,
    "float32": torch.float32,
    "double": torch.double,
    "float64": torch.float64,
    "int": torch.int,
    "int32": torch.int32,
    "long": torch.long,
    "int64": torch.int64,
    "bool": torch.bool,
}


def get_dtype(name: str):
    dtype = torch_dtypes.get(name)
    if not dtype:
        raise ValueError(f"No torch dtype available for {name}.")
    return dtype


torch_activations = {
    "Tanh": torch.nn.Tanh,
    "Sigmoid": torch.nn.Sigmoid,
    "ReLU": torch.nn.ReLU,
    "LeakyReLU": torch.nn.LeakyReLU,
}

torch_layers = {
    "Linear": torch.nn.Linear,
    "Conv1d": torch.nn.Conv1d,
    "Conv2d": torch.nn.Conv2d,
    "Dropout": torch.nn.Dropout,
    "ConvTranspose2d": torch.nn.ConvTranspose2d,
    "ConvTranspose1d": torch.nn.ConvTranspose1d,
    "Flatten": torch.nn.Flatten,
    "Unflatten": torch.nn.Unflatten,
    "Upsample": torch.nn.Upsample,
    "ConvTranspose2dStack": tcl.ConvTranspose2dStack,
    "ConvTranspose1dStack": tcl.ConvTranspose1dStack,
    "PixelShuffle": torch.nn.PixelShuffle,
    "MaxPool2d": torch.nn.MaxPool2d,
}

torch_layers.update(torch_activations)


def get_layer(name, *args, **kwargs):
    return torch_layers[name](*args, **kwargs)


def get_activation(name, *args, **kwargs):
    return torch_activations[name](*args, **kwargs)


torch_initializers = {
    "normal": torch.nn.init.normal_,
    "uniform": torch.nn.init.uniform_,
    "zeros": torch.nn.init.zeros_,
    "ones": torch.nn.init.ones_,
    "kaiming_normal": torch.nn.init.kaiming_normal_,
    "kaiming_uniform": torch.nn.init.kaiming_uniform_,
    "xavier_normal": torch.nn.init.xavier_normal_,
    "xavier_uniform": torch.nn.init.xavier_uniform_,
}


def get_initializer(name, *args, **kwargs):
    return torch_initializers[name](*args, **kwargs)


torch_losses = {
    "MSE": torch.nn.MSELoss,
    "BCE": torch.nn.BCELoss,
}


def get_loss(name, *args, **kwargs):
    return torch_losses[name](*args, **kwargs)


torch_optimizers = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "AdamW": torch.optim.AdamW,
}


def get_optimizer(name, *args, **kwargs):
    return torch_optimizers[name](*args, **kwargs)
