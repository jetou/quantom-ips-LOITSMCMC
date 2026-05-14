import torch
from collections import OrderedDict

from quantom_ips.utils import torch_nn_registry
import logging
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# TODO: Add activation to layer definition so you can check
# matching activation/initialization for layers


# Define model
class TorchSequentialModel(torch.nn.Module):
    def __init__(self, layers, torch_device, dtype):
        super(TorchSequentialModel, self).__init__()
        layer_dict = OrderedDict()
        layer_ordering = {
            layer.index: layer_name for layer_name, layer in layers.items()
        }
        for i in range(len(layer_ordering)):
            layer_name = layer_ordering.get(i)
            layer = layers[layer_name]
            logger.debug(f"Adding layer {layer_name} at index {i}")

            if layer.get("config") is None:
                torch_layer = torch_nn_registry.get_layer(layer["class"])
            else:
                layer_config = OmegaConf.to_container(layer["config"])
                logger.debug(f"Layer has configuration {layer_config}")
                torch_layer = torch_nn_registry.get_layer(
                    layer["class"], **layer_config
                )

            if layer.get("weight_init") is not None:
                torch_nn_registry.get_initializer(
                    layer["weight_init"], torch_layer.weight
                )

            if layer.get("bias_init") is not None:
                torch_nn_registry.get_initializer(layer["bias_init"], torch_layer.bias)

            layer_dict[layer_name] = torch_layer

            if layer.get("activation") is not None:
                activation = layer["activation"]
                if activation.get("config") is None:
                    torch_act = torch_nn_registry.get_activation(activation["class"])
                else:
                    act_config = OmegaConf.to_container(activation["config"])
                    torch_act = torch_nn_registry.get_activation(
                        activation["class"], **act_config
                    )
                layer_dict[layer_name + "_activation"] = torch_act

        self.model = torch.nn.Sequential(layer_dict).to(
            device=torch_device, dtype=dtype
        )

    def forward(self, x):
        out = self.model(x)
        return out
