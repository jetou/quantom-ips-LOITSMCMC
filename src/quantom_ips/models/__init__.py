from quantom_ips import register

register(id="MLP", entry_point="quantom_ips.models.mlp:MLP", group="model")

register(
    id="MLPDiscriminator1D",
    entry_point="quantom_ips.models.mlp:MLP",
    group="model",
    kwargs=dict(out_activation="Sigmoid"),
)

register(
    id="MLPDiscriminator2D",
    entry_point="quantom_ips.models.mlp:MLP",
    group="model",
    kwargs=dict(input_dim=2, out_activation="Sigmoid"),
)

register(
    id="GaussGeneratorV1",
    entry_point="quantom_ips.models.mlp:MLP",
    group="model",
    kwargs=dict(input_dim=10, output_dim=2),
)


register(
    id="Conv2DTGenerator",
    entry_point="quantom_ips.models.conv2d_generator:Conv2DTGenerator",
    group="model",
)

register(
    id="FSQ_VAE",
    entry_point="quantom_ips.models.fsq_vae:FSQ_VAE",
    group="model",
    defaults=[
        {"/model@encoder": "MLPPointCloud"},
        {"/model@decoder": "Conv2DTGenerator"},
        "_self_",
    ],
)

register(
    id="MLPPointCloud",
    entry_point="quantom_ips.models.mlp_pointcloud:MLPPointCloud",
    group="model",
)

register(
    id="Parameter",
    entry_point="quantom_ips.models.parameter_model:ParameterModel",
    group="model",
)
