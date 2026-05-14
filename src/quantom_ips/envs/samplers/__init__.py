from quantom_ips import register

base_entry_point = "quantom_ips.envs.samplers"
register(
    id="ITS",
    entry_point=base_entry_point + ".inverse_transform_sampler:InverseTransformSampler",
    group="env/sampler",
)

register(
    id="LOITS2D",
    entry_point=base_entry_point + ".loits_2d:LOInverseTransformSampler2D",
    group="env/sampler",
)

register(
    id="MCMCLOITS2D",
    entry_point=base_entry_point + ".mcmc_loits_2d:MCMCLOITS2D",
    group="env/sampler",
)

register(
    id="MVGS",
    entry_point=base_entry_point + ".multivariate_gaussian_sampler:MVGSampler",
    group="env/sampler",
)
