from quantom_ips import register

from . import samplers, theories


register(
    id="GaussianEnv",
    entry_point="quantom_ips.envs.gaussian_env:GaussianEnv",
    group="env",
)

register(
    id="QuantomEnv",
    entry_point="quantom_ips.envs.quantom_env:QuantomEnv",
    group="env",
)
