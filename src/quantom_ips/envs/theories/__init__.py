from quantom_ips import register

register(
    id="IdentityTheory",
    entry_point="quantom_ips.envs.theories.identity:IdentityTheory",
    group="env/theory",
)

register(
    id="JAMXTheory",
    entry_point="quantom_ips.envs.theories.jamx_theory:JAMXTheory",
    group="theory",
)
