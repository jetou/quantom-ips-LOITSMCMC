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

register(
    id="SIDIS",
    entry_point="quantom_ips.envs.theories.sidis_theory:SIDIS",
    group="env/theory",
)

register(
    id="SIDIS_masked",
    entry_point="quantom_ips.envs.theories.sidis_theory_masked:SIDIS",
    group="env/theory",
)

register(
    id="SIDIS_masked_cc",
    entry_point="quantom_ips.envs.theories.sidis_theory_masked_cc:SIDIS",
    group="env/theory",
)

register(
    id="SIDISFreeTMD",
    entry_point="quantom_ips.envs.theories.sidis_theory:SIDISFreeTMD",
    group="env/theory",
)
