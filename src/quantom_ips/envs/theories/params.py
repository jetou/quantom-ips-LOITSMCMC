import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PDG masses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#: charm mass (GeV)
mc   = 1.28
#: bottom mass (GeV)
mb   = 4.18
#: top quark mass (GeV)
mt   = 172.9
#: Z boson mass (GeV)
mZ   = 91.1876
mc2  = mc**2  
mb2  = mb**2
mt2  = mt**2
mZ2  = mZ**2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gauge couplings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#: alpha_em 
Alpha_EM = 1/137
#: alphaS at mZ 
alphaSMZ = 0.118


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# alphaS running setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#: alphaS running order: 0=LL, 1=NLL
alphaS_order=1
#: lowest value of mu20 (useful for fitting alphaS at input scale)
alphaS_mu20=1



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SU3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#: fundamental representation
CF = 4/3
#: adjoint representation
CA = 3
#: tbd
TF = 1/2
#: QCD number of colors
NC = 3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Units and conversion factors: By default, everything is in GeV
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GeV = 1
MeV = 1e-3 * GeV
TeV = 1e+3 * GeV

Pico_Barn = 1. / (0.3894e+9) * GeV**(-2)
Femto_Barn = 1e-3 * Pico_Barn
Nano_Barn = 1e+3 * Pico_Barn

M_proton = 0.938272 * GeV
M_pion = 0.135 * GeV

# pion decay constant
DC_pion = 0.130 * GeV


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Elementary particle properties: By default, everything is in GeV
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Q_u = 2/3
Q_d = -1/3

Q_e = -1
Q_p = 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mathematical constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
zeta2 = np.pi**2 / 6
zeta3 = 1.2020569031
zeta4 = np.pi**4 / 90
