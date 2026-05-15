import numpy as np
from scipy.integrate import quad
from numba import jit
from . import params as par


@jit(nopython=True)
def get_Nf(Q2):
    """
    Multiply two numbers.

    :param Q2: The scale
    :return: The number of flavors
    """
    Nf=3
    if Q2>=(par.mc2): Nf+=1
    if Q2>=(par.mb2): Nf+=1
    return Nf

@jit(nopython=True)
def beta_func(a,Nf,order):
    beta0=11.0-2.0/3.0*Nf 
    beta1=102.-38.0/3.0*Nf 
    beta2=2857.0/2.0-5033.0/18.0*Nf+325.0/54.0*Nf**2 
    betaf = -beta0
    if order>=1: betaf+=-a*beta1
    if order>=2: betaf+=-a*beta2
    return betaf*a**2

@jit(nopython=True)
def evolve_a(Q20,a,Q2,Nf,order):
    # Runge-Kutta 
    LR = np.log(Q2/Q20)/20.0
    for k in range(20):
        XK0 = LR * beta_func(a,Nf,order)
        XK1 = LR * beta_func(a + 0.5 * XK0,Nf,order)
        XK2 = LR * beta_func(a + 0.5 * XK1,Nf,order)
        XK3 = LR * beta_func(a + XK2,Nf,order)
        a+= (XK0 + 2.* XK1 + 2.* XK2 + XK3) * 0.166666666666666
    return a

order=par.alphaS_order
mu20=par.alphaS_mu20

aZ=par.alphaSMZ/(4*np.pi)
ab=evolve_a(par.mZ2,aZ,par.mb2,5,order)
ac=evolve_a(par.mb2,ab,par.mc2,4,order)
a0=evolve_a(par.mc2,ac,mu20,3,order)

@jit(nopython=True)
def get_a(Q2,order):
    if par.mb2<=Q2:
        return evolve_a(par.mb2,ab,Q2,5,order)
    elif par.mc2<=Q2 and Q2<par.mb2: 
        return evolve_a(par.mc2,ac,Q2,4,order)
    elif Q2<par.mc2:
        return evolve_a(mu20,a0,Q2,3,order)

@jit(nopython=True)
def get_alphaS(Q2):
    return get_a(Q2,order)*4*np.pi






