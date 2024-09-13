import numpy as np
from math import gamma
def ZofD(cosmos, D):
    # D is in unit of MpC
    Zs=np.linspace(0,30,100)
    Ds=cosmos.luminosity_distance(Zs).value
    return np.interp(D, Ds, Zs)


def mathcalI(alpha, beta, s):
    return -s**2*gamma(0.5*alpha+1+2/s)*gamma(-beta/s+1-2/s)/((alpha+2)*(beta+2)*s*gamma(-(-alpha+beta)/s))

