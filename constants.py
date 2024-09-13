from numpy import sqrt,pi
from astropy.constants import GM_sun,kpc,c

GM_sun = GM_sun.cgs.value
DKPC = kpc.cgs.value
c = c.cgs.value

CONST_AMPLITUDE_GW = sqrt(5./24.)*GM_sun**(5./6.)/pi**(2./3.)/c**1.5/(1e6*DKPC)

H0_1 = 70. # km/s/Mpc
Om0_1 = 0.3
