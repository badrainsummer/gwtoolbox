import numpy as np
from numpy import pi, cos, sin, nan
cimport cython

# From std/LALConstants.h

#define PI        3.141592653589793238462643383279502884
#define PI_4      0.785398163397448309615660845819875721
#define GAMMA     0.577215664901532860606512090082402431

# Gravitational constant, N m^2 kg^-2
cdef double G_SI = 6.67384e-11
# Speed of light in vacuo, m s^-1
cdef double C_SI = 299792458e0
# Parsec, m
cdef double PC_SI = 3.085677581491367278913937957796471611e16
# Sidereal year (2000), s
# see http://hpiers.obspm.fr/eop-pc/models/constants.html
cdef double YRSID_SI = 31558149.763545600
# Solar mass, kg
# MSUN_SI = LAL_GMSUN_SI / LAL_G_SI
cdef double MSUN_SI = 1.988546954961461467461011951140572744e30
# Geometrized solar mass, s
# MTSUN_SI = LAL_GMSUN_SI / (LAL_C_SI * LAL_C_SI * LAL_C_SI)
cdef double MTSUN_SI = 4.925491025543575903411922162094833998e-6

# m in solar masses, dist in pc, tc, t in s
@cython.cdivision(True)
cdef double hphcInsp2PN(double m, double eta, double tc, double phi0, double dist, double inc, double t):
    cdef double deltat, theta, Momega, prefactor, cinc, hp, hc
    deltat = tc - t
    if deltat<0:
        print "Error in hphcInsp2PN: must have t<tc."
        return nan
    theta = eta/(5*m*MTSUN_SI) * deltat
    Momega = 1./8 * theta**(-3./8) * (1. + (743./2688 + 11./32*eta) * theta**(-1./4) + (-3./10*pi) * theta**(-3./8) + (1855099./14450688 + 56975./258048*eta + 371./2048*(eta**2)) * theta**(-1./2))
    phi = phi0 - 1./(32*eta) * Momega**(-5./3) * (1 + (3715./1008 + 55./12*eta) * Momega**(2./3) + (-10.*pi) * Momega + (15293365./1016064 + 27145./1008*eta + 3085./144*(eta**2)) * Momega**(4./3) )
    prefactor = 2*G_SI*m*MSUN_SI*eta/(dist*PC_SI*C_SI**2) * (Momega)**(2./3)
    cinc = cos(inc)
    hp = prefactor * (1 + cinc*cinc) * cos(2*phi)
    hc = prefactor * 2*cinc * sin(2*phi)
    return hp, hc
