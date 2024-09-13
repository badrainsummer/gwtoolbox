#!/usr/bin/env python3

""" This script checks basic cosmological conversions and correspondance of pdfs """

import sys, re
import numpy
import math
import scipy.integrate as integrate
import scipy.optimize as optim
import LISAConstants as LC


#H0 = 73.0 # Mpc^{-1} km/sec
#Omegam  = 0.25  #  Omega matter
#Omegalam = 0.75  # Omega lambda


# values used by Emanuele
#H0 = 70.0 # Mpc^{-1} km/sec
#Omegam  = 0.28  #  Omega matter
#Omegalam = 0.72  # Omega lambda

# Planck 2015 data, let's udopt it here.
Omegam = 0.3175
Omegalam = 0.6825
H0 = 67.1
H0_SI = H0 * 1000 / (1e6*LC.pc)


def H(zp, w):
    fn = 1./(H0*math.sqrt( Omegam*math.pow(1.+zp, 3.) + Omegalam*math.pow(1.+zp, 3.*w) ))
    return(fn)


## computes DL(z, w) for LamdaCDM use w=0
def DL(zup, w):
   pd =  integrate.quad(H, 0., zup, args=(w))[0]
   res = (1.+zup) * pd  # in Mpc
   return res*LC.clight*1.e-3, pd*LC.clight*1.e-3



# f-n needed for finding z for given DL, w
def findz(zm, dlum, ww):
   dofzm = DL(zm, ww)
   return ( dlum - dofzm[0])


# computes z(DL, w), Assumes DL in Mpc
def zofDl(DL, w, tolerance):
   if (tolerance > 1.e-4):
      tolerance = 1.e-4
   zguess = DL/6.6e3
   zres =  optim.fsolve(findz, zguess, args=(DL, 0.0),  xtol=tolerance)
   return (zres)



"""
Emanuele

 if (SNRins>0.)
     {
   DLins = DL*SNRins/10.; // Horizon distance for inspiral
   DLroot DLinsp(nout, xg, wg, OmM, OmL, h, DLins);
   if (zbrac(DLinsp,z1,z2)) {
     //    cout << z1 << endl;
     //    cout << z2 << endl;
     zins = zriddr(DLinsp,z1,z2,1.e-15);
     Mins = Mz / (1.+zins);
     /*
       cout << "Ins: "
       << setw(18) << Mins
       << setw(18) << DLins/Mpc
       << setw(18) << zins
       << endl;
     */
   }
   else {
     throw("Can't bracket the root to find the initial frequency");
   };
     } else {
     DLmerge = zIMR = 0.;
     MIMR = Mz;
   }

"""
