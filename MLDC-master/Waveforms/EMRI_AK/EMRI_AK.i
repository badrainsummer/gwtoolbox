// EMRI_AK.i, required by swig
// requires GSL

%module EMRI_AK
%{
#include "AKWaveform.hh"
%}

%include numpy_typemaps.i
%typemap(in) (double *hPlus ,long hPlusLength ) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *hCross,long hCrossLength) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *tim,long Nt) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *et,long Net) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *nut,long Nnut) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *phit,long Nphit) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *alpt,long Nalpt) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *gamt,long Ngamt) = (double *numpyarray,long numpyarraysize);


%include "cpointer.i"
%pointer_class(double, doublep);

%include "AKWaveform.hh"

%pythoncode %{
import numpy
import math

import sys



class AK:
    """Returns a MLDC source that models the waveform emitted by the
    extreme mass ratio nspiral."""

    # this is the list of parameters that will be recorded in the lisaXML SourceData sections
    # give ParameterName, DefaultUnit, DefaultValue (a string), Description

    # outputlist = (('EclipticLatitude',                 'Radian',        None, 'standard ecliptic latitude'),
    #               ('EclipticLongitude',                'Radian',        None, 'standard ecliptic longitude'),
    #               ('Polarization',                     'Radian',        0,    'standard source polarization (degenerate for EMRIs)'),
    #               ('PolarAngleOfSpin',                 'Radian',        None, 'polar angle of MBH spin'),
    #               ('AzimuthalAngleOfSpin',             'Radian',        None, 'azimuthal direction of MBH spin'),
    #               ('Spin',                             'MassSquared',   None, 'magnitude of (specific) spin (S) of MBH'),
    #               ('MassOfCompactObject',              'SolarMass',     None, 'mass of the compact object'),
    #               ('MassOfSMBH',                       'SolarMass',     None, 'mass of the MBH'),
    #               ('InitialAzimuthalOrbitalFrequency', 'Hertz',         None, 'initial value of orbital azimuthal frequency'),
    #               ('InitialAzimuthalOrbitalPhase',     'Radian',        None, 'initial azimuthal orbital phase'),
    #               ('InitialEccentricity',              'Unit',          None, 'initial orbital eccentricity'),
    #               ('InitialTildeGamma',                'Radian',        None, 'initial position of pericenter, as angle between LxS and pericenter'),
    #               ('InitialAlphaAngle',                'Radian',        None, 'initial azimuthal direction of L (in the orbital plane)'),
    #               ('LambdaAngle',                      'Radian',        None, 'angle between L and S'),
    #               ('Distance',                         'Parsec',        None, 'standard source distance'),
    #               ('FixHarmonics',                     '1',             0,    'fixed number of harmonics used in waveform computation; use 0 for variable number'))

    def  __init__(self,name=''):

        self.__samples = None
        self.__deltat  = None
        self.__inittime = None
        self.EclipticLatitude = None
        self.EclipticLongitude = None
        self.Polarization = None
        self.PolarAngleOfSpin = None
        self.AzimuthalAngleOfSpin = None
        self.Spin = None
        self.MassOfCompactObject = None
        self.MassOfSMBH = None
        self.InitialAzimuthalOrbitalFrequency = None
        self.InitialAzimuthalOrbitalPhase = None
        self.InitialEccentricity = None
        self.InitialTildeGamma = None
        self.InitialAlphaAngle = None
        self.LambdaAngle = None
        self.Distance = None
        self.PlungeTime = None
        self.FixHarmonics = 5

    def waveforms(self,samples,deltat,inittime,debug=0):
        if samples != self.__samples or deltat != self.__deltat or inittime != self.__inittime:
            self.__samples, self.__deltat, self.__inittime = samples, deltat, inittime


            self.__emri = AKWaveform(self.Spin, self.MassOfCompactObject, self.MassOfSMBH, inittime + deltat*(samples-1), deltat)

            self.__emri.SetSourceLocation(math.pi/2.0 - self.EclipticLatitude, self.EclipticLongitude, self.PolarAngleOfSpin, \
                                        self.AzimuthalAngleOfSpin, self.Distance)

            self.__emri.EvolveOrbit(inittime, self.InitialAzimuthalOrbitalFrequency, self.InitialEccentricity, self.InitialTildeGamma,\
                                  self.InitialAzimuthalOrbitalPhase, self.InitialAlphaAngle, self.LambdaAngle)

        hp = numpy.empty(samples,'d')
        hc = numpy.empty(samples,'d')

        # do not apply polarization here since it comes in at the level of makebarycentric.py
        # wavelen = emri.GetWaveform(self.Polarization, hp, hc)

        wavelen = self.__emri.GetWaveform(0, hp, hc, debug, self.FixHarmonics)
        #if self.FixHarmonics > 0:
        #   wavelen = self.__emri.GetWaveform(0, hp, hc, debug, self.FixHarmonics)
        #else:
        #   wavelen = self.__emri.GetWaveform(0, hp, hc, debug, 0)

        hp[wavelen:] = 0.0
        hc[wavelen:] = 0.0

        return (hp,hc)

    def orbital_evolution(self,samples,deltat,inittime):
        self.__emri = AKWaveform(self.Spin, self.MassOfCompactObject, self.MassOfSMBH, inittime + deltat*(samples-1), deltat)

        self.__emri.SetSourceLocation(math.pi/2.0 - self.EclipticLatitude, self.EclipticLongitude, self.PolarAngleOfSpin, \
                                    self.AzimuthalAngleOfSpin, self.Distance)

        self.__emri.EvolveOrbit(inittime, self.InitialAzimuthalOrbitalFrequency, self.InitialEccentricity, self.InitialTildeGamma,\
                              self.InitialAzimuthalOrbitalPhase, self.InitialAlphaAngle, self.LambdaAngle)

        tim =  numpy.empty(samples,'d')
        et =  numpy.empty(samples,'d')
        nut =  numpy.empty(samples,'d')
        phit =  numpy.empty(samples,'d')
        alpt =  numpy.empty(samples,'d')
        gamt =  numpy.empty(samples,'d')

        sz_t = self.__emri.OrbEvol(tim, et, nut, phit, alpt, gamt)

        return(sz_t, tim, et, nut, phit, alpt, gamt)

# utility function

def EMRIEstimateInitialOrbit(spin,mu,MBHmass, Tend,e_lso,gamma_lso,phi_lso,alpha_lso, lam_lso):
   """Estimates the initial (t=0) orbital parameters for the orbit
specified at the plunge by e_lso, gamma_lso, alpha_lso, phi_lso, lambda, duration Tend."""

   ak = AKWaveform(spin, mu, MBHmass, 3.e8, 15.0)

   ak.EstimateInitialParams(Tend, e_lso, gamma_lso, phi_lso, alpha_lso, lam_lso)

   nut = doublep()
   et = doublep()
   gamt = doublep()
   pht = doublep()
   alt = doublep()

   ak.GetOrbitalParamsAt0(nut, et, gamt, pht, alt)

   return[nut.value(), et.value(), gamt.value(), pht.value(), alt.value()]

%}
