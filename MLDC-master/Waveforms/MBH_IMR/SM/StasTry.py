import numpy as np
import matplotlib
#matplotlib.use('GTK3Agg')
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import sys, os, math
#from scipy.signal import butter, lfilter
#from scipy import signal
import scipy
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from numpy import sqrt

from pyIMRPhenomD import IMRPhenomD, MSUN_SI, PC_SI
import Cosmology
import LISAConstants as LC
import pyIMRPhenomD
plt.style.use('Stas_plot')

YRSID_SI = 31558149.763545600
MTSUN_SI = LC.MTsun
#AU = LC.ua
light = LC.clight

[m1, m2, chi1, chi2, DL, psi, incl, lam, bet, Tc] =\
 [2000000.0, 640000.0, 0.81, 0.73, 6823090480.18, 2.4, 1.1, 4.647, -0.07, 16345319.4909]

pi = np.pi

arm = 2.5e9 ### armlength in m
arm_t = arm/light


def sinc(x):
    return(np.sinc(x/np.pi))

def LISAmotion(tm):
    AU = LC.ua/light  ## AU in sec.
    lamb = 0.0
    kappa = 0.0

    N = len(tm)

    a = AU
    e = arm_t/(2.0*np.sqrt(3.0)*a)
    print "orbital eccentricity", e
    nn = np.array([1,2,3])
    year = YRSID_SI

    Beta = (nn-1)*2.0*np.pi/3.0 + lamb
    alpha = 2.0*np.pi*tm/year + kappa

    x = np.zeros((3, N))
    y = np.zeros((3, N))
    z = np.zeros((3, N))
    for i in xrange(3):
        x[i, :] = a*np.cos(alpha) + a*e*(np.sin(alpha)*np.cos(alpha)*np.sin(Beta[i]) - (1.0 + (np.sin(alpha))**2)*np.cos(Beta[i]))
        y[i, :] = a*np.sin(alpha) + a*e*(np.sin(alpha)*np.cos(alpha)*np.cos(Beta[i]) - (1.0 + (np.cos(alpha))**2)*np.sin(Beta[i]))
        z[i, :] = -np.sqrt(3.0)*a*e*np.cos(alpha - Beta[i])

    return (x, y, z, alpha, a)


def PolarizationTensor(lam, bet, psi):
    sl = np.sin(lam)
    cl = np.cos(lam)
    sb = np.sin(bet)
    cb = np.cos(bet)
    u = -1.0*np.array([sb*cl, sb*sl, -cb])
    v = 1.0*np.array([sl, -cl, 0.0])
    Ep = np.zeros((3,3))
    Ec = np.zeros((3,3))
    for i in xrange(3):
        for j in xrange(3):
            Ep[i, j] = u[i]*u[j] - v[i]*v[j]
            #Ep[i, j] = -u[i]*u[j] + v[i]*v[j]
            Ec[i, j] = u[i]*v[j] + u[j]*v[i]
    cp = np.cos(2.0*psi)
    sp = np.sin(2.0*psi)

    plus = cp*Ep + sp*Ec
    cross = sp*Ep - cp*Ec

    return(plus, cross)

def Polarization2(lambd, beta, psi):
    clambd = np.cos(lambd); slambd = np.sin(lambd);
    cbeta = np.cos(beta); sbeta = np.sin(beta);
    cpsi = np.cos(psi); spsi = np.sin(psi);
    O1 =  np.array([[cpsi*slambd-clambd*sbeta*spsi,-clambd*cpsi*sbeta-slambd*spsi,-cbeta*clambd],\
                    [-clambd*cpsi-sbeta*slambd*spsi,-cpsi*sbeta*slambd+clambd*spsi,-cbeta*slambd],\
                    [cbeta*spsi,cbeta*cpsi,-sbeta]])
    invO1 = np.array([[cpsi*slambd-clambd*sbeta*spsi,-clambd*cpsi-sbeta*slambd*spsi,cbeta*spsi], \
                    [-clambd*cpsi*sbeta-slambd*spsi,-cpsi*sbeta*slambd+clambd*spsi,cbeta*cpsi],\
                    [-cbeta*clambd,-cbeta*slambd,-sbeta]])

    HSplus = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]])
    HScross = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])

    Hplus = np.dot(O1, np.dot(HSplus, invO1))
    Hcross = np.dot(O1, np.dot(HScross, invO1))

    return (Hplus, Hcross)


# Function reproducing XLALSpinWeightedSphericalHarmonic
# - Currently only supports s=-2, l=2,3,4,5 modes
def SpinWeightedSphericalHarmonic(s, l, m, theta, phi):
    func = "SpinWeightedSphericalHarmonic"
    # Sanity checks
    if ( l < abs(s) ):
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |s| <= l\n' % (func, s, l, m))
    if ( l < abs(m) ):
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |m| <= l\n' % (func, s, l, m))
    if not ( s == -2 ):
        raise ValueError('Error - %s: Invalid mode s=%d - only s=-2 implemented\n' % (func, s))
    fac = {
        # l=2
        (2,-2) : sqrt( 5.0 / ( 64.0 * pi ) ) * ( 1.0 - np.cos( theta ))*( 1.0 - np.cos( theta )),
        (2,1) : sqrt( 5.0 / ( 16.0 * pi ) ) * np.sin( theta )*( 1.0 - np.cos( theta )),
        (2,0) : sqrt( 15.0 / ( 32.0 * pi ) ) * np.sin( theta )*np.sin( theta ),
        (2,1) : sqrt( 5.0 / ( 16.0 * pi ) ) * np.sin( theta )*( 1.0 + np.cos( theta )),
        (2,2) : sqrt( 5.0 / ( 64.0 * pi ) ) * ( 1.0 + np.cos( theta ))*( 1.0 + np.cos( theta )),
        # l=3
        (3,-3) : sqrt(21.0/(2.0*pi))*np.cos(theta/2.0)*pow(np.sin(theta/2.0),5.0),
        (3,-2) : sqrt(7.0/4.0*pi)*(2.0 + 3.0*np.cos(theta))*pow(np.sin(theta/2.0),4.0),
        (3,-1) : sqrt(35.0/(2.0*pi))*(np.sin(theta) + 4.0*np.sin(2.0*theta) - 3.0*np.sin(3.0*theta))/32.0,
        (3,0) : (sqrt(105.0/(2.0*pi))*np.cos(theta)*pow(np.sin(theta),2.0))/4.0,
        (3,1) : -sqrt(35.0/(2.0*pi))*(np.sin(theta) - 4.0*np.sin(2.0*theta) - 3.0*np.sin(3.0*theta))/32.0,
        (3,2) : sqrt(7.0/pi)*pow(np.cos(theta/2.0),4.0)*(-2.0 + 3.0*np.cos(theta))/2.0,
        (3,3) : -sqrt(21.0/(2.0*pi))*pow(np.cos(theta/2.0),5.0)*np.sin(theta/2.0),
        # l=4
        (4,-4) : 3.0*sqrt(7.0/pi)*pow(np.cos(theta/2.0),2.0)*pow(np.sin(theta/2.0),6.0),
        (4,-3) : 3.0*sqrt(7.0/(2.0*pi))*np.cos(theta/2.0)*(1.0 + 2.0*np.cos(theta))*pow(np.sin(theta/2.0),5.0),
        (4,-2) : (3.0*(9.0 + 14.0*np.cos(theta) + 7.0*np.cos(2.0*theta))*pow(np.sin(theta/2.0),4.0))/(4.0*sqrt(pi)),
        (4,-1) : (3.0*(3.0*np.sin(theta) + 2.0*np.sin(2.0*theta) + 7.0*np.sin(3.0*theta) - 7.0*np.sin(4.0*theta)))/(32.0*sqrt(2.0*pi)),
        (4,0) : (3.0*sqrt(5.0/(2.0*pi))*(5.0 + 7.0*np.cos(2.0*theta))*pow(np.sin(theta),2.0))/16.0,
        (4,1) : (3.0*(3.0*np.sin(theta) - 2.0*np.sin(2.0*theta) + 7.0*np.sin(3.0*theta) + 7.0*np.sin(4.0*theta)))/(32.0*sqrt(2.0*pi)),
        (4,2) : (3.0*pow(np.cos(theta/2.0),4.0)*(9.0 - 14.0*np.cos(theta) + 7.0*np.cos(2.0*theta)))/(4.0*sqrt(pi)),
        (4,3) : -3.0*sqrt(7.0/(2.0*pi))*pow(np.cos(theta/2.0),5.0)*(-1.0 + 2.0*np.cos(theta))*np.sin(theta/2.0),
        (4,4) : 3.0*sqrt(7.0/pi)*pow(np.cos(theta/2.0),6.0)*pow(np.sin(theta/2.0),2.0),
        # l= 5
        (5,-5) : sqrt(330.0/pi)*pow(np.cos(theta/2.0),3.0)*pow(np.sin(theta/2.0),7.0),
        (5,-4) : sqrt(33.0/pi)*pow(np.cos(theta/2.0),2.0)*(2.0 + 5.0*np.cos(theta))*pow(np.sin(theta/2.0),6.0),
        (5,-3) : (sqrt(33.0/(2.0*pi))*np.cos(theta/2.0)*(17.0 + 24.0*np.cos(theta) + 15.0*np.cos(2.0*theta))*pow(np.sin(theta/2.0),5.0))/4.0,
        (5,-2) : (sqrt(11.0/pi)*(32.0 + 57.0*np.cos(theta) + 36.0*np.cos(2.0*theta) + 15.0*np.cos(3.0*theta))*pow(np.sin(theta/2.0),4.0))/8.0,
        (5,-1) : (sqrt(77.0/pi)*(2.0*np.sin(theta) + 8.0*np.sin(2.0*theta) + 3.0*np.sin(3.0*theta) + 12.0*np.sin(4.0*theta) - 15.0*np.sin(5.0*theta)))/256.0,
        (5,0) : (sqrt(1155.0/(2.0*pi))*(5.0*np.cos(theta) + 3.0*np.cos(3.0*theta))*pow(np.sin(theta),2.0))/32.0,
        (5,1) : sqrt(77.0/pi)*(-2.0*np.sin(theta) + 8.0*np.sin(2.0*theta) - 3.0*np.sin(3.0*theta) + 12.0*np.sin(4.0*theta) + 15.0*np.sin(5.0*theta))/256.0,
        (5,2) : sqrt(11.0/pi)*pow(np.cos(theta/2.0),4.0)*(-32.0 + 57.0*np.cos(theta) - 36.0*np.cos(2.0*theta) + 15.0*np.cos(3.0*theta))/8.0,
        (5,3) : -sqrt(33.0/(2.0*pi))*pow(np.cos(theta/2.0),5.0)*(17.0 - 24.0*np.cos(theta) + 15.0*np.cos(2.0*theta))*np.sin(theta/2.0)/4.0,
        (5,4) : sqrt(33.0/pi)*pow(np.cos(theta/2.0),6.0)*(-2.0 + 5.0*np.cos(theta))*pow(np.sin(theta/2.0),2.0),
        (5,5) : -sqrt(330.0/pi)*pow(np.cos(theta/2.0),7.0)*pow(np.sin(theta/2.0),3.0)
        }.get((l,m), None)
    if fac==None:
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |m| <= l\n' % (func, s, l, m))

    # Result
    if m==0:
        return fac
    else:
        return fac * np.exp(1j*m*phi)

# Compute chirp mass from m1 and m2 - same units as input m1,m2
def funcMchirpofm1m2(m1, m2):
  return pow(m1*m2, 3./5) / pow(m1+m2, 1./5)
# Newtonian estimate of the relation f(deltat) (for the 22 mode freq) - gives the starting geometric frequency for a given time to merger and chirp mass - output in Hz
# Input chirp mass (solar masses), time in years
def funcNewtonianfoft(Mchirp, t):
    if(t<=0.):
        return 0.
    return 1./pi * pow(Mchirp*MTSUN_SI, -5./8) * pow(256.*t*YRSID_SI/5, -3./8)
# Newtonian estimate of the relation deltat(f) (for the 22 mode freq) - gives the time to merger from a starting frequency for a given chirp mass - output in years
# Input chirp mass (solar masses), frequency in Hz
def funcNewtoniantoffchirp(Mchirp, f):
    return 5./256 * pow(Mchirp*MTSUN_SI, -5./3) * pow(pi*f, -8./3) / YRSID_SI

# Logarithmically spaced samples
def logsampling(start, stop, num=50.):
    res = np.exp(np.linspace(np.log(start), np.log(stop), num=num))
    res[0] = start # Eliminate possible rounding error
    res[-1] = stop # Eliminate possible rounding error
    return res


### Function from Sylvain
# Functions to generate PhenomD waveform and resample it in preparation for the FD response
# We try to ensure three conditions:
# - logarithmic sampling with Deltalnf=0.02 (natural log)
# - resampling at low-f such that Deltat<=1./24 yr (as measured from tf)
# - resampling at high-f such that Deltaf<=0.002 Hz
# Waveform generated for a given time to coalescence tobs (in years) (approximate, using tf)
# Also admits args for min, max frequency (in Hz)
# m1, m2 in solar masses, dist in Mpc
# Set minf and tobs to 0 to ignore (but not both !)
# If settRefAtfRef, arrange so that t=tRef at fRef FIXME: check fRef is in range
def GenerateResamplePhenomD(phi0, fRef, m1, m2, chi1, chi2, dist, inc, tobs=1., minf=1e-5, maxf=1., settRefAtfRef=False, tRef=0.):
    # Sampling targets
    Delta_lnf = 0.02
    Delta_t = 1./24 # yrs
    Delta_f = 0.002 # Hz
    # Maximum frequency covered by PhenomD
    # NOTE: we take some small margin of error here. Before this was interpreted as outside range by the C code and the last sample was not computed (was given a phase of 0, which messed up the interpolation afterwards)
    MfCUT_PhenomD = 0.2 - 1e-7

    # Check args
    if minf>=maxf:
        raise ValueError("Error in GenerateResamplePhenomDStartTime: incompatible minf and maxf.")
    if minf<=0 and tobs<=0:
        raise ValueError("Error in GenerateResamplePhenomDStartTime: both minf and tobs set to 0 and ignored, does not know where to start !")

    # Parameters
    m1_SI = m1*MSUN_SI
    m2_SI = m2*MSUN_SI
    dist_SI = dist*1e6*PC_SI
    Ms = (m1 + m2) * MTSUN_SI
    Mchirp = funcMchirpofm1m2(m1, m2) # in solar masses

    # First generate PhenomD waveform using log sampling
    # Here starting frequency estimated from Newtonian order, with a margin of 2 in duration since the estimate is approximate - also take minf into account
    if tobs>0:
        fstartN_twiceduration = funcNewtonianfoft(Mchirp, 2*tobs) # margin of a factor of 2 - Newtonian estimate is not very accurate
    else:
        fstartN_twiceduration = 0. # if tobs is zero, ignore it
    minf_PhD = max(minf, fstartN_twiceduration)
    cut_by_minf = (minf>fstartN_twiceduration) # Keep track of wether we cut by frequency or duration here
    fCUT_PhD = MfCUT_PhenomD/Ms
    maxf_PhD = min(maxf, fCUT_PhD)
    cut_by_maxf = (maxf<fCUT_PhD) # Keep track of wether we cut by argument maxf here
    # Here, ensure a minimum of 200 pts to do the first generation
    npt_PhD = max(int(np.ceil(np.log(maxf_PhD/minf_PhD) / Delta_lnf)), 400)
    freq_PhD = logsampling(minf_PhD, maxf_PhD, num=npt_PhD)
    wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq_PhD, phi0, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
    wf_PhD = wf_PhD_class.GetWaveform() # freq, amp, phase

    # Build tf function and f(t) by interpolation of the monotonous part
    # FIXME: can go wrong if the waveform is too short, add safety checks
    tfspline = spline(freq_PhD, 1/(2*pi)*wf_PhD[2]).derivative()
    tf = tfspline(freq_PhD)
    index_cuttf = 0
    tfdiff = np.diff(tf)
    while index_cuttf<len(tfdiff) and tfdiff[index_cuttf]>0:
        index_cuttf += 1
    nocuttf = (index_cuttf==len(freq_PhD)-1) # keep track of the case where the last tfr corresponds to maxf_rs=maxf_PhD
    tfr = tf[:index_cuttf+1]
    freq_PhDr = freq_PhD[:index_cuttf+1]
    ftspline = spline(tfr, freq_PhDr)

    # Build frequencies for resampling
    # Note: tobs is positive, duration of signal, hence -tobs - tobs in yrs, tf in SI (s)
    # If tobs=0., ignored
    if tobs>0:
        tstartobs = -tobs*YRSID_SI
        if (not cut_by_minf) and tfr[0]>tstartobs:
            raise ValueError("Error in GenerateResamplePhenomDStartTime: tf from initial waveform generation (with fstart decided by Newtonian estimate for duration of 2*tobs) does not include tobs.")
        if tfr[-1]<tstartobs:
            raise ValueError("Error in GenerateResamplePhenomDStartTime: tf for interpolation of ft (from monotonous part of tf) stops before tobs - indicates that tobs is too short.")
        if tfr[0]>tstartobs: # in this case, we can't evaluate ftspline for comparison as it is out of range
            minf_rs = minf_PhD
        else:
            minf_rs = ftspline(tstartobs)
    else:
        minf_rs = minf_PhD
    maxf_rs = maxf_PhD
    tstart = tfspline(minf_rs) # is either -tobs or tf(minf)
    # Original ln-sampling for comparison - ensure it is at least 100
    npt_lnsampling = max(100, int(np.ceil(np.log(maxf_rs/minf_rs) / Delta_lnf))) # npt if all the freq interval was covered with ln-sampling
    Delta_lnf_mod = np.log(maxf_rs/minf_rs) / npt_lnsampling # possibly lower than Delta_lnf, to ensure at least 100 ln-pts on [minf_rs, maxf_rs]
    # Deltat-resampling at low-f - limit the time range to where ftspline is defined, tfr[-1] should be close to 0
    times_deltatresampling = np.linspace(tstart, tfr[-1], int(np.ceil(abs(tstart - tfr[-1])/(Delta_t*YRSID_SI))))
    if len(times_deltatresampling)==0:
        freq_rsdeltat = np.array([])
    else:
        freqs_deltatresampling = ftspline(times_deltatresampling)
        index_samplingdeltat = 0
        while index_samplingdeltat<len(freqs_deltatresampling)-1 and  np.log(freqs_deltatresampling[index_samplingdeltat+1]/freqs_deltatresampling[index_samplingdeltat])<Delta_lnf_mod:
            index_samplingdeltat += 1
        if index_samplingdeltat==0:
            freq_rsdeltat = np.array([])
        else:
            freq_rsdeltat = freqs_deltatresampling[:index_samplingdeltat+1]
            # Re-adjust extremity -- this ensures the start point is minf_rs, without numerical rounding error
            freq_rsdeltat[0] = minf_rs
            # If last time not excluded and corresponds to maxf_rs, then ensure the end point is exactly maxf_rs without numerical rounding error
            if nocuttf and index_samplingdeltat==len(freqs_deltatresampling)-1:
                freq_rsdeltat[-1] = maxf_rs
    # Deltaf-resampling at high-f
    # Ensure f_startdeltafsampling in [minf_rs, maxf_rs]
    f_startdeltafsampling = max(minf_rs, Delta_f/((maxf_rs/minf_rs)**(1./(npt_lnsampling-1)) - 1))
    npt_deltafsampling = int(np.ceil((maxf_rs-f_startdeltafsampling)/Delta_f))
    if npt_deltafsampling<=0: # if f_startdeltafsampling>=maxfrs, do not do deltaf-sampling
        freq_rsdeltaf = np.array([])
    else:
        freq_rsdeltaf = np.linspace(f_startdeltafsampling, maxf_rs, num=npt_deltafsampling)
    # Original ln-sampling in the middle
    # FIXME: assumes there is always ln-sampling in the middle, or reduced to one extremity (no contact between deltat-sampling and deltaf-sampling) - will always be the case in practice, but to be made more robust
    if len(freq_rsdeltat)>0:
        minf_rsln = freq_rsdeltat[-1]
        freq_rsdeltat = freq_rsdeltat[:-1] # last freq will be included in the ln-sampling part
    else:
        minf_rsln = minf_rs
    if len(freq_rsdeltaf)>0:
        maxf_rsln = freq_rsdeltaf[0]
        freq_rsdeltaf = freq_rsdeltaf[1:] # first freq will be included in the ln-sampling part
    else:
        maxf_rsln = maxf_rs
    if minf_rsln<maxf_rsln:
        npt_rsln = int(np.ceil(np.log(maxf_rsln/minf_rsln) / Delta_lnf_mod)) + 1
        freq_rsln = logsampling(minf_rsln, maxf_rsln, num=npt_rsln*2)
    else:
        freq_rsln = np.array([maxf_rsln])
    freq_rs = np.concatenate((freq_rsdeltat, freq_rsln, freq_rsdeltaf))

    # Resampling
    amp_PhD_Int = spline(wf_PhD[0], wf_PhD[1])
    phase_PhD_Int = spline(wf_PhD[0], wf_PhD[2])
    amp_rs = amp_PhD_Int(freq_rs)
    phase_rs = phase_PhD_Int(freq_rs)

    # If settRefAtfRef, arrange so that t=tRef at fRef
    # FIXME: improve the check that fRef is in range - here simply forced back in
    if settRefAtfRef:
        fRef_inrange = min(max(freq_rs[0], fRef), freq_rs[-1])
        phase_rs += 2*pi * freq_rs * (tRef - tfspline(fRef_inrange))

    # Output
    return freq_rs, amp_rs, phase_rs


def ComputehphcFromAmpPhase(wf, freq, iota, phi):
    f, amp, phase = wf
    amp_Int = spline(f, amp)
    phase_Int = spline(f, phase)
    indices = (f[0]<=freq) & (freq<=f[-1])
    factorp = 1/2. * (SpinWeightedSphericalHarmonic(-2, 2, 2, iota, phi) + np.conj(SpinWeightedSphericalHarmonic(-2, 2, -2, iota, phi)))
    factorc = 1j/2. * (SpinWeightedSphericalHarmonic(-2, 2, 2, iota, phi) - np.conj(SpinWeightedSphericalHarmonic(-2, 2, -2, iota, phi)))
    h22tilde = np.zeros_like(freq, dtype=np.complex128)
    h22tilde[indices] = amp_Int(freq[indices]) * np.exp(1j*phase_Int(freq[indices]))
    hptilde = factorp * h22tilde
    hctilde = factorc * h22tilde
    print "compare phases"
    #tmph = np.angle(hptilde)
    # tmph = np.angle(np.exp(1j*phase_Int(freq[indices])))
    # tmph = np.unwrap(tmph)
    # plt.plot(f, phase)
    # plt.plot(freq[indices], phase_Int(freq[indices]), '--')
    # plt.plot(freq[indices], tmph, 'b.')
    # plt.show()
    # sys.exit(0)
    return freq, hptilde, hctilde


def ComputeYslr(send, link, rec, lsign=1):
    #hf = w_amp*np.exp(-1.0j*w_ph)
    if (link == 1):
        kn = lsign*kn1
        A = A1
    elif (link == 2):
        kn = lsign*kn2
        A = A2
    elif (link == 3):
        kn = lsign*kn3
        A = A3
    else:
        print "should never be here"
        raise ValueError
    if (rec == 1):
        kq = kq1
    elif (rec == 2):
        kq = kq2
    elif (rec == 3):
        kq = kq3
    else:
        print "should never be here"
        raise ValueError
    args = 0.5*omL*(1.0 - kn)

    y = 0.5j*omL*A*sinc(args)*np.exp(1.0j*(args + om*kq))

    #y = y*hf*np.exp(1.j*om*kR0)

    return(y)



def myEvaluateGslr(t, f, H, k, funcp0, funcp1L, funcp2L, funcp3L, funcn1, funcn2, funcn3):
    nt = len(t)
    p0 = funcp0(t)
    p1L = funcp1L(t)
    p2L = funcp2L(t)
    p3L = funcp3L(t)
    sn1 = funcn1(t)
    sn2 = funcn2(t)
    sn3 = funcn3(t)
    # Compute intermediate scalar products
    skn1 = np.dot(k, sn1)
    skn2 = np.dot(k, sn2)
    skn3 = np.dot(k, sn3)
    n1Hn1 = np.zeros(nt, dtype='complex128')
    n2Hn2 = np.zeros(nt, dtype='complex128')
    n3Hn3 = np.zeros(nt, dtype='complex128')
    for i in xrange(nt):
        n1Hn1[i] = np.dot(sn1[:,i], np.dot(H, sn1[:,i]))
        n2Hn2[i] = np.dot(sn2[:,i], np.dot(H, sn2[:,i]))
        n3Hn3[i] = np.dot(sn3[:,i], np.dot(H, sn3[:,i]))
    #n1Hn1 = np.dot(n1, np.dot(H, n1))
    #n2Hn2 = np.dot(n2, np.dot(H, n2))
    #n3Hn3 = np.dot(n3, np.dot(H, n3))
    kp0 = np.dot(k, p0)
    kp1Lp2L = np.dot(k, (p1L+p2L))
    kp2Lp3L = np.dot(k, (p2L+p3L))
    kp3Lp1L = np.dot(k, (p3L+p1L))
    # Prefactors
    prefactor = pi*f*L_SI/C_SI
    factorcexp12 = exp(1j*prefactor * (1.+kp1Lp2L/L_SI))
    factorcexp23 = exp(1j*prefactor * (1.+kp2Lp3L/L_SI))
    factorcexp31 = exp(1j*prefactor * (1.+kp3Lp1L/L_SI))
    factorsinc12 = sinc( prefactor * (1.-skn3))
    factorsinc21 = sinc( prefactor * (1.+skn3))
    factorsinc23 = sinc( prefactor * (1.-skn1))
    factorsinc32 = sinc( prefactor * (1.+skn1))
    factorsinc31 = sinc( prefactor * (1.-skn2))
    factorsinc13 = sinc( prefactor * (1.+skn2))

    # Compute the Gslr
    G12 = 1j*prefactor * n3Hn3 * factorsinc12 * factorcexp12
    G21 = 1j*prefactor * n3Hn3 * factorsinc21 * factorcexp12
    G23 = 1j*prefactor * n1Hn1 * factorsinc23 * factorcexp23
    G32 = 1j*prefactor * n1Hn1 * factorsinc32 * factorcexp23
    G31 = 1j*prefactor * n2Hn2 * factorsinc31 * factorcexp31
    G13 = 1j*prefactor * n2Hn2 * factorsinc13 * factorcexp31

    # Output dictionary
    Gslr = {}
    Gslr[(1,2)] = G12
    Gslr[(2,1)] = G21
    Gslr[(2,3)] = G23
    Gslr[(3,2)] = G32
    Gslr[(3,1)] = G31
    Gslr[(1,3)] = G13
    return Gslr
#"""

#Pl1, Cr1 = PolarizationTensor(lam, bet, -np.pi*0.25 + np.pi*0.5)
#Pl2, Cr2 = Polarization2(lam, bet, np.pi*0.25)
#Pl1, Cr1 = PolarizationTensor(lam, bet, 0.+np.pi*0.5)
#Pl2, Cr2 = Polarization2(lam, bet, 0.)
Pl1, Cr1 = PolarizationTensor(lam, bet, np.pi*0.5-psi)
Pl2, Cr2 = Polarization2(lam, bet, psi)   ### this is how Sylvain defines polarization

# print "plus"
# print Pl1
# print Pl2
# print Pl1 - Pl2
# print "\n"
# print "cross"
# print Cr1, "\n", Cr2
# print Cr1 - Cr2

phi0 = 2.21
w_fr, w_amp, w_ph = GenerateResamplePhenomD(phi0=phi0, fRef=0.0, m1=m1, m2=m2, chi1=chi1, \
                                            chi2=chi2, dist=DL*1.e-6, inc=incl, tobs=Tc/YRSID_SI)


# plt.loglog(w_fr, w_amp)
# plt.grid(True)
# plt.show()

#plt.semilogx(w_fr, w_ph, 'o')
#plt.show()

# Build function tf by interpolating FD phase
tfspline = spline(w_fr, 1/(2.*np.pi)*(w_ph-w_ph[0])).derivative() # get rid of possibly huge constant in the phase before interpolating
tfvec = tfspline(w_fr)

plt.loglog(w_fr, np.absolute(tfvec), 'o')
plt.show()

N = len(w_fr)
print "N = ", N
#print w_fr
#print np.diff(w_fr)

#  Let's try to have a look at the fourier transform of the waveform
fmin = w_fr[0]
fmax = w_fr[-1]
df = 1.0/YRSID_SI
Nf = int((fmax-fmin)/df)+1
freq = np.arange(Nf)*df + fmin
print len(w_fr), len(freq)
#print freq

fHz, hptilde, hctilde = ComputehphcFromAmpPhase([w_fr, w_amp, w_ph], freq, incl, 0.)

#print np.diff(fHz)
#sys.exit(0)

#hp = np.fft.irfft(hpf * np.exp(1.0j*freq*11000.0) )*(1.0/(np.sqrt(2.0)*del_t))
#hp = np.fft.irfft(hpf)
#hc = np.fft.irfft(hcf * np.exp(1.0j*freq*11000.0) )

del_t = 0.5/fHz[-1]
print "dt = ", del_t

# hp = np.fft.irfft(np.conjugate(hptilde)* np.exp(1.0j*fHz*11000.0) )*(1.0/(np.sqrt(2.0) *del_t))
#
# imax = np.argmax(hp)
# itc = int(Tc/del_t)
# shift = (itc - imax)*del_t
#
# plt.plot(np.arange(len(hp))*del_t+shift, hp)
# plt.show()


Y22 = SpinWeightedSphericalHarmonic(-2, 2, 2, incl, 0.)
Y2m2 = SpinWeightedSphericalHarmonic(-2, 2, -2, incl, 0.)
#print "Spher harmonics (22):", Y22, np.conjugate(Y22)
#print "Spher harmonics (2m2):", Y2m2, np.conjugate(Y2m2)
Yfactorplus = 1./2 * (Y22 + np.conjugate(Y2m2))
Yfactorcross = 1j/2 * (Y22 - np.conjugate(Y2m2))

x, y, z, alpha, a = LISAmotion(tfvec)

plt.plot(tfvec, x[2,:]-a*np.cos(alpha))
plt.plot(tfvec, y[2,:]-a*np.sin(alpha))
plt.plot(tfvec, z[2,:])
plt.show()

n1 = np.array([x[1,:]-x[2,:], y[1,:]-y[2,:], z[1,:] - z[2,:]])/arm_t
n2 = np.array([x[2,:]-x[0,:], y[2,:]-y[0,:], z[2,:] - z[0,:]])/arm_t
n3 = np.array([x[0,:]-x[1,:], y[0,:]-y[1,:], z[0,:] - z[1,:]])/arm_t
R0 = np.array([a*np.cos(alpha), a*np.sin(alpha), 0])
k = np.array([-np.cos(bet)*np.cos(lam), -np.cos(bet)*np.sin(lam), -np.sin(bet)])

#print np.shape(np.dot(k, n1))
print np.shape(n1)

kn1 = np.dot(k, n1)
kn2 = np.dot(k, n2)
kn3 = np.dot(k, n3)
kR0 = np.dot(k, R0)

#### NOTE we consider only the positive frequencies
Pij = Pl1*Yfactorplus + Cr1*Yfactorcross
print "Pij = ", Pij
A1 = np.zeros(N, dtype='complex128')
A2 = np.zeros(N, dtype='complex128')
A3 = np.zeros(N, dtype='complex128')
for i in xrange(N):
    A1[i] = np.dot(np.dot(n1[:, i], Pij), n1[:, i])
    A2[i] = np.dot(np.dot(n2[:, i], Pij), n2[:, i])
    A3[i] = np.dot(np.dot(n3[:, i], Pij), n3[:, i])

print np.shape(A1)

q1 = np.array([x[0,:]-a*np.cos(alpha), y[0, :]-a*np.sin(alpha), z[0, :]])
q2 = np.array([x[1,:]-a*np.cos(alpha), y[1, :]-a*np.sin(alpha), z[1, :]])
q3 = np.array([x[2,:]-a*np.cos(alpha), y[2, :]-a*np.sin(alpha), z[2, :]])

kq1 = np.dot(k, q1)
kq2 = np.dot(k, q2)
kq3 = np.dot(k, q3)

#print q1
#print kq3
#### Computing X and then interpolated
om = 2.0*np.pi*w_fr
omL = om*arm_t

y123 = ComputeYslr(send=1, link=2, rec=3, lsign=1.)
y231 = ComputeYslr(send=2, link=3, rec=1, lsign=1.)
y1m32 = ComputeYslr(send=1, link=3, rec=2, lsign=-1.)
y3m21 = ComputeYslr(send=3, link=2, rec=1, lsign=-1.)
y2m13 = ComputeYslr(send=2, link=1, rec=3, lsign=-1.)
y312 = ComputeYslr(send=3, link=1, rec=2, lsign=1.)

args = 0.5*omL*(1.0 - kn3)
print "plotting args"
f, ax = plt.subplots(2)
#ax[0].plot(w_fr, args)
#ax[1].plot(tfvec, args)
ax[0].plot(w_fr, np.real(0.5j*omL*A3*sinc(args)*np.exp(1.0j*args)*np.exp(1.0j*om*kq1)) )
#ax[0].plot(w_fr, np.real(0.5j*omL*A3*sinc(args)*np.exp(1.0j*(args + om*kq1))), '--')
ax[0].plot(w_fr, np.real(y231), '--')
ax[1].plot(tfvec, np.real(omL*A3*sinc(args)*np.exp(1.0j*args)*np.exp(1.0j*kq1)) )
plt.show()
#sys.exit(0)
#y = 0.5j*omL*A*sinc(args)*np.exp(1.0j*(args + om*kq))


print "plotting links"
f, ax = plt.subplots(2, 2)
ax[0, 0].plot(w_fr, np.real(y231))
ax[0, 0].plot(w_fr, np.real(y123))
ax[0, 0].plot(w_fr, np.real(y1m32))
ax[0, 0].plot(w_fr, np.real(y3m21))
ax[0, 1].plot(w_fr, np.imag(y231))
ax[0, 1].plot(w_fr, np.imag(y123))
ax[0, 1].plot(w_fr, np.imag(y1m32))
ax[0, 1].plot(w_fr, np.imag(y3m21))

ax[1,0].plot(tfvec, np.real(y231))
ax[1,0].plot(tfvec, np.real(y123))
ax[1,0].plot(tfvec, np.real(y1m32))
ax[1,0].plot(tfvec, np.real(y3m21))
ax[1,1].plot(tfvec, np.imag(y231))
ax[1,1].plot(tfvec, np.imag(y123))
ax[1,1].plot(tfvec, np.imag(y1m32))
ax[1,1].plot(tfvec, np.imag(y3m21))
plt.show()


RX = 2.0j*np.sin(omL)*(y231 + (y1m32 - y123)*np.exp(1.0j*omL) - y3m21)*np.exp(1.0j*omL)
RY = 2.0j*np.sin(omL)*(y312 + (y2m13 - y231)*np.exp(1.0j*omL) - y1m32)*np.exp(1.0j*omL)
RZ = 2.0j*np.sin(omL)*(y123 + (y3m21 - y312)*np.exp(1.0j*omL) - y2m13)*np.exp(1.0j*omL)

hf = w_amp*np.exp(-1.0j*w_ph)
hfD = w_amp*np.exp(1.0j*w_ph - 1.0j*om*kR0)
X = hfD*np.conjugate(RX)
Y = hfD*np.conjugate(RY)
Z = hfD*np.conjugate(RZ)



fstar = 1.0/arm_t
f, ax = plt.subplots(2)
ax[0].semilogx(w_fr, np.real(RX))
ax[1].semilogx(w_fr, (fstar/w_fr)**2 * np.imag(RX))
plt.show()


### Get Transfer f-n from Sylvain's code:
wf = [w_fr, w_amp, w_ph]
from pyFDresponse import *
[t0, trajdict, TDItag] = [0., trajdict_MLDC, "TDIXYZ"]
print "running "
wf_tdi = LISAFDresponseTDI(wf, incl, lam, bet, psi, t0=t0, trajdict=trajdict, TDItag=TDItag)
f, amp, phase, phaseRdelay, transferL1, transferL2, transferL3, TDI1, TDI2, TDI3, TDItag = \
wf_tdi['freq'], wf_tdi['amp'], wf_tdi['phase'], wf_tdi['phaseRdelay'], wf_tdi['transferL1'], wf_tdi['transferL2'], \
wf_tdi['transferL3'], wf_tdi['TDI1'], wf_tdi['TDI2'], wf_tdi['TDI3'], wf_tdi['TDItag']
    # Common settings
print "done"

### First I compare the spacecraft positions:
qS1 = funcp1L_MLDC(tfvec)/light
qS2 = funcp2L_MLDC(tfvec)/light
qS3 = funcp3L_MLDC(tfvec)/light
print "plotting q's"
f, ax = plt.subplots(3, sharex=True)
ax[0].plot(tfvec, q1[0, :])
ax[0].plot(tfvec, qS1[0,:], '--')
ax[0].plot(tfvec, q2[0, :])
ax[0].plot(tfvec, qS2[0,:], '--')
ax[0].plot(tfvec, q3[0, :])
ax[0].plot(tfvec, qS3[0,:], '--')

ax[1].plot(tfvec, q1[1, :])
ax[1].plot(tfvec, qS1[1,:], '--')
ax[1].plot(tfvec, q2[1, :])
ax[1].plot(tfvec, qS2[1,:], '--')
ax[1].plot(tfvec, q3[1, :])
ax[1].plot(tfvec, qS3[1,:], '--')

ax[2].plot(tfvec, q1[2, :])
ax[2].plot(tfvec, qS1[2,:], '--')
ax[2].plot(tfvec, q2[2, :])
ax[2].plot(tfvec, qS2[2,:], '--')
ax[2].plot(tfvec, q3[2, :])
ax[2].plot(tfvec, qS3[2,:], '--')

plt.show()

### checking the links:
trajdict_MLDC = {
    'funcp0': funcp0_MLDC,
    'funcp1L': funcp1L_MLDC,
    'funcp2L': funcp2L_MLDC,
    'funcp3L': funcp3L_MLDC,
    'funcn1': funcn1_MLDC,
    'funcn2': funcn2_MLDC,
    'funcn3': funcn3_MLDC,
    }
# Trajectories
funcp0 = trajdict['funcp0']
funcp1L = trajdict['funcp1L']
funcp2L = trajdict['funcp2L']
funcp3L = trajdict['funcp3L']
funcn1 = trajdict['funcn1']
funcn2 = trajdict['funcn2']
funcn3 = trajdict['funcn3']
Gslr = myEvaluateGslr(tfvec, w_fr, Pij, k, funcp0, funcp1L, funcp2L, funcp3L, funcn1, funcn2, funcn3)
#Gslr = EvaluateGslr(tfvec, w_fr, Pij, k, funcp0, funcp1L, funcp2L, funcp3L, funcn1, funcn2, funcn3)
Sy231 = np.zeros(N, dtype='complex128')
Sy3m21 = np.zeros(N, dtype='complex128')
Sy1m32 = np.zeros(N, dtype='complex128')
Sy123 = np.zeros(N, dtype='complex128')
for i in xrange(N):
    Gslr_o = EvaluateGslr(tfvec[i], w_fr[i], Pij, k, funcp0, funcp1L, funcp2L, funcp3L, funcn1, funcn2, funcn3)
    Sy231[i] = Gslr_o[(2,1)]
    Sy3m21[i] = Gslr_o[(3,1)]
    Sy1m32[i] = Gslr_o[(1,2)]
    Sy123[i] = Gslr_o[(1,3)]


f, ax = plt.subplots(2,2, sharex=True)
f.suptitle("Link responses")
#y231 + (y1m32 - y123)*np.exp(1.0j*omL) - y3m21
ax[0,0].plot(w_fr, np.real(y231))
#ax[0,0].plot(w_fr, np.real(Sy231))
ax[0,0].plot(w_fr, np.real(Gslr[(2,1)]), '--')
ax[0,0].plot(w_fr, np.imag(y231))
#ax[0,0].plot(w_fr, np.imag(Sy231))
ax[0,0].plot(w_fr, np.imag(Gslr[(2,1)]), '--')


ax[0,1].plot(w_fr, np.real(y3m21))
#ax[0,1].plot(w_fr, np.real(Sy3m21))
ax[0,1].plot(w_fr, np.real(Gslr[(3,1)]), '--')
ax[0,1].plot(w_fr, np.imag(y3m21))
#ax[0,1].plot(w_fr, np.imag(Sy3m21))
ax[0,1].plot(w_fr, np.imag(Gslr[(3,1)]), '--')


ax[1,0].plot(w_fr, np.real(y1m32))
#ax[1,0].plot(w_fr, np.real(Sy1m32))
ax[1,0].plot(w_fr, np.real(Gslr[(1,2)]), '--')
ax[1,0].plot(w_fr, np.imag(y1m32))
#ax[1,0].plot(w_fr, np.imag(Sy1m32))
ax[1,0].plot(w_fr, np.imag(Gslr[(1,2)]), '--')

ax[1,1].plot(w_fr, np.real(y123))
#ax[1,1].plot(w_fr, np.real(Sy123))
ax[1,1].plot(w_fr, np.real(Gslr[(1,3)]), '--')
ax[1,1].plot(w_fr, np.imag(y123))
#ax[1,1].plot(w_fr, np.imag(Sy123))
ax[1,1].plot(w_fr, np.imag(Gslr[(1,3)]), '--')

plt.show()

f, ax = plt.subplots(3, sharex=True)
f.suptitle("Response, $X, Y, Z (f_*/f)^2$")
fctr = (fstar/w_fr)**2
ax[0].plot(w_fr, np.real(RX)*fctr)
ax[0].plot(w_fr, np.real(transferL1)*fctr, '--')
ax[0].plot(w_fr, np.imag(RX)*fctr)
ax[0].plot(w_fr, np.imag(transferL1)*fctr, '--')

ax[1].plot(w_fr, np.real(RY)*fctr)
ax[1].plot(w_fr, np.real(transferL2)*fctr, '--')
ax[1].plot(w_fr, np.imag(RY)*fctr)
ax[1].plot(w_fr, np.imag(transferL2)*fctr, '--')

ax[2].plot(w_fr, np.real(RZ)*fctr)
ax[2].plot(w_fr, np.real(transferL3)*fctr, '--')
ax[2].plot(w_fr, np.imag(RZ)*fctr)
ax[2].plot(w_fr, np.imag(transferL3)*fctr, '--')
plt.show()

f, ax = plt.subplots(3, sharex=True)
f.suptitle("TDIs $X, Y, Z$")
ax[0].plot(w_fr, np.real(X))
ax[0].plot(w_fr, np.real(TDI1),'--')
ax[0].plot(w_fr, np.imag(X))
ax[0].plot(w_fr, np.imag(TDI1),'--')
ax[1].plot(w_fr, np.real(Y))
ax[1].plot(w_fr, np.real(TDI2),'--')
ax[1].plot(w_fr, np.imag(Y))
ax[1].plot(w_fr, np.imag(TDI2),'--')
ax[2].plot(w_fr, np.real(Z))
ax[2].plot(w_fr, np.real(TDI3),'--')
ax[2].plot(w_fr, np.imag(Z))
ax[2].plot(w_fr, np.imag(TDI3), '--')

plt.show()

Xabs = np.absolute(X)
Xph = np.angle(X)
Xph = np.unwrap(Xph)
Yabs = np.absolute(Y)
Yph = np.angle(Y)
Yph = np.unwrap(Yph)
Zabs = np.absolute(Z)
Zph = np.angle(Z)
Zph = np.unwrap(Zph)

ARX = np.absolute(RX)*fctr
ARY = np.absolute(RY)*fctr
ARZ = np.absolute(RZ)*fctr

RXph = np.angle(RX)
RXph = np.unwrap(RXph)
RYph = np.angle(RY)
RYph = np.unwrap(RYph)
RZph = np.angle(RZ)
RZph = np.unwrap(RZph)

f, ax = plt.subplots(2, 2, sharex=True)
col = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
bright=["#003FFF", "#03ED3A", "#E8000B", "#8A2BE2", "#FFC400", "#00D7FF"]
dark=["#001C7F", "#017517", "#8C0900", "#7600A1", "#B8860B", "#006374"]
muted=["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
pastel=["#92C6FF", "#97F0AA", "#FF9F9A","#D0BBFF", "#FFFEA3", "#B0E0E6"]
cblind=["#0072B2", "#009E73", "#D55E00","#CC79A7", "#F0E442", "#56B4E9"]
ax[0,0].plot(w_fr, Xabs, color=col[0])
ax[0,0].plot(w_fr, Yabs, color=col[1])
ax[0,0].plot(w_fr, Zabs, color=col[2])

ax[0,1].plot(w_fr, ARX, color=cblind[0])
ax[0,1].plot(w_fr, ARY, color=cblind[1])
ax[0,1].plot(w_fr, ARZ, color=cblind[2])

ax[1,0].plot(w_fr, Xph, color=col[0])
ax[1,0].plot(w_fr, Yph, color=col[1])
ax[1,0].plot(w_fr, Zph, color=col[2])

ax[1,1].plot(w_fr, RXph, color=muted[0])
ax[1,1].plot(w_fr, RYph, color=muted[1])
ax[1,1].plot(w_fr, RZph, color=muted[2])


#### Interpolate and replot
spl = spline(w_fr, ARX)
ARXf = spl(fHz)
spl = spline(w_fr, ARY)
ARYf = spl(fHz)
spl = spline(w_fr, ARZ)
ARZf = spl(fHz)

ax[0,1].plot(fHz, ARXf, color=cblind[3], ls='--')
ax[0,1].plot(fHz, ARYf, color=cblind[4], ls='--')
ax[0,1].plot(fHz, ARZf, color=cblind[5], ls='--')

spl = spline(w_fr, RXph)
RXphf = spl(fHz)
spl = spline(w_fr, RYph)
RYphf = spl(fHz)
spl = spline(w_fr, RZph)
RZphf = spl(fHz)

ax[1,1].plot(fHz, RXphf, color=muted[3], ls='--')
ax[1,1].plot(fHz, RYphf, color=muted[4], ls='--')
ax[1,1].plot(fHz, RZphf, color=muted[5], ls='--')

spl = spline(w_fr, w_amp)
w_ampf = spl(fHz)
spl = spline(w_fr, w_ph - om*kR0)
w_phf = spl(fHz)

#hfD_r = w_ampf*np.exp(1.0j*w_phf)
#hfD = w_amp*np.exp(1.0j*w_ph - 1.0j*om*kR0)

#fctr = (fstar/w_fr)**2
XAf = w_ampf*ARXf*(fHz/fstar)**2
YAf = w_ampf*ARYf*(fHz/fstar)**2
ZAf = w_ampf*ARZf*(fHz/fstar)**2
Xphf = w_phf - RXphf
Yphf = w_phf - RYphf
Zphf = w_phf - RZphf

ax[0,0].plot(fHz, XAf, color=col[3], ls='--')
ax[0,0].plot(fHz, YAf, color=col[4], ls='--')
ax[0,0].plot(fHz, ZAf, color=col[5], ls='--')


TrX = w_ampf**ARXf*(fHz/fstar)**2 * np.exp(1.j*w_phf)*np.exp(-1.j*RXphf)
tmp = np.angle(TrX)
tmp = np.unwrap(tmp)


ax[1,0].plot(fHz, tmp, color=col[3], ls='--')
#ax[1,0].plot(fHz, Xphf, color=col[3], ls='--')
#ax[1,0].plot(fHz, Yphf, color=col[4], ls='--')
#ax[1,0].plot(fHz, Zphf, color=col[5], ls='--')

plt.show()

hfD_r = w_ampf*np.exp(1.0j*w_phf)
RXr = ARXf*(fHz/fstar)**2*np.exp(1.j*RXphf)
Xr = hfD_r*np.conjugate(RXr)

aXr = np.absolute(Xr)
phXr = np.angle(Xr)
phXr = np.unwrap(phXr)

# tmp = np.angle(hptilde)
# tmp = np.unwrap(tmp)


# plt.plot(w_fr, w_ph-om*kR0 - (w_ph-om*kR0)[0])
# plt.plot(fHz, w_phf-w_ph[0], '--')
# plt.plot(fHz, tmp, '-.')
# plt.plot(w_fr, np.real(RX))
# plt.plot(fHz, np.real(RXr), '--')
# plt.plot(w_fr, np.imag(RX))
# plt.plot(fHz, np.imag(RXr), '--')
# plt.plot(w_fr, np.real(X))
# plt.plot(fHz, np.real(Xr), '--')
# plt.plot(w_fr, np.imag(X))
# plt.plot(fHz, np.imag(Xr), '--')
# plt.plot(w_fr, np.real(hfD))
# plt.plot(fHz, np.real(hfD_r), '--')
# plt.plot(w_fr, np.imag(hfD))
# plt.plot(fHz, np.imag(hfD_r), '--')
#plt.plot(w_fr, Xabs)
#plt.plot(fHz, aXr, '--')
plt.plot(w_fr, Xph)
plt.plot(fHz, phXr, '--')

plt.show()


#### Interpolate and then compute X
