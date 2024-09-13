## Import standard module
import time, sys, os
import numpy as np

## Import numerical module
import scipy
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline
from scipy import interpolate


import LISAConstants as LC
# from LISACommonFunctions import AziPolAngleL2PsiIncl
# from LISAhdf5 import LISAhdf5,ParsUnits,Str
#import Cosmology
import pyIMRPhenomD
import pyFDresponse as FD_Resp
# import LW_simple as LW

#import FastGB as FB


## FIXME remove/cvomment those lines before pushing!!!
# import matplotlib
# # matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# plt.style.use('Stas_plot')

### Common parameter
MfCUT_PhenomD = 0.2 - 1e-7

##################
# Main functions #
##################

####

def AziPolAngleL2PsiIncl(bet, lam, theL, phiL):
    """
    Convert Polar and Azimuthal angles of zS (typically orbital angular momentum L)
    to polarisation and inclination (see doc)
    @param bet is the ecliptic latitude of the source in sky [rad]
    @param lam is the ecliptic longitude of the source in sky [rad]
    @param theL is the polar angle of zS [rad]
    @param phiL is the azimuthal angle of zS [rad]
    @return polarisation and inclination
    """
    #inc = np.arccos( np.cos(theL)*np.sin(bet) + np.cos(bet)*np.sin(theL)*np.cos(lam - phiL) )
    #up_psi = np.cos(theL)*np.cos(bet) - np.sin(bet)*np.sin(theL)*np.cos(lam - phiL)
    #down_psi = np.sin(theL)*np.sin(lam - phiL)
    #psi = np.arctan2(up_psi, down_psi)

    inc = np.arccos( - np.cos(theL)*np.sin(bet) - np.cos(bet)*np.sin(theL)*np.cos(lam - phiL) )
    down_psi = np.sin(theL)*np.sin(lam - phiL)
    up_psi = -np.sin(bet)*np.sin(theL)*np.cos(lam - phiL) + np.cos(theL)*np.cos(bet)
    psi = np.arctan2(up_psi, down_psi)

    return psi, inc

def ComputeMBHBXYZ_FD(p, verbose=False, Larm=2.5e9, df=1e-5):
    """
    Generates TDIs (X, Y, Z) in Fourrier Domain using the Sylvain's approximation
    @param p is the parameters of GW MBHB as read from hdf5 file
    @return freq is array of frequency
    @return complex array of TDI X
    @return complex array of TDI Y
    @return complex array of TDI Z
    """
    #{{{

    st = time.time()

    ##### Read the parameters and compute derived quantity
    m1 = p.getConvert('Mass1',LC.convMass,'solarmass')  ### those are intrinsic masses
    m2 = p.getConvert('Mass2',LC.convMass,'solarmass')
    redshift = p.get('Redshift')
    #m1=m1*(1+redshift) # changed by yishuxu on Wed, 23rd April, 2020
    #m2=m2*(1+redshift) # ysx, 23/4/2020
    DL = p.getConvert('Distance',LC.convDistance,'mpc')
    tc = p.getConvert('CoalescenceTime',LC.convT,'sec')

    phi0 = p.getConvert('PhaseAtCoalescence',LC.convAngle,'rad')

    chi1s = p.get('Spin1')
    chi2s = p.get('Spin2')

    Stheta1s = p.getConvert('PolarAngleOfSpin1',LC.convAngle,'rad') #!!!! WRONG TODO What's wrong ?
    Stheta2s = p.getConvert('PolarAngleOfSpin2',LC.convAngle,'rad') #!!!! WRONG TODO

    a1 = np.cos(Stheta1s)*chi1s
    a2 = np.cos(Stheta2s)*chi2s

    ### Note that inc = iota_o
    bet, lam, inc, psi = GetSkyAndOrientation(p)


    ### To make FD or LW matching the LISACode polarization definition
    # psi_FD = 0.5*np.pi - psi   ### SB: to be consistent with LISACOde, LW and documentation.
    psi_FD = psi

    # print (bet, lam, theL, phiL, psi, inc, 'a')
    #### Compute polarisation angle
    ### psi and h+, hx is taken from https://arxiv.org/pdf/0806.2110.pdf -- there is typo there!!!

    #### Compute luminosity disctance
    dist=DL
    #dist = Cosmology.DL(redshift, w=0)[0]
    #if (dist != DL):
    #    print ("ComputeMBHBXYZ_FD: DL and redshift in the hdf5 file do not match adopted cosmological model")
    #    raise ValueError
    #print ("DL:", dist, DL, DL-dist)
    #sys.exit(0)

    # m1 =  m1s*(1+redshift)   ### redshifted masses
    # m2 =  m2s*(1+redshift)

    Tobs = p.get("ObservationDuration")
    del_t = p.get("Cadence")

    #df = 0.5e-8
    #print ("Assumed duration ", 1.0/df)
    # Mc = FD_Resp.funcMchirpofm1m2(m1, m2)
    # f0 not used.
    # f0 = FD_Resp.funcNewtonianfoft(Mc, 2.0*Tobs/LC.YRSID_SI)
    #### TODO FIXME hardcoded parameters -- not good!
    # fRef=0 means fRef=fpeak in PhenomD or maxf if out of range
    fRef = 0.0
    t0 = 0.0

    ### Define fmax based on the time step
    fmx = 0.5/del_t

    ##### Generate waveform as frequency, amplitude and the pahse of 2,2 mode
    if verbose:
        print("FD: parameters for PhenomD:", phi0, fRef, m1, m2, a1, a2, dist, inc, lam, bet, \
                       psi_FD, Tobs, tc, Tobs/LC.YRSID_SI)

    st = time.time()
    #print(fRef, m1, m2, a1, a2, df, fmx, "debugging")
    fr, X, Y, Z, wfTDI = MBHB_LISAGenerateTDI(phi0, fRef, m1, m2, a1, a2, dist, inc, lam, bet, \
                        psi_FD, Tobs, df, fmx, tc, settRefAtfRef=False, tRef=0.0, \
                        trajdict=FD_Resp.trajdict_MLDC, TDItag='TDIXYZ', nptmin=450, \
                        order_fresnel_stencil=0, verbose=verbose, Larm=Larm)

    # fr, X, Y, Z = GenTDIFD.ComputeMBHBXYZ_FD(p)

    # plt.loglog(fr, np.abs(X), 'o-')
    # plt.show()

    #### Build the frequency array
#    print('(debugging) fr=',fr)
    if isinstance(fr, int):
        df=fr 
    else:
        df = fr[1]-fr[0]
    Nf = int(fmx/df+1)
    freq = np.arange(Nf)*df
    # i_b = int(fr[0]/df)
    i_b = int(np.rint(fr[0]/df))
    if verbose:
        print ("Stas test:", fr[0], fr[0]/df, fr[0]-i_b*df, i_b, np.rint(fr[0]/df), df, 0.5/Tobs)
        print ("fr = ",fr)
        print ("freq = ",freq)
        print ("Starting bin i_b = ", i_b, " corresponding to fr[0] = ", fr[0], " => segment in frequency [ i_b*df = ", i_b*df, " i_b_b+len(X) = ", (i_b+len(X))*df,"] corresponding to bins [",i_b,",",len(X)+i_b,"]")

    ### Create the data array corersponding to the frequency

    Xf = np.zeros(Nf, dtype='complex128')
    Yf = np.zeros(Nf, dtype='complex128')
    Zf = np.zeros(Nf, dtype='complex128')

    ### Copy the results from MBHB_LISAGenerateTDI in the correct frequency bin segment
    Xf[i_b:len(X)+i_b] = X
    Yf[i_b:len(X)+i_b] = Y
    Zf[i_b:len(X)+i_b] = Z
    # Yf[:len(Xfr)] = np.copy(np.conjugate(Yfr))
    # Zf[:len(Xfr)] = np.copy(np.conjugate(Zfr))
    # sys.exit(0)

    en = time.time()
    if verbose:
        print ("ellapsed time ", en-st)

    # return (freq, Xf, Yf, Zf, wfTDI)  ### For debugging
    return (freq, Xf, Yf, Zf)

    #}}}

#def MBHB_LISAGenerateTDIfast(phi0, fRef, m1, m2, a1, a2, dist, inc, lam, beta, psi, Tobs, tc, del_t, tShift=0.0, fmin=1.e-5, fmax=0.1, frqs=None, resf=None):
#    #### Using new method from Sylvain
#    """
#    Compute X, Y, Z
#    @param phi0 azimuthal direction to the observer in the source frame (varphi_0)
#    @param fRef reference frequency at which the FD phase is specified, 0 to ignore (then defaults to min of peak frequency and max frequency)
#    @param m1 primary mass (observed) in solar masses
#    @param m2 secondary mass (observed) in solar mass
#    @param a1 spin of primary
#    @param a2 spin of secondary
#    @param dist luminosity distance in mpc
#    @param inc inclination polar direction to the observer in the source frame (iota_0)
#    @param lam ecliptic longitude
#    @param beta ecliptic latitude
#    @param psi polarization angle
#    @param Tobs observation duration in sec
#    @param tc time of coalescence in sec
#    @param del_t time cadence in sec
#    @param  tShift time shift of the template in sec
#    @param fmin min frequency for the waveform generation
#    @param fmax max frequency for the waveform generation
#    @param frqs frequency array for which the waveform will be generated
#    @param resf shift in frequency (in Hz) due to Stas' fuck up of the LDC MBHB generation
#    @return frequency array (df=1/Tobs) and Xf, Yf, Zf
#    """
#
#    acc_sampling = 1.e-5  ## hardcoded tolerance for the interpolation
#    m1_SI = m1*LC.MsunKG
#    m2_SI = m2*LC.MsunKG
#    Ms = (m1 + m2) * LC.MTsun
#    dist_SI = dist*1e6*LC.pc
#    chi1 = a1
#    chi2 = a2
#    Mc = FD_Resp.funcMchirpofm1m2(m1, m2)
#    eta = m1*m2/(m1+m2)**2
#    ### TODO DO we need 2*Tobs? MAybe 1.5Tobs is enough?
#    f0 = FD_Resp.funcNewtonianfoft(Mc, 2.0*Tobs/LC.YRSID_SI) ### We hardcode that we generate the waveform starting from f0
#    df = 0.5/Tobs   #### Need to try 1/Tobs
#    fstart_safe_duration = 0.003 / Ms
#    if (fstart_safe_duration < f0):
#        f0 = fstart_safe_duration
#    if (fmin > f0):
#        f0 = fmin
#
#    # print ("f0 = ", f0)
#
#    fCUT_PhD = MfCUT_PhenomD / Ms
#    fend = np.fmin(fmax, fCUT_PhD)
#
#    # Construct frequency grid with sampling adapted to waveform - converting between f and Mf
#    freq_PhD = 1/Ms * FD_Resp.WaveformFrequencyGridGeom(eta, Ms*f0, Ms*fend, acc=acc_sampling)
#
#    # Generate the waveform
#    phi_ref = 0.0
#    # print ("2:", freq_PhD, phi_ref, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
#    wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq_PhD, phi_ref, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
#    freq_PhD, amp_PhD, phase_PhD = wf_PhD_class.GetWaveform()
#    # print ("2", np.array(freq_PhD), np.array(phase_PhD))
#
#
#    # print ("num point in wvf generation:", len(freq_PhD))
#
#    tfspline = spline(freq_PhD, 1./(2.*np.pi) * phase_PhD).derivative()
#    tf = tfspline(freq_PhD)
#    Shift = tf[-1] - tc
#    tf = tf - Shift
#
#    index_cuttf = 0
#    tfdiff = np.diff(tf)
#    while index_cuttf<len(tfdiff)-1 and tfdiff[index_cuttf]>0:
#        index_cuttf += 1
#    tfr = tf[:index_cuttf+1]
#    # print ("cutoff:", index_cuttf, len(tf), tf[index_cuttf], tf[-1])
#    freq_PhDr = freq_PhD[:index_cuttf+1]
#    ftspline = spline(tfr, freq_PhDr)
#
#    # plt.plot(freq_PhD, tf)
#    # plt.plot(freq_PhD[:index_cuttf], tf[:index_cuttf], '--')
#    # plt.show()
#
#    # If tobs is within the range covered by f(t), use it to determine minf_rs
#    minf_rs = fmin
#    if (tfr[-1] - tfr[0] > Tobs):
#        #minf_rs = np.fmax(freq_PhD[0], ftspline(-Tobs))
#        minf_rs = np.fmax(freq_PhD[0], ftspline(tfr[-1]-Tobs))
#    else:
#        raise ValueError("Error in GenerateResamplePhenomD: tf for interpolation of ft (from monotonous part of tf) does not cover tobs.")
#
#    # print ("freq cut:", minf_rs, freq_PhD[0])
#
#    # If Tobs causes a cut of the waveform, build a new frequency grid and resample - note conversion between f and Mf
#    if minf_rs > freq_PhD[0]:
#        # print ("re-doing the freq grid")
#        freq_rs = 1/Ms * FD_Resp.WaveformFrequencyGridGeom(eta, Ms*minf_rs, Ms*freq_PhD[-1], acc=acc_sampling)
#        amp_int = spline(freq_PhD, amp_PhD)
#        phase_int = spline(freq_PhD, phase_PhD)
#        amp_rs = amp_int(freq_rs)
#        # The following is not necessary anynmore
#        # fRef=0 means fRef=fpeak in PhenomD or maxf if out of range
#        # # If fRef=0, in Phenom fRef is set to fstart. We have to propagate a phase shift due to the change of fstart coming from cutting with tobs
#        # if fRef <= 0.:
#        #     phase_shift = phase_int(freq_PhD[0]) - phase_int(freq_rs[0])
#        # else:
#        #     phase_shift = 0.
#        # phase_rs = phase_int(freq_rs) + phase_shift
#        phase_rs = phase_int(freq_rs)
#    else:
#        freq_rs, amp_rs, phase_rs = np.copy(freq_PhD), np.copy(amp_PhD), np.copy(phase_PhD)
#    # print ("num point in wvf generation:", len(freq_rs))
#
#    # If settRefAtfRef, arrange so that t=tRef at fRef
#    # FIXME: improve the check that fRef is in range - here simply forced back in
#    settRefAtfRef = False ### TODO not using for now, might want to use merger time as tref/freq_ref
#    if settRefAtfRef:
#        fRef_inrange = min(max(freq_rs[0], fRef), freq_rs[-1])
#        phase_rs += 2*pi * (freq_rs - fRef_inrange) * (tRef - tfspline(fRef_inrange))
#
#    # Output -> PhenomD  freq_rs, amp_rs, phase_rs
#    #### Computing set of frequencies  for the response
#    freq_response = FD_Resp.ResponseFrequencyGrid([freq_rs, amp_rs, phase_rs])
#
#    # print ("num freqs in resp", len(freq_response))
#    tm_response = tfspline(freq_response)-Shift
#
#    # plt.plot(freq_PhD, tf)
#    # plt.plot(freq_PhD[:index_cuttf], tf[:index_cuttf], '--')
#    # plt.plot(freq_response, tm_response, 'o')
#    # plt.show()
#
#    Tfvec_rs = 0.0
#    epsTfvec_rs = 0.0
#    wfTDI =  JustLISAFDresponseTDI(freq_response, tm_response, Tfvec_rs, epsTfvec_rs, inc, lam, beta, psi, phi0, t0=0.0, \
#                    order_fresnel_stencil=0)
#
#    # print ("freqs:", freq_rs)
#
#    df = 1.0/Tobs
#    fmax = 0.5/del_t
#    Nf = int(fmax/df)+1
#    # print ("resf= ", resf, 2.7770969708421946e-09)
#    # fr_ar =  np.arange(Nf)*df + 2.7770969708421946e-09
#    fr_F =  np.arange(Nf)*df   ### apply the frequency shift if given
#    # print (fr_ar)
#    # print (df, Nf)
#    i_st = int(np.ceil(freq_rs[0]/df))
#    i_en = int(np.floor(freq_rs[-1]/df))
#    # print ("indices", i_st, i_en)
#    fr_wvf = fr_F[i_st:i_en]
#    if isinstance(frqs, np.ndarray):  ### frqs if supplied by user
#        ib = 0
#        ie = len(frqs)
#        if (fr_F[i_st] > frqs[0]):
#            ib = np.argwhere(frqs >= fr_F[i_st])[0][0]
#        if (fr_F[i_en] < frqs[-1]):
#            ie = np.argwhere(frqs >= fr_F[i_en])[0][0]
#        # print ("i's:", ib, ie, frqs[0], frqs[ib], frqs[-1], frqs[ie])
#
#        fr_wvf = frqs[ib:ie]
#
#    amp_spl = spline(freq_rs, amp_rs)
#    ph_spl = spline(freq_rs, phase_rs)
#
#    amp_wvf = amp_spl(fr_wvf)
#    ph_wvf = ph_spl(fr_wvf)
#
#    # plt.loglog(freq_rs, amp_rs)
#    # plt.loglog(fr_wvf, amp_wvf, '.')
#    # plt.show()
#    #
#    # plt.semilogx(freq_rs, phase_rs, 'o-')
#    # plt.semilogx(fr_wvf, ph_wvf, '.')
#    # plt.show()
#
#    phaseRdelayspline = spline(freq_response, wfTDI['phaseRdelay'])
#    phaseRdelay = phaseRdelayspline(fr_wvf)
#
#    # plt.semilogx(freq_response, wfTDI['phaseRdelay'], 'o-')
#    # plt.semilogx(fr_wvf, phaseRdelay, '.')
#    # plt.show()
#
#    phasetimeshift = 2.*np.pi*(tc+tShift)*fr_wvf
#    fastPart = amp_wvf * np.exp(1j*(ph_wvf + phaseRdelay + phasetimeshift))
#
#    keytrs = ['transferL1', 'transferL2', 'transferL3']
#
#    for I, ky in enumerate(keytrs):
#        transferLRespline = spline(freq_response, np.real(wfTDI[ky]))
#        transferLImspline = spline(freq_response, np.imag(wfTDI[ky]))
#        transferLRe = transferLRespline(fr_wvf)
#        transferLIm = transferLImspline(fr_wvf)
#        if (ky == 'transferL1'):
#            X = np.conjugate((transferLRe+1j*transferLIm) * fastPart)
#            # X = np.conjugate(fastPart)
#        if (ky == 'transferL2'):
#            Y = np.conjugate((transferLRe+1j*transferLIm) * fastPart)
#        if (ky == 'transferL3'):
#            Z = np.conjugate((transferLRe+1j*transferLIm) * fastPart)
#
#    # print ("response:", en-st)
#    ### debugging info
#    # wfTDI['fr_resp'] = wfTDI['freq']
#    wfTDI['fr_resp'] = freq_response
#    wfTDI['tm_resp'] = wfTDI['tfvec']
#    wfTDI['freq'] = freq_rs
#    #wfTDI['fast'] = [fr_wvf, fastPart]
#    wfTDI['fast'] = [fr_wvf, ph_wvf + phaseRdelay + phasetimeshift]
#    # wfTDI['tfvec'] =
#    wfTDI['amp'] = amp_rs
#    wfTDI['phase'] = phase_rs
#    wfTDI['test'] = [np.array(freq_PhD), np.array(amp_PhD), np.array(phase_PhD)]
#
#    if isinstance(frqs, np.ndarray):
#        return(fr_wvf, X, Y, Z, wfTDI)
#    else:
#        Xfull = np.zeros(Nf, dtype=np.complex128)
#        Yfull = np.zeros(Nf, dtype=np.complex128)
#        Zfull = np.zeros(Nf, dtype=np.complex128)
#
#        Xfull[i_st:i_en] = X
#        Yfull[i_st:i_en] = Y
#        Zfull[i_st:i_en] = Z
#
#        # return (fr_ar - 2.7770969708421946e-09, Xfull, Yfull, Zfull, wfTDI)
#        # return (fr_ar - resf, Xfull, Yfull, Zfull, wfTDI)
#        return (fr_F, Xfull, Yfull, Zfull, wfTDI)
#
#
#
#
##### This code is obsolete for now, need to be improved/tested but has potentials to speed
## the signal production
#def ComputeMBHBXYZ_FD_old(p,verbose=False):
#    """
#    [Old version] Generates TDIs (X, Y, Z) in FD using the Sylvain's approximation
#    @param p is the parameters of GW MBHB as read from hdf5 file
#    @return freq is array of frequency
#    @return complex array of TDI X
#    @return complex array of TDI Y
#    @return complex array of TDI Z
#    """
#    #{{{
#
#    st = time.time()
#
#    ### Getting parameters:
#    m1 = p.getConvert('Mass1',LC.convMass,'solarmass')
#    m2 = p.getConvert('Mass2',LC.convMass,'solarmass')
#    redshift = p.get('Redshift')
#    DL = p.getConvert('Distance',LC.convDistance,'mpc')
#    tc = p.getConvert('CoalescenceTime',LC.convT,'sec')
#
#    phi0 = p.getConvert('PhaseAtCoalescence',LC.convAngle,'rad')
#
#    chi1s = p.get('Spin1')
#    chi2s = p.get('Spin2')
#
#    Stheta1s = p.getConvert('PolarAngleOfSpin1',LC.convAngle,'rad') #!!!! WRONG TODO
#    Stheta2s = p.getConvert('PolarAngleOfSpin2',LC.convAngle,'rad') #!!!! WRONG TODO
#
#    a1 = np.cos(Stheta1s)*chi1s
#    a2 = np.cos(Stheta2s)*chi2s
#
#
#    bet, lam, inc, psi = GetSkyAndOrientation(p)
#
#
#
#    # phiL = p.get("InitialAzimuthalAngleL")
#    # bet = p.get("EclipticLatitude")
#    # lam = p.get("EclipticLongitude")
#    # ### psi and h+, hx is taken from https://arxiv.org/pdf/0806.2110.pdf
#    # up = np.sin(bet)*np.cos(lam - phiL)*np.sin(inclination) - np.cos(inclination)*np.cos(bet)
#    # down = np.cos(bet)*np.sin(lam - phiL)
#    # psi = np.arctan2(up, down)
#
#
#    # m1 =  m1s*(1+redshift)   ### redshifted masses
#    # m2 =  m2s*(1+redshift)
#
#    Tobs = p.get("ObservationDuration")
#    del_t = p.get("Cadence")
#
#    df = 1.e-8
#    #df = 1.0/Tobs
#    Mc = FD_Resp.funcMchirpofm1m2(m1, m2)
#    f0 = FD_Resp.funcNewtonianfoft(Mc, 2.0*Tobs/LC.YRSID_SI)
#    # fRef=0 means fRef=fpeak in PhenomD or maxf if out of range
#    fRef = 0.
#
#    ### Generate waveform as frequency, amplitude and the pahse of 2,2 mode
#    if verbose:
#        print ("FD_old: parameters for PhenomD:", phi0, fRef, m1, m2, a1, a2, DL, inc,  lam, bet, psi, Tobs, tc, f0, Tobs/LC.YRSID_SI)
#    w_fr, w_amp, w_ph = FD_Resp.GenerateResamplePhenomD(phi0, fRef=fRef, m1=m1, m2=m2, chi1=a1, \
#                                            chi2=a2, dist=DL, inc=inc, minf=f0, tobs=Tobs/LC.YRSID_SI, nptmin=500)
#    tfspline = spline(w_fr, 1/(2.*np.pi)*(w_ph-w_ph[0])).derivative() # get rid of possibly huge constant in the phase before interpolating
#    tfvec = tfspline(w_fr) #### the time associated with the frequency
#
#    #### the shift:
#    spl = UnivariateSpline(w_fr, tfvec, k=4, s=0)
#    root = np.array(spl.derivative().roots())[0]
#    i_merg = np.argwhere(w_fr > root)[0][0]
#    #Shift = tfvec[i_merg] - tc
#    Shift = tfvec[-1] - tc
#    #print ("Shifts:", tfvec[i_merg]-tfvec[-1])
#    #print ("Shifts:", Shift, tfvec[-1])
#
#    # plt.semilogx(w_fr,tfvec-Shift)
#    # plt.show()
#
#    ### Now generate the TDI in FD
#
#    wvf = [w_fr, w_amp, w_ph]
#    #[t0, trajdict, TDItag] = [-Shift, FreqResp.trajdict_MLDC, "TDIXYZ"]
#    [t0, trajdict, TDItag] = [0., FD_Resp.trajdict_MLDC, "TDIXYZ"]
#
#
#    # psi = 0.5*np.pi - psi
#
#    #wf_tdi = FreqResp.LISAFDresponseTDI(wvf, incl, lam, bet, np.pi*0.5-psi, t0=t0, trajdict=trajdict, TDItag=TDItag)
#    #wf_tdi = FreqResp.LISAFDresponseTDI(wvf, inclination, lam, bet, psi, t0=t0, trajdict=trajdict, TDItag=TDItag)
#    #print ('paramsG:', inclination, lam, bet, psi, -Shift/LC.YRSID_SI)
#    #wf_tdi = FreqResp.LISAFDresponseTDI(w_fr, wvf, inclination, lam, bet, psi, t0=t0/LC.YRSID_SI, trajdict=trajdict, TDItag=TDItag)
#    # print ("old2", inclination, lam, bet, psi)
#    wf_tdi = FD_Resp.LISAFDresponseTDI(w_fr, wvf, inc, lam, bet, psi, t0=-Shift/LC.YRSID_SI, trajdict=trajdict, TDItag=TDItag)
#    f, amp, phase, phaseRdelay, transferL1, transferL2, transferL3, TDI1, TDI2, TDI3, TDItag = \
#    wf_tdi['freq'], wf_tdi['amp'], wf_tdi['phase'], wf_tdi['phaseRdelay'], wf_tdi['transferL1'], wf_tdi['transferL2'], \
#    wf_tdi['transferL3'], wf_tdi['TDI1'], wf_tdi['TDI2'], wf_tdi['TDI3'], wf_tdi['TDItag']
#
#    #### This corresponds to fmax = 0.5/dt
#    fmx = 0.5/del_t
#    Nf = int(fmx/df+1)
#    freq = np.arange(Nf)*df
#
#    Xfr, Yfr, Zfr = ResampleS(f, amp, phase, phaseRdelay, transferL1, transferL2, transferL3, -tc, freq)
#
#    # Xf = np.zeros(Nf, dtype='complex128')
#    # Yf = np.zeros(Nf, dtype='complex128')
#    # Zf = np.zeros(Nf, dtype='complex128')
#    # Xf[:len(Xfr)] = np.copy(np.conjugate(Xfr))
#    # Yf[:len(Xfr)] = np.copy(np.conjugate(Yfr))
#    # Zf[:len(Xfr)] = np.copy(np.conjugate(Zfr))
#
#    en = time.time()
#    if verbose:
#        print ("ellapsed time ", en-st)
#
#    return (freq, Xfr, Yfr, Zfr)
#
#    #}}}
#
#
#def ComputeMBHBXYZ_FD_LW(p, df=1.e-8, verbose=False):
#    """
#    Compute TDI XYZ with IMRPhenomD and low frequency approximation
#    @param p is the parameters of GW MBHB as read from hdf5 file
#    @param df is the frequency sampling
#    @return freq is array of frequency
#    @return complex array of TDI X
#    @return complex array of TDI Y
#    @return complex array of TDI Z
#    """
#
#    st = time.time()
#
#    ### Get parameters
#    m1_SI = p.getConvert('Mass1',LC.convMass,'kg')
#    m2_SI = p.getConvert('Mass2',LC.convMass,'kg')
#    m1 = p.getConvert('Mass1',LC.convMass,'solarmass')
#    m2 = p.getConvert('Mass2',LC.convMass,'solarmass')
#
#    z = p.get("Redshift")
#    DL = Cosmology.DL(z, w=0)[0]
#    dist_SI = DL * 1.e6 * LC.pc
#
#    chi1 = p.get('Spin1')
#    chi2 = p.get('Spin2')
#
#    phi0 = p.get('PhaseAtCoalescence')
#    tc = p.get('CoalescenceTime')
#
#    bet, lam, inc, psi = GetSkyAndOrientation(p)
#
#    Tobs = p.get("ObservationDuration")
#    del_t = p.get("Cadence")
#
#
#    ### Derived mass quantities
#    M = m1+m2
#    Mt = M*LC.MTsun
#    Mc = FD_Resp.funcMchirpofm1m2(m1, m2)
#    f0 = FD_Resp.funcNewtonianfoft(Mc, 2.0*Tobs/LC.YRSID_SI)
#
#    ### Set the frequency array
#    # fRef=0 means fRef=fpeak in PhenomD or maxf if out of range
#    fRef = 0.
#    fmax = MfCUT_PhenomD/Mt
#    Nf = int((fmax-f0)/df)
#    if verbose:
#        print ("Number of points:", Nf, fmax, f0)
#    freq_all = np.arange(Nf)*df + f0
#
#    ### Compute amplitude and phase for IMRPhenomD harmonic 22
#    wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq_all, phi0, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
#    wf_PhD = wf_PhD_class.GetWaveform() # freq, amp, phase
#    frS = wf_PhD[0]
#    phS = wf_PhD[2]
#    ampS = wf_PhD[1]
#
#    ### Reorganise phase and frequency
#    ph_nrm = 1/(2.*np.pi)*(phS-phS[0])
#    tfvec = np.diff(ph_nrm)/df
#    Shift = tfvec[-1] - tc
#    tfvec = tfvec - Shift
#    if (df > 1.0/(tfvec[-1]-tfvec[0])):
#        print ("ERROR in ComputeMBHBXYZ_FD_LW : you should either decrease df or increase f0, currently you undersample the waveform")
#        print ("Compare", 1.0/(tfvec[-1]-tfvec[0]), df)
#        raise ValueError
#
#    ### Define the Long Wavelength approximation
#    LW_MBHB = LW.LW(inc, bet, lam, psi, phi0)
#
#    ## Compute the response of LISA in frequency domain with long-wave approximation
#    phR, RX, RY, RZ = LW_MBHB.ComputeResponseMBHB(frS, ampS, phS, tc)
#
#    ### Applying scaling
#    phasetimeshift = -2.0*np.pi*tc*frS
#    common = ampS* np.exp(1.0j* (-phS + phR + phasetimeshift))
#    ### FIXME do not forget to remove the line below
#    # common = ampS* np.exp(1.0j* (-phS + phasetimeshift))
#
#    ### Apply the response
#    Xf_lw = RX*common
#    Yf_lw = RY*common
#    Zf_lw = RZ*common
#
#    en = time.time()
#    if verbose:
#        print ("ellapsed time ", en-st)
#
#    return frS, Xf_lw, Yf_lw, Zf_lw
#
#
#def SOBBH_LISAGenerateTDI(phi0, fRef, m1, m2, a1, a2, DL, inc, lam, bet, psi, Tobs, minf, maxf, t0, settRefAtfRef=True, tRef=0.0, trajdict=FD_Resp.trajdict_MLDC, TDItag='TDIXYZ', nptmin=500, order_fresnel_stencil=5):
#
#    #TODO Complete description
#    """
#    For generating SOBBH fRef and minf (starting frequency of GW signal) should be the same and correspond to t0
#    ...
#    @param phi0 (varphi_0) azimuthal direction to the observer in the source frame
#    @param fRef reference frequency at which the FD phase is specified, 0 to ignore (then defaults to min of peak frequency and max frequency)
#    @param m1 primary mass (solar masses)
#    @param m2 secondary mass (solar masses)
#    @param a1 dimensionless spin of primary (projection on the orbital angular momentum)
#    @param a2 dimensionless spin of secondary (projection on the orbital angular momentum)
#    @param DL luminocity distance (Mpc)
#    @param inc inclination of the orbital momentum to the line of sight
#    @param lam ecliptic longitude
#    @param bet ecliptic Latitude
#    @param psi polarisation angle
#    @param Tobs observation time (sec)
#    @param minf starting frequency of the GW signal
#    @param maxf desired (!) end frequency, in reality it will be defined by Tobs and minf
#    @param t0 initial time which corresponds to minf
#    @param settRefAtfRef if True moves the t0 to the fRef
#    @param trajdict uses analytic LDC orbit (so far the only option)
#    @param nptmin minimum number of points used in computation FD response function
#    @param order_frensel_stecil order of Frensel stencil in FD response (signal dependent)
#
#    output:
#    frequency array
#    X, Y, Z in frequency domain
#    wfTDI - debugging info dictionary containing auxilary info (response, phase, aplitude)
#    """
#    #{{{
#
#    ## Generate the PhenomD phase and amplitude
#    m1_SI = m1*LC.MsunKG
#    m2_SI = m2*LC.MsunKG
#    dist_SI = DL*1e6*LC.pc
#    chi1 = a1
#    chi2 = a2
#
#    OK = False
#    cnt = 2
#    df = 1.0/Tobs
#    #### Generate PhenomD amplitude and phase on the fine and regular grid of points in freq.
#    #### for efficiency we guess the end frequency and extend it if not enough
#    i_b = int(np.floor(minf/df))  ### FIXME
#    # print ("GenFD: ib = ", i_b)
#    phi_ref = 0.0
#    while not OK:
#        fmax = min(cnt*minf, maxf)
#        print ("freq range", minf, fmax)
#        Nf = int(np.rint(fmax/df))+1
#        freq_all = np.arange(Nf)*df
#        freq_fromf0 = freq_all[i_b:]
#        #Nf = int((fmax-minf)/df)+1
#        #freq_all = np.arange(Nf)*df + minf
#
#        st = time.time()
#        wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq_fromf0, phi_ref, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
#        wf_PhD = wf_PhD_class.GetWaveform() # freq, amp, phase
#        en = time.time()
#        print ("PhenomD elappsed", en-st)
#
#        fr_a = wf_PhD[0]
#        ph_a = wf_PhD[2]
#        amp_a = wf_PhD[1]
#        # print ("test f0", fr_a[0]-minf, freq_fromf0[0]-minf, df)
#        # print (fr_a[:20] - freq_fromf0[:20])
#
#        tfspline = spline(fr_a, 1/(2.*np.pi)*ph_a).derivative()
#        tf = tfspline(fr_a)
#
#        #tfspline = interpolate.splrep(fr_a, 1/(2.*np.pi)*ph_a, s=4)
#        #tf = interpolate.splev(fr_a, tfspline, der=1)
#
#        #print ("tf", tf)
#        #print ("We have applied shift:", tf[0], Tobs)
#        tf = tf - tf[0]
#
#        if (tf[-1] > Tobs):
#            ind = np.argwhere(tf > Tobs)[0][0]
#            frspl = spline(tf[:ind], fr_a[:ind])
#            fr_end = frspl(Tobs)
#
#            print ("signal end fr = ", fr_end)
#            OK = True
#        else:
#            if (fmax == maxf):
#                #print ("here")
#                # print (np.diff(tf))
#                ind =np.argwhere(np.diff(tf) <0)
#                # print (len(ind))
#                if (len(ind) > 0):
#                    ind = ind[0][0]
#                else:
#                    ind = len(tf)-1
#                # ind =np.argwhere(np.diff(tf) <0)[0][0]
#                # print ("ind= ", ind, len(tf))
#                # print (np.diff(tf[:ind]))
#                # print (np.diff(tf[ind:]))
#                frspl = spline(tf[:ind], fr_a[:ind])
#                fr_end = fr_a[ind]
#
#                # fr_end = fr_a[-1]
#                # ind = len(fr_a)-1
#                # plt.plot(fr_a, tf)
#                # plt.show()
#                # frspl = spline(tf, fr_a)
#                OK = True
#            else:
#                cnt += 1
#
#
#
#        #plt.semilogy(fr_a, tf/LC.YRSID_SI)
#        # plt.plot(fr_a, tf/LC.YRSID_SI)
#        # plt.show()
#
#    #print (fr_a, fr_end)
#    dffphasespline = spline(fr_a[:ind], ph_a[:ind]).derivative(2)
#    epsTf2vec = 1./(4*np.pi*np.pi)*dffphasespline(fr_a[:ind+2])
#    Tfvec = np.sqrt(np.abs(epsTf2vec))
#    epsTfvec = np.sign(epsTf2vec)
#
#    ### start and the end of the signal
#    frS_start = fr_a[0]
#    frS_end = fr_end
#    i_end = int(np.rint((frS_end-fr_a[0])/df))
#    #print ("check", fr_end- fr_a[i_end], fr_end-fr_a[i_end+1])
#
#    frS = fr_a[:i_end]
#    ampS = amp_a[:i_end]
#    phS = ph_a[:i_end]
#
#    ### Constructing the reduced freq. array for computing the response f-n
#    # uniform in time
#    Delta_t = 1.0/24.0
#    times_deltatresampling = np.linspace(tf[0], tf[ind], int(np.ceil(abs(tf[0] - tf[ind])/(Delta_t*LC.YRSID_SI)))+1)
#
#    if len(times_deltatresampling)==0:
#        freq_rsdeltat = np.array([])
#    else:
#        freqs_deltatresampling = frspl(times_deltatresampling)
#        freqs_deltatresampling[0] = minf
#        freqs_deltatresampling[-1] = frS[-1]
#
#    inp_pos = np.argwhere(freqs_deltatresampling-frS[-1] <= 0.0)[:,0]
#    freqs_deltatresampling =  freqs_deltatresampling[inp_pos]
#    # print (inp_pos)
#
#    # print ("from time", freqs_deltatresampling-frS[-1])
#    # print ("from time-2", freqs_deltatresampling[inp_pos]-frS[-1])
#
#    # UNiform in freq.
#    Delta_f = 0.0005
#    npt_deltafsampling = int(np.ceil((frS[-1] - frS[0])/Delta_f))
#    if npt_deltafsampling<=0: # if f_startdeltafsampling>=maxfrs, do not do deltaf-sampling
#        freq_rsdeltaf = np.array([])
#    else:
#        ind_l = np.linspace(0, len(frS)-1, num=npt_deltafsampling, dtype='int')
#        freq_rsdeltaf = frS[ind_l[1:-1]]
#
#    # print("from uniform fr", freq_rsdeltaf-frS[-1])
#
#    # uniform in log
#    freq_rsln = np.logspace(np.log10(minf,), np.log10(frS[-1]), num=nptmin)
#
#    # print ("from uniform log-fr", freq_rsln-frS[-1])
#
#    freq_tmp = np.concatenate((freqs_deltatresampling, freq_rsdeltaf, freq_rsln[1:-1]))
#
#    # print (np.shape(freq_tmp))
#    freq_tmp = np.sort(freq_tmp)
#    freq_tmp = np.unique(freq_tmp)
#
#    # print ("all downselelcted freqs", freq_tmp - frS[-1])
#    # sys.exit(0)
#
#    ind_fi = np.zeros(len(freq_tmp), dtype='int')
#    freq_rs = np.zeros(len(freq_tmp))
#    freq_rs[0] = freq_tmp[0]
#    freq_rs[-1] = freq_tmp[-1]
#    ind_fi[-1] = len(frS)-1
#    # print (np.shape(freq_tmp), np.shape(frS))
#    for i in range(1, len(freq_tmp)-1):
#        fi = freq_tmp[i]
#        i_f = int(np.floor((fi-frS[0])/df))
#        # print (i, fi, i_f)
#        # print ("df", i, i_f, fi - frS[-1], frS[1]-frS[0])
#        freq_rs[i] = frS[i_f]
#        ind_fi[i] = i_f
#    #en1 = time.time()
#
#    freq_rs, ind_u = np.unique(freq_rs, return_index=True)
#    ind_fi = ind_fi[ind_u]
#    timeFr = tf[ind_fi]
#
#    Tfvec_rs = Tfvec[ind_fi]
#    epsTfvec_rs = epsTfvec[ind_fi]
#
#    # Ar = ampS[ind_fi]
#    # Phr = phS[ind_fi]
#
#    ### Shifting the to the reference
#    #settRefAtfRef = True
#    # fRef = f0
#    # tRef = 0.0
#    if settRefAtfRef:
#        fRef_inrange = min( max(frS[0], fRef), frS[-1] )
#        # print ("Important:", fRef_inrange)
#        Del_Phase = 2.*np.pi * (frS - fRef_inrange) * (tRef - tfspline(fRef_inrange))
#
#        #tf_time = interpolate.splev(fRef_inrange, tfspline, der=1)
#        #Del_Phase = 2.*np.pi * (frS - fRef_inrange) * (tRef - tf_time)
#        # Del_shift = (tRef - tfspline(fRef_inrange))
#        # print ("Shift = ", Del_shift)
#        phS += Del_Phase
#
#    # Compute the response
#    wfTDI =  JustLISAFDresponseTDI(freq_rs, timeFr, Tfvec_rs, epsTfvec_rs, inc, lam, bet, psi, phi0, t0=0.0, \
#                    order_fresnel_stencil=order_fresnel_stencil)
#
#    wfTDI['phase'] = phS[ind_fi]
#    wfTDI['amp'] = ampS[ind_fi]
#
#    ### Interpolating and building the TDIs
#
#    phaseRdelayspline = spline(freq_rs, wfTDI['phaseRdelay'])
#    phaseRdelay = phaseRdelayspline(frS)
#    phasetimeshift = 2.*np.pi*t0*frS
#    fastPart = ampS * np.exp(1j*(phS+phaseRdelay+phasetimeshift))
#    # fastPart = ampS * np.exp(1j*(phS+phaseRdelay))
#
#    # plt.plot(freq_rs, np.sin(wfTDI['phaseRdelay']))
#    # plt.plot(frS, np.sin(phaseRdelay))
#    # plt.show()
#    # sys.exit()
#
#    keytrs = ['transferL1', 'transferL2', 'transferL3']
#    for I, ky in enumerate(keytrs):
#        transferLRespline = spline(freq_rs, np.real(wfTDI[ky]))
#        transferLImspline = spline(freq_rs, np.imag(wfTDI[ky]))
#        transferLRe = transferLRespline(frS)
#        transferLIm = transferLImspline(frS)
#        if (ky == 'transferL1'):
#            X = np.conjugate((transferLRe+1j*transferLIm) * fastPart)
#            # X = np.conjugate(fastPart)
#        if (ky == 'transferL2'):
#            Y = np.conjugate((transferLRe+1j*transferLIm) * fastPart)
#        if (ky == 'transferL3'):
#            Z = np.conjugate((transferLRe+1j*transferLIm) * fastPart)
#        # wfTDI[ky] = transferLRe+1j*transferLIm
#    ####
#    #### DEBUGGING FIXME
#    # wfTDI['freq'] = frS
#    # wfTDI['phase'] = phS
#    # wfTDI['amp'] = ampS
#    # wfTDI['phaseRdelay'] = phaseRdelay
#
#    #return (frS, X, Y, Z)
#    return (frS, X, Y, Z, wfTDI)
#
#
#def GenerateGBXYZ_FD(p,oversample=4,verbose=False):
#    """
#    Generates TDIs (X, Y, Z) in Fourrier Domain using the FastGB code
#    @param p is the parameters of GW GB as read from hdf5 file
#    @return oversample is ...
#    @return complex array of TDI X
#    @return complex array of TDI Y
#    @return complex array of TDI Z
#    """
#
#    ### Get the parameters
#    Amp = p.get("Amplitude")
#    f0 = p.get("Frequency")
#    fdot = p.get("FrequencyDerivative")
#    inc = p.get("Inclination")
#    psi = p.get("Polarization")
#    phi0 = p.get("InitialPhase")
#    bet = p.getConvert('EclipticLatitude',LC.convAngle,'rad')
#    lam = p.getConvert('EclipticLongitude',LC.convAngle,'rad')
#    Tobs = float(p.get("ObservationDuration"))
#    del_t = float(p.get("Cadence"))
#
#    if verbose:
#        print ("Tobs =",Tobs," , dt =", del_t)
#
#    Ns = len(Amp)
#    if verbose:
#        print ("Found ", Ns, "source(s)")
#
#    ### Define FastGB
#    GB = FB.FastGB("Test", dt=del_t, Tobs=Tobs, orbit="analytic")
#
#    ### Collect parameters
#    for i in range(Ns):
#        prm = np.array([f0[i], fdot[i], bet[i], lat[i], Amp[i], inc[i], psi[i], phi0[i]])
#        if verbose:
#            print (prm)
#        Xftmp, Yftmp, Zftmp = GB.onefourier(simulator='synthlisa', params=prm, buffer=None, T=Tobs, dt=del_t, algorithm='Michele', oversample=oversample)
#        if i==0:
#            Xf, Yf, Zf = Xftmp, Yftmp, Zftmp
#        else:
#            Xf += Xftmp
#            Yf += Yftmp
#            Zf += Zftmp
#
#    return(Xf.f,Xf[:],Yf[:],Zf[:])
#
#

#######################
# Auxiliary functions #
#######################

def GetSkyAndOrientation(p):
    """
    Get sky postion and orientation of the source from parameters
    If Initial Polar Angle L and Initial Azimuthal Angle L are the parameters,
    it will do the conversion
    @param p is the parameters of GW MBHB as read from hdf5 file
    @return beta, lamdba, inclination and polarisation
    """
    bet = p.getConvert("EclipticLatitude",LC.convAngle,'rad')
    lam = p.getConvert("EclipticLongitude",LC.convAngle,'rad')

    OrientationDefined = False
    if ('InitialPolarAngleL' in p.pars) and ('InitialAzimuthalAngleL' in p.pars):
        OrientationDefined = True
        theL = p.getConvert('InitialPolarAngleL',LC.convAngle,'rad') #!!!! WRONG TODO
        phiL = p.getConvert("InitialAzimuthalAngleL",LC.convAngle,'rad')
        psi, inc = AziPolAngleL2PsiIncl(bet, lam, theL, phiL)
    if ('Polarisation' in p.pars) and ('Inclination' in p.pars):
        psi_n = p.getConvert('Polarisation',LC.convAngle,'rad') #!!!! WRONG TODO
        inc_n = p.getConvert("Inclination",LC.convAngle,'rad')
        if OrientationDefined and ( (not np.isclose(psi_n,psi)) or (not np.isclose(inc_n,inc))) :
            print("Error in ComputeMBHBXYZ_FD: Incompatibility between multiple definition of orientation:")
            print("\t - from Polarisation and Inclination : psi = ",psi_n," inc = ",inc_n)
            print("\t - from InitialPolarAngleL and InitialAzimuthalAngleL : psi = ",psi," inc = ",inc)
            raise ValueError
        psi = psi_n
        inc = inc_n
        OrientationDefined = True
    if not OrientationDefined :
        print("Error in ComputeMBHBXYZ_FD: Orientation not defined: use either InitialPolarAngleL and InitialAzimuthalAngleL or Polarisation and Inclination.")
        raise ValueError
    return bet, lam, inc, psi


####
#def ResampleS(f, amp, phase, phaseRdelay, transferL1, transferL2, transferL3, shift, freq):
#    #TODO Describe this function
#    """
#    Interpolation function
#    @param is
#    @param is
#    @param is
#    @param is
#    @param is
#    @param is
#    @param is
#    @param is
#    @param is
#    @return complex array of TDI X
#    @return complex array of TDI Y
#    @return complex array of TDI Z
#    """
#    #{{{
#    fbeg = max(freq[0], f[0])
#    fend = min(freq[-1], f[-1])
#    ibeg = np.where(freq>=fbeg)[0][0]
#    iend = np.where(freq<=fend)[0][-1]
#    fs = freq[ibeg:iend+1]
#
#    spl = spline(f, amp)
#    amp_i = spl(fs)
#    spl = spline(f, phase)
#    phase_i = spl(fs)
#    spl = spline(f, phaseRdelay)
#    phaseRdelay_i = spl(fs)
#    spl_r = spline(f, np.real(transferL1))
#    spl_i = spline(f, np.imag(transferL1))
#    transferL1_i = spl_r(fs) + 1.0j*spl_i(fs)
#
#    spl_r = spline(f, np.real(transferL2))
#    spl_i = spline(f, np.imag(transferL2))
#    transferL2_i = spl_r(fs) + 1.0j*spl_i(fs)
#
#    spl_r = spline(f, np.real(transferL3))
#    spl_i = spline(f, np.imag(transferL3))
#    transferL3_i = spl_r(fs) + 1.0j*spl_i(fs)
#
#    t0 = shift
#    phasetimeshift = -2.0*np.pi*t0*fs
#    #vals = (transferLRe+1j*transferLIm) * amp * np.exp(1j*(phase+phaseRdelay+phasetimeshift))
#    Xf_i = np.zeros(len(freq), dtype='complex128')
#    vals = transferL1_i * amp_i * np.exp(1.0j* (phase_i + phaseRdelay_i + phasetimeshift))
#    #vals = transferL1_i * amp_i * np.exp(1.0j* (phase_i - phaseRdelay_i + phasetimeshift))
#    Xf_i[ibeg:iend+1] = vals
#    #Xf_i[ibeg:iend+1] =  amp_i * np.exp(1.0j* (phase_i + phaseRdelay_i + phasetimeshift))
#    Yf_i = np.zeros(len(freq), dtype='complex128')
#    vals = transferL2_i * amp_i * np.exp(1.0j* (phase_i + phaseRdelay_i + phasetimeshift))
#    #vals = transferL2_i * amp_i * np.exp(1.0j* (phase_i - phaseRdelay_i + phasetimeshift))
#    Yf_i[ibeg:iend+1] = vals
#    Zf_i = np.zeros(len(freq), dtype='complex128')
#    #vals = transferL3_i * amp_i * np.exp(1.0j* (phase_i - phaseRdelay_i + phasetimeshift))
#    vals = transferL3_i * amp_i * np.exp(1.0j* (phase_i + phaseRdelay_i + phasetimeshift))
#    Zf_i[ibeg:iend+1] = vals
#
#    return (np.conjugate(Xf_i), np.conjugate(Yf_i), np.conjugate(Zf_i))
#
#    #}}}
#
#####
#def ComputeTD(Xf, Yf, Zf, del_t):
#    #TODO Complete description
#    """
#    Performs inverse FFT
#    @param Xf is the array of TDI X in Fourier domain
#    @param Yf is the array of TDI Y in Fourier domain
#    @param Zf is the array of TDI Z in Fourier domain
#    @param del_t is time step (should be consitent with fmax used to generate Xf, Yf, Zf)
#
#    output
#    time array and X, Y, Z in time domain (iFFT)
#    """
#    #{{{
#    # Xta = np.fft.irfft(Xf)*(1.0/(np.sqrt(2.0)*del_t))
#    # Yta = np.fft.irfft(Yf)*(1.0/(np.sqrt(2.0)*del_t))
#    # Zta = np.fft.irfft(Zf)*(1.0/(np.sqrt(2.0)*del_t))
#    ### FIXME Check normalization again
#    Xta = np.fft.irfft(Xf)*(1.0/del_t)
#    Yta = np.fft.irfft(Yf)*(1.0/del_t)
#    Zta = np.fft.irfft(Zf)*(1.0/del_t)
#    tma = np.arange(len(Xta))*del_t
#    return (tma, Xta, Yta, Zta)
#    #}}}
#
#
#
def JustLISAFDresponseTDI(fr_ds, tm_ds, Tfvec, epsTfvec, inc, lam, bet, psi, phi0, t0=0.0, trajdict=FD_Resp.trajdict_MLDC, \
                            TDItag='TDIXYZ', order_fresnel_stencil=5, Larm=2.5e9):
    #{{{
    ##### Based on the f-n by Sylvain   #####

    HSplus = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]])
    HScross = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])


    # Wave unit vector
    kvec = FD_Resp.funck(lam, bet)

    # Trajectories
    funcp0 = trajdict['funcp0']

    # Compute constant matrices Hplus and Hcross in the SSB frame
    # print ("FD, psi, incl = ", psi, inc)
    O1 = FD_Resp.funcO1(lam, bet, psi) # doesn't contain L
    invO1 = FD_Resp.funcinverseO1(lam, bet, psi) # no L
    Hplus = np.dot(O1, np.dot(HSplus, invO1))
    Hcross = np.dot(O1, np.dot(HScross, invO1))
    # print ("lam,. bet, psi, incl", lam, bet, psi, inc)
    Y22 = FD_Resp.SpinWeightedSphericalHarmonic(-2, 2, 2, inc, phi0) # No L
    Y2m2star = np.conjugate(FD_Resp.SpinWeightedSphericalHarmonic(-2, 2, -2, inc, phi0))
    Yfactorplus = 1./2 * (Y22 + Y2m2star)
    # Yfactorcross = 1j/2 * (Y22 - Y2m2)  ### SB, should be for correct phase conventions
    Yfactorcross = 1j/2 * (Y22 - Y2m2star)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
    # Yfactorcross = -1j/2 * (Y22 - Y2m2)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
    # Yfactorcross = 1j/2 * (Y22 - Y2m2)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
    # The matrix H is now complex

    # H = np.conjugate((Yfactorplus*Hplus + Yfactorcross*Hcross))  ### SB: H_ij = H A_22 exp(i\Psi(f))
    H = (Yfactorplus*Hplus + Yfactorcross*Hcross)  ### SB: H_ij = H A_22 exp(i\Psi(f))

    # print ("FD, Hplus, ", Hplus)
    # print ("FD, Hcross", Hcross)
    # print ("FD:, H", H)

    #print ("order here is ", order_fresnel_stencil)
    if order_fresnel_stencil>20:
        raise ValueError('Only order_fresnel_stencil<=20 is supported for now.')
    if order_fresnel_stencil>=1:
        coeffs = np.array(FD_Resp.coeffs_fresnel_stencil[order_fresnel_stencil])
        coeffs_array = np.concatenate((coeffs[1:][::-1], 2*coeffs[0:1], coeffs[1:])) / 2
    n = len(fr_ds)

    #print ("n=", n)

    #print ("Stas, order = ", order_fresnel_stencil)
    ### loop over freqs
    wfTDI = {}
    wfTDI['freq'] = fr_ds
    wfTDI['tfvec'] = tm_ds
    wfTDI['phaseRdelay'] = np.zeros(n, dtype=np.float64)
    wfTDI['transferL1'] = np.zeros(n, dtype=np.complex128)
    wfTDI['transferL2'] = np.zeros(n, dtype=np.complex128)
    wfTDI['transferL3'] = np.zeros(n, dtype=np.complex128)

    #print (np.shape(fr_ds), np.shape(tm_ds))

    for i in range(n):
        f = fr_ds[i]
        t = tm_ds[i] + t0*LC.YRSID_SI
        p0 = funcp0(t)
        kR = np.dot(kvec, p0)
        phaseRdelay = 2.*np.pi/LC.clight *f*kR

        # if (i> n - 10):
        #     print ("t=", t, "f=", f, "p0 = ", p0, "kR=", kR/LC.clight, "phRD = ", phaseRdelay)

        if order_fresnel_stencil>=1:
            Gslr = {}
            Tf = Tfvec[i] # used for higher-order correction of Fresnel type
            epsTf = epsTfvec[i] # keeping track of the sign associated with Tf
            if epsTf==1. or epsTf==0.:
                coeffs_array_signeps = coeffs_array
            else:
                coeffs_array_signeps = np.conj(coeffs_array)
            tvec = t + Tf * np.arange(-order_fresnel_stencil, order_fresnel_stencil+1)
            Gslrvec = FD_Resp.EvaluateGslr(tvec, f, H, kvec, trajdict=trajdict, responseapprox='full', L=Larm)
            # this have L
            for key in Gslrvec:
                Gslr[key] = np.dot(coeffs_array_signeps, Gslrvec[key])
        else:
            Gslr = FD_Resp.EvaluateGslr(t, f, H, kvec, trajdict=trajdict, responseapprox='full', L=Larm)

        # Scale out the leading-order correction from the orbital delay term, that we keep separate
        Tslr = {}
        for key in Gslr:
            Tslr[key] = Gslr[key] * np.exp(-1j*phaseRdelay)
        # if (i >= n-10):
        #     print ("FD Stas", Tslr[(1, 3)])
        # Build TDI combinations
        tdi = FD_Resp.TDICombinationFD(Tslr, f, TDItag=TDItag, rescaled=False, L=Larm)
        wfTDI['phaseRdelay'][i] = phaseRdelay
        wfTDI['transferL1'][i] = tdi['transferL1']
        wfTDI['transferL2'][i] = tdi['transferL2']
        wfTDI['transferL3'][i] = tdi['transferL3']

    return (wfTDI)
#
#    #}}}
#
#
def MBHB_LISAGenerateTDI(phi0, fRef, m1, m2, a1, a2, DL, inc, lam, bet, psi, Tobs, minf, maxf, tc, settRefAtfRef=False, \
        tRef=0.0, trajdict=FD_Resp.trajdict_MLDC, TDItag='TDIXYZ', nptmin=500, order_fresnel_stencil=0, verbose=False, Larm=2.5e9):
    """
    @param phi0 observer's phase
    @param fRef reference frequency at which the FD phase is specified, 0 to ignore (then defaults to min of peak frequency and max frequency)
    @param m1 primary mass (solar masses)
    @param m2 secondary mass (solar masses)
    @param a1 dimensionless spin of primary (projection on the orbital angular momentum)
    @param a2 dimensionless spin of secondary (projection on the orbital angular momentum)
    @param DL luminocity distance (Mpc)
    @param inc inclination: azimuthal direction to the observer in the source frame
    @param lam ecliptic longitude
    @param bet ecliptic Latitude
    @param psi polarisation angle
    @param Tobs observation time (sec)
    @param minf starting frequency of the GW signal (will be redefined based on Tobs)
    @param maxf desired (!) end frequency, in reality it will be redefined if binary merges within Tobs
    @param tRef initial time which corresponds to minf -- not used if Tobs is given!!!!
    @param trajdict uses analytic LDC orbit (so far the only option)
    @param nptmin minimum number of points used in computation FD response function
    @param order_frensel_stecil order of Frensel stencil in FD response (signal dependent)

    output:
    frequency array
    X, Y, Z in frequency domain
    wfTDI - debugging info, dictionary containing auxilary info (responses, GW phase, aplitude, doppler modulation)
    """


    #{{{

    ## Generate the PhenomD phase and amplitude
    m1_SI = m1*LC.MsunKG
    m2_SI = m2*LC.MsunKG
    Ms = (m1 + m2) * LC.MTsun
    dist_SI = DL*1e6*LC.pc
    chi1 = a1
    chi2 = a2

    #### Generate PhenomD amplitude and phase on the fine and regular grid of points in freq.
    #df = 1.0/Tobs
    #df = 8.e-9
    # df = 1.e-8
    #df = 0.5/Tobs ### generating waveform for duration 2Tobs, will trancate after FFT
    df = minf # changed by yishuxu @ 21 April, 2020
#    print("m1=%f, m2=%f" % (m1,m2))
    Mc = FD_Resp.funcMchirpofm1m2(m1, m2) 
    # redshifted Chirpmass
    #f0 = FD_Resp.funcNewtonianfoft(Mc, 3.0*Tobs/LC.YRSID_SI)
    f0 = FD_Resp.funcNewtonianfoft(Mc, 2.0*Tobs/LC.YRSID_SI)
    #print("f0=%f, df=%f" % (f0,df))
    if f0<=0.4*df: return [-1,-1,-1,-1,-1]
    # Redefine f0 to be an integer x df
    f0 = np.floor(f0/df)*df

    # fRef=0 means fRef=fpeak in PhenomD or maxf if out of range
    fRef = 0.0
    fmax = min(MfCUT_PhenomD/Ms, maxf)
    ### FIXME Do we want to restrict low freq???? say to 1.e-5? yishuxu already did it.
    Nf = int((fmax-f0)/df)

    # freq_all = (n + Nf) x df where n = floor(f0/df) with f0 obtained from FD_Resp.
    # freq_all is therefore an integer x df
    freq_all = np.arange(Nf)*df + f0
    if verbose:
        print ("\tdf = "+str(df)+"\n\tf0 = "+str(f0)+"\n\tfmax = "+str(fmax)+"\n\tNpts = "+str(Nf))
        print(freq_all)

    phi_ref = 0.0 ### Fixed initial angle at the reference time/frequency
    st = time.time()
    # print ("FD wvf:", freq_all, phi_ref, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
    wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq_all, phi_ref, fRef, m1_SI, m2_SI, chi1, \
                                                      chi2, dist_SI)
    wf_PhD = wf_PhD_class.GetWaveform() # freq, amp, phase
    en = time.time()
    #print ("PhenomD elappsed", en-st)

    frS = wf_PhD[0]
    phS = wf_PhD[2]
    ampS = wf_PhD[1]
    if verbose:
        print("\tInitial frequency from pyIMRPhenomD.IMRPhenomDh22AmpPhase :",frS[0])

    # print ("ini freq here", frS[0], f0)

    tfspline = spline(frS, 1/(2.*np.pi)*(phS-phS[0])).derivative()
    tf = tfspline(frS)
    # tfspline = interpolate.splrep(frS, 1/(2.*np.pi)*(phS-phS[0]), s=4)
    # tf = interpolate.splev(frS, tfspline, der=1)
    # plt.plot(tf, frS)
    # plt.show()
    # sys.exit(0)

    Shift = tf[-1] - tc
    tf = tf - Shift
    # print ("FD, tf\n", tf)
    #tf = interpolate.spalde(frS, tfspline)
    # df = frS[1] - frS[0]
    # ph_nrm = 1/(2.*np.pi)*(phS-phS[0])
    # tfvec = np.diff(ph_nrm)/df
    # Shift = tfvec[-1] - tc
    # tfvec = tfvec - Shift
    #
    # tf = np.zeros(len(frS))
    # tf[:-1] = tfvec
    # tf[-1] = tf[-2]


    #print ("Shift = ", Shift, tf[-1])
    #print ("total time extend", tf[-1]-tf[0], (tf[-1]-tf[0])/Tobs)

    #ind = np.argmax(tf)-200
    # print (np.diff(tf[:ind])[-30:])

    index_cuttf = 0
    tfdiff = np.diff(tf)
    while index_cuttf<len(tfdiff)-1 and tfdiff[index_cuttf]>0:
        index_cuttf += 1
    tfr = tf[:index_cuttf+1]
    freq_r = frS[:index_cuttf+1]
    frspl = spline(tf[:index_cuttf], frS[:index_cuttf])

    # plt.semilogx(frS, tf)
    # plt.semilogx(frS[:ind], tf[:ind])
    # plt.grid(True)
    # plt.show()
    # frspl = spline(tf[:ind], frS[:ind])

    # sys.exit(0)

    #print ("freqs", frS)

    ### SB. downsample the frequency array

    if (order_fresnel_stencil > 0):
        dffphasespline = spline(frS, phS).derivative(2)
        epsTf2vec = 1./(4*np.pi*np.pi)*dffphasespline(frS)
        Tfvec = np.sqrt(np.abs(epsTf2vec))
        epsTfvec = np.sign(epsTf2vec)

    ### Constructing the reduced freq. array for computing the response f-n
    # uniform in time
    Delta_t = 1.0/48.0
    times_deltatresampling = np.linspace(tf[0], tf[index_cuttf], \
                                int(np.ceil(abs(tf[0] - tf[index_cuttf])/(Delta_t*LC.YRSID_SI)))+1)

    if len(times_deltatresampling)==0:
        freq_rsdeltat = np.array([])
    else:
        freqs_deltatresampling = frspl(times_deltatresampling)
        freqs_deltatresampling[0] = frS[0]
        freqs_deltatresampling[-1] = frS[-1]

    #print ("Check for negative", freqs_deltatresampling - frS[0])

    # Uniform in freq.
    Delta_f = 0.0005
    npt_deltafsampling = int(np.ceil((frS[-1] - frS[0])/Delta_f))
    if npt_deltafsampling<=0: # if f_startdeltafsampling>=maxfrs, do not do deltaf-sampling
        freq_rsdeltaf = np.array([])
    else:
        ind_l = np.linspace(0, len(frS)-1, num=npt_deltafsampling, dtype='int')
        freq_rsdeltaf = frS[ind_l[1:]]

    # uniform in log
    freq_rsln = np.logspace(np.log10(frS[0]), np.log10(frS[-1]), num=nptmin)

    freq_tmp = np.concatenate((freqs_deltatresampling, freq_rsdeltaf, freq_rsln[1:-1]))

    # print (np.shape(freq_tmp))
    freq_tmp = np.unique(freq_tmp)
    freq_tmp = np.sort(freq_tmp)
    #print ("df", np.diff(freq_tmp))


    ind_fi = np.zeros(len(freq_tmp), dtype='int')
    freq_rs = np.zeros(len(freq_tmp))
    freq_rs[0] = freq_tmp[0]
    freq_rs[-1] = freq_tmp[-1]
    ind_fi[-1] = len(frS)-1
    for i in range(1, len(freq_tmp)-1):
        fi = freq_tmp[i]
        i_f = int((fi-frS[0])/df)+1
        #print ("i_f = ", i_f)
        freq_rs[i] = frS[i_f]
        ind_fi[i] = i_f
    #en1 = time.time()

    #print (len(freq_rs), len(ind_fi))

    freq_rs, ind_u = np.unique(freq_rs, return_index=True)
    #print ("ind_u", np.diff(ind_u))!!!!!!!
    ind_fi = ind_fi[ind_u]
    #timeFr, ind_ut = np.unique(tf[ind_fi], return_index=True)
    #print (ind_u - ind_ut)
    timeFr = tf[ind_fi]
    #print ("MUST look at it!!!!", len(freq_rs), len(timeFr), len(tf[ind_fi]), len(ind_fi))
    if (len(freq_rs) != len(timeFr)):
        print ("Something is seriously wrong: ", len(freq_rs), len(timeFr), len(tf[ind_fi]))
        raise ValueError

    # print ("diff", np.diff(timeFr))
    # plt.semilogx(freq_rs, timeFr, 'o-')
    # plt.grid(True)
    # plt.show()
    # sys.exit(0)

    if (order_fresnel_stencil > 0):
        Tfvec_rs = Tfvec[ind_fi]
        epsTfvec_rs = epsTfvec[ind_fi]
    else:
        Tfvec_rs = 0.0
        epsTfvec_rs = 0.0


    if settRefAtfRef:
        fRef_inrange = min( max(frS[0], fRef), frS[-1] )
        # print ("Important:", fRef_inrange)
        tf_time = interpolate.splev(fRef_inrange, tfspline, der=1)
        Del_Phase = 2.*np.pi * (frS - fRef_inrange) * (tRef - tf_time)
        #Del_Phase = 2.*np.pi * (frS - fRef_inrange) * (tRef - tfspline(fRef_inrange))
        # Del_shift = (tRef - tfspline(fRef_inrange))
        # print ("Shift = ", Del_shift)
        phS += Del_Phase

    # st = time.time()
    # print ("new2", inc, lam, bet, psi)
    # print ("FD, passing to resp:\n", freq_rs[:10], timeFr[:10], freq_rs[-10:], timeFr[-10:])
    # print ("FD, passing to resp:\n", freq_rs, timeFr)
    # Tfvec_rs, epsTfvec_rs, inc, lam, bet, psi, phi0, order_fresnel_stencil )
    wfTDI =  JustLISAFDresponseTDI(freq_rs, timeFr, Tfvec_rs, epsTfvec_rs, inc, lam, bet, psi, phi0, t0=0.0, \
                    order_fresnel_stencil=order_fresnel_stencil, Larm=Larm)

    wfTDI['phase'] = phS[ind_fi]
    # en = time.time()

    ### Interpolating and building the TDIs

    phaseRdelayspline = spline(freq_rs, wfTDI['phaseRdelay'])
    phaseRdelay = phaseRdelayspline(frS)
    #phasetimeshift = -2.*np.pi*tc*frS
    # print ("tc = ", tc)
    phasetimeshift = 2.*np.pi*tc*frS
    # fastPart = ampS * np.exp(1j*(-phS+phaseRdelay-phasetimeshift))
    fastPart = ampS * np.exp(1j*(phS+phaseRdelay+phasetimeshift))

    #wfTDI["fast"] = fastPart
    wfTDI["fast"] = phS+phaseRdelay+phasetimeshift

    keytrs = ['transferL1', 'transferL2', 'transferL3']

    # plt.loglog(freq_rs, np.abs(wfTDI['transferL1']))
    # plt.loglog(freq_rs, np.abs(wfTDI['transferL2']))
    # plt.loglog(freq_rs, np.abs(wfTDI['transferL3']))
    # plt.grid(True)
    # plt.show()

    # print ("Here FD checks:", len(frS), len(wfTDI['phaseRdelay']))
    # print ("frS\n", frS, "phS \n", phS, "ampS \n", ampS)
    # print ("phR\n", wfTDI['phaseRdelay'][-10:], "RespX\n", wfTDI['transferL1'][-10:])
    # print ("---")

    for I, ky in enumerate(keytrs):
        transferLRespline = spline(freq_rs, np.real(wfTDI[ky]))
        transferLImspline = spline(freq_rs, np.imag(wfTDI[ky]))
        transferLRe = transferLRespline(frS)
        transferLIm = transferLImspline(frS)
        if (ky == 'transferL1'):
            # X = np.conjugate((transferLRe+1j*transferLIm) * fastPart)
            X = np.conjugate((transferLRe+1j*transferLIm) * fastPart)  ## SB: to make up fpr the wrong phase sign convention
            # print("ky")
            # print ("FD 1\n", frS[-10:])
            # print ("FD 2\n", (transferLRe+1j*transferLIm)[-10:])
            # print ("FD 3\n", fastPart[-10:])
            # X = np.conjugate(fastPart)
        if (ky == 'transferL2'):
            Y = np.conjugate((transferLRe+1j*transferLIm) * fastPart)
            # Y = -1.0*(transferLRe+1j*transferLIm) * fastPart
        if (ky == 'transferL3'):
            Z = np.conjugate((transferLRe+1j*transferLIm) * fastPart)

    # print ("response:", en-st)

    return (frS, X, Y, Z, wfTDI)
