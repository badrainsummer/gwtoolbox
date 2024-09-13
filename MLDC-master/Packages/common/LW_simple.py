import numpy as np
import sys, os
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as spline

import LISAConstants as LC

import FrequencyArray
from pyIMRPhenomD import IMRPhenomD
import pyIMRPhenomD
from scipy import interpolate

#
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000
# plt.style.use('Stas_plot')

import GenerateFD_SignalTDIs as GenTDIFD
import tdi
import pyFDresponse as FD_Resp



def sinc(x):
    return(np.sinc(x/np.pi))


class LW:
    """ Generates the rigid adiabatic TDI (XYZ) for MBH, GBs, SOBBH """
    def __init__(self, incl, bet, lam, psi, phi0, arm = 2.5e9):
        self.incl = incl     ### Inclination
        self.bet = bet       ### Ecliptic Latitude
        self.lam = lam       ### Ecliptic Longitude
        self.psi = psi       ### polarization
        self.armt = arm/LC.clight  ### armlength in sec
        self.settRefAtfRef = False

        Y22 = FD_Resp.SpinWeightedSphericalHarmonic(-2, 2, 2, incl, phi0)
        Y2m2 = FD_Resp.SpinWeightedSphericalHarmonic(-2, 2, -2, incl, phi0)
        # self.Yfactorplus = 0.5 * (Y22 + np.conjugate(Y2m2))
        # self.Yfactorcross = -0.5j * (Y22 - np.conjugate(Y2m2))
        self.Yfactorplus = 0.5 * (Y22 + np.conjugate(Y2m2))
        self.Yfactorcross = 0.5j * (Y22 - np.conjugate(Y2m2))
        # print ("+, x factors", self.Yfactorplus, self.Yfactorcross)

    def ComputeHpHc(self, freq, psi,  fRef, m1_si, m2_si, a1, a2, dist, del_t, Tobs, tc):
        wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq, phi0, fRef, m1_si, m2_si, a1, a2, dist)
        wf_PhD = wf_PhD_class.GetWaveform() # freq, amp, phase

        frS = wf_PhD[0]
        phS = wf_PhD[2]
        ampS = wf_PhD[1]

        phasetimeshift = 2.0*np.pi*frS*tc

        CommonPart = ampS * np.exp(1.0j*(phS+phasetimeshift)) ### 22 mode

        df = frS[1] - frS[0]
        fmx = 0.5/del_t
        # I want samplign rate to be del_t
        Nf = int(fmx/df+1)
        hp_f = np.zeros(Nf, dtype='complex128')
        hc_f = np.zeros(Nf, dtype='complex128')

        i_b = int(frS[0]/df)
        i_e = i_b + len(frS)

        hp_f[i_b:i_e] = np.conjugate(self.Yfactorplus * CommonPart)
        hc_f[i_b:i_e] = np.conjugate(self.Yfactorcross * CommonPart)

        # print ("computing fft")
        hp = np.fft.irfft(hp_f)*(1.0/del_t)
        hc = np.fft.irfft(hc_f)*(1.0/del_t)
        # print ("done")

        ### Need to trancate the waveform to the duration tc
        Nt = len(hp)
        tm = np.arange(Nt)*del_t
        i0 = np.argwhere((tm-Tobs)>=0 )[0][0]
        Mt = (m1_si+m2_si)*LC.MTsun/LC.MsunKG

        plt.plot(tm, hp)
        plt.plot(tm, hc)
        plt.grid(True)
        plt.show()

        tap = tc + 500.0*Mt
        window = taper = 0.5*(1.0 - np.tanh( 0.001*( tm  -  tap)) )
        #plt.plot(tm, window)

        # plt.plot(tm, hp)
        # plt.plot(tm, hp*window)
        # plt.grid(True)
        # plt.show()

        hp = window*hp
        hc = window*hc


        hot_p = hp[:i0]
        hot_c = hc[:i0]
        tmf = tm[:i0]

        #psi = 0.0
        #print ("polarization, psi = ", psi)
        cpsi = np.cos(2.0*self.psi)
        spsi = np.sin(2.0*self.psi)
        hphcDat = np.zeros([len(tmf),3])
        #print "writing the h+, hx", tmf, based on https://arxiv.org/pdf/0806.2110.pdf
        hphcDat[:,0] = np.copy(tmf)
        hphcDat[:,1] = np.copy(hot_p*cpsi - hot_c*spsi)
        hphcDat[:,2] = np.copy(hot_p*spsi + hot_c*cpsi)

        #return (hphcDat)
        return (tmf, hot_p, hot_c)

    def ComputeSOBBH_HpHc(self, fstart, phi0, m1, m2, a1, a2, DL, incl, Tobs, del_t):

        fRef = fstart
        tRef = 0.0
        fny = 0.5/del_t
        minf = 1.e-5
        maxf = 1.e-1
        fend = 4.*fstart

        m1_SI = m1*LC.MsunKG
        m2_SI = m2*LC.MsunKG
        dist_SI = DL*1.e6*LC.pc
        print ("masses", m1, m2)
        w_fr, w_amp, w_ph = FD_Resp.GenerateResamplePhenomD(phi0, fRef, m1, m2, a1, a2, DL, incl, minf=fstart, maxf=0.0015, \
                            settRefAtfRef=True, tRef=tRef, tobs=0.0, nptmin=1000)

        tfspline = spline(w_fr, 1/(2.*np.pi)*(w_ph-w_ph[0])).derivative() # get rid of possibly huge constant in the phase before interpolating
        tfvec = tfspline(w_fr)

        # plt.plot(w_fr, tfvec)
        # plt.show()

        fspl = spline(tfvec, w_fr)
        fend = fspl(Tobs)

        print ("freq. extend", fstart, fend)

        # I want samplign rate to be del_t
        df = 1.0/Tobs
        Nf = int(fny/df+1)
        freq = np.arange(Nf)*df
        i_b = int(np.floor(fstart/df))
        i_e = int(np.ceil(fend/df))

        # print (freq[i_b:i_e])
        # print ("LW pars", phi0, fRef, m1_SI, m2_SI, a1, a2, dist_SI)
        wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq[i_b:i_e], phi0, fRef, m1_SI, m2_SI, a1, a2, dist_SI)
        wf_PhD = wf_PhD_class.GetWaveform() # freq, amp, phase


        frS = wf_PhD[0]
        phS = wf_PhD[2]
        ampS = wf_PhD[1]

        tfspline2 = spline(frS, 1/(2.*np.pi)*(phS-phS[0])).derivative() # get rid of possibly huge constant in the phase before interpolating

        fRef_inrange = fstart
        tRef = 0.0
        phS += 2.0*np.pi * (frS - fRef_inrange) * (tRef - tfspline2(fRef_inrange))
        print ("compare", fRef_inrange, tRef, tfspline2(fRef_inrange), (tRef - tfspline2(fRef_inrange)))

        # phspl = spline(w_fr, w_ph)
        # ph_i = phspl(frS)
        #
        # plt.plot(frS, phS, '-o')
        # plt.plot(w_fr, w_ph, '.')
        # plt.plot(frS, ph_i, '.')
        # plt.show()
        #
        #
        # sys.exit(0)

        freq, hptilde, hctilde = FD_Resp.ComputehphcFromAmpPhase([frS, ampS, phS], freq, incl, 0.)
        hptilde = np.conjugate(hptilde)
        hctilde = np.conjugate(hctilde)

        plt.plot(freq, np.real(hptilde))
        plt.show()
        # sys.exit(0)

        # fbeg = max(freq[0], w_fr[0])
        # fend = min(freq[-1], fend)
        #
        # ibeg = int((fbeg - freq[0])/df)
        # iend = int((fend - freq[0])/df)
        #
        # hptilde[:ibeg] = 0.0 + 0.0j
        # hctilde[:ibeg] = 0.0 + 0.0j
        # hptilde[iend:] = 0.0 + 0.0j
        # hctilde[iend:] = 0.0 + 0.0j

        hp = np.fft.irfft(hptilde)*(1.0/del_t)
        hc = np.fft.irfft(hctilde)*(1.0/del_t)
        Nt = len(hp)
        tm = np.arange(Nt)*del_t

        psi = self.psi
        cpsi = np.cos(2.0*psi)
        spsi = np.sin(2.0*psi)
        hphcDat = np.zeros([len(tm),3])
        #print "writing the h+, hx", tmf, based on https://arxiv.org/pdf/0806.2110.pdf
        hphcDat[:,0] = np.copy(tm)
        hphcDat[:,1] = np.copy(-hp*cpsi - hc*spsi)
        hphcDat[:,2] = np.copy(hp*spsi - hc*cpsi)

        plt.plot(tm/LC.YRSID_SI, hphcDat[:, 1])
        plt.show()

        return (hphcDat)





    def ComputeProjections(self, tvec, wv="modes", Ap=None, Ac=None, verbose=False):

        #### We assume the analytic eccentric orbits.
        self.tfvec = tvec
        x, y, z, alpha, a = self.LISAmotion(tvec, verbose=verbose)
        n1 = np.array([x[1,:]-x[2,:], y[1,:]-y[2,:], z[1,:] - z[2,:]])/self.armt
        n2 = np.array([x[2,:]-x[0,:], y[2,:]-y[0,:], z[2,:] - z[0,:]])/self.armt
        n3 = np.array([x[0,:]-x[1,:], y[0,:]-y[1,:], z[0,:] - z[1,:]])/self.armt
        R0 = np.array([a*np.cos(alpha), a*np.sin(alpha), 0])
        self.k = 1.0*np.array([-np.cos(self.bet)*np.cos(self.lam), -np.cos(self.bet)*np.sin(self.lam), -np.sin(self.bet)])
        self.kR0 = np.dot(self.k, R0)
        self.kn1 = np.dot(self.k, n1)
        self.kn2 = np.dot(self.k, n2)
        self.kn3 = np.dot(self.k, n3)

        # print ("LW: k, t, kR0", self.k, tvec[-10:], self.kR0[-10:])


        q1 = np.array([x[0,:]-a*np.cos(alpha), y[0, :]-a*np.sin(alpha), z[0, :]])
        q2 = np.array([x[1,:]-a*np.cos(alpha), y[1, :]-a*np.sin(alpha), z[1, :]])
        q3 = np.array([x[2,:]-a*np.cos(alpha), y[2, :]-a*np.sin(alpha), z[2, :]])

        self.kq1 = np.dot(self.k, q1)
        self.kq2 = np.dot(self.k, q2)
        self.kq3 = np.dot(self.k, q3)

        # print ("LW angles:, lam, bet, psi, incl", self.lam, self.bet, self.psi, self.incl)
        Pl1, Cr1 = self.PolarizationTensor(self.lam, self.bet, self.psi)
        # Pij = 1.0*(Pl1*self.Yfactorplus + Cr1*self.Yfactorcross) ### SB: assumes that h_ij = P_ij A e^{i\Psi} in FD

        Pij = np.conjugate(Pl1*self.Yfactorplus + Cr1*self.Yfactorcross) ### SB because h_ij = P_ij A e^{-i\Psi}
        # print ("polariz", Pl1, Cr1)
        # print ("Pij", Pij)
        if (wv == "hphc"):
            # print ("incl = ", self.incl/np.pi)
            # Ap = (1.0 + (np.cos(self.incl))**2)
            Ap = -(1.0 + (np.cos(self.incl))**2)  ### New conventions
            Ac = -2.0*np.cos(self.incl)
            Pij = Ap*Pl1 - 1.0j*Ac*Cr1 ### SB: here I assume that h_{ij} = Re( Pij * A * exp(i\Phi(t)-phi0) )
            # Pij = Ap*Pl1 + 1.0j*Ac*Cr1 ### SB: here I assume that h_{ij} = Re( Pij * A * exp(-i\Phi(t)) )
            ## This assumes we are working in the time domain and we need to take real part of $p_{ij} A e^{i \phi}$

        N = len(tvec)
        self.A1 = np.zeros(N, dtype='complex128')
        self.A2 = np.zeros(N, dtype='complex128')
        self.A3 = np.zeros(N, dtype='complex128')
        for i in range(N):
            if (wv == 'emri'):
                 Pij = Ap[i]*Pl1 + Ac[i]*Cr1
            self.A1[i] = np.dot(np.dot(n1[:, i], Pij), n1[:, i])
            self.A2[i] = np.dot(np.dot(n2[:, i], Pij), n2[:, i])
            self.A3[i] = np.dot(np.dot(n3[:, i], Pij), n3[:, i])
        # print (np.shape(self.A1))
        #
        # print (self.lam, self.bet, self.psi)
        # print (Pl1)
        # print (Cr1)
        # print (self.Yfactorplus, self.Yfactorcross)

        # print ('A1', self.A1)

    def ComputeResponse(self, om, method="full"):
        ## om - single freq, or array of freq: 2*pi*f_gw
        ## method - ["low", "full"], "low" - using the leading order only, "full" - goes beond the leading order

        self.om = om
        self.omL = om*self.armt
        phR = om*self.kR0

        y123 = self.ComputeYslr(send=1, link=2, rec=3, lsign=1.)
        y231 = self.ComputeYslr(send=2, link=3, rec=1, lsign=1.)
        y1m32 = self.ComputeYslr(send=1, link=3, rec=2, lsign=-1.)
        y3m21 = self.ComputeYslr(send=3, link=2, rec=1, lsign=-1.)
        y2m13 = self.ComputeYslr(send=2, link=1, rec=3, lsign=-1.)
        y312 = self.ComputeYslr(send=3, link=1, rec=2, lsign=1.)


        omL = self.omL
        RX = -2.0j*np.sin(omL)*(y231 + (y1m32 - y123)*np.exp(-1.0j*omL) - y3m21)*np.exp(-1.0j*omL)
        RY = -2.0j*np.sin(omL)*(y312 + (y2m13 - y231)*np.exp(-1.0j*omL) - y1m32)*np.exp(-1.0j*omL)
        RZ = -2.0j*np.sin(omL)*(y123 + (y3m21 - y312)*np.exp(-1.0j*omL) - y2m13)*np.exp(-1.0j*omL)

        # print ("LW Y123", y123[-10:])

        ## NOTE the sign of delay phase phR !!!
        return(-phR, RX, RY, RZ)

    def ComputeResponseMBHB(self, frS, ampS, phS, tc, verbose=False):
        #### Here we get freq, amp and phase of PhenomD and construct the response by
        #### first downsampling the freqs -> compute response -> interpolate
        #### here we assume stencil order = 0, so we have used simple time-freq relation

        # tfspline = interpolate.splrep(frS, 1/(2.*np.pi)*(phS-phS[0]), s=4)
        # tf = interpolate.splev(frS, tfspline, der=1)
        df = frS[1] - frS[0]
        ph_nrm = 1/(2.*np.pi)*(phS-phS[0])
        tfvec = np.diff(ph_nrm)/df
        Shift = tfvec[-1] - tc
        tfvec = tfvec - Shift

        tf = np.zeros(len(frS))
        tf[:-1] = tfvec
        tf[-1] = tf[-2]

        # Shift = tf[-1] - tc
        # tf = tf - Shift
        index_cuttf = 0
        tfdiff = np.diff(tf)
        while index_cuttf<len(tfdiff)-1 and tfdiff[index_cuttf]>0:
            index_cuttf += 1
        tfr = tf[:index_cuttf+1]
        #freq_r = frS[:index_cuttf+1]
        freq_r = frS[:index_cuttf]
        frspl = spline(tf[:index_cuttf], frS[:index_cuttf])
        # print ("LW, tf\n", tf)

        ### Downselecting the frequencies
        Delta_t = 1.0/48.0
        times_deltatresampling = np.linspace(tf[0], tf[index_cuttf], int(np.ceil(abs(tf[0] - tf[index_cuttf])/(Delta_t*LC.YRSID_SI)))+1)

        if len(times_deltatresampling)==0:
            freq_rsdeltat = np.array([])
        else:
            freqs_deltatresampling = frspl(times_deltatresampling)
            freqs_deltatresampling[0] = frS[0]
            freqs_deltatresampling[-1] = frS[-1]

        # Uniform in freq.
        Delta_f = 0.0005
        npt_deltafsampling = int(np.ceil((frS[-1] - frS[0])/Delta_f))
        if npt_deltafsampling<=0: # if f_startdeltafsampling>=maxfrs, do not do deltaf-sampling
            freq_rsdeltaf = np.array([])
        else:
            ind_l = np.linspace(0, len(frS)-1, num=npt_deltafsampling, dtype='int')
            freq_rsdeltaf = frS[ind_l[1:]]

        # uniform in log
        freq_rsln = np.logspace(np.log10(frS[0]), np.log10(frS[-1]), 500)

        freq_tmp = np.concatenate((freqs_deltatresampling, freq_rsdeltaf, freq_rsln[1:-1]))

        # print (np.shape(freq_tmp))
        freq_tmp = np.sort(freq_tmp)
        freq_tmp = np.unique(freq_tmp)

        ind_fi = np.zeros(len(freq_tmp), dtype='int')
        freq_rs = np.zeros(len(freq_tmp))
        freq_rs[0] = freq_tmp[0]
        freq_rs[-1] = freq_tmp[-1]
        ind_fi[-1] = len(frS)-1
        df = frS[1]-frS[0]
        for i in range(1, len(freq_tmp)-1):
            fi = freq_tmp[i]
            i_f = int((fi-frS[0])/df)+1
            freq_rs[i] = frS[i_f]
            ind_fi[i] = i_f

        freq_rs, ind_u = np.unique(freq_rs, return_index=True)
        ind_fi = ind_fi[ind_u]
        timeFr = tf[ind_fi]


        if self.settRefAtfRef:
            fRef_inrange = min( max(frS[0], fRef), frS[-1] )
            print ("Important:", fRef_inrange)
            tf_time = interpolate.splev(fRef_inrange, tfspline, der=1)
            Del_Phase = 2.*np.pi * (frS - fRef_inrange) * (tRef - tf_time)
            #Del_Phase = 2.*np.pi * (frS - fRef_inrange) * (tRef - tfspline(fRef_inrange))
            # Del_shift = (tRef - tfspline(fRef_inrange))
            # print ("Shift = ", Del_shift)
            phS += Del_Phase

        self.ComputeProjections(timeFr, verbose=verbose)
        # print ("LW in resp:", freq_rs[:10], timeFr[:10], freq_rs[-10:], timeFr[-10:])
        om = 2.0*np.pi*freq_rs
        phR, RX, RY, RZ = self.ComputeResponse(om)

        ### interpolating
        spl = spline(freq_rs, phR)
        phR_i = spl(frS)

        splR = spline(freq_rs, np.real(RX))
        splI = spline(freq_rs, np.imag(RX))
        RX_i = splR(frS) + 1.0j*splI(frS)

        splR = spline(freq_rs, np.real(RY))
        splI = spline(freq_rs, np.imag(RY))
        RY_i = splR(frS) + 1.0j*splI(frS)

        splR = spline(freq_rs, np.real(RZ))
        splI = spline(freq_rs, np.imag(RZ))
        RZ_i = splR(frS) + 1.0j*splI(frS)

        plotIt = False
        if plotIt:
            fig, ax = plt.subplots(nrows=3, sharex=True)
            ax[0].loglog(frS, np.abs(np.real(RX_i)))
            ax[0].loglog(freq_rs, np.abs(np.real(RX)), 'o')
            ax[0].loglog(frS, np.abs(np.imag(RX_i)))
            ax[0].loglog(freq_rs, np.abs(np.imag(RX)), 'o')
            ax[1].loglog(frS, np.abs(np.real(RY_i)))
            ax[1].loglog(freq_rs, np.abs(np.real(RY)), 'o')
            ax[1].loglog(frS, np.abs(np.imag(RY_i)))
            ax[1].loglog(freq_rs, np.abs(np.imag(RY)), 'o')
            ax[2].loglog(frS, np.abs(np.real(RZ_i)))
            ax[2].loglog(freq_rs, np.abs(np.real(RZ)), 'o')
            ax[2].loglog(frS, np.abs(np.imag(RZ_i)))
            ax[2].loglog(freq_rs, np.abs(np.imag(RZ)), 'o')
            plt.show()

        #sys.exit(0)

        return (phR_i, RX_i, RY_i, RZ_i)

    def ComputeResponseSOBBH(self, frS, ampS, phS, fRef, tRef, verbose=False):
        #{{{  Computes the responses. FIrst we down-sample the data, then compute response and interpolate
        tfspline = spline(frS, 1/(2.*np.pi)*phS).derivative()
        tf = tfspline(frS)
        tf = tf - tf[0]

        frspl = spline(tf, frS)

        ### Constructing the reduced freq. array for computing the response f-n
        # uniform in time
        Delta_t = 1.0/24.0
        times_deltatresampling = np.linspace(tf[0], tf[-1], int(np.ceil(abs(tf[0] - tf[-1])/(Delta_t*LC.YRSID_SI)))+1)

        if len(times_deltatresampling)==0:
            freq_rsdeltat = np.array([])
        else:
            freqs_deltatresampling = frspl(times_deltatresampling)
            freqs_deltatresampling[0] = frS[0]
            freqs_deltatresampling[-1] = frS[-1]

        # UNiform in freq.
        Delta_f = 0.0005
        npt_deltafsampling = int(np.ceil((frS[-1] - frS[0])/Delta_f))
        # print ("Check this", npt_deltafsampling)
        if npt_deltafsampling<=1: # if f_startdeltafsampling>=maxfrs, do not do deltaf-sampling
            freq_rsdeltaf = np.array([])
        else:
            ind_l = np.linspace(0, len(frS)-1, num=npt_deltafsampling, dtype='int')
            # print ("Check", ind_l)
            freq_rsdeltaf = frS[ind_l[1:-1]]

        # uniform in log
        freq_rsln = np.logspace(np.log10(frS[0],), np.log10(frS[-1]), num=500)
        # print (np.shape(freqs_deltatresampling), np.shape(freq_rsdeltaf), np.shape(freq_rsln))
        freq_tmp = np.concatenate((freqs_deltatresampling, freq_rsdeltaf, freq_rsln[1:-1]))

        # print (np.shape(freq_tmp))
        freq_tmp = np.sort(freq_tmp)
        freq_tmp = np.unique(freq_tmp)

        ind_fi = np.zeros(len(freq_tmp), dtype='int')
        freq_rs = np.zeros(len(freq_tmp))
        freq_rs[0] = freq_tmp[0]
        freq_rs[-1] = freq_tmp[-1]
        ind_fi[-1] = len(frS)-1
        df = frS[1]-frS[0]
        for i in range(1, len(freq_tmp)-1):
            fi = freq_tmp[i]
            i_f = int((fi-frS[0])/df)
            freq_rs[i] = frS[i_f]
            ind_fi[i] = i_f
        #en1 = time.time()

        freq_rs, ind_u = np.unique(freq_rs, return_index=True)
        ind_fi = ind_fi[ind_u]
        timeFr = tf[ind_fi]

        if self.settRefAtfRef:
            fRef_inrange = min( max(frS[0], fRef), frS[-1] )
            print ("Important:", fRef_inrange)
            #tf_time = interpolate.splev(fRef_inrange, tfspline, der=1)
            tf_time = tfspline(fRef_inrange)
            # print (tRef, tf_time, fRef_inrange, np.shape(frS))
            Del_Phase = 2.*np.pi * (frS - fRef_inrange) * (tRef - tf_time)
            # print (np.shape(Del_Phase), np.shape(phS))
            #Del_Phase = 2.*np.pi * (frS - fRef_inrange) * (tRef - tfspline(fRef_inrange))
            # Del_shift = (tRef - tfspline(fRef_inrange))
            # print ("Shift = ", Del_shift)
            phS += Del_Phase

        self.ComputeProjections(timeFr, verbose=verbose)

        om = 2.0*np.pi*freq_rs
        phR, RX, RY, RZ = self.ComputeResponse(om)

        ### interpolating
        spl = spline(freq_rs, phR)
        phR_i = spl(frS)

        splR = spline(freq_rs, np.real(RX))
        splI = spline(freq_rs, np.imag(RX))
        RX_i = splR(frS) + 1.0j*splI(frS)

        splR = spline(freq_rs, np.real(RY))
        splI = spline(freq_rs, np.imag(RY))
        RY_i = splR(frS) + 1.0j*splI(frS)

        splR = spline(freq_rs, np.real(RZ))
        splI = spline(freq_rs, np.imag(RZ))
        RZ_i = splR(frS) + 1.0j*splI(frS)

        plotIt = False
        if plotIt:
            fig, ax = plt.subplots(nrows=3, sharex=True)
            ax[0].loglog(frS, np.abs(np.real(RX_i)))
            ax[0].loglog(freq_rs, np.abs(np.real(RX)), 'o')
            ax[0].loglog(frS, np.abs(np.imag(RX_i)))
            ax[0].loglog(freq_rs, np.abs(np.imag(RX)), 'o')
            ax[1].loglog(frS, np.abs(np.real(RY_i)))
            ax[1].loglog(freq_rs, np.abs(np.real(RY)), 'o')
            ax[1].loglog(frS, np.abs(np.imag(RY_i)))
            ax[1].loglog(freq_rs, np.abs(np.imag(RY)), 'o')
            ax[2].loglog(frS, np.abs(np.real(RZ_i)))
            ax[2].loglog(freq_rs, np.abs(np.real(RZ)), 'o')
            ax[2].loglog(frS, np.abs(np.imag(RZ_i)))
            ax[2].loglog(freq_rs, np.abs(np.imag(RZ)), 'o')
            plt.show()

        return (phR_i, RX_i, RY_i, RZ_i)

        #}}}


    def InterpolateFreq(self, f, amp, ph, kR0, RX, RY, RZ, tShift, freq, fbeg, fend):
        """ Interpolates in the freq. domain from coarse freqs given by f, to even sampled freq """
        ### amp: amplitude
        ### ph: phase
        ### kR0: delay phase
        ### RX, RY, RZ responses
        ### fbeg: initial frequency of the signal
        ### fend: end frequency of the signal

        ibeg = np.where(freq>=fbeg)[0][0]
        iend = np.where(freq<=fend)[0][-1]
        # print ("freq range:", fbeg, fend, ibeg, iend)
        fs = freq[ibeg:iend+1]

        om = 2.0*np.pi*fs
        omL = om*self.armt
        spl = spline(f, amp)
        amp_i = spl(fs)
        spl = spline(f, ph)
        phase_i = spl(fs)
        #print (np.shape(f), np.shape(kR0))
        spl = spline(f, kR0)
        #phaseRdelay_i = om*spl(fs)
        phaseRdelay_i = spl(fs)
        spl_r = spline(f, np.real(RX))
        spl_i = spline(f, np.imag(RX))

        phasetimeshift = -2.0*np.pi*tShift*fs
        #phasetimeshift = 2.0*np.pi*t0*fs
        common = amp_i* np.exp(1.0j* (phase_i + phaseRdelay_i + phasetimeshift))

        Xf_i = np.zeros(len(freq), dtype='complex128')
        Yf_i = np.zeros(len(freq), dtype='complex128')
        Zf_i = np.zeros(len(freq), dtype='complex128')

        Xf_i[ibeg:iend+1] = (spl_r(fs)+ 1.j*spl_i(fs))*common

        spl_r = spline(f, np.real(RY))
        spl_i = spline(f, np.imag(RY))

        Yf_i[ibeg:iend+1] = (spl_r(fs)+ 1.j*spl_i(fs))*common

        spl_r = spline(f, np.real(RZ))
        spl_i = spline(f, np.imag(RZ))

        Zf_i[ibeg:iend+1] = (spl_r(fs)+ 1.j*spl_i(fs))*common

        ### NOTE that the size of fs and Xf do not match !!!!, the size of tdis is the same as freq
        return(fs, Xf_i, Yf_i, Zf_i)


    def FrtoTm(self, del_t, Xf, Yf, Zf):
        Xtlw = np.fft.irfft(Xf)*(1.0/del_t)
        Ytlw = np.fft.irfft(Yf)*(1.0/del_t)
        Ztlw = np.fft.irfft(Zf)*(1.0/del_t)

        tlw = np.arange(len(Xtlw))*del_t

        return (tlw, Xtlw, Ytlw, Ztlw)


    def ComputeTDTDI(self, tm, Amp, phi0, om, omdot, RX, RY, RZ, src="GB"):
        """ So far only for GBs, for other sources I need phase in time domain either numerically or analytic """
        if (src != "GB"):
            raise NotImplementedError

        #### TODO Check if we need to interpolate alreadyprecomputed projections
        xi = tm - self.kR0
        #print ("stas:", np.shape(xi), np.shape(tm), np.shape(self.kR0))

        phase = -phi0 + om*xi + 0.5*omdot*xi**2
        # phase = phi0 + om*xi + 0.5*omdot*xi**2
        #print (np.shape(phase))

        hfast = Amp*np.exp(1.0j*phase)
        #print (np.shape(hfast))

        X = np.real(hfast*RX)
        Y = np.real(hfast*RY)
        Z = np.real(hfast*RZ)
        #print (np.shape(X), np.shape(Y))

        return(X, Y, Z)



    ####################################
    #    INternal functions
    ###################################

    def PolarizationTensor(self, lam, bet, psi):
       sl = np.sin(lam)
       cl = np.cos(lam)
       sb = np.sin(bet)
       cb = np.cos(bet)
       # u = 1.0*np.array([sb*cl, sb*sl, -cb])
       # v = 1.0*np.array([sl, -cl, 0.0])
       v = -1.0*np.array([sb*cl, sb*sl, -cb])
       u = 1.0*np.array([sl, -cl, 0.0])
       Ep = np.zeros((3,3))
       Ec = np.zeros((3,3))
       for i in range(3):
           for j in range(3):
               Ep[i, j] = u[i]*u[j] - v[i]*v[j]   ### Polarization basis in SSB
               #Ep[i, j] = -u[i]*u[j] + v[i]*v[j]
               Ec[i, j] = u[i]*v[j] + u[j]*v[i]

       #psi = 0.5*np.pi - psi
       cp = np.cos(2.0*psi)
       sp = np.sin(2.0*psi)

       # plus = -cp*Ep + sp*Ec
       # cross = -sp*Ep - cp*Ec
       plus = cp*Ep + sp*Ec
       cross = -sp*Ep + cp*Ec   ### polarization basis in the source frame

       return(plus, cross)


    def LISAmotion(self, tm, verbose=False):
        AU = LC.ua/LC.clight  ## AU in sec.
        lamb = 0.0
        kappa = 0.0

        N = len(tm)

        a = AU
        e = self.armt/(2.0*np.sqrt(3.0)*a)
        if verbose:
            print ("orbital eccentricity", e)
        nn = np.array([1,2,3])
        year = LC.YRSID_SI

        Beta = (nn-1)*2.0*np.pi/3.0 + lamb
        alpha = 2.0*np.pi*tm/year + kappa

        x = np.zeros((3, N))
        y = np.zeros((3, N))
        z = np.zeros((3, N))
        for i in range(3):
            x[i, :] = a*np.cos(alpha) + a*e*(np.sin(alpha)*np.cos(alpha)*np.sin(Beta[i]) - (1.0 + (np.sin(alpha))**2)*np.cos(Beta[i]))
            y[i, :] = a*np.sin(alpha) + a*e*(np.sin(alpha)*np.cos(alpha)*np.cos(Beta[i]) - (1.0 + (np.cos(alpha))**2)*np.sin(Beta[i]))
            z[i, :] = -np.sqrt(3.0)*a*e*np.cos(alpha - Beta[i])

        return (x, y, z, alpha, a)

    def ComputeYslr(self, send, link, rec, lsign=1):
        ### Does not include fast varying part of the GW signal (only response part)
        if (link == 1):
            kn = lsign*self.kn1
            A = self.A1
        elif (link == 2):
            kn = lsign*self.kn2
            A = self.A2
        elif (link == 3):
            kn = lsign*self.kn3
            A = self.A3
        else:
            print ("should never be here")
            raise ValueError
        if (rec == 1):
            kq = self.kq1
        elif (rec == 2):
            kq = self.kq2
        elif (rec == 3):
            kq = self.kq3
        else:
            print ("should never be here")
            raise ValueError
        args = 0.5*self.omL*(1.0 - kn)

        # print ("A", A)
        # print ("args", args)
        # print ("kq", kq)
        y = -0.5j*self.omL*A*sinc(args)*np.exp(-1.0j*(args + self.om*kq))
        # print ("LW", -0.5j*self.omL, A, sinc(args), np.exp(-1.0j*(args + self.om*kq)) )
        ### FIXME
        #y = 0.5j*omL*A*sinc(args)*np.exp(-1.0j*(args + om*kq))

        #y = y*hf*np.exp(1.j*om*kR0)

        return(y)
