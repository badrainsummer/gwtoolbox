from multiprocessing import Pool
from .functions_space import *
#from MLDC_tools import *
import numpy as np
import sys
import ctypes
from ctypes import CDLL
from ctypes import RTLD_GLOBAL
from .parameters import *
import pandas as pd
import platform
import random
import json
import os
import glob

def _find_libs():
    """
    Automatically search for GSL and GSL CBLAS shared libraries.
    The function selects appropriate search paths based on the OS.
    """
    system = platform.system()

    candidates = []

    # ----- macOS (supports Homebrew Intel/M1/M2 and MacPorts) -----
    if system == "Darwin":
        search_dirs = [
            "/usr/local/opt/gsl/lib",
            "/opt/homebrew/opt/gsl/lib",
            "/usr/local/lib",
            "/opt/local/lib",
        ]
        patterns = ["libgsl*.dylib", "libgslcblas*.dylib"]

    # ----- Linux (various distributions including /usr/lib, /usr/local/lib, and conda environments) -----
    else:
        search_dirs = [
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib",
            "/usr/local/lib64",
        ]

        # Conda environment
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            search_dirs.append(os.path.join(conda_prefix, "lib"))

        patterns = ["libgsl*.so*", "libgslcblas*.so*"]

    # Collect candidate library files
    libs = []
    for d in search_dirs:
        for p in patterns:
            libs.extend(glob.glob(os.path.join(d, p)))

    # Remove duplicates and sort (prefer unversioned libraries first)
    libs = sorted(set(libs), key=lambda x: (".so." in x) or (".dylib." in x))

    return libs


def load_gsl():
    """
    Load GSL and GSL CBLAS libraries.
    RTLD_GLOBAL is used on Linux to ensure symbols are visible
    to subsequently loaded C extensions.
    """
    libs = _find_libs()
    if not libs:
        raise RuntimeError(
            "GSL libraries not found on this system.\n"
            "Please install GSL or set DYLD_LIBRARY_PATH / LD_LIBRARY_PATH."
        )

    loaded = []
    for lib in libs:
        try:
            # macOS does not require RTLD_GLOBAL; Linux does for symbol visibility
            if platform.system() == "Darwin":
                ctypes.CDLL(lib)
            else:
                ctypes.CDLL(lib, mode=RTLD_GLOBAL)

            loaded.append(lib)
        except Exception:
            continue

    if not loaded:
        raise RuntimeError("GSL libraries were found but none could be loaded.")

    return loaded


# ---- Invocation (load automatically at import) ----

loaded_paths = load_gsl()
# print("Loaded GSL libs:", loaded_paths)


# if platform.uname().system == 'Darwin':
#     gslcblas = CDLL('/usr/local/Cellar/gsl/2.6/lib/libgslcblas.dylib')
#     gsl = CDLL('/usr/local/Cellar/gsl/2.6/lib/libgsl.dylib')
# else:
#     gslcblas = CDLL('/usr/lib64/libgslcblas.so',mode=RTLD_GLOBAL)
#     gsl = CDLL('/usr/lib64/libgsl.so',mode=RTLD_GLOBAL)
path=os.path.dirname(os.path.abspath(__file__))
import AAKwrapper

#def listSNR_unit(setting):
#        ps, det= setting
#        return [SMBHB.SNR(p,det) for p in ps]

class SMBHB:
    """
    This is a class to describe super massive black hole black hole mergers.

    Parameters:
      cosmos: define cosmological model

    """
    def __init__(self, pop_model, cosmos):
        """
       @param: pop_model, string, can be 'pop3', 'Q3_delays', 'Q3_nodelays';
       @param: SNR_thres, float, threshold
       @param: cosmos: cosmology model
       @param: duration: in unit of year
        """
        #self.det=det

        self.pop_model=pop_model
        #self.SNR_thres=SNR_thres
        #self.duration=duration
        self.cosmos=cosmos
        self.Zs=np.logspace(-3,np.log10(20),200)
        self.Ds=cosmos.luminosity_distance(self.Zs).value/1e3

    def SNR(self, p,det):
        """
        @param: MLDC p, parameter object.
        @param: det, detector object
        returns:
        @float, SNR of the TDI_X of given p and noise curves.
        """
        #masslimit=1e3 # solar masses
        #massupper=1e7
        Larm=det.lisaLT*2.99792458e8 # arm length in meter
        z=p.pars['Redshift']
        # debugging ysx: 6th Oct.:
        #p.display()
        mass1,mass2,distance=p.pars['Mass1'],p.pars['Mass2'],p.pars['Distance']
        #p.pars.update({'Cadence':5})
        #print("m1=%f,m2=%f,z=%f,D=%f" % (mass1,mass2,z,distance))
        if (1e8>(mass1+mass2) >1e6): # red-shifted
            df=1e-7
            cadence=100
        elif ((mass1+mass2)>=1e8):
            df=1e-7
            cadence=100
        else:
            df=1e-5
            cadence=5
        try:
            p.pars.update({'Cadence':cadence})
            freq, Xf, Yf, Zf=ComputeMBHBXYZ_FD(p=p, Larm=Larm, df=df)
        except:
            freq=[0]
        if len(freq)==1:
            return 0
        else:
            freq=freq[1::]# /(1+z) # don't need to do that!  manually redshift, ysx 23/4/2020
            Xf=Xf[1::] #*(1+z)**(5./6.) # manually redshift, ysx 23/4/2020; ysx 6/10/2020: I believe there's no need to redshift the X,Y,Z because in p there's already the red-shifted masses!
            Yf=Yf[1::] #*(1+z)**(5./6.) # manually redshift, ysx 23/4/2020
            Zf=Zf[1::] #*(1+z)**(5./6.) # manually redshift, ysx 23/4/2020
            y=det.Stdix(freq)
            SNR2=4.*np.trapz(np.abs(Xf)**2/y,freq)
    #     print("SNR=%f" % np.sqrt(SNR2))
            return np.sqrt(SNR2)

    def listSNR_unit(self, setting):
        ps, det=setting
        return [self.SNR(p,det) for p in ps]
    def listSNR(self, ps, det):
        n_cores=3
        settings=[]
        unit_len=int(len(ps)/n_cores)+1
        for i in range(n_cores):
            setting=[ps[i*unit_len:min((i+1)*unit_len,len(ps))],det]
            settings.append(setting)
        #print(__name__)
        #if __name__ == 'gwtoolbox.sources_mHz':
        with Pool(n_cores) as pool:
            #pool=Pool(n_cores)
            SNRlist_multi=pool.map(self.listSNR_unit, settings)
        #pool.close()
        SNRlist=np.concatenate(tuple(SNRlist_multi))
        return SNRlist

    def givelist(self, duration, SNR_thres, det, Nmax):
        """
        @param: duration, in unit of year
        @SNR_threshold, float
        @det: detector class
        """
        #Larm=det.lisaLT*299792458. # in meter
        #lisaD=det.lisaD
        #lisaP=det.lisaP
        #pardicts=[]
    #if sourceType=='MBHB':
        kys=MBHBKys+["rho"]
        #kys=['Mass1', 'Mass2', 'Redshift', 'Spin1', 'Spin2', 'AzimuthalAngleOfSpin1', 'AzimuthalAngleOfSpin2', 'CoalescenceTime', 'Distance', 'EclipticLatitude', 'EclipticLongitude', 'PhaseAtCoalescence']
        PS=EventsUniverse(cosmos=self.cosmos, duration=duration, sourceType='MBHB', model=self.pop_model)
        SNRs=self.listSNR(PS, det)
        OK_indices=np.array(range(0,len(SNRs)))[np.array(SNRs)>=SNR_thres]
        tot_numbers=len(OK_indices)
        Nshow=min(tot_numbers, Nmax)
        cata=np.empty([Nshow,len(kys)])
        #for i in range(0,Nshow):
            #pardicts.append(PS[OK_indices[i]])
        #    for ind_key in range(0,len(kys)-1):
        cata[:,:len(kys)-1]=np.array([[PS[OK_indices[i]].pars[kys[ind_key]] for ind_key in range(len(kys)-1)] for i in range(Nshow)])
        cata[:, len(kys)-1]=[SNRs[OK_indices[i]] for i in range(Nshow)]
        return [cata, tot_numbers]
    def givedataframe(self, duration, SNR_thres, det, Nmax):
        cata, tot_numbers= self.givelist(duration, SNR_thres, det, Nmax)
        try:
            df= pd.DataFrame(data=cata[:,[0,1,2,3,4,8,9,10,12]], columns=['M1', 'M2', 'z', 's1', 's2','D', 'β', '𝝺', 'SNR'])
        except:
            df=None
            tot_numbers=31415926
        return [df, tot_numbers]


    def errors(self, p, det, steps):
        """
        @param: p, parameter dict
        @param: detector
        @param: steps, derivative steps
        """
        Larm=det.lisaLT*299792458 # in meters
        keys=['Mass1','Mass2','Spin1','Spin2','Distance','CoalescenceTime','PhaseAtCoalescence','EclipticLatitude','EclipticLongitude','InitialPolarAngleL','InitialAzimuthalAngleL'] # 11 parameters that want to be calculated with uncertainties.
        mass1, mass2 = p.pars['Mass1'],p.pars['Mass2']
        if (1e8>(mass1+mass2) >1e6): # red-shifted
            df=1e-7
            cadence=100
        elif ((mass1+mass2)>=1e8):
            df=1e-7
            cadence=100
        else:
            df=1e-5
            cadence=5

        p.pars.update({'Cadence': cadence})
        freq, Xf, Xf, Zf=ComputeMBHBXYZ_FD(p=p, Larm=Larm,df=df)
        freq=freq[1::]
        partials=np.empty(shape=(len(freq),len(keys)),dtype=np.complex128)
        for i in range(len(keys)):
            p_left, p_right, strid =leapfrog(p,keys[i],steps, self.Zs, self.Ds)
    #        partials[:,i]=/(2.*steps)
            freq, Xf_left, Yf_left, Zf_left=ComputeMBHBXYZ_FD(p=p_left, Larm=Larm, df=df)
            freq, Xf_right,Yf_right,Zf_left=ComputeMBHBXYZ_FD(p=p_right, Larm=Larm, df=df)
            freq=freq[1::]
            Xf_left=Xf_left[1::]
            Xf_right=Xf_right[1::]
    #        Yf_=Yf_[1::]
    #        Zf_=Zf_[1::]
            Xf_delta=Xf_right-Xf_left
            partials[:,i]=Xf_delta/(2.*strid)
            #if keys[i]=='Redshift':
        y=det.Stdix(freq)
        to_intg_1=np.einsum('...i,...j->...ij', np.conj(partials), partials)
        to_intg_2=np.einsum('i...,i->i...',to_intg_1, 1./y)
        fisher=4.*np.real(np.trapz(to_intg_2,freq,axis=0))
        #return fisher
        #print("Fisher Matrix", fisher)
        sigma=np.linalg.inv(fisher)
        deltaOmega=2*np.pi*np.cos(p.pars['EclipticLatitude'])*(np.sqrt(np.abs(sigma[7,7]*sigma[8,8]))-np.abs(sigma[7,8]))
        # Cutler 1998, PRD
        deltaD=np.sqrt(np.abs(sigma[4,4]))
        deltaD=deltaD #if deltaD<p.pars['Distance'] else p.pars['Distance']
        #z_center=p.pars['Redshift']
        #z_left=z_center-np.abs(error_z)
        #z_right=z_center+np.abs(error_z)
        #D_left=cosmos.luminosity_distance(z_left).value/1e3
        #D_right=cosmos.luminosity_distance(z_right).value/1e3
        #deltaD=(D_right-D_left)*0.5
        m1=p.pars['Mass1']
        m2=p.pars['Mass2']
        deltam1=np.sqrt(np.abs(sigma[0,0]))
        deltam2=np.sqrt(np.abs(sigma[1,1]))
        deltas1=np.sqrt(np.abs(sigma[2,2]))
        deltas1=deltas1 if deltas1<p.pars['Spin1'] else p.pars['Spin1']
        deltas2=np.sqrt(np.abs(sigma[3,3]))
        deltas2=deltas2 if deltas2<p.pars['Spin2'] else p.pars['Spin2']
        return [deltam1, deltam2, deltas1, deltas2, deltaD, deltaOmega/(4.*np.pi)*41252.96]

    def errors_debug(self, p, det, steps, step_dist):
        """
        @param: p, parameter dict
        @param: detector
        @param: steps, derivative steps
        """
        Larm=det.lisaLT*299792458 # in meters
        keys=['Mass1','Mass2','Spin1','Spin2','Distance','CoalescenceTime','PhaseAtCoalescence','EclipticLatitude','EclipticLongitude','InitialPolarAngleL','InitialAzimuthalAngleL'] # 11 parameters that want to be calculated with uncertainties.
        mass1, mass2 = p.pars['Mass1'],p.pars['Mass2']
        if (1e8>(mass1+mass2) >1e6): # red-shifted
            df=1e-7
            cadence=100
        elif ((mass1+mass2)>=1e8):
            df=1e-7
            cadence=100
        else:
            df=1e-5
            cadence=5

        p.pars.update({'Cadence': cadence})
        freq, Xf, Xf, Zf=ComputeMBHBXYZ_FD(p=p, Larm=Larm,df=df)
        freq=freq[1::]
        partials=np.empty(shape=(len(freq),len(keys)),dtype=np.complex128)
        for i in range(len(keys)):
            p_left, p_right, strid =leapfrog_debug(p,keys[i],steps,step_dist, self.Zs, self.Ds)
    #        partials[:,i]=/(2.*steps)
            freq, Xf_left, Yf_left, Zf_left=ComputeMBHBXYZ_FD(p=p_left, Larm=Larm, df=df)
            freq, Xf_right,Yf_right,Zf_left=ComputeMBHBXYZ_FD(p=p_right, Larm=Larm, df=df)
            freq=freq[1::]
            Xf_left=Xf_left[1::]
            Xf_right=Xf_right[1::]
    #        Yf_=Yf_[1::]
    #        Zf_=Zf_[1::]
            Xf_delta=Xf_right-Xf_left
            partials[:,i]=Xf_delta/(2.*strid)
            #if keys[i]=='Redshift':
        y=det.Stdix(freq)
        to_intg_1=np.einsum('...i,...j->...ij', np.conj(partials), partials)
        to_intg_2=np.einsum('i...,i->i...',to_intg_1, 1./y)
        fisher=4.*np.real(np.trapz(to_intg_2,freq,axis=0))
        #return fisher
        #print("Fisher Matrix", fisher)
        sigma=np.linalg.inv(fisher)
        deltaOmega=2*np.pi*np.cos(p.pars['EclipticLatitude'])*(np.sqrt(np.abs(sigma[7,7]*sigma[8,8]))-np.abs(sigma[7,8]))
        # Cutler 1998, PRD
        deltaD=np.sqrt(np.abs(sigma[4,4]))
        deltaD=deltaD #if deltaD<p.pars['Distance'] else p.pars['Distance']
        #z_center=p.pars['Redshift']
        #z_left=z_center-np.abs(error_z)
        #z_right=z_center+np.abs(error_z)
        #D_left=cosmos.luminosity_distance(z_left).value/1e3
        #D_right=cosmos.luminosity_distance(z_right).value/1e3
        #deltaD=(D_right-D_left)*0.5
        m1=p.pars['Mass1']
        m2=p.pars['Mass2']
        deltam1=np.sqrt(np.abs(sigma[0,0]))
        deltam2=np.sqrt(np.abs(sigma[1,1]))
        deltas1=np.sqrt(np.abs(sigma[2,2]))
        deltas1=deltas1 if deltas1<p.pars['Spin1'] else p.pars['Spin1']
        deltas2=np.sqrt(np.abs(sigma[3,3]))
        deltas2=deltas2 if deltas2<p.pars['Spin2'] else p.pars['Spin2']
        return [deltam1, deltam2, deltas1, deltas2, deltaD, deltaOmega/(4.*np.pi)*41252.96]


    def givelisterrors(self, duration, SNR_thres, det, Nmax):
        Larm=det.lisaLT*299792458. # in meter
        pardicts=[]
        kys=['Mass1', 'Mass2', 'Redshift', 'Spin1', 'Spin2', 'AzimuthalAngleOfSpin1', 'AzimuthalAngleOfSpin2', 'CoalescenceTime', 'Distance', 'EclipticLatitude', 'EclipticLongitude', 'PhaseAtCoalescence','dm1','dm2','ds1','ds2','dD','dOme','rho']
        PS=EventsUniverse(cosmos=self.cosmos, duration=duration, sourceType='MBHB', model=self.pop_model)
        SNRs=self.listSNR(PS, det)
        OK_indices=np.array(range(0,len(SNRs)))[np.array(SNRs)>=SNR_thres]
        tot_numbers=len(OK_indices)
        N_show=min(tot_numbers, Nmax)
        cata=np.empty([N_show,len(kys)])
        #for i in range(0,N_show):
            #pardicts.append(PS[OK_indices[i]])
        #    for ind_key in range(0,len(kys)-7):
        #        cata[i, ind_key]=PS[OK_indices[i]].pars[kys[ind_key]]
        cata[:,:12]=np.array([[PS[OK_indices[i]].pars[kys[ind_key]] for ind_key in range(len(kys)-7)] for i in range(N_show)])
        cata[:,18]=np.array([SNRs[OK_indices[i]] for i in range(N_show)])
        # calculate the error in parallel
        #n_core=3
        #n_unit=int(N_show/n_core)+1
        #for i in range(n_core):
        #    p=[PS[OK_indices[]
        PS_to_error=np.array([PS[OK_indices[i]] for i in range(N_show)])
        cata[:, 12:18]=self.para_np_error(PS_to_error, det, 1e-5)
        #cata[:, 12:18]=np.array([self.errors(PS[OK_indices[i]],det,1e-5) for i in range(N_show)])
            #cata[i,18]=SNRs[OK_indices[i]]
        return [cata, tot_numbers]
    def para_np_error(self, PS_to_error, det, steps):
        n_cores=3
        n_unit=int(len(PS_to_error)/n_cores)+1
        settings=[]
        for i in range(n_cores):
            settings.append([PS_to_error[i*n_unit:min((i+1)*n_unit,len(PS_to_error))], det, steps])
        with Pool(n_cores) as pool2:
        #pool2=Pool(n_cores)
        #try:
            #return None
            result_multi=pool2.map(self.unit_np_error, settings)
        #print(result_multi)
        #except:
        #    results=None
        results=np.concatenate(result_multi, axis=0)
        #pool2.close()
        return results
    def unit_np_error(self, setting):
        pars, det, steps=setting
        #print(pars[0])
        results=np.array([self.errors(p, det, steps) for p in pars])
        return results
    def giveerrordataframe(self, duration, SNR_thres, det,Nmax):
        kys=['M1', 'M2', 'z', 's1', 's2', 'AzimuthalAngleOfSpin1', 'AzimuthalAngleOfSpin2', 'CoalescenceTime', 'D', 'β', '𝝺' , 'PhaseAtCoalescence','dm1','dm2','ds1','ds2','dD','dΩ','SNR','dz']
        cata, tot_numbers=self.givelisterrors(duration, SNR_thres, det, Nmax)
        dz_column=cata[:,16]*1e3/dDovdz(cata[:,2],self.cosmos)
        cata_1more=np.column_stack((cata, dz_column))
        toshow=[0,1,2,3,4,8,9,10,12,13,14,15,16,17,18,19]
        df=pd.DataFrame(data=cata_1more[:,toshow], columns=np.array(kys)[toshow])
        #print("Debugging")
        return [df,tot_numbers]
class GB:
    def __init__(self, pop_model, cosmos):
        #sys.path.append('./MLDC-master/software/Waveforms/fastGB/')
        #sys.path.append('/usr/local/lib64/python3.6/site-packages/')
        import FastBinary as FB
        self.pop_model=pop_model
        #self.SNR_thres=SNR_thres
        #self.duration=duration
        self.cosmos=cosmos
        self.Zs=np.logspace(-3,np.log10(20),200)
        self.Ds=cosmos.luminosity_distance(self.Zs).value/1e3
        #working_path='/Library/WebServer/gwtoolbox-website/gwtoolbox/gwtoolbox'
        self.year=31558149.8
        self.fm=3.168753575e-8
        if pop_model=='Gijs':
            from astropy import units as u
            from astropy.coordinates import SkyCoord
            catafile=path+'/catalogues_mHz/Galaxy/huge_cat_1.dat'
            temp_cube=np.loadtxt(catafile,dtype=float)
            m1=temp_cube[:,0]
            m2=temp_cube[:,1]
            P=temp_cube[:,2] # orbital period in second
            Pdot=temp_cube[:,3]
            #Mdot=temp_cube[:,4]
            l=temp_cube[:,5] # galactic longitude (deg)
            b=temp_cube[:,6] # galactic latitude (deg)
            cords=SkyCoord(l=l*u.deg,b=b*u.deg,frame='galactic')
            ra=cords.icrs.ra.rad
            dec=cords.icrs.dec.rad
            d=temp_cube[:,7] # distance (kpc)
            #V1/V2 are V-band magnitudes (useful for EM part) A_V is reddening;
            fgw=2./P
            amp=intriamp(m1,m2,d,fgw*1e3) # this function wants fgw in mHz
            inc=np.random.uniform(low=0, high=0.5*np.pi, size=len(fgw))
            fdot=-2.*Pdot/P**2
            indice=np.arange(len(fgw))
            self.data_cube_huge=np.stack((fgw, fdot, dec, ra, amp, inc, np.ones(len(fgw)), np.ones(len(fgw)),indice, m1,m2,d), axis=1)
        elif pop_model=='VBs':
            catafile=path+'/catalogues_mHz/Galaxy/VB.dat'
            temp_cube=np.array(np.loadtxt(catafile, dtype=str, delimiter=','))
            var_name=temp_cube[:,0]
            var_long=temp_cube[:,1].astype(np.float64)
            var_lati=temp_cube[:,2].astype(np.float64)
            var_fgw=temp_cube[:,3].astype(np.float64)
            var_d = temp_cube[:,4].astype(np.float64)
            var_m1= temp_cube[:,5].astype(np.float64)
            var_m2= temp_cube[:,6].astype(np.float64)
            var_inc= temp_cube[:,7].astype(np.float64)
            var_amp= intriamp(var_m1, var_m2, var_d, var_fgw)
            var_indice=np.arange(len(var_name))
            self.data_cube_huge=np.stack((var_fgw*1e-3, np.ones(len(var_fgw))*1e-23, var_lati, var_long, var_amp, var_inc, np.ones(len(var_fgw)), np.ones(len(var_fgw)),var_indice, var_m1, var_m2, var_d), axis=1)
            #print(self.data_cube_huge.shape)
        self.ks=['Frequency', 'FrequencyDerivative', 'EclipticLatitude', 'EclipticLongitude', 'Amplitude', 'Inclination', 'Polarization', 'InitialPhase']
        self.kyerr=['deltaF', 'deltaAmp', 'deltaOmega']
        bnr = {}
        for i in range(len(self.ks)):
            bnr[self.ks[i]] = 0
        self.fastB=FB.FastGalacticBinary("Stas", bnr)

    def SNR_acc(self,pars, Tobs, dt, det):
        """
            @param: pars list of parameters
            @param: Tobs, in unit of years
            @param: dt, time resolve of waveform in unit of seconds.
            @param: det, detector class
        """
        Tobs=Tobs*self.year

        Xf, Yf, Zf = self.fastB.onefourier(simulator='synthlisa', vector=pars, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
        Xf=Xf*(det.lisaLT/16.6782)**2*4 # with a different Larm # yishuxu 12th Oct.
        frq = np.arange(Xf.kmin, Xf.kmin + len(Xf))*Xf.df
#    y=np.interp(frq, noise_f, noise_y)
        y = det.Stdix(frq)
        SNR2=4.*np.trapz(np.abs(Xf)**2/y,frq)*Tobs**2
        return np.sqrt(SNR2)

    def SNR_speed(self,pars, Tobs, dt, det, Michelson=False):
        """
            @param: pars list of parameters
            @param: Tobs, in unit of years
            @param: dt, time resolution of waveform in unit of seconds.
            @param: det, detector class
            @param: Michelson, whether to use Michelson SNR
        """
        Tobs=Tobs*self.year
        if len(pars.shape)==2:
            amp=pars[:,4]
            f=pars[:,0]
            phi=pars[:,2]
            theta=1.5707963-pars[:,3]
            psi=pars[:,6]
            iota=pars[:,5]
        else: 
            amp=pars[4]
            f=pars[0]
            phi=pars[2]
            theta=1.5707963-pars[3]
            psi=pars[6]
            iota=pars[5]
        Dpt=243./512.*np.cos(theta)*np.sin(2*phi)*(2.*np.cos(phi)**2-1)*(1.+np.cos(theta)**2)
        Dt2=3./512.*(120.*np.sin(theta)**2+np.cos(theta)**2+162.*np.sin(2.*phi)**2*np.cos(theta)**2)
        Dp2=3./2048.*(487.+158.*np.cos(theta)**2+7.*np.cos(theta)**4-162.*np.sin(2.*phi)**2*(1+np.cos(theta)**2)**2)
        Fp2=0.25*(np.cos(2.*psi)**2*Dp2-np.sin(4.*psi)*Dpt+np.sin(2.*psi)**2*Dt2)
        Ft2=0.25*(np.cos(2.*psi)**2*Dt2+np.sin(4.*psi)*Dpt+np.sin(2.*psi)**2*Dp2)
        A2=0.5*amp**2*((1.+np.cos(iota)**2)**2*Fp2+4.*np.cos(iota)**2*Ft2)
        
        if Michelson == True:
            Sf=det.Snoise(f)
        else:
            Sf=det.Stdix(f)
        if Michelson == True:
            return np.sqrt(A2/Sf*Tobs) #Wouter edit
        else:
            return np.sqrt(16.*A2/Sf*Tobs)*np.abs(np.sin(f/det.tf))*(f/det.tf)
        
    def tot_num(self, Tobs, det, rhostar):
        """
            @param: Tobs should in unit of year
        """
        #Tobs=Tobs*self.year
        SNRs=self.SNR_speed(self.data_cube_huge[0:8], Tobs,15.0,det)
        sample_length=len(self.data_cube_huge)
        N_Universe=26084411
        data_sm=self.data_cube_huge[SNRs>rhostar] # a smaller sample filtered with speed algothrism
#        SNRs2=np.array([self.SNR_acc(pars, Tobs, 15.0, det) for pars in data_sm])
#        result=len(SNRs2[SNRs2>rhostar])/float(sample_length)*N_Universe
        result=len(data_sm)/float(sample_length)*N_Universe
        return int(result)

    def givelist(self, Tobs, det, rhostar,size):
        """
            @param: Tobs, Observation duration, in unit of year
            @param: det , detector
            @param: rhostar, SNR_threshold
        """
        N_Universe=26084411
        sample_length=len(self.data_cube_huge)
        data_cube=self.data_cube_huge
        SNRs=self.SNR_speed(data_cube, Tobs, 15.0, det)
        sample_length=len(SNRs)
        data_sm=data_cube[SNRs>rhostar]
#        SNRs2=np.array([self.SNR_acc(pars, Tobs, 15.0, det) for pars in data_sm])
#        data_final=data_sm[SNRs2>rhostar]
        data_final=np.column_stack((data_sm, SNRs[SNRs>rhostar]))
        if self.pop_model=='Gijs':
            tot=len(data_final)/float(sample_length)*N_Universe
        elif self.pop_model=='VBs':
            tot=len(data_final)
        return [data_final[:size],tot]

    def givedataframe(self, Tobs, det, rhostar, size):
        cata,tot=self.givelist(Tobs, det, rhostar,size)
        cata_str=cata.astype(str)
        kys=['f', 'fdot', 'β', '𝝺','Amp', 'Inclination', 'Polarization', 'InitialPhase']
        if self.pop_model=='Gijs':
            kys=kys+['indice','m1','m2','D','SNR']
            toshow=[0,1,2,3,4,12,9,10,11]
        elif self.pop_model=='VBs':
            indices=cata[:,8].astype(int)
            Names=np.array(VBsNames)[indices]
            cata_str[:,8]=Names
            kys=kys+['name','m1','m2','D','SNR']
            toshow=[8,0,1,2,3,4,12,9,10,11]
        df=pd.DataFrame(data=cata_str[:,toshow], columns=np.array(kys)[toshow])
        dtype_dict={}
        for ind in toshow:
            if kys[ind]=='name':
                dtype_dict[kys[ind]]='str'
            else:
                dtype_dict[kys[ind]]='float'
        #df.astype(dtype_dict)
        return [df.astype(dtype_dict),tot]
    def errors(self, pars, det, Tobs, dt, steps=1e-8):
        """
            @param: Tobs, Observation duration, in unit of year
            @param: det , detector
            @param: rhostar, SNR_threshold            
        """
        pars=pars[0:8]
        Tobs=Tobs*self.year
        keys = ['Frequency','FrequencyDerivative','EclipticLatitude', 'EclipticLongitude', 'Amplitude', 'Inclination', 'Polarization', 'InitialPhase']
        Xf, Yf, Zf = self.fastB.onefourier(simulator='synthlisa', vector=pars, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
        Xf=Xf*(det.lisaLT/16.6782)**2 # with a different Larm
        freq = np.arange(Xf.kmin, Xf.kmin + len(Xf))*Xf.df
        partials=np.empty(shape=(len(freq),len(keys)),dtype=np.complex128)
        for i in range(len(keys)):
            p_left=pars-pars*np.reshape(np.eye(1,len(keys),i), (8,))*steps
            p_right=pars+pars*np.reshape(np.eye(1,len(keys),i), (8,))*steps
            Xf_left, Yf_left, Zf_left=self.fastB.onefourier(simulator='synthlisa', vector=p_left, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
            Xf_right, Yf_right, Zf_right=self.fastB.onefourier(simulator='synthlisa', vector=p_right, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
            Xf_delta=Xf_right-Xf_left
            Xf_delta=Xf_delta*(det.lisaLT/16.6782)**2 #with a different Larm
            freq_temp=np.arange(Xf_delta.kmin, Xf_delta.kmin + len(Xf_delta))*Xf_delta.df
            #partials[:,i]=Xf_delta/(2.*steps*pars[i])
            partials[:,i]=np.interp(freq, freq_temp, Xf_delta/(2.*steps*pars[i]))
        y=det.Stdix(freq)
        to_intg_1=np.einsum('...i,...j->...ij', np.conj(partials), partials)
        to_intg_2=np.einsum('i...,i->i...',to_intg_1, 1./y)
        fisher=4.*np.real(np.trapz(to_intg_2,freq,axis=0))*Tobs**2*16. # yishuxu 12th. Oct
        sigma=np.linalg.inv(fisher)
        beta=pars[2]
        deltaOmega=2.*np.pi*np.cos(beta)*(np.sqrt(np.abs(sigma[2,2]*sigma[3,3]))-np.abs(sigma[2,3]))
        deltaAmp=np.sqrt(np.abs(sigma[4,4]))
        deltaF=1./Tobs
        #deltaF=np.sqrt(np.abs(sigma[0,0]))
        deltaLat=sigma[2,2]
        deltaLong=sigma[3,3]
        return [deltaF, deltaAmp, np.abs(deltaOmega)/(4.*np.pi)*41252.96,
                deltaLat, deltaLong]  
    
    def cov_matrix(self, pars, det, Tobs, dt, steps=1e-8):
        """
            @param: pars, parameters of the source
            @param: Tobs, Observation duration, in unit of year
            @param: det , detector
            @param: dt, sampling
            @param: steps, size of the step to do numerical derivative 
        """
        pars=pars[0:8]
        Tobs=Tobs*self.year
        keys = ['Frequency','FrequencyDerivative','EclipticLatitude', 'EclipticLongitude', 'Amplitude', 'Inclination', 'Polarization', 'InitialPhase']
        Xf, Yf, Zf = self.fastB.onefourier(simulator='synthlisa', vector=pars, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
        Xf=Xf*(det.lisaLT/16.6782)**2 # with a different Larm
        freq = np.arange(Xf.kmin, Xf.kmin + len(Xf))*Xf.df
        partials=np.empty(shape=(len(freq),len(keys)),dtype=np.complex128)
        for i in range(len(keys)):
            p_left=pars-pars*np.reshape(np.eye(1,len(keys),i), (8,))*steps
            p_right=pars+pars*np.reshape(np.eye(1,len(keys),i), (8,))*steps
            Xf_left, Yf_left, Zf_left=self.fastB.onefourier(simulator='synthlisa', vector=p_left, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
            Xf_right, Yf_right, Zf_right=self.fastB.onefourier(simulator='synthlisa', vector=p_right, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
            Xf_delta=Xf_right-Xf_left
            Xf_delta=Xf_delta*(det.lisaLT/16.6782)**2 #with a different Larm
            freq_temp=np.arange(Xf_delta.kmin, Xf_delta.kmin + len(Xf_delta))*Xf_delta.df
            #partials[:,i]=Xf_delta/(2.*steps*pars[i])
            partials[:,i]=np.interp(freq, freq_temp, Xf_delta/(2.*steps*pars[i]))
        y=det.Stdix(freq)
        to_intg_1=np.einsum('...i,...j->...ij', np.conj(partials), partials)
        to_intg_2=np.einsum('i...,i->i...',to_intg_1, 1./y)
        fisher=4.*np.real(np.trapz(to_intg_2,freq,axis=0))*Tobs**2*16. # yishuxu 12th. Oct
        sigma=np.linalg.inv(fisher)
        return sigma

    def givelisterrors(self,Tobs, det, rhostar,size):
        data_final, tot=self.givelist(Tobs, det, rhostar, size)
        lengthlist=len(data_final)
        #ErrorCube=np.empty((lengthlist,3))
        ErrorCube=np.array([self.errors(list(data_final[i,:]), det, Tobs, 15.0) for i in range(lengthlist)])
        #for i in range(lengthlist):
        #    pars=list(data_final[i,:])
        #    ErrorCube[i,:]=self.errors(pars, det, Tobs, 15.0)
        result=np.concatenate((data_final, ErrorCube), axis=1)

        return [result, tot]
    def giveerrordataframe(self, Tobs, det, rhostar,size):
        cata, tot=self.givelisterrors(Tobs, det, rhostar,size)
        cata_str=cata.astype(str)
        kys=['f', 'fdot', 'β', '𝝺','Amp', 'Inclination', 'Polarization', 'InitialPhase']
        kyserror=['df', 'dAmp', 'dΩ']
        if self.pop_model=='Gijs':
            kys=kys+['indice','m1','m2','D','SNR']+kyserror
            toshow=[0,13,1,2,3,4,14,15,9,10,11]
        elif self.pop_model=='VBs':
            kys=kys+['name','M1','M2','D','SNR']+kyserror
            indices=cata[:,8].astype(int)
            Names=np.array(VBsNames)[indices]
            cata_str[:,8]=Names
            toshow=[8,0,13,1,2,3,4,14,15,9,10,11]
        df=pd.DataFrame(data=cata_str[:,toshow], columns=np.array(kys)[toshow])
        dtype_dict={}
        for ind in toshow:
            if kys[ind]=='name':
                dtype_dict[kys[ind]]='str'
            else:
                dtype_dict[kys[ind]]='float'
        return [df.astype(dtype_dict),tot]
class EMRI:
    """
    This is a class to describe extreme mass ratio inspirals.

    Parameters:
      cosmos: define cosmological model

    """
    def __init__(self, pop_model, cosmos):
        self.pop_model=pop_model
        #self.SNR_thres=SNR_thres
        #self.duration=duration
        self.cosmos=cosmos
        self.Zs=np.logspace(-3,np.log10(20),200)
        self.Ds=cosmos.luminosity_distance(self.Zs).value/1e3
#        from ctypes import CDLL
#        from ctypes import RTLD_GLOBAL
#        gslcblas = CDLL('libgslcblas.dylib',mode=RTLD_GLOBAL)
#        gsl = CDLL('libgsl.dylib')
#        import AAKwrapper
        self.year=31558149.8
        self.fm=3.168753575e-8
        #working_path='/Library/WebServer/gwtoolbox-website/gwtoolbox/gwtoolbox'
        self.cat_path=path+'/catalogues_mHz/EMRIs/'
        self.wave_path=path+'/EMRI_waveform/'
        self.kys= ['backint','LISA','length','dt','p','T','f','T_fit','mu','M','s','e','iota','gamma','psi','theta_S','phi_S','theta_K','phi_K','alpha','D']
    def SNR(self, pars_cat, det, Tobs, fast=True):
        """
            @param: pars_st, is an array of parameters with the STRANGE format.
        """
        if fast:
            length=400000
            deltat=30
        else:
            length=1000000
            deltat=5
        kys=self.kys
        pars={} # the dictionary form of pars that AAKwrapper want to eat in.
        t_to_pluge = pars_cat[0] # in second
        year_to_pl = pars_cat[0]/31556926. # convert to years
        Obs_duration=min(Tobs, year_to_pl)
        pars[kys[0]] = False # backint
        pars[kys[1]] = False # LISA
        pars[kys[2]] = length # 1e6; length , length * dt equals to Obs_duration
        #pars[kys[3]] = Obs_duration/float(pars[kys[2]]) # dt
        #pars[kys[3]]= deltat #Obs_duration*31556926./length #deltat #dt yishuxu: it's one works....
        pars[kys[4]] = 8. # p
        pars[kys[5]] = Obs_duration # T in unit year
        pars[kys[6]] = 5.e-3 # f
        pars[kys[7]] = Obs_duration # T_fit
        mu_intrinsic=np.exp(pars_cat[1]+12.2209) # pars_cat[1] logrithm (in base e) of mu in unit of second! log(mu)*2.03e5,
						 # Equivalent of exp(..+12.2209)
        pars[kys[20]]= mu_intrinsic*4.7869e-23/pars_cat[14] # D in unit of Gpc
        z=np.interp(pars[kys[20]], self.Ds, self.Zs) # redshift
        pars[kys[8]] = mu_intrinsic #*(1+z) # mu, in solar masses, yishuxu: redshifted 23Oct/2020
        pars[kys[9]] = np.exp(pars_cat[2]+12.2209) #*(1+z) # M, in soloar masses, yishuxu: redshifted 23Oct/2020
        # above two lines I'm not certain
        pars[kys[3]]= deltat*pars[kys[9]]/1e6
        pars[kys[10]]= pars_cat[11] # s
        pars[kys[11]]= 0.01 #pars_cat[3] # e
        pars[kys[12]]= np.arccos(pars_cat[9]) # iota
        pars[kys[13]]= pars_cat[5] # gamma
        pars[kys[14]]= np.random.uniform(low=0,high=6.283,size=None) # psi
        pars[kys[15]]= np.arccos(pars_cat[7]) # theta_S
        pars[kys[16]]= pars_cat[8] # phi_S
        pars[kys[17]]= np.arccos(pars_cat[12]) # theta_K
        pars[kys[18]]= pars_cat[13] # phi_K
        pars[kys[19]]= pars_cat[10] # alpha
#        pars[kys[20]]= pars[kys[8]]*4.7869e-23/pars_cat[14] # D in unit of Gpc
#        z=np.interp(pars[kys[20]], self.Ds, self.Zs) # redshift
        #frq, Xf_r, Xf_im, Yf_r, Yf_im, Zf_r, Zf_im, timing = AAKwrapper.tdi(pars)
        frq, Xf_r, Xf_im, Yf_r, Yf_im, Zf_r, Zf_im, timing = AAKwrapper.aktdi(pars)
        frq=frq[1::1000]
        y = det.Stdix(frq)
        Xf=Xf_r[1::1000]+1j*Xf_im[1::1000]
        # rescaling according to different lisaLT
        x0=8.33910*2.*np.pi*frq
        x1=det.lisaLT*2.*np.pi*frq
        Xf=Xf*x1/x0*np.sin(x1)/np.sin(x0)
        SNR2=4.*np.trapz(np.absolute(Xf)**2/y,frq)
        #return np.sqrt(SNR2)
        return np.sqrt(SNR2)

    def errors(self, pars_cat, det, Tobs, fast=True):
        """
            @param: pars_st, is an array of parameters with the STRANGE format.
            """
        if fast:
            length=1000000
            deltat=5
        else:
            length=1000000
            deltat=5.184
        kys=self.kys
        pars={} # the dictionary form of pars that AAKwrapper want to eat in.
        t_to_pluge = pars_cat[0] # in second
        year_to_pl = pars_cat[0]/31556926. # convert to years
        Obs_duration=min(Tobs, year_to_pl)
        pars[kys[0]] = False # backint
        pars[kys[1]] = False # LISA
        pars[kys[2]] = length# 1e6; length , length * dt equals to Obs_duration
        #pars[kys[3]] = Obs_duration/float(pars[kys[2]]) # dt
        pars[kys[3]]= deltat #dt yishuxu: it's one works....
        pars[kys[4]] = 8. # p
        pars[kys[5]] = Obs_duration # T in unit year
        pars[kys[6]] = 2.e-3 # f
        pars[kys[7]] = 1. # T_fit
        mu_intrinsic=np.exp(pars_cat[1]+12.2209)
        pars[kys[20]]= mu_intrinsic*4.7869e-23/pars_cat[14] # D in unit of Gpc
        z=np.interp(pars[kys[20]], self.Ds, self.Zs) # redshift
        pars[kys[8]] = mu_intrinsic#*(1+z) # mu, in solar masses, yishuxu: redshifted 23Oct/2020
        pars[kys[9]] = np.exp(pars_cat[2]+12.2209)#*(1+z) # M, in soloar masses, yishuxu: redshifted 23Oct/2020
        # above two lines I'm not certain
        pars[kys[10]]= max(pars_cat[11],0.01) # s
        pars[kys[11]]= max(pars_cat[3],0.01) # e
        pars[kys[12]]= np.arccos(pars_cat[9]) # iota
        pars[kys[13]]= pars_cat[5] # gamma
        pars[kys[14]]= np.random.uniform(low=0,high=6.283,size=None) # psi
        pars[kys[15]]= np.arccos(pars_cat[7]) # theta_S
        pars[kys[16]]= pars_cat[8] # phi_S
        pars[kys[17]]= np.arccos(pars_cat[12]) # theta_K
        pars[kys[18]]= pars_cat[13] # phi_K
        pars[kys[19]]= pars_cat[10] # alpha
        frq, Xf_r, Xf_im, Yf_r, Yf_im, Zf_r, Zf_im, timing = AAKwrapper.aktdi(pars)
        frq=frq[1::100]
        y = det.Stdix(frq)
        Xf=Xf_r[1::100]+1j*Xf_im[1::100]
        # rescaling according to different lisaLT
        x0=8.33910*2.*np.pi*frq
        x1=det.lisaLT*2.*np.pi*frq
        Xf=Xf*x1/x0*np.sin(x1)/np.sin(x0)
        #Xf=Xf_r[1:]+1j*Xf_im[1:]
        kys_error=['mu','M','s','e','iota','gamma','psi','theta_S','phi_S','theta_K','phi_K','alpha','D']
        partials=np.empty(shape=(len(frq), len(kys_error)), dtype=np.complex128)
        steps=1e-6
        for i in range(len(kys_error)):
            pars_left =pars.copy()
            pars_right=pars.copy()
            #abstep=pars[kys_error[i]]*steps
            pars_left[kys_error[i]]=pars[kys_error[i]]*(1-steps)
            pars_right[kys_error[i]]=pars[kys_error[i]]*(1+steps)
            frq_l, Xf_rl, Xf_iml, Yf_rl, Yf_iml, Zf_rl, Zf_iml, timingl = AAKwrapper.aktdi(pars_left)
            frq_r, Xf_rr, Xf_imr, Yf_rr, Yf_imr, Zf_rr, Zf_imr, timingr = AAKwrapper.aktdi(pars_right)
            Xf_left =Xf_rl[1::100]+1j*Xf_iml[1::100]
            Xf_right=Xf_rr[1::100]+1j*Xf_imr[1::100]
            Xf_delta=Xf_right-Xf_left
            Xf_delta=Xf_delta*x1/x0*np.sin(x1)/np.sin(x0)
            #frq=frq_r[1:]
            #print(Xf_delta,pars[kys_error[i]])
            partials[:,i]=Xf_delta/(2.*steps*pars[kys_error[i]])
        y=det.Stdix(frq)
        to_intg_1=np.einsum('...i,...j->...ij', np.conj(partials), partials)
        to_intg_2=np.einsum('i...,i->i...',to_intg_1, 1./y)
        fisher=4.*np.real(np.trapz(to_intg_2,frq,axis=0))
        sigma=np.linalg.inv(fisher)
        beta=pars['theta_S']
        deltamu=np.sqrt(abs(sigma[0,0]))
        deltaM =np.sqrt(abs(sigma[1,1]))
        deltaD =np.sqrt(abs(sigma[12,12]))
        deltaOmega=2.*np.pi*np.cos(beta)*(np.sqrt(np.abs(sigma[7,7]*sigma[8,8]))-np.abs(sigma[7,8]))
        # Sr
        return [deltamu, deltaM, deltaD, np.abs(deltaOmega)/3.0462e-4]
    def givelist(self, Tobs, det, rhostar, Nmax, fast):
        popmodel=self.pop_model
        if popmodel not in ['M1','M2','M3','M4','M5','M6','M8','M9','M10','M11']:
            raise ValueError('We donnot know this model!')
        wave_dir=self.wave_path+popmodel+'/'
        if popmodel=='M11':
            yearMax=4
        else:
            yearMax=10
        if Tobs>=10:
            raise ValueError('Tobs too long!')
        Tobs_int=int(Tobs)
        Tobs_frac=Tobs-Tobs_int
        arr=np.arange(yearMax)+100
        np.random.shuffle(arr)
        cata_dir=self.cat_path+popmodel+'/'
        cat_names=sorted(os.listdir(cata_dir))
        #print(cat_names)
        data_cube=np.empty(shape=(0,20))
        totnum=0
        for i in range(Tobs_int):
            filename=str(arr[i])+'.wv'
            with open(wave_dir+filename, "r") as read_file:
                data=json.load(read_file)
            cat_filename=cat_names[arr[i]-100] # find the corresponding catalogue file
            #print(cat_filename, filename)
            events=np.loadtxt(cata_dir+cat_filename,skiprows=1)
            for j in range(len(data)):
                Xf=data[j]['Xf']
                f0,f1=data[j]['f0f1']
                freq=np.arange(f0, len(Xf)*(f1-f0), f1-f0)
                x0=2*np.pi*freq*8.3391
                x1=2*np.pi*freq*det.lisaLT
                Xf=Xf/(x0*np.sin(x0))*x1*np.sin(x1)
                Sx=det.Stdix(freq)
                SNR=np.sqrt(np.trapz(Xf**2/Sx, freq))
                if SNR>=rhostar:
                    to_stack=np.append(events[j],[SNR])
                    data_cube=np.vstack((data_cube,to_stack))
                    totnum+=1
                else:
                    pass
        # dealing with Tobs fraction
        filename=str(arr[Tobs_int])+'.wv'
        with open(wave_dir+filename, "r") as read_file:
            data=json.load(read_file)
        cat_filename=cat_names[arr[Tobs_int]-100] # find the corresponding catalogue file
        events=np.loadtxt(cata_dir+cat_filename, skiprows=1)
        for j in range(int(Tobs_frac*len(data))):
            Xf=data[j]['Xf']
            f0,f1=data[j]['f0f1']
            freq=np.arange(f0, len(Xf)*(f1-f0), f1-f0)
            x0=2*np.pi*freq*8.3391
            x1=2*np.pi*freq*det.lisaLT
            Xf=Xf/(x0*np.sin(x0))*x1*np.sin(x1)
            Sx=det.Stdix(freq)
            SNR=np.sqrt(np.trapz(Xf**2/Sx, freq))
            if SNR>=rhostar:
                to_stack=np.append(events[j],[SNR])
                data_cube=np.vstack((data_cube,to_stack))
                totnum+=1
            else:
                pass
        return [data_cube[:Nmax], totnum]


    def givelist_old(self, Tobs, det, rhostar, Nmax, fast=True):
        """
            @param: Tobs in unit of year!
            Return
                observed is an array of param in Strange Format
        """
        popmodel=self.pop_model
        if popmodel not in ['M1','M2','M3','M4','M5','M6','M8','M9','M10','M11']:
            raise ValueError('We donnot know this model!')
        cata_dir=self.cat_path+popmodel+'/'
        #print(os.listdir(cata_dir))
        file_names=os.listdir(cata_dir)
        random.shuffle(file_names)
        if Tobs>=len(file_names):
            raise ValueError('Tobs too long!')
        Tobs_int=int(Tobs)
        Tobs_frac=Tobs-Tobs_int
        data_cube=np.empty(shape=(0,19))
        for i in range(Tobs_int):
            data_cube=np.vstack((data_cube,np.loadtxt(cata_dir+file_names[i],skiprows=1)))
        data_spare=np.loadtxt(cata_dir+file_names[Tobs_int],skiprows=1)
        lines_num=int(len(data_spare)*Tobs_frac)
        #print("debugging", lines_num)
        data_cube=np.vstack((data_cube,data_spare[:lines_num:]))
        #SNRs=[self.SNR(pars_cat, det, Tobs, fast) for pars_cat in data_cube]
        SNRs=self.giveSNRlist_multi(data_cube, det, Tobs, fast)
        observed=data_cube[np.array(SNRs)>rhostar]
        observed_cat=np.column_stack((observed,np.array(SNRs)[np.array(SNRs)>rhostar]))
        totnum=len(observed)
        Nshow=min(totnum, Nmax)
        #if Nmax<totnum:
        #    observed_cat=observed_cat[:Nmax]
        return [observed_cat[:Nshow],totnum]
    def giveSNRlist_multi(self, manypars, det, Tobs, fast):
        n_cores=3
        N=len(manypars)
        n_unit=int(N/n_cores)+1 # lengh of each processing
        settings=[]
        for i in range(n_cores):
            settings.append([manypars[i*n_unit : min((i+1)*n_unit,N)], det, Tobs, fast])
        with Pool(n_cores) as pool3:
            result_multi=pool3.map(self.listSNR_unit, settings)
        results=np.concatenate(result_multi, axis=0)
        return results

    def listSNR_unit(self, setting):
        pars, det, Tobs, fast= setting
        SNRlist_unit=[self.SNR(par, det, Tobs, fast) for par in pars]
        return SNRlist_unit


    def givelisterrors(self, Tobs, det, rhostar, Nmax, fast):
        data_final, tot=self.givelist(Tobs, det, rhostar, Nmax, fast)
        N=len(data_final)
        ErrorCube=self.giveErrorCube_multi(data_final, N, det, Tobs, fast)
        #ErrorCube=np.empty((N,4))
        #ErrorCube=np.array([self.errors(data_final[i,:], det, Tobs, fast) for i in range(N)])
        #for i in range(N):
        #    pars_cat=data_final[i,:]
        #    ErrorCube[i,:]=self.errors(pars_cat,det, Tobs)
        datalist=self.translate(data_final)
        result=np.concatenate((datalist, ErrorCube), axis=1)
        return [result, tot]

    def giveErrorCube_multi(self, data, N, det, Tobs, fast):
        n_cores=3
        settings=[]
        if N<=3:
            n_cores=1
        n_unit=int(N/n_cores)+1
        for i in range(n_cores):
            settings.append([data[i*n_unit:min((i+1)*n_unit,N)], det, Tobs, fast])
        with Pool(n_cores) as pool4:
            results_multi=pool4.map(self.giveErrorCube_unit, settings)
        results=np.concatenate(results_multi, axis=0)
        return results
    def giveErrorCube_unit(self, setting):
        data, det, Tobs, fast = setting
        results=[self.errors(datum, det, Tobs, fast) for datum in data]
        return results

    def translate(self, data_final):
        """
        Translate the 'Strange format' into format that makes sense
        """
        #Names=['mu','M','s','e','iota','gamma','psi','theta_S','phi_S','theta_K','phi_K','alpha','D','SNR']
        parcube=np.empty(data_final.shape)
        parcube[:,0]=np.exp(data_final[:,1]+12.22096) # intrinsic
        parcube[:,1]=np.exp(data_final[:,2]+12.22096)
        parcube[:,2]= data_final[:,11] # s
        parcube[:,3]= data_final[:,3] # e
        parcube[:,4]= np.arccos(data_final[:,9]) # iota
        parcube[:,5]= data_final[:,5] # gamma
        #np.random.seed(0)
        parcube[:,6]= np.random.uniform(low=0,high=6.283,size=len(parcube)) # psi
        parcube[:,7]= np.arccos(data_final[:,7]) # theta_S
        parcube[:,8]= data_final[:,8] # phi_S
        parcube[:,9]= np.arccos(data_final[:,12]) # theta_K
        parcube[:,10]= data_final[:,13] # phi_K
        parcube[:,11]= data_final[:,10] # alpha
        parcube[:,12]=parcube[:,0]*4.7869e-23/data_final[:,14] # D
        parcube[:,13]=data_final[:,19] # SNR caculated by me
        parcube[:,14]=data_final[:,18] # published SNR_tot
        return parcube[:,:15]

    def givedataframe(self,Tobs, det, rhostar, Nmax, fast):
        #print("In EMRI.givedataframe")
        data_final, tot=self.givelist(Tobs, det, rhostar, Nmax, fast)
        catalogue=self.translate(data_final)
        #kys=['mu','M','s','e','iota','gamma','psi','theta_S','phi_S','theta_K','phi_K','alpha','D','SNR']
        kys=['μ','M','s','e','ι','γ','ψ','θ_S','φ_S','θ_K','φ_K','α','D','SNR','snr_pri']
        df=pd.DataFrame(data=catalogue, columns=kys)
        return [df, tot]

    def giveerrordataframe(self, Tobs, det, rhostar, Nmax, fast):
        catalogue, tot=self.givelisterrors(Tobs, det, rhostar, Nmax, fast)
        #kys=['mu','M','s','e','iota','gamma','psi','theta_S','phi_S','theta_K','phi_K','alpha','D','SNR','dmu','dM','dD','dOmega']
        kys=['μ','M','s','e','ι','γ','ψ','θ_S','φ_S','θ_K','φ_K','α','D','SNR','snr_pri','dμ','dΜ','dD','dΩ']
        df=pd.DataFrame(data=catalogue, columns=kys)
        return [df, tot]
