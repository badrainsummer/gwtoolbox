import numpy as np
from itertools import combinations
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from .functions_earth import chirp_mass
def GetOne(samples, sty='num'):
    # draw one sample from the distribution underlying the given sample 
    two=np.random.choice(samples,2)
    if sty=='dms':
        # convert hms to arc
        first_str_dms=two[0]
        second_str_dms=two[1]
        first_angle=Angle(first_str_dms,unit=u.degree)
        second_angle=Angle(second_str_dms, unit=u.degree)
        first_str_degree=first_angle.to_string(unit=u.degree, decimal=True)
        second_str_degree=second_angle.to_string(unit=u.degree, decimal=True)
        first_num_degree=float(first_str_degree)
        second_num_degree=float(second_str_degree)
        #one_num_degree=0.5*(first_num_degree+second_num_degree)
        weight=np.random.uniform(0,1,size=None)
        one_num_degree=weight*first_num_degree+(1-weight)*second_num_degree
        one_angle=Angle(one_num_degree, unit=u.degree)
        one_str_dms=one_angle.to_string(unit=u.degree, sep=":", decimal=False)
        one=one_str_dms

        # convert arc back to hms
    elif sty=='hms': 
        first_str_hms=two[0]
        second_str_hms=two[1]
        first_angle=Angle(first_str_hms,unit=u.hourangle)
        second_angle=Angle(second_str_hms, unit=u.hourangle)
        first_str_degree=first_angle.to_string(unit=u.degree, decimal=True)
        second_str_degree=second_angle.to_string(unit=u.degree, decimal=True)
        first_num_degree=float(first_str_degree)
        second_num_degree=float(second_str_degree)
        weight=np.random.uniform(0,1,size=None)
        one_num_degree=weight*first_num_degree+(1-weight)*second_num_degree
        #one_num_degree=0.5*(first_num_degree+second_num_degree)
        one_angle=Angle(one_num_degree, unit=u.degree)
        one_str_hms=one_angle.to_string(unit=u.hourangle, sep=":")
        one=one_str_hms
    else:
        weight=np.random.uniform(0,1,size=None)
        one=weight*two[0]+(1-weight)*two[1]
    return one 

def SNR_pair(pair, Omega, H0):
    """
        pair is a pair of pulsars (two dicts). 
        Omega is a function of frequency (arraylike)
        H0 is the Hubble constant in unit of km/s/Mpc
    """
    psr1=pair[0]
    psr2=pair[1]
    Tyear_1=psr1["Tyear"]
    Tyear_2=psr2["Tyear"]
    deltat_1=psr1["deltat"]
    deltat_2=psr2["deltat"]
    Ared_1, Ared_2=[psr1["Ared"],psr2["Ared"]]
    gamma_1, gamma_2=[psr1["gamma"],psr2["gamma"]]
    flow_1, flow_2= [1./Tyear_1, 1./Tyear_2]
    sigw_1, sigw_2= [psr1["sigw"],psr2["sigw"]]
 #   f_low_1=1./Tyear_1
 #   f_low_2=1./Tyear_2
    fup_1=0.5/deltat_1
    fup_2=0.5/deltat_2
    fhigh_1, fhigh_2=[fup_1*365.2, fup_2*365.2]
    f_up=min(fup_1,fup_2)
    T=min(Tyear_1, Tyear_2)*365.2 # in unit of days
    f_low=1./T
    fs=np.linspace(f_low, f_up, int(f_up/f_low)) # frequency array in unit of 1/day
    # we should make all time in [day] and all frequency in [day^-1]
    Omegasquare_array=Omega(fs)**2 # Omega is dimensionless!
    rms_1_us=np.sqrt(sigw_1+Ared_1**2/12./np.pi**2/(1-gamma_1)*(fhigh_1**(1-gamma_1)-flow_1**(1-gamma_1))*1e27)
    rms_2_us=np.sqrt(sigw_2+Ared_2**2/12./np.pi**2/(1-gamma_2)*(fhigh_2**(1-gamma_2)-flow_2**(1-gamma_2))*1e27)
    rms_1_day=rms_1_us*1.157e-11 # us to day
    rms_2_day=rms_1_us*1.157e-11 # us to day
    #delta=max(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high**(1-gamma)-f_low**(1-gamma))*1e27,1e-4)
    #P1s=Noise_array(psr1,fs)/rms_1_day**2
    #P2s=Noise_array(psr2,fs)/rms_2_day**2
    P1s=Noise_array(psr1,fs*365.2)*fs**2
    P2s=Noise_array(psr2,fs*365.2)*fs**2
    #print(fs)
    integrans=Omegasquare_array/(fs**6*P1s*P2s)
    integral=np.trapz(integrans, fs)
    sqroot=np.sqrt(2.*integral)
    Gamma_0=Gammanode(pair) # dimensionless
    #H0_day=1.02269032e-12*H0 # convert from unit km/s/Mpc to 1/day
    H0_day=2.8e-15*H0
    #print(H0_day, np.sqrt(T), sqroot, Gamma_0)
    factor=H0_day**2*np.sqrt(T)/(4.*np.pi**2)*Gamma_0
    return factor*sqroot

def Gammanode(pair):
    psr1=pair[0]
    psr2=pair[1]
    RA_1=psr1["RA"] # str
#    print("RA_1=",RA_1)
    DEC_1=psr1["DEC"]
#    print("DEC_1=",DEC_1)
    RA_2=psr2["RA"]
#    print("RA_2=",RA_2)
    DEC_2=psr2["DEC"]
#    print("DEC_2=", DEC_2)
    coor_1=RA_1+" "+DEC_1
    coor_2=RA_2+" "+DEC_2
    c1=SkyCoord(coor_1, unit=(u.hourangle, u.degree))
    c2=SkyCoord(coor_2, unit=(u.hourangle, u.degree))
    separa=c1.separation(c2)
    xi=float(separa.to_string(unit=u.degree, decimal=True))
    result=3.*(1./3.+(1-np.cos(xi))/2.*(np.log(0.5*(1-np.cos(xi)))-1./6.)) 
    return result
        
def Noise_array(psr, farray):
    # red-in farray is in unit of year^-1
    sigw=psr["sigw"] # in unit [us]
    rednoiseindex=psr["gamma"] 
    rednor=psr["Ared"] # in unit of yr^3/2!!
    fhigh=0.5/psr["deltat"] # should keep it in unit of [day^-1]
    T_year=psr["Tyear"]
    flow=1./(psr["Tyear"]*365) # should keep it in unit of [day^-1]
    f_year=1./365. # 1 year^-1 to 1 day^-1  ysx: you should not covert it!, because the reference frequency is 1 yr^-1
    # convert everything's unit to day or day^-1
    rednor2=rednor**2*365**3 # from year**3 to day**3
    sigw_day=1.157e-11*sigw # from us to day
    factor=lambda f: np.where(T_year*f>1, 1, np.Inf)*np.where(f*f_year<fhigh, 1, np.Inf)
    if isinstance(farray, np.ndarray):
        results=np.ones(len(farray))*sigw_day**2/(fhigh-flow)*factor(farray)+rednor2/(12.*np.pi**2)*(farray)**-rednoiseindex    
    else:
        results=sigw_day**2/(fhigh-flow)*factor(farray)+rednor2/(12.*np.pi**2)*(farray)**-rednoiseindex
    return results # arraylike, unit day**3

def SNR_PTA(PTA, Omega, H0):
    """
        PTA a list of pulsars (dicts). 
        Omega is a function of frequency (arraylike)
        H0 is the Hubble constant in unit of km/s/Mpc
    """
    SNRsq_tot=0
    for pair in combinations(PTA,2):
        SNRsq_tot+=SNR_pair(list(pair), Omega, H0)**2
    return np.sqrt(SNRsq_tot)

def SNR_indi_single(psr, frequency, hs, RA_gw, DEC_gw, phi, iota):
    """
       This function is to calculate the SNR of individual GW source induced in the timing residual of a single pulsar
       psr is a single pulsar
       frequency is the fr of GW, in unit of yr^-1
       #res_ampt is the amplitude of the individual GW induced timing residuals, unit of day!
       hs is the amplitude of the gw, 
       RA_gw, DEC_gw are the coordinate of the source, it's str 
       phi is the phase of the gw source. in radian
       iota is the inclination angle of the GW source, in radian
    """
    T_obs=psr["Tyear"]*365 # convert to days
    RA_PSR=Angle(psr["RA"], unit=u.hourangle)
    DEC_PSR=Angle(psr["DEC"], unit=u.degree)
    RA_GW=Angle(RA_gw, unit=u.hourangle)
    DEC_GW=Angle(DEC_gw, unit=u.degree)
    Cord_PSR=SkyCoord(RA_PSR, DEC_PSR)
    Cord_GW=SkyCoord(RA_GW,DEC_GW)
    Ang_sep=Cord_PSR.separation(Cord_GW)
    #phi=np.random.uniform(0,2.*np.pi)
    theta=float(Ang_sep.to_string(unit=u.degree, decimal=True))
    w=2*np.pi*frequency/365.2 # convert it to day^-1
    sqr=np.sqrt(np.cos(2*phi)**2*((1.+np.cos(iota)**2)/2.)**2+np.sin(2*phi)**2*np.cos(iota)**2)
    res_ampt=hs/w*(1+np.cos(theta))*sqr
    N_f=Noise_array(psr, frequency)
    snrsq=res_ampt**2/N_f*T_obs
    result=np.sqrt(snrsq)
    return result

def SNR_indi_PTA(PTA, frequency, hs, RA_gw, DEC_gw,phi,iota):
    #SNRsq_tot=0
    SNRsqs=np.array([SNR_indi_single(psr, frequency, hs, RA_gw, DEC_gw,phi, iota)**2 for psr in PTA])
    #for psr in PTA:
    #    SNRsq_tot+=SNR_indi_single(psr, frequency, hs, RA_gw, DEC_gw,phi, iota)**2
    SNRsq_tot=np.sum(SNRsqs, axis=0)
    return np.sqrt(SNRsq_tot)

#def resi_ampt(hs, frequency, ra, dec):
#    """
#        ra is in format of hms (str), single
#        dec in format of dms (str)
#        frequency in unit of day^-1
#        hs is demensionless
#    """
#    w=2*np.pi*frequency
#    ra_angle=Angle(ra, unit=u.hourangle)
#    dec_angle=Angle(dec, unit=u.degree)
#    theta=float(ra_angle.to_string(unit=u.degree, decimal=True))
#    phi=float(dec_angle.to_string(unit=u.degree, decimal=True))
#    res_amp=hs/w*(1+np.cos(theta))*np.sin(2*phi)
#    return res_amp   # in unit of day!
      
def give_uniform_sphere(size):
    costheta=np.random.uniform(-1,1,size)
    phi=np.random.uniform(0,2*np.pi,size)
    phi_wrap=np.where(phi<=np.pi, phi, phi-2*np.pi)
    theta=np.pi/2.-np.arccos(costheta)
    return [theta, phi_wrap]


def strain_amplitude(m1, m2, Dl, fz):
    '''
    m1,m2: masses (red-shifted) of the binaries, in unit of soloar  mass
    Dl: Luminosity distance
    fz: red-shifted frequency of GW in Hz
    '''
    M_chirp=chirp_mass(m1,m2)   # in soloar mass
    GMsolar_c3=4.93e-6 # second
    c=3e10 # cm/s
    Mpc=3.086e24 # cm 
    hs=4.*np.sqrt(0.4)*(M_chirp*GMsolar_c3)**(5/3)*c/(Dl*Mpc)*(np.pi*fz)**(2./3.)    
    return hs 
