"""
classes and functions copied from MLDC codes
"""
import os, random
import sys
#path='../gwtoolbox/gwtoolbox'
sys.path.insert(0,'/Library/WebServer/gwtoolbox-website/gwtoolbox/gwtoolbox/MLDC-master/Packages/common/')
sys.path.insert(0,'/usr/local/lib/python3.6/site-packages/')
#sys.path.append(path+'/MLDC-master/software/Waveforms/MBH_IMR/IMRPhenomD/')
#sys.path.insert(0, '../gwtoolbox/gwtoolbox/')
import numpy as np
from astropy.constants import c
from .parameters import *
#from time import time
#import LISAConstants as LC
from GenerateFD_SignalTDIs import *
#from .sources_mHz import SMBHB
#def listSNR_unit(ps, det):
#        return [SMBHB.SNR(p,det) for p in ps]
def Str(a):
    if type(a)==bytes:
        return a.decode('utf8')
    if type(a)!=str:
        return str(a)
    return a

def dDovdz(z,cosmos):
    D=cosmos.luminosity_distance(z).value
    dz=1e-10*np.ones(len(z))
    D_right=cosmos.luminosity_distance(z+dz).value
    D_left=cosmos.luminosity_distance(z-dz).value
    return (D_right-D_left)/dz/2.

class ParsUnits():
    """
    This class defines the small object to manage parameters and units.
    """

    def __init__(self, pars_i=None, units_i=None, name='', value=0., unit=''):
        """
        Initalize `ParsUnits`
        @param pars_i is an optional dictionary of parameters
        @param units_i is an optional dictionary of units (same size as pars_i)
        @param name is an optional name
        @param value is an optional value
        @param unit is an optional unit
        """
        self.pars = {}
        self.units = {}
        if pars_i is not None:
            self.addDict(pars_i,units_i)
        elif name!='':
            self.addPar(name,value,unit)
    def copy(self, p):
        self.pars=p.pars.copy()
        self.units=p.units.copy()
        pass
    def __del__(self):
        # so far is empty, see what we need to add here
        pass

    def display(self,ReturnStr=False):
        """
        Display all parameters
        @param ReturnStr is true to return string instead of display
        """
        r = ""
        for i,k in enumerate(self.pars):
            r = r + "\t"+str(k)+" "
            if type(self.pars[k])==str:
                r = r + self.pars[k]
            else:
                r = r + str(self.pars[k])
            r = r + " ["+Str(self.units[k])+"]\n"
        if ReturnStr:
            return r
        else:
            print(r)

    def addPar(self,name,value,unit):
        """
        Add parameter, its value and unit
        @param name is the name of the parameter
        @param value is the value of the parameter
        @param unit is the unit of the parameter
        """
        self.pars.update({name : value})
        self.units.update({name : unit})

    def addDict(self,pars_i,units_i):
        """
        Add dictionnary and its unit
        @param pars_i is a dictionary of parameters
        @param units_i is a dictionary of units(same size as pars_i)
        """
        if len(pars_i)==len(units_i):
            self.pars.update(pars_i)
            self.units.update(units_i)
        else:
            raise Exception('addDict : parameters and units should have the same number of elements.')

    def get(self,parName):
        """
        Get parameter value
        @param parName parameter name
        @return value
        """
        ### TODO : Use proper error system
        if parName not in self.pars :
            print("WARNING: ParsUnits.get :",parName,"is not a parameter name (",self.pars,")")
            return None
        else:
            return self.pars[parName]

    def getConvert(self,parName,conversion,requiredUnit):
        """
        Get parameter value for distance parameters
        @param parName is the parameter name
        @param conversion is the conversion dictionary:
            + LC.convT for mass, time and distance (everything in time)
            + LC.convMass for mass only
            + LC.convTime for time only
            + LC.convDistance for distance only
        @param requiredUnit is the required unit [default:s]
        @return value
        """
        ### TODO : Use proper error system
        v = self.get(parName)
        if type(v)!=type(np.zeros(10)) and v == None:
            return None
        else:
            uV = self.units[parName]
            uV = uV.lower()
            requiredUnit = requiredUnit.lower()
            if requiredUnit not in conversion:
                print("WARNING: ParsUnits.getConvert : parameter unit",requiredUnit,"is not in the conversion list (",conversion,")")
                return None
            if uV not in conversion:
                print("WARNING: ParsUnits.getConvert : required unit",uV,"is not in the conversion list (",conversion,")")
                return None
            return v * ( conversion[uV] / conversion[requiredUnit] )

def readfromcat(cosmos=None, SourceType='MBHB', Approximant='IMRPhenomD', Cadence=10, duration=1, file_name=None, n=0):
    """
read event from catalgoue, and return a p object
@cosmos: an astropy.cosmo object
@param file_name: filename of catalogue
@param n: number of event reading from the catalogue file
@param duration: observation duration in unit of year, defalt is 1 year.
return a p object, with 20 parameters in it.
    """
#    masscut=1e4
    Cat={}
    data_cube=np.loadtxt(file_name, dtype=float, max_rows=n)
#    print("hello")
#    print('filename',file_name)
    zs=data_cube[:,0]
    Cat['SourceType']=SourceType       # 1
    Cat['Approximant']=Approximant     # 2
    Cat['Cadence']=Cadence             # 3
    Cat['Redshift']=zs                 # 4
    #mass1=(1.0+zs)*data_cube[:,1] # read in intrinsic masses, convert to red-shifted: may be need not # yishuxu: 28 Apr
    #mass2=(1.0+zs)*data_cube[:,2]
    mass1=data_cube[:,1]*(1.0+zs)
    mass2=data_cube[:,2]*(1.0+zs)
    #indices= np.array(mass1)>1e4 and np.array(mass2)>1e4
    #Cat['Mass1']=mass1[(mass1>masscut)*(mass2>masscut)]                 # 5
    Cat['Mass1']=mass1
    Cat['Mass2']=mass2 #[(mass1>masscut)*(mass2>masscut)]                 # 6
    spin1=data_cube[:,3]
    Cat['Spin1']=spin1 #[(mass1>masscut)*(mass2>masscut)]                 # 7
    spin2=data_cube[:,4]
    Cat['Spin2']=spin2 #[(mass1>masscut)*(mass2>masscut)]                 # 8
    PolarAngleOfSpin1=data_cube[:,5]
    PolarAngleOfSpin2=data_cube[:,6]
    Cat['PolarAngleOfSpin1']=PolarAngleOfSpin1 #[(mass1>masscut)*(mass2>masscut)] # 9
    Cat['PolarAngleOfSpin2']=PolarAngleOfSpin2 #[(mass1>masscut)*(mass2>masscut)] #10
    Sphi12s=data_cube[:,7]
    CoalescenceTime=np.random.uniform(low=0,high=duration,size=len(mass1))*31556926. # want seconds
    Cat['CoalescenceTime']= CoalescenceTime*0.5 #[(mass1>masscut)*(mass2>masscut)] #11
    EclipticLatitude=0.5*np.pi - data_cube[:, 9]
    Cat['EclipticLatitude']=EclipticLatitude #[(mass1>masscut)*(mass2>masscut)] #12
    EclipticLongitude=data_cube[:,10]
    Cat['EclipticLongitude']=EclipticLongitude #[(mass1>masscut)*(mass2>masscut)] #13
    InitialPolarAngleL=data_cube[:,11]
    Cat['InitialPolarAngleL']=InitialPolarAngleL #[(mass1>masscut)*(mass2>masscut)] #14
    ObservationDuration=CoalescenceTime
    Cat['ObservationDuration']=ObservationDuration #[(mass1>masscut)*(mass2>masscut)] #15
    AzimuthalAngleOfSpin1, AzimuthalAngleOfSpin2, InitialAzimuthalAngleL, PhaseAtCoalescence =  np.random.uniform(low=0.0, high=2.*np.pi, size=(4,len(mass1)))
    Cat['AzimuthalAngleOfSpin1']=AzimuthalAngleOfSpin1 #[(mass1>masscut)*(mass2>masscut)] # 16
    Cat['AzimuthalAngleOfSpin2']=AzimuthalAngleOfSpin2 #[(mass1>masscut)*(mass2>masscut)] # 17
    Cat['InitialAzimuthalAngleL']=InitialAzimuthalAngleL #[(mass1>masscut)*(mass2>masscut)] #18
    Cat['PhaseAtCoalescence']=PhaseAtCoalescence #[(mass1>masscut)*(mass2>masscut)]  #19
    Cat['Distance']=[cosmos.luminosity_distance(z).value/1e3 for z in zs] #[(mass1>masscut)*(mass2>masscut)]]
    ps=[]

    for i in range(0,min(n,len(data_cube))): # [(mass1>masscut)*(mass2>masscut)]))):
       # ParsValues=MBHBunits
        ParsValues={}
        for kys in MBHBunits.keys():
            if type(Cat[kys])==np.ndarray or type(Cat[kys])==list:
                ParsValues[kys]=Cat[kys][i]
            else:
                #print(i)
                ParsValues[kys]=Cat[kys]
        p=ParsUnits(pars_i=ParsValues, units_i=MBHBunits)
        ps.append(p)
    return ps


def lisasn(f, lisaLT=16.6782, lisaD=0.4, lisaP=1.0):
    """
    Return the analytic approximation of LISA noise curve, from A. Klein et al. (2016).
    """
    defaultLT = 16.6782
    defaultD = 0.4
    defaultP = 2.0
    Sacc=9e-30/(2.*3.1415926*f)**4*(1+1e-4/f) # unit m^2/Hz for N2
    Ssn=2.96e-23*(lisaLT/defaultLT)**2*(defaultD/lisaD)**2*(defaultP/lisaP) # unit m^2/Hz for A5
    Somn=2.65e-23 # unit m^2/Hz for all conf
    L=c.value*lisaLT
    result=20./3.*(4.*Sacc+Ssn+Somn)/L**2*(1+(f/0.41*2.*lisaLT)**2)
    Agal=3.266e-44
    alpha=1.183
    s1=1.426e-3
    s2=4.835e-3
    f0=2.412e-3
    Sgal=Agal*f**(-7./3.)*np.exp(-(f/s1)**alpha)*0.5*(1+np.tanh(-(f-f0)/s2))
    #print(Sgal,result)
    #print(len(Sgal),len(result))
    result=result+Sgal
    return result

def lisaStdiX(f, lisaLT=8.3391, lisaD=0.3, lisaP=2.0,Sacc0=3.9e-44, Sopo0=2.81e-38, Sops0=5.3e-38, Tobs=1):
    """
    Return the analytic approximation of LISA TDI X noise spectrum.
    """

    #Sn=lisasn(f, lisaLT=lisaLT, lisaD=lisaD, lisaP=lisaP)
    #x = 2.0*np.pi*lisaLT*f
    #Sx=4.*np.sin(x)**2*Sn
    #Sx=noisepsd_X(f, model='SciRDv1', includewd=None,lisaLT=16.6782, lisaD=0.4, lisaP=1.0)
    #return Sx
    defaultD=0.3
    defaultP=2.0
    defaultL=8.3391
    Sacc=Sacc0*(1+(4e-4/f)**2)*((8e-3/f)**2+(f/8e-3)**2)
    #Sacc0*(1+(f/3e-3)**2)**2*(f/1e-4)**(-8./3.) # fractional frequency in Hz^-1
    #Sacc=Sacc0*(1+1e-4/f)/(f/1e-4)**2
    Sopo=Sopo0*f**2 # fractional frequency in Hz^-1
    Sops=Sops0*f**2*(lisaLT/defaultL)**2 * (defaultD/lisaD)**2*(defaultP/lisaP)
    # the D dependence should be 2, rather than 4. But in MLDC's code, common/tdi.py:lisanoises, it writes 4, I don't know why
    x=2*np.pi*f*lisaLT
    Sopt=Sopo+Sops
    SX=(4*np.sin(2*x)**2+32*np.sin(x)**2)*Sacc+16*np.sin(x)**2*Sopt
    return SX+Sgdw_X(f,lisaLT, Tobs)

def Sgdw_X(f,lisaLT, Tobs):
    x=2*np.pi*f*lisaLT
    #result=20/3.*(np.piecewise(f,(f >= 1.0e-5  ) & (f < 1.0e-3  ),[lambda f: 10**-44.62 * f**-2.3, 0]) + \
    #                 np.piecewise(f,(f >= 1.0e-3  ) & (f < 10**-2.7),[lambda f: 10**-50.92 * f**-4.4, 0]) + \
    #                 np.piecewise(f,(f >= 10**-2.7) & (f < 10**-2.4),[lambda f: 10**-62.8  * f**-8.8, 0]) + \
    #                 np.piecewise(f,(f >= 10**-2.4) & (f < 10**-2.0),[lambda f: 10**-89.68 * f**-20.0,0]))
    # above is relative displacement squared.
    #return result*lisaLT**2*4*np.pi**2*f**2*16*np.sin(x)**2 # convert to relative frequency in Hz^-1
    return lisaLT**2*4*np.pi**2*f**2*16*np.sin(x)**2*Sgwd(f, Tobs)
def lisanoises(f, model="SciRDv1",unit='relativeFrequency',lisaLT=16.6782, lisaD=0.4, lisaP=1.0):
    """
    Return the analytic approximation of the two components of LISA noise,
    i.e. the acceleration and the
    @param f is the frequency array
    @param model is the noise model:
        * 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
    @param unit is the unit of the output: 'relativeFrequency' or 'displacement'
    @param lisaLT is the arm length divided by the speed of light in second
    @param lisaD is the diameter of the telescope in metre
    @param lisaP is the laser power in Watt
    """
    if model=='newdrs':
        #Spm = 6.00314e-48 * f**(-2)              #this number need to be check                   # 4.6e-15 m/s^2/sqrt(Hz)
        #Spm= 4.6**2*1e-30*f**-2
        Spm= 6.00314e-54*f**(-4)
        defaultL = 16.6782
        defaultD = 0.4
        defaultP = 1.0
        Sops = 6.15e-38 * (lisaLT/defaultL)**2 * (defaultD/lisaD)**4 * (defaultP/lisaP)      # 11.83 pm/sqrt(Hz)
        Sopo = 2.81e-38                                                                                                         # 8 pm/sqrt(Hz)
        Sop = (Sops + Sopo) * f**2
    elif model=='SciRDv1' or model=='MRDv1':
        frq = f
        ### Acceleration noise
        ## In acceleration
        LPSa_a={'Proposal':(3.e-15)**2, 'SciRDv1': (3.e-15)**2, 'MRDv1': (2.4e-15)**2}
        Sa_a = LPSa_a[model] *(1.0 +(0.4e-3/frq)**2)*(1.0+(frq/8e-3)**4)
        ## In displacement
        Sa_d = Sa_a*(2.*np.pi*frq)**(-4.)
        ## In relative frequency unit
        Sa_nu = Sa_d*(2.0*np.pi*frq/c.value)**2
        Spm =  Sa_nu

        ### Optical Metrology System
        ## In displacement
        LPSoms_d={'Proposal':(10.e-12)**2, 'SciRDv1': (15.e-12)**2, 'MRDv1': (10.e-12)**2}
        Soms_d = LPSoms_d[model] * (1. + (2.e-3/f)**4)
        ## In relative frequency unit
        Soms_nu=Soms_d*(2.0*np.pi*frq/c.value)**2
        Sop =  Soms_nu
    else:
        raise NotImplementedError(model)
    if unit=='displacement':
        return Sa_d, Soms_d
    elif unit=='relativeFrequency':
        return Spm, Sop
    else:
        raise NotImplementedError(unit)

def lisa_michealson_noise(f, lisaLT=8.3391, lisaD=0.3, lisaP=2.0,Sacc0=3.9e-44, Sopo0=2.81e-38, Sops0=5.3e-38, Tobs=1):
    Larm=lisaLT*c.value
    x = 2.0*np.pi*lisaLT*f
    dis_to_y=4.0*np.pi**2*f**2/c.value**2 # convert factor from displacement to Doppler (fractional frequency shift)
    y_to_a=4.0*np.pi**2*c.value**2*f**2 # convert from Doppler to acceleration
    defaultD=0.3
    defaultP=2.0
    defaultL=8.3391
    Sacc=Sacc0*(1+(4e-4/f)**2)*((8e-3/f)**2+(f/8e-3)**2)
    #Sacc0*(1+(f/3e-3)**2)**2*(f/1e-4)**(-8./3.) # fractional frequency in Hz^-1
    #Sacc=Sacc0*(1+1e-4/f)/(f/1e-4)**2
    Sopo=Sopo0*f**2 # fractional frequency in Hz^-1
    Sops=Sops0*f**2*(lisaLT/defaultL)**2 * (defaultD/lisaD)**2*(defaultP/lisaP)
    Sop_dis=(Sopo+Sops)/dis_to_y # convert from Doppler to displacement
    Sacc_a=Sacc*y_to_a # convert from Dopper to acceleration
    Sn=10./(3.*Larm**2)*(Sop_dis+2.*(1.+np.cos(x)**2)*Sacc_a/(2.*np.pi*f)**4)*(1.+0.6*x**2)
    Sn+=Sgwd(f, Tobs)
    return Sn

def Sgwd(f, Tobs):
    if Tobs<=0.5:
        A, alpha, beta, kappa, gamma, fk = [9e-45, 0.133, 243., 482., 917., 0.00258]
    elif Tobs<=1:
        A, alpha, beta, kappa, gamma, fk = [9e-45, 0.171, 292., 1020., 1680., 0.00215]
    elif Tobs<=2:
        A, alpha, beta, kappa, gamma, fk = [9e-45, 0.165, 299., 611., 1340., 0.00173]
    elif Tobs<=4:
        A, alpha, beta, kappa, gamma, fk = [9e-45, 0.138, -221., 521., 1680., 0.00113]
    else:
        A, alpha, beta, kappa, gamma, fk = [0, 0.138, -221., 521., 1680., 0.00113]
    result=A*f**(-7./3.)*np.exp(-f**alpha+beta*f*np.sin(kappa*f))*(1.+np.tanh(gamma*(fk-f)))
    return result

def noisepsd_X(f, model='SciRDv1', includewd=None,lisaLT=16.6782, lisaD=0.4, lisaP=1.0):
    """
Compute and return analytic PSD of noise for TDI X
@param f: np array of frequencies
@param model: same as in function lisanoises.
@param includewd whether to include  GB confusion, if yes should give a duration of observations in years.
example: includewd=2.3 - 2.3 yeras of observations
if includewd == None: includewd = model.lisaWD
    """
    x = 2.0*np.pi*lisaLT*f
    Spm, Sop = lisanoises(f=f, model=model, lisaLT=lisaLT, lisaD=lisaD, lisaP=lisaP)
    Sx = 16.0 * np.sin(x)**2 * (2.0 * (1.0 + np.cos(x)**2) * Spm + Sop)
    if includewd != None:
        Sx += WDconfusionX(f, includewd, model=model)
    return Sx

def WDconfusionX(f, includewd, model):
    """
    not defined yet!
    """
    return 0

def EventsUniverse(cosmos=None, duration=1, sourceType='MBHB', model=None):
    # the duration needs to be rescaled, be cause the numbers in the catalogues seems 5 times larger than needed
    duration=duration/5.
    if duration<=0.1:
        raise ValueError('duration too short (>=0.1 years)!')
    elif duration>=10:
        raise ValueError('duration too long (<=10 years)!')
    if sourceType=='MBHB':
        path_sourcetype=os.path.join(os.path.dirname(os.path.abspath(__file__)),'catalogues_mHz','MBHs/')
    else: return None
    path_catalogue=path_sourcetype+model+'/'
    all_names=os.listdir(path_catalogue)
    years_int=int(duration) # duration in unit of years, integer parts
    years_fraction=duration-years_int
    name_selected=random.sample(all_names, k=years_int+1)
    ps=[]
    cadence=10
    for file_ in name_selected[1:]:
        ps=ps+readfromcat(cosmos=cosmos, SourceType=sourceType, Approximant='IMRPhenomD', Cadence=cadence, duration=duration*5, file_name=path_catalogue+file_, n=1000)
    file_=name_selected[0]
    data_cube=np.loadtxt(path_catalogue+file_)
    num=int(years_fraction*len(data_cube))+2
    #print(num)
    ps=ps+readfromcat(cosmos=cosmos, SourceType=sourceType, Approximant='IMRPhenomD', Cadence=cadence, duration=duration*5, file_name=path_catalogue+file_, n=num)
    return ps

def leapfrog_debug(p,key,step,step_dist,Zs, Ds):
    """
    @p is a dict of GW parameters,
    @key is a string, indicating the name of the parameter that you want to leap back and forth.
    @step: double, this is the relative step length of parameters.
    @step_dist: is the step specially for the luminosity_distance parameter
    @Zs, Ds are used to interpolate
    return: two p, p_left and p_right, being p_left=p-step*p, p_right=p+step*p
    """
    if key=='Distance':
        step=step_dist
    else:
        pass
    cvalue=p.pars[key]
    p_left=ParsUnits()
    p_right=ParsUnits()
    p_left.copy(p)
    p_right.copy(p)
#    print(cvalue)
    left_value=cvalue*(1-step)
    right_value=cvalue*(1+step)
    p_left.pars.update({key : left_value})
    p_right.pars.update({key : right_value})
    if key=='Distance':
#        print("leftValue=%f" % left_value)
        z_left=np.interp(left_value,Ds,Zs)
        z_right=np.interp(right_value,Ds,Zs)
        p_left.pars.update({'Redshift' : z_left})
        p_right.pars.update({'Redshift': z_right})
    return p_left, p_right, cvalue*step

def leapfrog(p,key,step,Zs, Ds):
    """
    @p is a dict of GW parameters,
    @key is a string, indicating the name of the parameter that you want to leap back and forth.
    @step: double
    return: two p, p_left and p_right, being p_left=p-step, p_right=p+step
    """
    if key=='Distance':
        step=step
    else:
        pass
    cvalue=p.pars[key]
    p_left=ParsUnits()
    p_right=ParsUnits()
    p_left.copy(p)
    p_right.copy(p)
#    print(cvalue)
    left_value=cvalue*(1-step)
    right_value=cvalue*(1+step)
    p_left.pars.update({key : left_value})
    p_right.pars.update({key : right_value})
    if key=='Distance':
#        print("leftValue=%f" % left_value)
        z_left=np.interp(left_value,Ds,Zs)
        z_right=np.interp(right_value,Ds,Zs)
        p_left.pars.update({'Redshift' : z_left})
        p_right.pars.update({'Redshift': z_right})
    return p_left, p_right, cvalue*step

def intriamp(m1,m2,d,f):
    # convert all quantities in UI units!
    GMsolar=1.327e20
    c=3e8
    dkpc=3e19
    fmHz=1e-3
    mchirp=(m1*m2)**0.6/(m1+m2)**0.2
    ampl=2*(GMsolar*mchirp)**(5./3.)/(c**4*d*dkpc)*(np.pi*f*fmHz)**(2./3.)
    return ampl

from astropy.cosmology import Planck18  # or replace with your cosmology model
from scipy.optimize import root_scalar
import astropy.units as u

def redshift_from_luminosity_distance(DL, cosmos=Planck18, z_max=20):
    """
    Calculate redshift (z) given a luminosity distance (DL) using numerical root finding.

    Parameters
    ----------
    DL : astropy.units.Quantity
        Luminosity distance with units (e.g., Mpc).
    cosmo : astropy.cosmology.FLRW, optional
        Cosmological model to use for the calculation. Default is Planck18.
    z_max : float, optional
        Maximum redshift to search for the root. Default is 20.

    Returns
    -------
    float
        The redshift corresponding to the given luminosity distance.

    Raises
    ------
    RuntimeError
        If the numerical root finding fails to converge.
    ValueError
        If the input luminosity distance does not have proper units.
    """
    if not isinstance(DL, u.Quantity):
        raise ValueError("Input luminosity distance must be an astropy Quantity with units.")

    DL = DL.to(u.Mpc)  # Convert to Mpc for consistency

    # Define the function whose root corresponds to the correct redshift
    def func(z):
        return (cosmos.luminosity_distance(z) - DL).value

    # Use root_scalar to find the redshift where func(z) = 0
    sol = root_scalar(func, bracket=[1e-5, z_max], method='brentq')

    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("Root finding did not converge: cannot determine redshift for given luminosity distance.")

