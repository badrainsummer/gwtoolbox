from gwtoolbox.cosmology import Cosmology
from gwtoolbox.functions_pta import *
from gwtoolbox.pta_pulsars import *
def set_cosmology(cosmoID=None, H0=None, Om0=None,Tcmb=None):
    cosmos_class = Cosmology()
    cosmos = cosmos_class.set_cosmo(cosmoID,H0,Om0,Tcmb)
    return cosmos
class PTA_individual:
    def __init__(self, obs_plan="A", delta_t=1, T=10, Ndot=10, which_PTA='IPTA'):
        self.PTA=newPulsars(meal=obs_plan, deltat=int(delta_t), T=int(T), Ndot=int(Ndot), which=which_PTA)
        return None 
    def SNR(self, ra,dec,hs,frequency,phi, iota):
        """
            ra is in format of hms (str), single, it's the RA of the GW source
            dec in format of dms (str) 
            frequency in unit of yr^-1, it's the frequency of the source
            hs is demensionless, it's the dimensionless 
        """
        if not hs>0:
            raise ValueError("hs should >0")
        #if not frequency:
        #    raise ValueError("frequency should >0")
        #res_ampt=resi_ampt(hs, frequency/365., ra, dec) # res_ampt wants frequency in days^-1
        result=SNR_indi_PTA(self.PTA, frequency, hs, ra, dec, phi, iota) 
        return result
    
    def sens_curve_directional(self,ra, dec, psi, iota, freq, rhostar):
        if not rhostar>0:
            raise ValueError("rhostar should be larger than 0")
        hs_fidu=1e-15
        snr_fidu=self.SNR(ra=ra, dec=dec, hs=hs_fidu, frequency=freq, phi=psi, iota=iota)
        hs_sen=np.where(snr_fidu==0,1e-8, np.sqrt(rhostar/snr_fidu)*hs_fidu)
        #if snr_fidu==0:
        #    hs_sen=1e-8
        #else:
        #    hs_sen=np.sqrt(rhostar/snr_fidu)*hs_fidu
        return hs_sen
    def sens_curve_skyave(self, rhostar):
        if not rhostar>0:
            raise ValueError("rhostar should be larger than 0")     
        Skypositions=100
        FreqBins=20
        theta_s, phi_s=give_uniform_sphere(Skypositions)
        iota_s, psi_s=give_uniform_sphere(Skypositions)
        iota_s=np.pi/2.-iota_s # 90 degree convert to iota=0 degree
        Beta_s=Angle(theta_s,unit=u.radian).to_string(unit=u.degree,sep=':') # ecliptic latitude
        Lambda_s=Angle(phi_s, unit=u.radian).to_string(unit=u.hour,sep=':') # eclipitc longitude
        Freqs=np.logspace(-2,3,20) # in year^-1
        sens_cube=np.array([self.sens_curve_directional(ra=Lambda_s[i], dec=Beta_s[i], psi=psi_s[i], iota=iota_s[i], freq=Freqs, rhostar=1) for i in range(Skypositions)])
        #sens_cube=np.array([[self.sens_curve_directional(ra=Lambda_s[i], dec=Beta_s[i], psi=psi_s[i], iota=iota_s[i], freq=Freqs[j], rhostar=1) for i in range(Skypositions)] for j in range(FreqBins)]) 
        sens_ave=np.mean(sens_cube, axis=0)
        return [rhostar*sens_ave, Freqs*3.1689e-8]

class PTA_SGWB:
    # main reference Anholm, Melissa et al. (2009), downloaded! 
    def __init__(self, cosmos, obs_plan="A", delta_t=1, T=10, Ndot=10, which_PTA='IPTA', which_SGWB='SBHBH', index=2./3.):
        self.PTA=newPulsars(meal=obs_plan, deltat=delta_t, T=T, Ndot=Ndot, which=which_PTA)
        self.cosmos=cosmos
        if which_SGWB=='SBHBH': # incoherent overlapping of SMBHBs
            self.SGWBIndex=2./3. # this is the index for hs, not for Omega!
        elif which_SGWB=='CS': # cosmic string
            self.SGWBIndex=7./6.
        elif which_SGWB=='primordial': # primordial relic
            self.SGWBIndex=1.
        elif which_SGWB=='selfdefine': 
            self.SGWBIndex=index
        #print(self.cosmos.H(0).value)
        else:
            raise ValueError("don't know that SGWB source")
        return None
    def Omega(self, norm, fs):
        """
        norm is the Omega SGBW in the frequency of 1/year
        fs is the arraylike in unit of day^-1
        """
        index_for_omega=2.*self.SGWBIndex-2.
        f_year=1./365. # convert frequency from day^-1 to year^-1
        density_ratio=norm*(fs/f_year)**-index_for_omega
        return density_ratio
    
    def SNR(self, norm):
        result=SNR_PTA(self.PTA, lambda fs: self.Omega(norm,fs), self.cosmos.H(0).value)
        return result
   
    def UpperLimit(self, rho_cri): # the upperlimit of return h^2*Omega_year
        if not rho_cri:
            raise ValueError("rho_cri should >0")
        Omega_ref=1e-10 # the reference Omega_year
        SNR_ref=self.SNR(Omega_ref)
        uplmt_omega=rho_cri/SNR_ref*Omega_ref
        h=self.cosmos.H(0).value/100.
        uplmt=uplmt_omega*h**2 
        return uplmt
    def Upper_A(self, rho_cri):
        Upper_Omega=self.UpperLimit(rho_cri)
        result=np.sqrt(Upper_Omega*100**2*3./(2.*np.pi**2)*(1.02e-12)**2)
        return result
         
