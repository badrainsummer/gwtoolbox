import numpy as np
from pkg_resources import resource_filename
#from MLDC_tools  import *
from .functions_space import *
class LisaLike:
    """
    This class describes LISA-like detectors
    
    Parameters:
      det (int): detector ID (3 - LISA, -3 - LISA-like)

    """
    def __init__(self, model='SciRDv1', lisaLT=8.3391,lisaD=0.3, lisaP=2.0,Sacc0=3.9e-44, Sopo0=2.81e-38, Sops0=5.3e-38, Tobs=1):
        """
        Parameters:
            model='newdrs' or '' or 'SciRDv1'
            lisaLT: light travel time in LISA's armlength (unit s)
            lisaD : Diameter of LISA's telescope (unit meter)
            lisaP : Power of LISA's laser (unit W)
        """
        #self.det = det
        #self.lisa_file = resource_filename('gwtoolbox','data_detectors/LISA.txt')
        #self.Sx = 
        self.fq=np.logspace(-5,0,50) # frequencies  
        self.Sn=lisasn(self.fq,lisaLT=lisaLT, lisaD=lisaD, lisaP=lisaP)
        self.Sx=lisaStdiX(self.fq,lisaLT=lisaLT, lisaD=lisaD, lisaP=lisaP, Sacc0=Sacc0, Sopo0=Sopo0, Sops0=Sops0)
        self.Sacc0=Sacc0
        self.Sopo0=Sopo0
        self.Sops0=Sops0 
        self.tf=1./(lisaLT*6.283)
        self.lisaLT=lisaLT
        self.model=model
        self.lisaD=lisaD
        self.lisaP=lisaP
        self.Tobs=Tobs  # in years
        #np.savetxt('arm.dat', [self.lisaLT],fmt='%.5f')
    
    def change_tobs(self, Tobs): # Shuxu Changed in Jul26
        self.Tobs=Tobs
        return None

    def ante_pattern(self, theta, varphi, psi, Nnest=2):
        """
        The Antenna Pattern of LISA-like Gravitational Waves Observatories.
        
        Parameters:
            theta  (float): the polar angle of the GW source in the detector coordinates frame
            varphi (float): the azimuth angle of the GW source in the detector coordiantes frame
            to see the frame defination, see figure in ref 46. in Klein et al.  
            psi (float): the polarization angle of the GW
            Nnest (int): The number of nested detectors

        
        Returns:
          (array of dtype float): antenna pattern
          see equations (6-12) of Klein et al. 2016 "PhysRevD. 93.024003"
        """
        
        plus1 = np.sqrt(3)/2.*(0.5*(1+np.cos(theta)**2)*np.cos(2.*varphi)*np.cos(2.*psi)-np.cos(theta)*np.sin(2.*varphi)*np.sin(2.*psi))
        psip=psi-np.pi/4.
        cross1= np.sqrt(3)/2.*(0.5*(1+np.cos(theta)**2)*np.cos(2.*varphi)*np.cos(2.*psip)-np.cos(theta)*np.sin(2.*varphi)*np.sin(2.*psip))
        plus_joint_sq = plus1**2
        cross_joint_sq = cross1**2
        
        if Nnest >=2:
            varphi2 = varphi-np.pi/4.
            plus2 = np.sqrt(3)/2.*(0.5*(1+np.cos(theta)**2)*np.cos(2.*varphi2)*np.cos(2.*psi)-np.cos(theta)*np.sin(2.*varphi2)*np.sin(2.*psi))
            
            cross2=np.sqrt(3)/2.*(0.5*(1+np.cos(theta)**2)*np.cos(2.*varphi2)*np.cos(2.*psip)-np.cos(theta)*np.sin(2.*varphi2)*np.sin(2.*psip))
            
            plus_joint_sq += plus2**2
            cross_joint_sq += cross2**2

            F=np.sqrt([plus_joint_sq,cross_joint_sq])
        return F
   
    def angles_convert(self, alpha_0, phi_0, Beta, Lambda, t):
        """
        To convert the angles in ecliptical system to detector-frame.
        
        Parameters:
            alpha_0 (float, or numpy array): alpha(t) at t=0
            phi_0 (float, or numpy array): phi(t), the position of the space craft, at t=0
            Beta (float, or numpy array): ecliptical latitude
            Lambda (float, or numpy array): ecliptical longitude
            t (float): time in unit of days
            
        Returns:
            theta_s (float, or numpy array): source latitude
            phi_s (float, or numpy array): source longitude
            see equations (3.3,3.6,3.16-3.17) of Cutler (1998).
        """
        T=365.242 # year in days
        alpha_t=2.*np.pi*t/T-np.pi/12.+alpha_0
        phi_t=phi_0+2.*np.pi*t/T
        cth_s=0.5*np.cos(Beta)-np.sqrt(3)/2.*np.sin(Beta)*np.cos(phi_t-Lambda)
        theta_s=np.arccos(cth_s)
        phi_s=alpha_t+np.pi/12.+np.arctan((np.sqrt(3)*np.cos(Beta)+np.sin(Beta)*np.cos(phi_t-Lambda))/(2.*np.sin(Beta)*np.sin(phi_t-Lambda)))
        
        return [theta_s, phi_s]
        

    def noise_curve(self, pars=None):
        """
        The Noise power spectrum of LISA-like GW observatories.
        
        Parameters:
          pars (Optional[list of floats]): which defines the configuration of the LISA-like detector
        
        Returns:
          (array of dtype float): noise vs. frequency, size=(,2), the 0-th column are frequencies, and 1-st column are corresponding noise power
        """
        if self.det == 3:
            data=np.loadtxt(self.lisa_file)
        else: raise ValueError('I dont know this detector, sorry')
        
        noise = [data[:,0],data[:,1]**2]
            
        return noise
    def Stdix(self, freqs):
        """
        Parameters:
           freqs: frequency or an ArrayLike frequencies. 
        """
        tdisn=lisaStdiX(freqs, lisaLT=self.lisaLT, lisaD=self.lisaD, lisaP=self.lisaP, Sacc0=self.Sacc0, Sopo0=self.Sopo0, Sops0=self.Sops0, Tobs=self.Tobs)
        return tdisn
        #return np.interp(freqs, self.fq, self.Sx)
    
    def Snoise(self, freqs):
        return np.interp(freqs, self.fq, self.Sn)

    def S_michaelson(self, freqs):
        Sn=lisa_michealson_noise(freqs, lisaLT=self.lisaLT, lisaD=self.lisaD, lisaP=self.lisaP, Sacc0=self.Sacc0, Sopo0=self.Sopo0, Sops0=self.Sops0, Tobs=self.Tobs)
        return Sn
