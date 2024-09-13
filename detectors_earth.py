import numpy as np
from pkg_resources import resource_filename

class LigoLike:
    """
    This class describes LIGO-like detectors
    
    Parameters:
      det (int): detector 'string' indicating the name of the detectors

    """
    def __init__(self, det):
        """
        Parameters:
            det (int): detector ID
        """
        self.det = det
        self.ligo_file = resource_filename('gwtoolbox','data_detectors/aLIGO.txt')
        self.ligo_o4_file = resource_filename('gwtoolbox','data_detectors/aLIGO_o4.txt')
        self.virgo_file = resource_filename('gwtoolbox','data_detectors/aVirgo.txt')
        self.kagra_file = resource_filename('gwtoolbox','data_detectors/KAGRA.txt')
        self.ligo_o3_file= resource_filename('gwtoolbox','data_detectors/o3_h1.txt')
        self.ce1_file= resource_filename('gwtoolbox','data_detectors/ce1.txt')
        self.ce2_file= resource_filename('gwtoolbox','data_detectors/ce2.txt')
    def ante_pattern(self, theta, varphi, psi):
        """
        The Antenna Pattern of LIGO-like (two arms with 90 deg) Gravitational Waves (GW) Observatories.
        
        Parameters:
          theta (float): the polar angle of the GW source in the detector coordinates frame
          varphi (float): the azimuth angle of the GW source in the detector coordiantes frame
          psi (float): the polarization angle of the GW
        
        Returns:
          (array of dtype float): size=2, [Fplus, Fcross]
        """
        plus = 0.5*(1.+np.cos(theta)**2)*np.cos(2.*varphi)*np.cos(2.*psi)-np.cos(theta)*np.sin(2.*varphi)*np.sin(2.*psi)
        cross = 0.5*(1.+np.cos(theta)**2)*np.cos(2.*varphi)*np.sin(2.*psi)+np.cos(theta)*np.sin(2.*varphi)*np.cos(2.*psi)
        F = [plus,cross]
        return F

    def noise_curve(self, pars=None):
        """
        The Noise power spectrum of LIGO-like GW observatories.
        
        Parameters:
          pars (Optional[list of floats]): which defines the configuration of the LIGO-like detector, this list of parameters will be passed to FINESSE
        
        Returns:
          (array of dtype float): noise vs. frequency, size=(,2), the 0-th column are frequencies, and 1-st column are corresponding noise power
        """
        if self.det == 'virgo':
            data = np.loadtxt(self.virgo_file)
        elif self.det =='ligo-o3':
            data = np.loadtxt(self.ligo_o3_file)
        elif self.det =='ligo-o4':
            data = np.loadtxt(self.ligo_o4_file)
        elif self.det == 'ligo':
            data = np.loadtxt(self.ligo_file)
        elif self.det == 'kagra':
            data = np.loadtxt(self.kagra_file)
        elif self.det == 'ce1':
            data = np.loadtxt(self.ce1_file)
        elif self.det == 'ce2':
            data = np.loadtxt(self.ce2_file)
        elif self.det == 'ligo-like' and pars != None:
            import gwtoolbox.simulate_ligo as sim # We import the simulator here because it is faster: loading it takes about one second, and we don't want to slow our code down when it is not necessary.
            session = sim.Session(pars) # Instantiate a Session, which needs us to supply it with the katCode at this point
            out_finesse = session.run() # returns the result from PyKat, which has a .x component (frequencies) and a .y component (strain noises).
            data = np.column_stack([out_finesse.x, np.ravel(out_finesse.y)])
        elif self.det == 'ligo-like' and pars == None:
            import gwtoolbox.simulate_ligo as sim # We import the simulator here because it is faster: loading it takes about one second, and we don't want to slow our code down when it is not necessary.
            session = sim.Session() # Instantiate a Session, use the default code here
            out_finesse = session.run() # returns the result from PyKat, which has a .x component (frequencies) and a .y component (strain noises).
            data = np.column_stack([out_finesse.x, np.ravel(out_finesse.y)])
        else: raise ValueError('ligo-like selected but no arm length and laser power provided')

        data[:,1] = data[:,1]**2.
        if self.det == 'ligo-like' and pars != None:
            # add seismic noise
            data[:,1]=data[:,1]+1e-27*data[:,0]**-18
        return [data[:,0], data[:,1]]

class ETLike:
    """
    This class describes ET-like detectors
    
    Parameters:
      det (int): detector ID (2 - ET, -2 - ET-like)
      
    """

    def __init__(self, det):
        """
        Parameters:
            det (int): detector ID (2 - ET, -2 - ET-like)
        """
        self.det = det
        self.et_file = resource_filename('gwtoolbox','data_detectors/ET.txt')

    def ante_pattern(self, theta, varphi, psi, Nnest=3):
        """
        The Antenna Pattern of ET-like (multiple detectors nested, each has two arms with 60 deg) Gravitational Waves (GW) Observatories.
        
        Parameters:
          theta (float): the polar angle of the GW source in the detector coordinates frame
          varphi (float): the azimuth angle of the GW source in the detector coordiantes frame
          psi (float): the polarization angle of the GW
          Nnest (int): The number of nested detectors
        
        Returns:
          (array of dtype float): size=(Nnext,2). The i-th row are the Antenna Patterns [Fplus,Fcross] of the i-th nested detector
        """
        plus1 = -np.sqrt(3.)/2.*(0.5*(1.+np.cos(theta)**2)*np.sin(2.*varphi)*np.cos(2.*psi)+np.cos(theta)*np.cos(2.*varphi)*np.sin(2.*psi))
        cross1 = np.sqrt(3.)/2.*(0.5*(1.+np.cos(theta)**2)*np.sin(2.*varphi)*np.sin(2.*psi)-np.cos(theta)*np.cos(2.*varphi)*np.cos(2.*psi))
        #F1=[plus1, cross1] # the first detector
        plus_joint_sq = plus1**2
        cross_joint_sq = cross1**2

        if Nnest >= 2:
            varphi2 = varphi+2.*np.pi/3.
            plus2 = -np.sqrt(3.)/2.*(0.5*(1.+np.cos(theta)**2)*np.sin(2.*varphi2)*np.cos(2.*psi)+np.cos(theta)*np.cos(2.*varphi2)*np.sin(2.*psi))
            cross2 = np.sqrt(3.)/2.*(0.5*(1.+np.cos(theta)**2)*np.sin(2.*varphi2)*np.sin(2.*psi)-np.cos(theta)*np.cos(2.*varphi2)*np.cos(2.*psi))
            #F2=[plus2, cross2]
            plus_joint_sq += plus2**2
            cross_joint_sq += cross2**2
            
        if Nnest >=3 :
            varphi3 = varphi-2.*np.pi/3.
            plus3 = -np.sqrt(3.)/2.*(0.5*(1.+np.cos(theta)**2)*np.sin(2.*varphi3)*np.cos(2.*psi)+np.cos(theta)*np.cos(2.*varphi3)*np.sin(2.*psi))
            cross3 = np.sqrt(3.)/2.*(0.5*(1.+np.cos(theta)**2)*np.sin(2.*varphi3)*np.sin(2.*psi)-np.cos(theta)*np.cos(2.*varphi3)*np.cos(2.*psi))
            #F3=[plus3, cross3]
            plus_joint_sq += plus3**2
            cross_joint_sq += cross3**2

        F=np.sqrt([plus_joint_sq,cross_joint_sq])
        return F

    def noise_curve(self, pars=None):
        """
        The Noise power spectrum of ET-like GW observatories.
        
        Parameters:
          pars (Optional[list of floats]): define the configuration of the ET-like detector, this list of parameters will be passed to FINESSE ???
        
        Returns:
          (array of dtype float): noise vs. frequency, size=(,2), the 0-th column are frequencies, and 1-st column are corresponding noise power
        """

        if self.det == 'et':
            data = np.loadtxt(self.et_file)
        elif self.det == 'et-like' and pars != None:
            import gwtoolbox.simulate_ligo as sim # We import the simulator here because it is faster: loading it takes about one second, and we don't want to slow our code down when it is not necessary.
            session = sim.Session(pars) # Instantiate a Session, which needs us to supply it with the katCode at this point
            out_finesse = session.run() # returns the result from PyKat, which has a .x component (frequencies) and a .y component (strain noises).
            data = np.column_stack([out_finesse.x, np.ravel(out_finesse.y)])
        elif self.det == 'et-like' and pars == None:
            import gwtoolbox.simulate_ligo as sim # We import the simulator here because it is faster: loading it takes about one second, and we don't want to slow our code down when it is not necessary.
            session = sim.Session() # Instantiate a Session, use the default code here
            out_finesse = session.run() # returns the result from PyKat, which has a .x component (frequencies) and a .y component (strain noises).
            data = np.column_stack([out_finesse.x, np.ravel(out_finesse.y)])
        else: raise ValueError('katCode is not given!')

        noise = [data[:,0],data[:,1]**2]

        return noise
