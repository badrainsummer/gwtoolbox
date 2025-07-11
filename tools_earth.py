# Try to modify the tools_earth object to support networks.
# Changes are marked in comments.

from pkg_resources import resource_filename

import astropy.utils.data as aud
aud.conf.show_progress = False

from astropy import units as u
from astropy.coordinates import EarthLocation
import numpy as np
from gwtoolbox.sources_kHz import *
from gwtoolbox.detectors_earth import *
from gwtoolbox.functions_earth import *
from gwtoolbox.cosmology import Cosmology
from gwtoolbox.constants import *
from gwtoolbox.parameters import *
import pandas as pd
from scipy import integrate
from itertools import accumulate
from astropy.cosmology import z_at_value
import warnings

def set_cosmology(cosmoID=None, H0=None, Om0=None, Tcmb=None):
    """
    Set the cosmological model.

    Parameters:
      cosmoID (optional): Cosmology identifier.
      H0 (optional): Hubble constant.
      Om0 (optional): Matter density parameter.
      Tcmb (optional): CMB temperature.

    Returns:
      cosmos: Cosmology instance.
    """
    cosmos_class = Cosmology()
    cosmos = cosmos_class.set_cosmo(cosmoID,H0,Om0, Tcmb)
    return cosmos

class Tools:
    """
    This is a class with tools to manipulate GW detectors and sources on Earth.

    Args:
      detID (int): detector id
      popID (int): population id
      cosmos (class): cosmological model
      det_setup (Optional[list of floats]): parameters to modify the detector noise

    Attributes:
      detector (class): detector setup class
      ant_pat (array of dtype float): detector shape
      noise (array of dtype float): detector noise

    """

    def __init__(self, detector_type, event_type, population, cosmos, det_setup=None, scale=None, new_theta=None):
        """
        Initialize the Tools class.

        Parameters:
          detector_type (string or list): Detector type(s).
          event_type (string): Event type.
          population (string): Population type.
          cosmos (class): Cosmological model.
          det_setup (optional): Detector noise parameters.
          scale (optional): Noise scaling.
          new_theta (optional): New population parameters.
        """
        
        self.cosmos = cosmos
        self.detector_type = detector_type #Allow this to be a list of strings too.
        self.event_type = event_type
        self.population = population
        self.network = isinstance(detector_type, list) #Checks whether a network is being used.
        if det_setup!=None:
            if not float(det_setup[0][1])>0:
                raise ValueError("laser armlength must be positive.");
            if not float(det_setup[1][1])>0:
                raise ValueError("laser power must be positive.")
            if (not float(det_setup[2][1])>0 or not	float(det_setup[2][1])<1):
                raise ValueError("cavity mirror transmssion must in between 0 and 1.")
            if (not float(det_setup[3][1])>0 or not float(det_setup[3][1])<1):
                raise ValueError("signal recycling mirror transmssion must in between 0 and 1.")
            if (not float(det_setup[4][1])>0 or not float(det_setup[4][1])<1):
                raise ValueError("power recycling mirror transmssion must in between 0 and 1.")
            if not float(det_setup[5][1])>0:
                raise ValueError("Mirror Mass must be postive.")
            if not float(det_setup[7][1])>0:
                raise ValueError("Power recycling length must be positive")
            if not float(det_setup[8][1])>0:
                raise ValueError("Signal recycling length must be positive")

        #chi_ranges = m1_ranges = m2_ranges = z_ranges = None
        #chi_ranges = [-0.5,0.5]
        # determine noise, ant, and detector generations...
        #if event_type in ['bg_dns', 'bg_bbh'] and detector_type!='ligo':
        #    raise ValueError("Currently, Background estimation only valid for LIGO")

        if not self.network: #In case no network is being used, same old same old.
            self.location = None # Indicates to some function(s) (tel_fun) that no locations are being used.
            if detector_type in ['virgo', 'ligo', 'kagra','ligo-o3','ligo-o4']:
                self.detector = LigoLike(self.detector_type)
                # with current generation of detectors like LIGO/VIRGO/KAGRA, the peak detectable redshift is at local Universe.
                self.ant_pat = self.detector.ante_pattern
                self.noise = self.detector.noise_curve(det_setup)
                self.generation='2G'
                if scale!=None:
                    self.noise = self.noise[0], self.noise[1]*scale
                    minsens=min(self.noise[1])
                    if np.sqrt(minsens)>2e-24:
                        self.generation='2G';
                    elif np.sqrt(minsens)>1e-24 and np.sqrt(minsens)<=2e-24:
                        self.generation='med';
                    else:
                        self.generation='3G';
            elif detector_type in ['ce1','ce2']:
                self.detector = LigoLike(self.detector_type)
                # with current generation of detectors like LIGO/VIRGO/KAGRA, the peak detectable redshift is at local Universe.
                self.ant_pat = self.detector.ante_pattern
                self.noise = self.detector.noise_curve(det_setup)
                self.generation='3G'
            elif detector_type=='ligo-like':
                self.detector = LigoLike(self.detector_type)
                self.ant_pat = self.detector.ante_pattern
                self.noise = self.detector.noise_curve(det_setup)
                minsens=min(self.noise[1])
                if np.sqrt(minsens)>2e-24:
                    self.generation='2G';
                elif np.sqrt(minsens)>1e-24 and np.sqrt(minsens)<=2e-24:
                    self.generation='med';
                else:
                    self.generation='3G';
            elif detector_type in ['et', 'et-like']:
                # the Next generation of detectors, the peak detectable red-shift is at z=1-2. and the range needs to be larger.
                self.detector = ETLike(detector_type)
                self.ant_pat = self.detector.ante_pattern
                self.noise = self.detector.noise_curve(det_setup)
                self.generation='3G';
                if scale!=None:
                    self.noise = self.noise[0], self.noise[1]*scale
                    minsens=min(self.noise[1])
                    if np.sqrt(minsens)>2e-24:
                        self.generation='2G';
                    elif np.sqrt(minsens)>1e-24 and np.sqrt(minsens)<=2e-24:
                        self.generation='med';
                    else:
                        self.generation='3G';
            else:
                raise ValueError("unknown detector type {}".format(detector_type))
        
        else: # If a network is being used.
            self.detector = [] # List for detectors in network
            self.ant_pat = [] # List of detector antenna patterns
            self.noise = [] #  list of detector  noise profiles
            self.generation = [] # List  of detector generations
            self.location = [] # List of detector locations. NEED CONVERSION STILL.
            for i, det in enumerate(self.detector_type): # Repeat process for all detectors.
                if det == 'virgo': # Set location of virgo observatory
                    loc = EarthLocation.of_site('Virgo')
                    loc_m = EL_to_np(loc) # ONLY WORKS WHEN MY CODE IS AVAILABLE.
                    self.location.append(m_to_ls(loc_m)) # ONLY WORKS WHEN MY CODE IS AVAILABLE.
                if det in ['hanford','handford-o3']: # Set location of ligo observatory
                    loc = EarthLocation.of_site('LHO_4k')
                    loc_m = EL_to_np(loc) # ONLY WORKS WHEN MY CODE IS AVAILABLE.
                    self.location.append(m_to_ls(loc_m))
                if det ==['livingston','livingston-o3']: # Set location of ligo observatory
                    loc = EarthLocation.of_site('LLO_4k')
                    loc_m = EL_to_np(loc) # ONLY WORKS WHEN MY CODE IS AVAILABLE.
                    self.location.append(m_to_ls(loc_m))
                if det == 'kagra': # Set location of kagra observatory
                    loc = EarthLocation.of_site('KAGRA')
                    loc_m = EL_to_np(loc) # ONLY WORKS WHEN MY CODE IS AVAILABLE.
                    self.location.append(m_to_ls(loc_m)) # ONLY WORKS WHEN MY CODE IS AVAILABLE.
                if det == 'et': # Set location of ET observatory (Aachen)
                    loc = EarthLocation(6.08453, 50.77234) # Somewhere in Aachen
                    loc_m = EL_to_np(loc)
                    self.location.append(m_to_ls(loc_m))
#                else: # When detector does not yet have a location
#                    raise ValueError("Location of detector type {} not yet determined".format(det))
                if det in ['virgo', 'ligo', 'kagra','ligo-o3','ligo-o4', 'hanford', 'livingston','hanford-o3', 'livingston-o3']:
                    if det in['hanford', 'livingston']:
                        self.detector.append(LigoLike('ligo')) # touched by shuxu @ 10th Nov,2023
                    elif det in ['hanford-o3', 'livingston-o3']:
                        self.detector.append(LigoLike('ligo')) # touched by shuxu @ 10th Nov,2023
                    else:
                        self.detector.append(LigoLike(det))
                    # with current generation of detectors like LIGO/VIRGO/KAGRA, the peak detectable redshift is at local Universe.
                    self.ant_pat.append(self.detector[i].ante_pattern)
                    self.noise.append(self.detector[i].noise_curve(det_setup))
                    self.generation.append('2G')
                    if scale!=None:
                        self.noise[i] = self.noise[i][0], self.noise[i][1]*scale
                        minsens=min(self.noise[i][1])
                        if np.sqrt(minsens)>2e-24:
                            self.generation[i]='2G';
                        elif np.sqrt(minsens)>1e-24 and np.sqrt(minsens)<=2e-24:
                            self.generation[i]='med';
                        else:
                            self.generation[i]='3G';
                elif det in ['ce1','ce2']:
                    self.detector.append(LigoLike(det))
                    # with current generation of detectors like LIGO/VIRGO/KAGRA, the peak detectable redshift is at local Universe.
                    self.ant_pat.append(self.detector[i].ante_pattern)
                    self.noise.append(self.detector[i].noise_curve(det_setup))
                    self.generation.append('3G')
                elif det=='ligo-like':
                    self.detector.append(LigoLike(det))
                    self.ant_pat.append(self.detector[i].ante_pattern)
                    self.noise.append(self.detector[i].noise_curve(det_setup))
                    minsens=min(self.noise[i][1])
                    if np.sqrt(minsens)>2e-24:
                        self.generation.append('2G');
                    elif np.sqrt(minsens)>1e-24 and np.sqrt(minsens)<=2e-24:
                        self.generation.append('med');
                    else:
                        self.generation.append('3G');
                elif det in ['et', 'et-like']:
                    # the Next generation of detectors, the peak detectable red-shift is at z=1-2. and the range needs to be larger.
                    self.detector.append(ETLike(det))
                    self.ant_pat.append(self.detector[i].ante_pattern)
                    self.noise.append(self.detector[i].noise_curve(det_setup))
                    self.generation.append('3G');
                    if scale!=None:
                        self.noise[i] = self.noise[i][0], self.noise[i][1]*scale
                        minsens=min(self.noise[i][1])
                        if np.sqrt(minsens)>2e-24:
                            self.generation[i]='2G';
                        elif np.sqrt(minsens)>1e-24 and np.sqrt(minsens)<=2e-24:
                            self.generation[i]='med';
                        else:
                            self.generation[i]='3G';
                else:
                    raise ValueError("unknown detector type {}".format(det))


        ##### deal with population
        if '2G' in self.generation: # Checks through the generation list
            if event_type == 'bhbh':
                self.population_class = BHB(self.cosmos) # Modified BHB object
                m1_ranges = m2_ranges = [3.01, 100]
                self.param_init = [0.25, 10., 10., 0.]
                self.population_class.set_model_theta(pop=population, new_theta=new_theta)
            elif event_type == 'nsns':
                self.population_class = DNS(self.cosmos)
                self.population_class.set_model_theta(new_theta=new_theta)
                self.param_init = [0.01, 1.4, 1.4, 0.]

            elif event_type == 'bhns':
                self.population_class = BHNS(self.cosmos)
                self.population_class.set_model_theta(pop=population, new_theta=new_theta)
                self.param_init = [0.01, 10., 1., 0.]

            elif event_type == 'bg_dns':
                self.population_class= Background(self.cosmos)
                self.param_init = [0.3, 1.4, 1.4, 0.]
                self.population_class.set_model_theta(pop='DNS')
            elif event_type == 'bg_bbh':
                self.population_class= Background(self.cosmos)
                self.param_init = [0.25, 10, 10, 0.]
                self.population_class.set_model_theta(pop='BHB')


            else:
                raise ValueError("event type must be bhbh, nsns, bhns, or bg_dns, or bg_bbh")
        elif '3G' in self.generation: # Checks through generation list
            if event_type == 'bhbh':
                self.population_class = BHB(self.cosmos) # Modified BHB object
                self.param_init=[2.,10.,10.,0.]
                self.population_class.set_model_theta(pop=population, new_theta=new_theta)
            elif event_type == 'nsns':
                self.population_class = DNS(self.cosmos)
                self.population_class.set_model_theta(new_theta=new_theta)
                self.param_init = [2.,1.,1.,0.]

            elif event_type == 'bhns':
                self.population_class = BHNS(self.cosmos)
                self.population_class.set_model_theta(pop=population, new_theta=new_theta)
                self.param_init = [2,5.,1.,0.]

            elif event_type == 'bg_dns':
                self.population_class= Background(self.cosmos)
                self.param_init = [2.,1.,1.,0.]
                self.population_class.set_model_theta(pop='DNS')
            elif event_type == 'bg_bbh':
                self.population_class= Background(self.cosmos)
                self.param_init = [2, 10, 10, 0.]
                self.population_class.set_model_theta(pop='BHB')

            else:
                raise ValueError("event type must be bhbh, nsns or bhns")

        elif 'med' in self.generation: # Checks through generation list.
            if event_type == 'bhbh':
                self.population_class = BHB(self.cosmos) # Modified BHB object
                self.param_init=[0.8,10.,10.,0.]
                self.population_class.set_model_theta(pop=population, new_theta=new_theta)
            elif event_type == 'nsns':
                self.population_class = DNS(self.cosmos)
                self.population_class.set_model_theta(new_theta=new_theta)
                self.param_init = [0.14, 1.,1.,0.]

            elif event_type == 'bhns':
                self.population_class = BHNS(self.cosmos)
                self.population_class.set_model_theta(pop=population, new_theta=new_theta)
                self.param_init = [0.4 ,5.,1.,0.]

            elif event_type == 'bg_dns':
                self.population_class= Background(self.cosmos)
                self.param_init = [0.14, 1.,1.,0.]
                self.population_class.set_model_theta(pop='DNS')
            elif event_type == 'bg_bbh':
                self.population_class= Background(self.cosmos)
                self.param_init = [0.8,10.,10.,0.]
                self.population_class.set_model_theta(pop='BHB')

            else:
                raise ValueError("event type must be bhbh, nsns or bhns")


    def total_number(self, time_obs=TIME_OBS_EARTH, rho_cri=RHO_CRIT_EARTH):
        """
        Return the expected number of detections.

        Parameters:
          time_obs (float): Observation time, in unit of minute
          rho_cri (float): The SNR threshold of detection

        Returns:
          (float): expected number of detections
        """
        if not time_obs>=0:
            raise ValueError("Observationn durationn must be positive.")
        if not rho_cri>=0:
            raise ValueError("SNR criterion should be larger than zero.")
        self.ndet = self.population_class.tot_num(time_obs,rho_cri,self.ant_pat,self.noise,self.generation,self.detector_type)
        return self.ndet

    def sensitivity_curve(self):
        """
        Return the sensitivity curve (frequency, sqrt(power)) of the detector.
        """
        sen_curve = []
        if self.network:
            for noise in self.noise:
                freq, power=noise
                sen_curve.append([freq, np.sqrt(power)])
        else:  # If a network is being used, return the sensitivity curve of the first detector.
            freq, power=self.noise
            sen_curve=[freq, np.sqrt(power)]
        return sen_curve
    def sensitivity_curve_sjoerd(self):
        """
        Return the sensitivity curve of the experiment using default detector settings.
        """
        if self.network:
            warnings.warn("networks detetors are all default detectors, use sensitivity_curve() instead.")
            return self.sensitivity_curve()
        else:
            detector_class = type(self.detector)
            default_type = self.detector_type[:-5]
            default_detector = detector_class(default_type)
            default_noise = default_detector.noise_curve()
            freq, power=default_noise
            return [freq, np.sqrt(power)]

    def _list_params(self, time_obs=TIME_OBS_EARTH, rho_cri=RHO_CRIT_EARTH, size=None, dtp=False, withangles=False, witherrors=False):
        """
        Return an array of final parameters, optionally with angles and errors.

        Parameters:
          time_obs (float): Observation time, in unit of minute
          rho_cri (float): The SNR threshold of detection
          size (Optional[int]): sample size
          dtp (bool): Include dtp column.
          withangles (bool): Include angles.
          witherrors (bool): Include parameter errors.

        Returns:
          (array or tuple): samples or (samples, loc) if network and withangles.
        """
        if size == None:
            self.number = np.random.poisson(lam=self.population_class.tot_num(time_obs,rho_cri,self.ant_pat,self.noise,self.generation,self.detector_type))
        else:
            self.number = size
        samples = self.population_class.MCMCsample(self.number,self.param_init,time_obs,rho_cri,self.ant_pat,self.noise, name=self.detector_type)
        if withangles:
            if not self.network:
                z=np.array(samples[1,:])
                m1=np.array(samples[2,:])
                m2=np.array(samples[3,:])
                dtps, thetheta, thevarphi, theiota, thepsi, therho2=self.population_class.tel_fun(z, m1, m2, rho_cri, self.ant_pat, self.noise, accurate=False, withangles=withangles)
                if dtp:
                    samples=np.vstack((samples,thetheta, thevarphi, theiota, thepsi, np.sqrt(therho2), dtps))
                else:
                    samples=np.vstack((samples,thetheta, thevarphi, theiota, thepsi, np.sqrt(therho2)))
            if self.network:
                z=np.array(samples[1,:])
                m1=np.array(samples[2,:])
                m2=np.array(samples[3,:])
                dtps, thetheta, thevarphi, theiota, thepsi, therho2, area, thetheta_dets, thephi_dets,theiota_dets, thepsi_dets=self.population_class.tel_fun(z, m1, m2, rho_cri, self.ant_pat, self.noise, accurate=False, name=self.detector_type, withangles=withangles)
                if dtp:
                    samples=np.vstack((samples,thetheta, thevarphi, theiota, thepsi, np.sqrt(therho2), area, dtps))
                else:
                    samples=np.vstack((samples,thetheta, thevarphi, theiota, thepsi, np.sqrt(therho2), area))
                loc = [thetheta_dets, thephi_dets, theiota_dets, thepsi_dets]
        elif dtp:
            z=np.array(samples[1,:])
            m1=np.array(samples[2,:])
            m2=np.array(samples[3,:])
            dtps=self.population_class.tel_fun(z, m1, m2, rho_cri, self.ant_pat, self.noise, accurate=False, name=self.detector_type, withangles=False)
            samples=np.vstack((samples,dtps))
        if witherrors:
            if self.network:
                errors = self.population_class.errors_FIM(self.number, samples, self.noise[0])
            else:
                errors = self.population_class.errors_FIM(self.number, samples, self.noise)
            samples=np.vstack((samples,errors))
        if self.network and withangles:
            return samples, loc
        else:
            return samples

    def list_params(self, time_obs=TIME_OBS_EARTH, rho_cri=RHO_CRIT_EARTH, size=None, dtp=False, withangles=False, witherrors=False):
        """
        Return a DataFrame of parameters, optionally with angles and errors.

        Parameters:
          time_obs (float): Observation time, in unit of minute
          rho_cri (float): The SNR threshold of detection
          size (Optional[int]): sample size
          dtp (bool): Include dtp column.
          withangles (bool): Include angles.
          witherrors (bool): Include parameter errors.

        Returns:
          DataFrame or (DataFrame, loc) if network and withangles.
        """
        if self.network and withangles:
            cata, loc = self._list_params(time_obs=time_obs, rho_cri=rho_cri, size=size, dtp=dtp, withangles=withangles, witherrors=witherrors)
        else:
            cata = self._list_params(time_obs=time_obs, rho_cri=rho_cri, size=size, dtp=dtp, withangles=withangles, witherrors=witherrors)
        cata = cata.transpose()
        keys = ['z','D','m1','m2','χ']
        if withangles:
            keys += ['theta','varphi','inc','psi','rho']
            if self.network:
                keys += ['area']
        if dtp:
            keys += ['dtb']
        if witherrors:
            keys += ['dz','dm1','dm2','dχ','dD']
            z_error_3G=lambda z:0.017*z+0.012
            z_error_med=lambda z:0.1*z+0.01
            z_error_2G=lambda z:0.5*z
            if self.generation=='3G':
                z_errors=z_error_3G(cata[:,1])
            elif self.generation=='2G':
                z_errors=z_error_2G(cata[:,1])
            else:
                z_errors=z_error_med(cata[:,1])
            D_errors=dDovdz(cata[:,1],self.cosmos)*z_errors
            z = cata[:,1]
            m1z=cata[:,2]*(1+z)
            m2z=cata[:,3]*(1+z)
            dlnm1z=cata[:,-3]/m1z
            dlnm2z=cata[:,-2]/m2z
            dln1pz=z_errors/(1+z)
            dm1=cata[:,2]*np.sqrt(dlnm1z**2+dln1pz**2)
            dm2=cata[:,3]*np.sqrt(dlnm2z**2+dln1pz*2)
            cata=np.column_stack((cata[:,:-3],z_errors,dm1,dm2,cata[:,-1],D_errors))
        cols = list(range(1,cata.shape[1]))
        cols[:5] = [1, 5, 2, 3, 4]
        df=pd.DataFrame(data=cata[:,cols], columns=keys)
        if self.network and withangles:
            return df, loc
        else:
            return df

    def list_with_iota(self, dataframe):
        """
        Add inclination ('inc') column to the dataframe.

        Parameters:
          dataframe (pd.DataFrame): Input dataframe.

        Returns:
          pd.DataFrame: Dataframe with 'inc' column.
        """
        dtbs=dataframe['dtb']
        det=self.detector
        pop=self.population_class
        iotas=give_iota(det, pop, dtbs)
        dataframe['inc']=iotas
        return dataframe

    def list_with_angles(self, dataframe):
        """
        Add angle columns ('theta', 'varphi', 'inc', 'psi') to the dataframe.

        Parameters:
          dataframe (pd.DataFrame): Input dataframe.

        Returns:
          pd.DataFrame: Dataframe with angle columns.
        """
        dtbs=dataframe['dtb']
        det=self.detector
        pop=self.population_class
        thetas, varphis, iotas, psis=give_angles(det, pop, dtbs)
        dataframe['theta']=thetas
        dataframe['varphi']=varphis
        dataframe['inc']=iotas
        dataframe['psi']=psis
        return dataframe

    def get_better_fisher(self, dataframe, know_loc=1):
        """
        Attach Fisher matrix errors to the dataframe.

        Parameters:
          dataframe (pd.DataFrame): Input dataframe.
          know_loc (int): Whether location is known.

        Returns:
          pd.DataFrame: Dataframe with error columns.
        """
        from gwtoolbox.function_earth_fisher import fisher
        my_fisher=fisher(det=self.detector, approximant='IMRPhenomD', cosmos=self.cosmos);
        for key in ['theta', 'varphi', 'inc', 'psi']:
            if not key in dataframe:
                self.list_with_angles(dataframe)
                break;
        new_DF=my_fisher.attach_error_dataframe(dataframe, know_loc=know_loc)
        return new_DF
    
    def snr(self, theta=None):
        """
        Compute the SNR for given parameters.

        Parameters:
          theta (tuple): (z, D, chi, m1, m2)

        Returns:
          array: SNR squared values.
        """
        ## individual SNR earth
        z, D, chi, m1, m2 = theta
        if not z:
            z = redshift_from_luminosity_distance(D*u.Mpc,cosmos=self.cosmos)
        elif not D:
            D = self.cosmos.luminosity_distance(z).to(u.Mpc).value
        Mch = chirp_mass(m1,m2)
        if self.network:
            rho_sq_array = []
            population = self.population_class
            for i in range(len(self.detector)):
                detector = self.detector[i]
                noise = self.noise[i]

                f_up = population.freq_limit(m1=m1*(1+z), m2=m2*(1+z), chi=chi)
                f2=population.freq_limit_merger(m1=m1*(1+z), m2=m2*(1+z), chi=chi)
                f3=population.freq_limit_ringdown(m1=m1*(1+z), m2=m2*(1+z), chi=chi)
                freq_sig=population.freq_sigma(m1=m1*(1+z), m2=m2*(1+z), chi=chi)
                f1 = f_up

                rho_sq_core_value = rho_sq_core(noise, population.mod_shape, f_up=f_up)+rho_sq_core_merger(noise, population.mod_shape_merger, f1=f1, f2=f2)+rho_sq_core_ringdown(noise, population.mod_shape_ringdown,f1=f1, f2=f2, freq_sig=freq_sig, f3=f3)

                theta_array = np.array([np.arccos(0.)])
                varphi_array = np.array([np.pi])
                iota_array = np.array([np.arccos(0.)])
                psi_array = np.array([np.arccos(0.)])

                F = detector.ante_pattern(theta_array, varphi_array, psi_array)
                A_array = population.mod_norm(Mch*(1+z), F, iota_array, z)

                rho_sq_array.append(np.array(4.*A_array**2*rho_sq_core_value))
        else:
            detector = self.detector
            noise = self.noise
            population = BHB(self.cosmos)

            f_up = population.freq_limit(m1=m1*(1+z), m2=m2*(1+z), chi=chi)
            f2=population.freq_limit_merger(m1=m1*(1+z), m2=m2*(1+z), chi=chi)
            f3=population.freq_limit_ringdown(m1=m1*(1+z), m2=m2*(1+z), chi=chi)
            freq_sig=population.freq_sigma(m1=m1*(1+z), m2=m2*(1+z), chi=chi)
            f1 = f_up

            rho_sq_core_value = rho_sq_core(noise, population.mod_shape, f_up=f_up)+rho_sq_core_merger(noise, population.mod_shape_merger, f1=f1, f2=f2)+rho_sq_core_ringdown(noise, population.mod_shape_ringdown,f1=f1, f2=f2, freq_sig=freq_sig, f3=f3)

            theta_array = np.array([np.arccos(0.)])
            varphi_array = np.array([np.pi])
            iota_array = np.array([np.arccos(0.)])
            psi_array = np.array([np.arccos(0.)])

            F = detector.ante_pattern(theta_array, varphi_array, psi_array)
            A_array = population.mod_norm(Mch*(1+z), F, iota_array, z)

            rho_sq_array = np.array(4.*A_array**2*rho_sq_core_value)
        return rho_sq_array


        # params = np.zeros((5,2))
        # params[:,0] = np.array([Mch,z,m1,m2,chi])
        # params[:,1] = np.array([Mch,z,m1,m2,chi])

        # errs = population.errors_FIM(1,params,noise)
        # print(errs)
    
    def create_BNScatalog(self, T=525600, popu=[50,3,1.4,0.5,1.1,2.5,0.1], pathname=None):
        """
        Create a BNS catalog.

        Parameters:
          T (int): Total observation time in minutes.
          popu (list): Population parameters.
          pathname (str): Optional path to save CSV.

        Returns:
          pd.DataFrame: BNS catalog.
        """
        Ti=T/525600 #year
        R0=popu[0]    
        def rm(z):
            y=4*np.pi*Ti*R0*R(z, 2.7, 5.6, 2.9, tau=popu[1], cosmos=self.cosmos)*self.cosmos.differential_comoving_volume(z).to_value(u.Gpc**3/u.sr)/(1+z)
            return y
        def tot_bns_number(z):
            result, error = integrate.quad(rm, 0, z, epsabs=1e-1)
            return int(result)
        N=tot_bns_number(z=3)
        # print(N)
        low=popu[4]
        loc=popu[2]
        high=popu[5]
        scal=popu[3]
        aNS,bNS = (low-loc)/scal, (high-loc)/scal
        pm1 = truncnorm.rvs(aNS, bNS, loc=loc, scale=scal, size=N)
        pm2 = truncnorm.rvs(aNS, bNS, loc=loc, scale=scal, size=N)
        zs=np.linspace(1e-3,3,200)
        RM=list(accumulate(rm(zs)))
        def inver(x):
            return np.interp(x, np.array(RM), zs)
        aa=np.random.uniform(0,RM[199],size=N)
        PZ=inver(aa)
        PZ=np.array(PZ)
        theta=np.arccos(np.random.uniform(-1,1,size=N))
        varphi=np.random.uniform(0,2*np.pi,size=N)
        psi=np.random.uniform(0,np.pi,size=N)
        iota=np.arccos(np.random.uniform(0,1,size=N))
        
        DIS=self.cosmos.luminosity_distance(PZ).value
        Time=np.random.randint(0,T,size=N)
        #print(DIS)
        rho=[]
        area=[]
        for i in range(N):
            re1, re2 = BHB(self.cosmos).area_snr(z=PZ[i], m1=pm1[i], m2=pm2[i], theta=theta[i], varphi=varphi[i], iota=iota[i], psi=psi[i], ant_fun=self.ant_pat, noise_tab=self.noise, name=self.detector_type)
            rho.append(re1)
            area.append(re2)
        dD=np.array(DIS)/rho
        column_names = ['z', 'D', 'm1', 'm2', 'rho', 'theta', 'varphi', 'inc', 'psi', 'area', 'dD', 'time']
        df_BNS=pd.DataFrame([PZ, DIS, pm1, pm2, rho, theta, varphi, iota, psi, area, dD, Time],index=column_names).T
        self.BNScatalog = df_BNS
        if pathname:
            df_BNS.to_csv(f'{pathname}.csv')
        return df_BNS
    
    def create_backgroundcatalog(self, rho, T=4, pathname=None):
        """
        Create a background catalog.

        Parameters:
          rho (float): SNR threshold.
          T (int): Observation time in years.
          pathname (str): Optional path to save CSV.

        Returns:
          pd.DataFrame: Background catalog.
        """
        FAR8=30000
        cri_dis=2493.8699694225065
        alpha=0.13
        bg_number=T*FAR8*np.exp((8-rho)/alpha)
        Mch=np.random.uniform(0.435, 2.61, int(bg_number))
        #calculate the luminosity distance
        Time=np.random.randint(0,525600*T,size=int(bg_number))
        Dis=Mch**(5/6)*cri_dis/rho
        theta=np.arccos(np.random.uniform(-1,1,size=int(bg_number)))
        varphi=np.random.uniform(0,2*np.pi,size=int(bg_number))
        psi=np.random.uniform(0,np.pi,size=int(bg_number))
        iota=np.arccos(np.random.uniform(0,1,size=int(bg_number)))
        SNR, DIS=[], []
        for i, dis in enumerate(Dis):
            SNR.append(BHB(self.cosmos).SNR(z=z_at_value(self.cosmos.luminosity_distance, Dis[i]*u.Mpc).value, Mch=Mch[i], theta=theta[i], varphi=varphi[i], iota=iota[i], psi=psi[i], name=self.detector_type, ant_fun=self.ant_pat, noise_tab=self.noise))
            DIS.append(Dis[i]*SNR[i]/rho)
        Z=z_at_value(self.cosmos.luminosity_distance, DIS*u.Mpc).value
        m1=Mch*2.0**0.2
        m2=m1
        dD=np.array(DIS)/rho
        area=[]
        for i in range(int(bg_number)):
            re1, re2 = BHB(self.cosmos).area_snr(z=Z[i], m1=m1[i], m2=m2[i], theta=theta[i], varphi=varphi[i], iota=iota[i], psi=psi[i], ant_fun=self.ant_pat, noise_tab=self.noise, name=self.detector_type)
            area.append(re2)
        column_names = ['z', 'D', 'm1', 'm2', 'theta', 'varphi', 'inc', 'psi', 'area', 'dD', 'time']
        df=pd.DataFrame([Z, DIS, m1, m2, theta, varphi, iota, psi, area, dD, Time],index=column_names).T
        if pathname:
            df.to_csv(f'{pathname}.csv')
        return df
    
    def create_GWcatalog(self,rho_cri, pathname=None):
        """
        Create a GW catalog from BNS catalog with SNR cut.

        Parameters:
          rho_cri (float): SNR threshold.
          pathname (str): Optional path to save CSV.

        Returns:
          pd.DataFrame: GW catalog.
        """
        df_BNS=self.BNScatalog[self.BNScatalog['z']<=0.4].reset_index(drop=True)
        N=len(df_BNS)
        z,D,m1,m2,rho,theta,varphi,inc,psi,area,dD,time=[],[],[],[],[],[],[],[],[],[],[],[]
        for i in range(N):
            re1, re2 = BHB(self.cosmos).area_snr(z=df_BNS['z'][i], m1=df_BNS['m1'][i], m2=df_BNS['m2'][i], theta=df_BNS['theta'][i], varphi=df_BNS['varphi'][i], iota=df_BNS['inc'][i], psi=df_BNS['psi'][i], ant_fun=self.ant_pat, noise_tab=self.noise, name=self.detector_type)
            if re1>=rho_cri:
                z.append(df_BNS.iloc[i]['z'])
                D.append(df_BNS.iloc[i]['D'])
                m1.append(df_BNS.iloc[i]['m1'])
                m2.append(df_BNS.iloc[i]['m2'])
                rho.append(re1)
                theta.append(df_BNS.iloc[i]['theta'])
                varphi.append(df_BNS.iloc[i]['varphi'])
                inc.append(df_BNS.iloc[i]['inc'])
                psi.append(df_BNS.iloc[i]['psi'])
                area.append(re2)
                dD.append(df_BNS.iloc[i]['D']/re1)
                time.append(df_BNS.iloc[i]['time'])
        column_names = ['z', 'D', 'm1', 'm2', 'rho', 'theta', 'varphi', 'inc', 'psi', 'area', 'dD', 'time']
        df_gw=pd.DataFrame([z, D, m1, m2, rho, theta, varphi, inc, psi, area, dD, time],index=column_names).T
        if pathname:
            df_gw.to_csv(f'{pathname}.csv')
        return df_gw
    
    def create_kilonovacatalog(self,z1=0.036,z2=0.022, pathname=None):
        """
        Create a kilonova catalog.

        Parameters:
          z1 (float): Redshift threshold for full detection.
          z2 (float): Redshift width for detection probability.
          pathname (str): Optional path to save CSV.

        Returns:
          pd.DataFrame: Kilonova catalog.
        """
        df_BNS=self.BNScatalog[self.BNScatalog['z']<=0.4].reset_index(drop=True)
        N=len(df_BNS)
        def Pdet_kilo(z,z1=z1,z2=z2):
            if z<z1:
                return 1
            if z>=z1:
                return np.exp(-1/2*((z-z1)/z2)**2) 
        z,D,m1,m2,theta,varphi,inc,psi,time=[],[],[],[],[],[],[],[],[]
        for i in range(N):
            random1=np.random.uniform(0,1)
            if Pdet_kilo(df_BNS['z'][i])>=random1:
                z.append(df_BNS.iloc[i]['z'])
                D.append(df_BNS.iloc[i]['D'])
                m1.append(df_BNS.iloc[i]['m1'])
                m2.append(df_BNS.iloc[i]['m2'])
                theta.append(df_BNS.iloc[i]['theta'])
                varphi.append(df_BNS.iloc[i]['varphi'])
                inc.append(df_BNS.iloc[i]['inc'])
                psi.append(df_BNS.iloc[i]['psi'])
                time.append(df_BNS.iloc[i]['time'])
        column_names = ['z', 'D', 'm1', 'm2', 'theta', 'varphi', 'inc', 'psi', 'time']
        df_kilo=pd.DataFrame([z, D, m1, m2, theta, varphi, inc, psi, time],index=column_names).T
        if pathname:
            df_kilo.to_csv(f'{pathname}.csv')
        return df_kilo

    def create_sgrbcatalog(self,flim=2*10**(-7), lam=0.6, pathname=None):
        """
        Create an sGRB catalog.

        Parameters:
          flim (float): Flux limit.
          lam (float): Detection efficiency parameter.
          pathname (str): Optional path to save CSV.

        Returns:
          pd.DataFrame: sGRB catalog.
        """
        N=len(self.BNScatalog)
        def Pdet_sgrb(z, thetav, thetac, flim=flim, lam=lam):
            L_star=2*10**52
            L_lim=flim*(4*np.pi)**2*(self.cosmos.comoving_distance(z=z).to_value(u.cm))**2*np.exp(thetav**2/thetac**2)
            if L_star>L_lim:
                pvalue=np.log(L_star/L_lim)+1
                return pvalue/(15.5-np.log((2*10**(-7)/flim)))*lam*20#20年
            else:
                pvalue1=L_star/L_lim
                return pvalue1/(15.5-np.log((2*10**(-7)/flim)))*lam*20
        z,D,m1,m2,theta,varphi,inc,psi,time,thetac=[],[],[],[],[],[],[],[],[],[]
        for i in range(N):
            random1=np.random.uniform(0,1)
            thetacs=np.random.normal(5*np.pi/180, 2*np.pi/180)
            if Pdet_sgrb(z=self.BNScatalog['z'][i],thetav=self.BNScatalog['inc'][i], thetac=thetacs)>=random1:
                z.append(self.BNScatalog.iloc[i]['z'])
                D.append(self.BNScatalog.iloc[i]['D'])
                m1.append(self.BNScatalog.iloc[i]['m1'])
                m2.append(self.BNScatalog.iloc[i]['m2'])
                theta.append(self.BNScatalog.iloc[i]['theta'])
                varphi.append(self.BNScatalog.iloc[i]['varphi'])
                inc.append(self.BNScatalog.iloc[i]['inc'])
                psi.append(self.BNScatalog.iloc[i]['psi'])
                time.append(self.BNScatalog.iloc[i]['time'])
                thetac.append(thetacs)
        column_names = ['z', 'D', 'm1', 'm2', 'theta', 'varphi', 'inc', 'psi', 'time', 'thetac']
        df_sgrb=pd.DataFrame([z, D, m1, m2, theta, varphi, inc, psi, time, thetac],index=column_names).T
        if pathname:
            df_sgrb.to_csv(f'{pathname}.csv')
        return df_sgrb
    
    def kilonovae(self, pathname=None):
        """
        Simulate GW-EM coincidence for kilonovae.

        Parameters:
          pathname (str): Optional path to save CSV.

        Returns:
          pd.DataFrame: Results of coincidence simulation.
        """
        MISID=[]
        rho_list = np.linspace(8.0,9.3,14)
        df_kilo = self.create_kilonovacatalog(z1=0.067,z2=0.056)
        for rho in rho_list:
            DF_astro_tool = self.create_GWcatalog(rho)
            DF_back_tool= self.create_backgroundcatalog(rho)
            DF_gw_all = pd.concat([DF_astro_tool, DF_back_tool])
            df_gwall = DF_gw_all.reset_index(drop=True)

            gw=np.linspace(0,len(df_gwall)-1,len(df_gwall))
            
            emc=np.linspace(0,len(df_kilo)-1,len(df_kilo))
            gw=list(gw)
            emc=list(emc)
            GE=[]
            N_gw_em=0
            mis=0
            for i in gw:
                GW_EMC=[]
                gwi_emc=[]
                D1=df_gwall['D'][i] - df_gwall['dD'][i]
                D2=df_gwall['D'][i] + df_gwall['dD'][i]
                Time=df_gwall['time'][i]
                area = df_gwall['area'][i]
                theta = df_gwall['theta'][i]
                varphi = df_gwall['varphi'][i]
                for j in emc:
                    theta1=df_kilo['theta'][j]
                    varphi1=df_kilo['varphi'][j]
                    angel1=np.arccos(np.cos(theta)*np.cos(theta1)*np.cos(varphi-varphi1)+np.sin(theta)*np.sin(theta1))
                    angel2=np.sqrt(area*(np.pi/180)**2/np.pi)
                    if D1<=df_kilo['D'][j]<=D2 and 0<=(df_kilo['time'][j]-Time)<=10080 and angel1<=angel2:
                        gwi_emc.append(j)
                        emc.remove(j)
                if len(gwi_emc)>1:
                    GW_EMC.append(random.choice(gwi_emc))
                elif len(gwi_emc)==1:
                    GW_EMC.append(gwi_emc[0])
                if len(GW_EMC)==1:
                    N_gw_em+=1
                    if df_gwall['z'][i]!=df_kilo['z'][GW_EMC[0]]:
                        mis+=1
                GE.append([i, gwi_emc, GW_EMC])
            MISID.append([N_gw_em, mis, len(gw), len(emc), len(DF_astro_tool)])
            
        COLUMNS=['Ngwemc','N_mis','Ngw', 'Nemc', 'Ngwast'] 
        #INDEX=[ast_em]
        df=pd.DataFrame(MISID,columns=COLUMNS, index=None)
        if pathname:
            df.to_csv(f'{pathname}.csv')
        return df
    
    def sgrb(self, pathname=None):
        """
        Simulate GW-EM coincidence for sgrb.

        Parameters:
          pathname (str): Optional path to save CSV.

        Returns:
          pd.DataFrame: Results of coincidence simulation.
        """
        MISID=[]
        rho_list = np.linspace(8.0,9.3,14)
        df_kilo = self.create_sgrbcatalog(flim=2*10**(-8), lam=1)
        for rho in rho_list:
            DF_astro_tool = self.create_GWcatalog(rho)
            DF_back_tool= self.create_backgroundcatalog(rho)
            DF_gw_all = pd.concat([DF_astro_tool, DF_back_tool])
            df_gwall = DF_gw_all.reset_index(drop=True)
            gw=np.linspace(0,len(df_gwall)-1,len(df_gwall))
            
            emc=np.linspace(0,len(df_kilo)-1,len(df_kilo))
            gw=list(gw)
            emc=list(emc)
            GE=[]
            N_gw_em=0
            mis=0
            for i in gw:
                GW_EMC=[]
                gwi_emc=[]
                area = df_gwall['area'][i]
                theta = df_gwall['theta'][i]
                varphi = df_gwall['varphi'][i]
                Time=df_gwall['time'][i]
                for j in emc:
                    theta1=df_kilo['theta'][j]
                    varphi1=df_kilo['varphi'][j]
                    angel1=np.arccos(np.cos(theta)*np.cos(theta1)*np.cos(varphi-varphi1)+np.sin(theta)*np.sin(theta1))
                    angel2=np.sqrt(area*(np.pi/180)**2/np.pi)+5*np.pi/180
                    if -2<=(df_kilo['time'][j]-Time)<=10 and angel1<=angel2:
                        gwi_emc.append(j)
                        emc.remove(j)
                if len(gwi_emc)>1:
                    GW_EMC.append(random.choice(gwi_emc))
                elif len(gwi_emc)==1:
                    GW_EMC.append(gwi_emc[0])
                if len(GW_EMC)==1:
                    N_gw_em+=1
                    if df_gwall['z'][i]!=df_kilo['z'][GW_EMC[0]]:
                        mis+=1
                GE.append([i, gwi_emc, GW_EMC])
            MISID.append([N_gw_em, mis, len(gw), len(emc), len(DF_astro_tool)])
            
        COLUMNS=['Ngwemc','N_mis','Ngw', 'Nemc', 'Ngwast'] 
        #INDEX=[ast_em]
        df=pd.DataFrame(MISID,columns=COLUMNS, index=None)
        if pathname:
            df.to_csv(f'{pathname}.csv')
        return df