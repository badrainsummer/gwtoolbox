from astropy import units as u
import numpy as np
from gwtoolbox.sources_kHz import *
from gwtoolbox.detectors_earth import *
from gwtoolbox.functions_earth import *
from gwtoolbox.cosmology import Cosmology
from gwtoolbox.constants import *
from gwtoolbox.parameters import *
import pandas as pd

def set_cosmology(cosmoID=None, H0=None, Om0=None, Tcmb=None):
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
        Parameters:
          detector_type (string): one of 'virgo', 'ligo', 'et', 'kagra', 'ce1','ce2','ligo-like or 'et-like'
          event_type (string): one of 'nsns', 'bhns' or 'bhbh','bg_dns', 'bg_bbh', 'bg_nsbh'
          population (string): either 'I' or 'II'
          cosmos (class): cosmological model
          det_setup (Optional[list of dtype floats]): parameters to modify the detector noise
        """
        self.cosmos = cosmos
        self.detector_type = detector_type
        self.event_type = event_type
        self.population = population
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
        if event_type in ['bg_dns', 'bg_bbh'] and detector_type!='ligo':
            raise ValueError("Currently, Background estimation only valid for LIGO")

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

        ##### deal with population
        if self.generation=='2G':
            if event_type == 'bhbh':
                self.population_class = BHB(self.cosmos)
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
        elif self.generation=='3G':
            if event_type == 'bhbh':
                self.population_class = BHB(self.cosmos)
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

        elif self.generation=='med':
            if event_type == 'bhbh':
                self.population_class = BHB(self.cosmos)
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
        self.ndet = self.population_class.tot_num(time_obs,rho_cri,self.ant_pat,self.noise,self.generation)
        return self.ndet

    def sensitivity_curve(self):
        freq, power=self.noise
        return [freq, np.sqrt(power)]
    def sensitivity_curve_sjoerd(self):
        """
        Return the sensitivity curve of the experiment
        """
        #if not default:
        #    freq, power=self.noise
        #    return [freq, np.sqrt(power)]

        #if not detector_type.endswith('-like'):
        #    raise ValueError('Selected detector is already a default detector')

        detector_class = type(self.detector)
        default_type = self.detector_type[:-5]           # cut the '-like' part
        default_detector = detector_class(default_type)  # instantiate a detector of the default type
        default_noise = default_detector.noise_curve()
        freq, power=default_noise
        return [freq, np.sqrt(power)]
#    def sensitivity_curve(self, default=False):
#        """
#        Return the sensitivity curve of the experiment
#        """
#        if not default:
#            freq, power=self.noise
#            return [freq, np.sqrt(power)]
#
#        if not detector_type.endswith('-like'):
#            raise ValueError('Selected detector is already a default detector')
#
#        detector_class = type(self.detector)
#        default_type = self.detector_type[:-5]           # cut the '-like' part
#        default_detector = detector_class(default_type)  # instantiate a detector of the default type
#        default_noise = default_detector.noise_curve()
#        freq, power=default_noise
#        return [freq, np.sqrt(power)]



    def list_params(self, time_obs=TIME_OBS_EARTH, rho_cri=RHO_CRIT_EARTH, size=None, dtp=False, withangles=False):
        """
        Return an array of final parameters.

        Parameters:
          time_obs (float): Observation time, in unit of minute
          rho_cri (float): The SNR threshold of detection
          size (Optional[int]): sample size

        Returns:
          (list of arrays of dtype float): samples
        """
        if size == None:
            self.number = np.random.poisson(lam=self.population_class.tot_num(time_obs,rho_cri,self.ant_pat,self.noise,self.generation))
        else:
            self.number = size
        #if self.event_type == 'bhbh' and  self.population == 'III':
        #    self.samples = self.population_class.MCMCsample_Gibbs(self.number,self.param_init,self.param_ranges,time_obs,rho_cri,self.ant_pat,self.noise)
        #else:
        self.samples = self.population_class.MCMCsample(self.number,self.param_init,time_obs,rho_cri,self.ant_pat,self.noise)
        if dtp:
            # if dtp is true, withangles must not be true
            z=np.array(self.samples[1,:])
            m1=np.array(self.samples[2,:])
            m2=np.array(self.samples[3,:])
            dtps=self.population_class.tel_fun(z, m1, m2, rho_cri, self.ant_pat, self.noise, accurate=False)
            samples=np.vstack((self.samples,dtps))
            return samples
        elif withangles:
            z=np.array(self.samples[1,:])
            m1=np.array(self.samples[2,:])
            m2=np.array(self.samples[3,:])
            dtps, thetheta, thevarphi, theiota, thepsi, therho2=self.population_class.tel_fun(z, m1, m2, rho_cri, self.ant_pat, self.noise, accurate=False, withangles=withangles)
            samples=np.vstack((self.samples,thetheta, thevarphi, theiota, thepsi, np.sqrt(therho2)))
            return samples
        else:
            return self.samples

    def list_param_errors(self):
        """
        Return array of final parameter errors.
        It can be used only after list_params is used.

        Returns:
          (list of arrays of dtype float): errors m1,m2,chi
        """
        errors = self.population_class.errors_FIM(self.number, self.samples, self.noise)

        return errors
    def list_params_df(self, time_obs=TIME_OBS_EARTH, rho_cri=RHO_CRIT_EARTH, size=None, dtp=False, withangles=False):
        cata=np.transpose(self.list_params(time_obs=time_obs, rho_cri=rho_cri, size=size, dtp=dtp, withangles=withangles))
        if dtp:
            keys=np.array(['z','D','m1','m2','χ','dtb'])
            df=pd.DataFrame(data=cata[:,[1,5,2,3,4,6]], columns=keys)
        elif withangles:
            keys=np.array(['z','D','m1','m2','χ','theta','varphi','inc','psi','rho'])
            df=pd.DataFrame(data=cata[:,[1,5,2,3,4,6,7,8,9,10]], columns=keys)
        else:
            keys=np.array(['z','D','m1','m2','χ'])
            df=pd.DataFrame(data=cata[:,[1,5,2,3,4]], columns=keys)

        return df
    def list_with_errors_df(self, time_obs=TIME_OBS_EARTH, rho_cri=RHO_CRIT_EARTH, size=None, dtp=False, withangles=False):

        cata=np.transpose(self.list_params(time_obs=time_obs, rho_cri=rho_cri, size=size, dtp=dtp, withangles=withangles))
        errors=np.transpose(self.list_param_errors())
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
        data_append=np.column_stack((cata,z_errors,errors,D_errors))
        if dtp:
            keys=np.array(['z','D','m1','m2','χ','dz','dm1','dm2','dχ','dD','dtb'])
            df=pd.DataFrame(data=data_append[:,[1,5,2,3,4,7,8,9,10,11,6]], columns=keys)
        elif withangles:
            keys=np.array(['z','D','m1','m2','χ','theta','varphi','inc','psi','rho','dz','dm1','dm2','dχ','dD'])
            df=pd.DataFrame(data=data_append[:,[1,5,2,3,4,6,7,8,9,10,12,13,14,15,16]], columns=keys)
        else:
            keys=np.array(['z','D','m1','m2','χ','dz','dm1','dm2','dχ','dD'])
            df=pd.DataFrame(data=data_append[:,[1,5,2,3,4,6,7,8,9,10]], columns=keys)
        # propogation of z uncertainties into intrinsic mass, added on 9th-Feb, 2022
        m1z=df['m1']*(1+df['z'])
        m2z=df['m2']*(1+df['z'])
        dlnm1z=df['dm1']/m1z # relative mass error
        dlnm2z=df['dm2']/m2z # relative mass error
        dln1pz=df['dz']/(1+df['z'])
        dm1=df['m1']*np.sqrt(dlnm1z**2+dln1pz**2)
        dm2=df['m2']*np.sqrt(dlnm2z**2+dln1pz**2)
        df['dm1']=dm1
        df['dm2']=dm2
        return df

    def list_with_iota(self, dataframe):
        dtbs=dataframe['dtb']
        det=self.detector
        pop=self.population_class
        iotas=give_iota(det, pop, dtbs)
        dataframe['inc']=iotas
        return dataframe

    def list_with_angles(self, dataframe):
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
        from gwtoolbox.function_earth_fisher import fisher
        my_fisher=fisher(det=self.detector, approximant='IMRPhenomD', cosmos=self.cosmos);
        for key in ['theta', 'varphi', 'inc', 'psi']:
            if not key in dataframe:
                self.list_with_angles(dataframe)
                break;
        new_DF=my_fisher.attach_error_dataframe(dataframe, know_loc=know_loc)
        return new_DF
