# Try to modify the tools_earth object to support networks.
# Changes are marked in comments.

from pkg_resources import resource_filename

import astropy.utils.data as aud
aud.conf.show_progress = False

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
import numpy as np
from gwtoolbox.sources_kHz import *
from gwtoolbox.detectors_earth import *
from gwtoolbox.functions_earth import *
from gwtoolbox.cosmology import Cosmology
from gwtoolbox.constants import *
from gwtoolbox.parameters import *
from gwtoolbox.HE_detectors import *
import pandas as pd
from scipy import integrate
from itertools import accumulate
from astropy.cosmology import z_at_value
import warnings

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
    _DETECTOR_INF = {
        "virgo": ["2G", LigoLike('virgo'), EarthLocation.of_site('Virgo')], 
        "kagra": ["2G", LigoLike('kagra'), EarthLocation.of_site('KAGRA')], 
        "ligo": [],
        "hanford": ["2G", LigoLike('ligo'), EarthLocation.of_site('LHO_4K')], 
        "livingston": ["2G", LigoLike('ligo'), EarthLocation.of_site('LLO_4K')], 
        "ligo-o3": [],
        "hanford-o3": ["2G", LigoLike('ligo-o3'), EarthLocation.of_site('LHO_4K')], 
        "livingston-o3": ["2G", LigoLike('logo-o3'), EarthLocation.of_site('LLO_4K')], 
        "ligo-o4": [],
        "hanford-o4": ["2G", LigoLike('ligo-o4'), EarthLocation.of_site('LHO_4K')], 
        "livingston-o4": ["2G", LigoLike('logo-o4'), EarthLocation.of_site('LLO_4K')], 
        "ligo-like": ["2G", LigoLike('ligo-like'), EarthLocation.from_geodetic(lon=0., lat=0., height=0.)],
        "ce1": ["3G", LigoLike('ce1'), EarthLocation.from_geodetic(lon=-98.0, lat=45.0, height=0.)], 
        "ce2": ["3G", LigoLike('ce2'), EarthLocation.from_geodetic(lon=-106.0, lat=32.0, height=0.)], 
        "et": ["3G", ETLike('et'), EarthLocation.from_geodetic(lon=5.99, lat=50.63, height=250)], 
        "et-like": ["3G", ETLike('et-like'), EarthLocation.from_geodetic(lon=0., lat=0., height=0.)]
    }
    _DEFAULT_DET_SETUP = {
        "ligo": {"LARM_VALUE":3995,"PIN_VALUE":125,"ITMT_VALUE":0.014,"SRMT_VALUE":0.2,"PRMT_VALUE":0.03,"MTM_VALUE":40,"SRMPHI_VALUE":-90,"LPRC_VALUE":53,"LSRC_VALUE":50.525},
        "et": {"LARM_VALUE":10000,"PIN_VALUE":500,"ITMT_VALUE":0.014,"SRMT_VALUE":0.2,"PRMT_VALUE":0.03,"MTM_VALUE":200,"SRMPHI_VALUE":-90,"LPRC_VALUE":53,"LSRC_VALUE":50.525}
    }
    _DEFAULT_NEW_THETA = {
        "bhbhI": {'R0': 13, 'tau': 3, 'mu': 3, 'c': 6, 'gamma': 2.5, 'mcut': 95, 'ql': 0.4, 'sig_x': 0.1},
        "bhbhII": { 'R0': 13, 'tau': 3, 'mu': 3, 'c': 6, 'gamma': 2.5, 'mcut': 95, 'm_peak': 40, 'm_peak_scale': .002, 'm_peak_sig': 1, 'ql': 0.4, 'sig_x': 0.1},
        "nsnsI": {'R0': 300, 'tau': 3, 'm_mean': 1.4, 'm_scale': 0.5, 'm_low': 1.1, 'm_high': 2.5, 'chi_sigma': 0.1},
        "bhnsI": {'R0': 45, 'tau': 3, 'm_mean': 1.4, 'm_scale': 0.5, 'm_low': 1.1, 'm_high': 2.5, 'mu': 3, 'c': 15, 'gamma': 2.5, 'mcut': 95, 'chi_sigma': 0.1},
        "bhnsII": {'R0': 5, 'tau': 3, 'm_mean': 1.4, 'm_sclae': 0.5, 'm_low': 1.1, 'm_high': 2.5, 'mu': 3, 'c': 15, 'gamma': 2.5, 'mcut': 95, 'm_peak': 40, 'm_peak_scale': 0.02, 'm_peak_sig': 0.4, 'chi_sigma': 0.1}
    }
    _DEFAULT_GRB_PARAMETER = {"L_star": 2e52, "theta_mean": 5*np.pi/180, "theta_std": 2*np.pi/180}

    def __init__(self, detector_type, event_type, population):
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
        self._init_detector_type(detector_type)
        self._event_type = event_type
        self._population = population
        self._scale = 1

        # Default cosmology
        self._cosmoID = "Planck18"
        self._cosmo_params = {}
        self._update_cosmos()

        # Initialize containers
        self._init_detectors()

        # Initialize population
        self._init_population()

    # ----------- Properties -----------

    @property
    def cosmos(self): return self._cosmos

    @cosmos.setter
    def cosmos(self, value):
        if isinstance(value, astropy.cosmology.core.Cosmology):
            self._cosmos = value
        else:
            raise ValueError('cosmos must be gwtoolbox.cosmology type.')

    @property
    def cosmoID(self): return self._cosmoID

    @cosmoID.setter
    def cosmoID(self, value):
        self._cosmoID = value
        self._update_cosmos()

    @property
    def cosmo_params(self): return self._cosmo_params

    @cosmo_params.setter
    def cosmo_params(self, value):
        if not isinstance(value, dict):
            raise TypeError("cosmo_params must be a dict, e.g. {'H0':70, 'Om0':0.3, 'Tcmb':2.725}")
        self._cosmo_params = value
        self._update_cosmos()

    @property
    def network(self): return len(self.detector_type) > 1

    @property
    def scale(self): return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self._update_generation()

    @property
    def new_theta(self): return self._new_theta

    @new_theta.setter
    def new_theta(self, value):
        if not isinstance(value, dict):
            raise TypeError("new_theta must be a dictionary.")
        value = self._remove_merge(value, self._DEFAULT_NEW_THETA[self.event_type+self.population])
        self._new_theta = value
        self._update_population()

    @property
    def detector_type(self): 
        return self._detector_type

    @detector_type.setter
    def detector_type(self, value):
        self._init_detector_type(value)
        self._init_detectors()
        self._init_population()

    @property
    def event_type(self):
        return self._event_type

    @event_type.setter
    def event_type(self, value):
        self._event_type = value
        self._init_population()

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value):
        self._population = value
        self._init_population()

    # ----------- Helpers -----------

    def _init_detector_type(self, detector_type):
        detector_type = detector_type if isinstance(detector_type, list) else [detector_type]
        i = 0
        while i < len(detector_type):
            if detector_type[i] not in self._DETECTOR_INF.keys():
                raise ValueError(f"{detector_type[i]} is not aviable detector type")
            if detector_type[i] in ['ligo', 'ligo-o3', 'ligo-o4']:
                run_version = detector_type[i].replace('ligo', '')
                detector_type.pop(i)
                detector_type[i:i] = ['hanford'+run_version, 'livingston'+run_version]
                i += 2
            else:
                i += 1
        self._detector_type = detector_type

    def _update_cosmos(self):
        cosmos_class = Cosmology()
        self._cosmos = cosmos_class.set_cosmo(
            self._cosmoID,
            self._cosmo_params.get("H0"),
            self._cosmo_params.get("Om0"),
            self._cosmo_params.get("Tcmb"),
        )

    def _update_detector_noise(self, index=None):
        if not index:
            index = np.arange(len(self.detector))
        for i in index:
            self.noise[i] = self.detector[i].noise_curve(self.det_setup[i])

    def _update_generation(self, index=None):
        if not index:
            index = np.arange(len(self.detector))
        for i in index:
            self.noise[i][1] *= self.scale
            minsens = np.sqrt(min(self.noise[i][1]))

            if minsens > 2e-24:
                self.generation[i] = "2G"
            elif 1e-24 < minsens <= 2e-24:
                self.generation[i] = "med"
            else:
                self.generation[i] = "3G"

    def _update_population(self):
        new_theta_order = self._get_order(self._new_theta)
        self.population_class.set_model_theta(pop=self.population, new_theta=new_theta_order)

    def _get_order(self, new_theta):
        """Reorder new_theta to match the expected order."""
        if self.event_type == "bhbh":
            if self.population == "I":
                return [new_theta['R0'], new_theta['tau'], new_theta['mu'], new_theta['c'], new_theta['gamma'], new_theta['mcut'], new_theta['ql'], new_theta['sig_x']]
            elif self.population == "II":
                return [new_theta['R0'], new_theta['tau'], new_theta['mu'], new_theta['c'], new_theta['gamma'], new_theta['mcut'], new_theta['m_peak'], new_theta['m_peak_scale'], new_theta['m_peak_sig'], new_theta['ql'], new_theta['sig_x']]
        elif self.event_type == "nsns":
            return [new_theta['R0'], new_theta['tau'], new_theta['m_mean'], new_theta['m_scale'], new_theta['m_low'], new_theta['m_high'], new_theta['chi_sigma']]
        elif self.event_type == "bhns":
            if self.population == "I":
                return [new_theta['R0'], new_theta['tau'], new_theta['m_mean'], new_theta['m_scale'], new_theta['m_low'], new_theta['m_high'], new_theta['mu'], new_theta['c'], new_theta['gamma'], new_theta['mcut'], new_theta['chi_sigma']]
            elif self.population == "II":
                return [new_theta['R0'], new_theta['tau'], new_theta['m_mean'], new_theta['m_scale'], new_theta['m_low'], new_theta['m_high'], new_theta['mu'], new_theta['c'], new_theta['gamma'], new_theta['mcut'], new_theta['m_peak'], new_theta['m_peak_scale'], new_theta['m_peak_sig'], new_theta['chi_sigma']]
        else:
            raise ValueError(f"Unsupported event type: {self.event_type}")

    def _remove_merge(self, value, defaults):
        """Remove keys not in DEFAULT_DET_SETUP and merge the dict."""
        valid_keys = set(defaults.keys())
        invalid_keys = [k for k in value.keys() if k not in valid_keys]

        if invalid_keys:
            for k in invalid_keys:
                warnings.warn(f"Invalid key '{k}' found in det_setup. It will be ignored.")
                del value[k]

        merged_value = defaults.copy()
        merged_value.update(value)

        return merged_value

    def _check_det_setup(self, det_setup):
        """Check detector setup parameters."""
        checks = [
            (det_setup["LARM_VALUE"] > 0, "Laser armlength must be positive."),
            (det_setup["PIN_VALUE"] > 0, "Laser power must be positive."),
            (0 < det_setup["ITMT_VALUE"] < 1, "Cavity mirror transmission must be between 0 and 1."),
            (0 < det_setup["SRMT_VALUE"] < 1, "Signal recycling mirror transmission must be between 0 and 1."),
            (0 < det_setup["PRMT_VALUE"] < 1, "Power recycling mirror transmission must be between 0 and 1."),
            (det_setup["MTM_VALUE"] > 0, "Mirror mass must be positive."),
            (det_setup["LPRC_VALUE"] > 0, "Power recycling length must be positive."),
            (det_setup["LSRC_VALUE"] > 0, "Signal recycling length must be positive."),
        ]
        for cond, msg in checks:
            if not cond:
                raise ValueError(msg)

    def _init_population(self):
        """Initialize population class depending on generation and event type."""
        gen_type = "2G" if "2G" in self.generation else "3G" if "3G" in self.generation else "med"

        if gen_type == "2G":
            init_map = {
                "bhbh": ([0.25, 10., 10., 0.], BHB(self.cosmos), self.population),
                "nsns": ([0.01, 1.4, 1.4, 0.], DNS(self.cosmos), None),
                "bhns": ([0.01, 10., 1., 0.], BHNS(self.cosmos), self.population),
                "bg_dns": ([0.3, 1.4, 1.4, 0.], Background(self.cosmos), "DNS"),
                "bg_bbh": ([0.25, 10., 10., 0.], Background(self.cosmos), "BHB"),
            }
        elif gen_type == "3G":
            init_map = {
                "bhbh": ([2., 10., 10., 0.], BHB(self.cosmos), self.population),
                "nsns": ([2., 1., 1., 0.], DNS(self.cosmos), None),
                "bhns": ([2., 5., 1., 0.], BHNS(self.cosmos), self.population),
                "bg_dns": ([2., 1., 1., 0.], Background(self.cosmos), "DNS"),
                "bg_bbh": ([2., 10., 10., 0.], Background(self.cosmos), "BHB"),
            }
        else:  # med
            init_map = {
                "bhbh": ([0.8, 10., 10., 0.], BHB(self.cosmos), self.population),
                "nsns": ([0.14, 1., 1., 0.], DNS(self.cosmos), None),
                "bhns": ([0.4, 5., 1., 0.], BHNS(self.cosmos), self.population),
                "bg_dns": ([0.14, 1., 1., 0.], Background(self.cosmos), "DNS"),
                "bg_bbh": ([0.8, 10., 10., 0.], Background(self.cosmos), "BHB"),
            }

        if self.event_type not in init_map:
            raise ValueError(f"Unsupported event type: {self.event_type}")

        self.param_init, self.population_class, pop_arg = init_map[self.event_type]
        if pop_arg is None:
            self.population_class.set_model_theta()
        else:
            self.population_class.set_model_theta(pop=pop_arg)

    def _transform_location(self, location):
        """Initialize location"""
        return m_to_ls(EL_to_np(location))

    def _init_detectors(self):
        """Initialize detectors based on detector_type"""
        n_det = len(self.detector_type)
        self.detector = [None] * n_det
        self.ant_pat = [None] * n_det
        self.noise = [None] * n_det
        self.generation = [None] * n_det
        self.location = [None] * n_det
        self.det_setup = [None] * n_det

        for i, det in enumerate(self.detector_type):
            self.detector[i] = self._DETECTOR_INF[det][1]
            self.ant_pat[i] = self.detector[i].ante_pattern
            self.noise[i] = self.detector[i].noise_curve()
            self.generation[i] = self._DETECTOR_INF[det][0]
            self.location[i] = self._transform_location(self._DETECTOR_INF[det][2])
            if '-like' in det:
                self.det_setup[i] = self._DEFAULT_DET_SETUP[det[:-5]]

    def _get_HE_detector_parameters(self, det_type):
        """Get HE detector information"""
        if det_type in HE_detectors.keys():
            E_min, E_max = HE_detectors[det_type]["Erange"]
            sens, FoV, correction = HE_detectors[det_type]["sens"], HE_detectors[det_type]["FoV"], HE_detectors[det_type]["corr"]
            E_avg = np.log(E_max / E_min) / (1 / E_min - 1 / E_max)
            E_avg_erg = E_avg * 1e6 * u.eV.to(u.erg)
            flim = sens * E_avg_erg  # erg/cm^2/s
            lam = FoV * correction
            return [flim, lam]
        else:
            raise ValueError(f"Unkown detector type {det_type}")

    # ----------- Functions -----------

    def update_detector(self, det_setup=None, location=None, index=None):
        """
        Update the configuration and/or location of specified detectors.

        Parameters:
            det_setup (list, optional): List of new detector setups.
            location (list, optional): List of new detector locations.
            index (list, optional): List of detector indices to update. 
                                    If None, updates all definable detectors in order.

        Functionality:
            - Validates input types and lengths.
            - Updates specified detectors, or all definable detectors if index is not provided.
            - Merges new detector setup with default configuration and validates it.
            - Transforms location if provided.
        """
        if det_setup is None:
            det_setup = []
        if location is None:
            location = []
        if index is None:
            index = []

        definable_detectors = [i for i, dtype in enumerate(self.detector_type) if '-like' in dtype]
        max_detectors = len(definable_detectors)

        for name, value in {"det_setup": det_setup, "location": location, "index": index}.items():
            if not isinstance(value, list):
                raise TypeError(f"{name} must be a list, got {type(value).__name__}")
            if len(value) > max_detectors:
                raise ValueError(f"Only {max_detectors} definable detectors, but got {len(value)} in {name}")

        if not index:
            index = definable_detectors

        if not set(index).issubset(set(definable_detectors)):
            raise ValueError(f"The index of definable detectors is {definable_detectors}, please provide index within this range.")

        param_lengths = set(len(lst) for lst in [det_setup, location] if lst)
        if param_lengths and param_lengths != {len(index)}:
            raise ValueError(f"The length of det_setup/location must match the length of index ({len(index)}).")

        for i, det_index in enumerate(index):
            if det_setup:
                det = self._remove_merge(det_setup[i], self._DEFAULT_DET_SETUP[self.detector_type[det_index][:-5]])
                self._check_det_setup(det)
                self.det_setup[det_index] = det
                self._update_detector_noise(index=[det_index])
                self._update_generation(index=[det_index])

            if location:
                self.location[det_index] = self._transform_location(location[i])
                    

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
        if self.network:
            self.ndet = self.population_class.tot_num(time_obs,rho_cri,self.ant_pat,self.noise,self.generation,self.detector_type)
        else:
            self.ndet = self.population_class.tot_num(time_obs, rho_cri, self.ant_pat[0], self.noise[0], self.generation[0], self.detector_type[0])
        return self.ndet
    
    def sensitivity_curve(self, use_default=False):
        """
        Return the sensitivity curve(s) (frequency, sqrt(power)) of the detector(s).

        Parameters:
            use_default (bool): If True, returns the sensitivity curve using default detector settings.
                                Only valid for single detectors. Networks will ignore this flag.
        
        Returns:
            list or tuple: For a network, returns a list of [freq, sqrt(power)] for each detector.
                        For a single detector, returns [freq, sqrt(power)].
        """
        if self.network:
            # For a network, always use current detector noise
            sen_curves = []
            for noise in self.noise:
                freq, power = noise
                sen_curves.append([freq, np.sqrt(power)])
            return sen_curves

        # Single detector
        det = self.detector[0]
        if use_default:
            # Use default settings
            detector_class = type(det)
            default_type = self.detector_type[0].replace('-o3', '').replace('-like', '')
            det = detector_class(default_type)
            freq, power = det.noise_curve()
        else:
            freq, power = self.noise[0]

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
            self.number = np.random.poisson(lam=self.total_number(time_obs,rho_cri))
        else:
            self.number = size
        if self.network:
            samples = self.population_class.MCMCsample(self.number,self.param_init,time_obs,rho_cri,self.ant_pat,self.noise, name=self.detector_type)
        else:
            samples = self.population_class.MCMCsample(self.number,self.param_init,time_obs,rho_cri,self.ant_pat[0],self.noise[0], name=self.detector_type[0])
        if withangles:
            if not self.network:
                z=np.array(samples[1,:])
                m1=np.array(samples[2,:])
                m2=np.array(samples[3,:])
                dtps, thetheta, thevarphi, theiota, thepsi, therho2=self.population_class.tel_fun(z, m1, m2, rho_cri, self.ant_pat[0], self.noise[0], accurate=False, withangles=withangles)
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
        elif dtp:
            z=np.array(samples[1,:])
            m1=np.array(samples[2,:])
            m2=np.array(samples[3,:])
            dtps=self.population_class.tel_fun(z, m1, m2, rho_cri, self.ant_pat, self.noise, accurate=False, name=self.detector_type, withangles=False)
            samples=np.vstack((samples,dtps))
        if witherrors:
            errors = self.population_class.errors_FIM(self.number, samples, self.noise[0])
            samples=np.vstack((samples,errors))
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
    
    def snr(self, m1, m2, chi, z=None, D=None, theta=0, varphi=0, psi=np.pi/2, iota=np.pi/2, time=Time("2020-01-01 00:00:00", scale="utc")):
        """
        Compute the SNR for given parameters.

        Parameters:
            m1 (float): Mass of the first object.
            m2 (float): Mass of the second object.
            chi (float): Spin parameter.
            z (float, optional): Redshift. If not provided, D must be given.
            D (float, optional): Luminosity distance in Mpc. If not provided, z must be given.
            theta (float): Polar angle.
            varphi (float): Azimuthal angle.
            psi (float): Polarization angle.
            iota (float): Inclination angle.

        Returns:
            float: SNR values.
        """
        if not z:
            if not D:
                raise ValueError("Luminosity distance D must be provided if redshift z is not given.")
            z = redshift_from_luminosity_distance(D*u.Mpc,cosmos=self.cosmos)
        return self.population_class.snr(z, m1, m2, chi, theta, varphi, psi, iota, time, self.ant_pat, self.location, self.noise)
    
    def snr_area(self, m1, m2, chi, z=None, D=None, theta=0, varphi=0, psi=np.pi/2, iota=np.pi/2, time=Time("2020-01-01 00:00:00", scale="utc")):
        """
        Compute the SNR and area for given parameters.
        Parameters:
            m1 (float): Mass of the first object.
            m2 (float): Mass of the second object.
            chi (float): Spin parameter.
            z (float, optional): Redshift. If not provided, D must be given.
            D (float, optional): Luminosity distance in Mpc. If not provided, z must be given.
            theta (float): Polar angle.
            varphi (float): Azimuthal angle.
            psi (float): Polarization angle.
            iota (float): Inclination angle.

        Returns:
            tuple: SNR and area.
        """
        if not self.network:
            raise ValueError("snr_area is only available for networks.")
        if not z:
            if not D:
                raise ValueError("Luminosity distance D must be provided if redshift z is not given.")
            z = redshift_from_luminosity_distance(D*u.Mpc,cosmos=self.cosmos)
        return self.population_class.area_snr(z, m1, m2, chi, theta, varphi, psi, iota, time, self.ant_pat, self.location, self.noise)

    # ----------- Catalog Helpers -----------

    def _sample_angles(self, N):
        """Sample random angles for N events."""
        theta = np.arccos(np.random.uniform(-1, 1, size=N))
        varphi = np.random.uniform(0, 2*np.pi, size=N)
        psi = np.random.uniform(0, np.pi, size=N)
        iota = np.arccos(np.random.uniform(0, 1, size=N))
        return theta, varphi, psi, iota

    def _catalog_dataframe(self, columns, data, pathname=None):
        """Create and optionally save a DataFrame."""
        df = pd.DataFrame(data, index=columns).T
        if pathname:
            df.to_csv(f'{pathname}.csv', index=False)
        return df

    def _snr_and_area(self, m1, m2, chi, z, theta, varphi, psi, iota):
        """Compute SNR and area for a single event."""
        if self.network:
            snr, area = self.snr_area(m1, m2, chi, z, None, theta, varphi, psi, iota)
            return snr, area
        else:
            snr = self.snr(m1, m2, chi, z, None, theta, varphi, psi, iota)
            return snr, None

    # ----------- Catalog Functions -----------

    def create_BNScatalog(self, T=525600, N=None, pathname=None):
        """Create a BNS (Binary Neutron Star) catalog."""
        Ti = T / 525600
        popu = self.population_class.theta
        R0 = popu[0]
        def rm(z):
            return (
                4 * np.pi * Ti * R0 *
                R(z, 2.7, 5.6, 2.9, tau=popu[1], cosmos=self.cosmos) *
                self.cosmos.differential_comoving_volume(z).to_value(u.Gpc**3/u.sr) /
                (1 + z)
            )
        if N is None:
            result, _ = integrate.quad(rm, 0, 3, epsabs=1e-1)
            N = int(result)
        low, loc, high, scal = popu[4], popu[2], popu[5], popu[3]
        aNS, bNS = (low - loc) / scal, (high - loc) / scal
        pm1 = truncnorm.rvs(aNS, bNS, loc=loc, scale=scal, size=N)
        pm2 = truncnorm.rvs(aNS, bNS, loc=loc, scale=scal, size=N)
        zs = np.linspace(1e-3, 3, 200)
        RM = list(accumulate(rm(zs)))
        def inverse_cdf(x): 
            return np.interp(x, np.array(RM), zs)
        PZ = inverse_cdf(np.random.uniform(0, RM[-1], size=N))
        theta, varphi, psi, iota = self._sample_angles(N)
        DIS = self.cosmos.luminosity_distance(PZ).value
        Time = np.random.randint(0, T, size=N)
        rho, area = [], []
        for i in range(N):
            snr, ar = self._snr_and_area(pm1[i], pm2[i], 0, PZ[i], theta[i], varphi[i], psi[i], iota[i])
            rho.append(snr)
            if ar is not None:
                area.append(ar)
        rho = np.array(rho)
        dD = DIS / rho
        data_list = [PZ, DIS, pm1, pm2, rho, theta, varphi, iota, psi]
        column_names = ['z', 'D', 'm1', 'm2', 'rho', 'theta', 'varphi', 'inc', 'psi']
        if self.network:
            data_list += [area, dD, Time]
            column_names += ['area', 'dD', 'time']
        else:
            data_list += [dD, Time]
            column_names += ['dD', 'time']
        df_BNS = self._catalog_dataframe(column_names, data_list, pathname)
        self.BNScatalog = df_BNS
        return df_BNS

    def create_backgroundcatalog(self, rho=8, T=525600, pathname=None):
        """Create a background catalog."""
        FAR8 = 30000
        cri_dis = 2493.8699694225065
        alpha = 0.13
        bg_number = int(T / 525600 * FAR8 * np.exp((8 - rho) / alpha))
        Mch = np.random.uniform(0.435, 2.61, bg_number)
        Time = np.random.randint(0, T, size=bg_number)
        theta, varphi, psi, iota = self._sample_angles(bg_number)
        m1 = Mch * 2.0**0.2
        m2 = m1
        DIS, area_list = [], []
        z_val = z_at_value(self.cosmos.luminosity_distance, cri_dis * u.Mpc).value
        for i in range(bg_number):
            snr, ar = self._snr_and_area(m1[i], m2[i], 0, z_val, theta[i], varphi[i], psi[i], iota[i])
            if ar is not None:
                area_list.append(ar)
            DIS.append(cri_dis * snr / rho)
        DIS = np.array(DIS)
        Z = z_at_value(self.cosmos.luminosity_distance, DIS * u.Mpc).value
        dD = DIS / rho
        data_list = [Z, DIS, m1, m2, theta, varphi, iota, psi]
        column_names = ['z', 'D', 'm1', 'm2', 'theta', 'varphi', 'inc', 'psi']
        if self.network:
            data_list += [area_list, dD, Time]
            column_names += ['area', 'dD', 'time']
        else:
            data_list += [dD, Time]
            column_names += ['dD', 'time']
        return self._catalog_dataframe(column_names, data_list, pathname)

    def create_GWcatalog(self, rho_cri=8, pathname=None):
        """Create a GW catalog from BNS catalog with SNR cut."""
        df_BNS = self.BNScatalog[self.BNScatalog['z'] <= 0.4].reset_index(drop=True)
        N = len(df_BNS)
        z_list, D_list, m1_list, m2_list, rho_list = [], [], [], [], []
        theta_list, varphi_list, inc_list, psi_list = [], [], [], []
        area_list, dD_list, time_list = [], [], []
        for i in range(N):
            snr, ar = self._snr_and_area(
                df_BNS['m1'][i], df_BNS['m2'][i], 0, df_BNS['z'][i],
                df_BNS['theta'][i], df_BNS['varphi'][i], df_BNS['psi'][i], df_BNS['inc'][i]
            )
            if snr >= rho_cri:
                z_list.append(df_BNS['z'][i])
                D_list.append(df_BNS['D'][i])
                m1_list.append(df_BNS['m1'][i])
                m2_list.append(df_BNS['m2'][i])
                rho_list.append(snr)
                theta_list.append(df_BNS['theta'][i])
                varphi_list.append(df_BNS['varphi'][i])
                inc_list.append(df_BNS['inc'][i])
                psi_list.append(df_BNS['psi'][i])
                dD_list.append(df_BNS['D'][i] / snr)
                time_list.append(df_BNS['time'][i])
                if ar is not None:
                    area_list.append(ar)
        data_list = [z_list, D_list, m1_list, m2_list, rho_list, theta_list, varphi_list, inc_list, psi_list]
        column_names = ['z', 'D', 'm1', 'm2', 'rho', 'theta', 'varphi', 'inc', 'psi']
        if self.network:
            data_list += [area_list, dD_list, time_list]
            column_names += ['area', 'dD', 'time']
        else:
            data_list += [dD_list, time_list]
            column_names += ['dD', 'time']
        return self._catalog_dataframe(column_names, data_list, pathname)

    def create_kilonovacatalog(self, z1=0.036, z2=0.022, pathname=None):
        """Create a kilonova catalog."""
        df_BNS = self.BNScatalog[self.BNScatalog['z'] <= 0.4].reset_index(drop=True)
        N = len(df_BNS)
        def Pdet_kilo(z):
            return np.where(z < z1, 1.0, np.exp(-0.5 * ((z - z1) / z2) ** 2))
        probs = Pdet_kilo(df_BNS['z'].values)
        detected_mask = probs >= np.random.uniform(0, 1, size=N)
        df_kilo = df_BNS.loc[detected_mask, ['z', 'D', 'm1', 'm2', 'theta', 'varphi', 'inc', 'psi', 'time']].copy()
        if pathname:
            df_kilo.to_csv(f'{pathname}.csv', index=False)
        return df_kilo

    def create_sgrbcatalog(self, HE_detector='Fermi_GBM', GRB_parameter={"L_star": 2e52, "theta_mean": 5*np.pi/180, "theta_std": 2*np.pi/180}, pathname=None):
        """Create an sGRB catalog."""
        if isinstance(HE_detector, str):
            flim, lam = self._get_HE_detector_parameters[HE_detector]
        elif isinstance(HE_detector, dict):
            HE_detector_parameter = self._remove_merge(GRB_parameter, {"flim": 2e-7, "lam": 0.6})
            flim, lam = HE_detector_parameter["flim"], HE_detector_parameter["flim"]
        else:
            raise TypeError(f"HE_detector should rather be name of detector or a dict of detector parameter, but now is {type(HE_detector).__name__}")
        GRB_parameter = self._remove_merge(GRB_parameter, {"L_star": 2e52, "theta_mean": 5*np.pi/180, "theta_std": 2*np.pi/180})
        L_star, mu, sigma = GRB_parameter["L_star"], GRB_parameter["theta_mean"], GRB_parameter["theta_std"]
        df_BNS = self.BNScatalog
        N = len(df_BNS)
        lower, upper = 0, np.pi/2
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        thetacs_array = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=N)
        def Pdet_sgrb(z, thetav, thetac):
            d_c = self.cosmos.comoving_distance(z=z).to_value(u.cm)
            L_lim = flim * (4 * np.pi * d_c)**2 * np.exp(thetav**2 / thetac**2)
            if L_star > L_lim:
                pval = np.log(L_star / L_lim) + 1
            else:
                pval = L_star / L_lim
            return pval / (15.5 - np.log(2e-7 / flim)) * lam * 20
        probs = np.array([Pdet_sgrb(z, inc, thetac) for z, inc, thetac in zip(df_BNS['z'], df_BNS['inc'], thetacs_array)])
        detected_mask = probs >= np.random.uniform(0, 1, size=N)
        df_detected = df_BNS.loc[detected_mask].copy()
        df_detected['thetac'] = thetacs_array[detected_mask]
        df_sgrb = df_detected[['z', 'D', 'm1', 'm2', 'theta', 'varphi', 'inc', 'psi', 'time', 'thetac']]
        if pathname:
            df_sgrb.to_csv(f'{pathname}.csv', index=False)
        return df_sgrb

    # ----------- Coincidence Functions -----------

    def kilonovae(self, pathname=None):
        """Simulate GW-EM coincidence for kilonovae."""
        if not self.network:
            raise ValueError('Only network detectors can perform GW-EM coincidence.')
        MISID = []
        rho_list = np.linspace(8.0, 9.3, 14)
        df_kilo = self.create_kilonovacatalog(z1=0.067, z2=0.056)
        EM_theta = df_kilo['theta'].values
        EM_varphi = df_kilo['varphi'].values
        EM_D = df_kilo['D'].values
        EM_time = df_kilo['time'].values
        for rho in rho_list:
            df_gw = self.create_GWcatalog(rho)
            df_back = self.create_backgroundcatalog(rho)
            df_gwall = pd.concat([df_gw, df_back]).reset_index(drop=True)
            GW_theta = df_gwall['theta'].values
            GW_varphi = df_gwall['varphi'].values
            GW_D = df_gwall['D'].values
            GW_dD = df_gwall['dD'].values
            GW_time = df_gwall['time'].values
            GW_area = df_gwall['area'].values
            GW_z = df_gwall['z'].values
            N_gw_em = 0
            mis = 0
            em_matched_mask = np.zeros(len(df_kilo), dtype=bool)
            for i in range(len(df_gwall)):
                D1, D2 = GW_D[i] - GW_dD[i], GW_D[i] + GW_dD[i]
                area_thresh = np.sqrt(GW_area[i] * (np.pi / 180)**2 / np.pi)
                varphi_diff = GW_varphi[i] - EM_varphi
                cos_ang = np.cos(GW_theta[i]) * np.cos(EM_theta) + np.sin(GW_theta[i]) * np.sin(EM_theta) * np.cos(varphi_diff)
                ang_sep = np.arccos(np.clip(cos_ang, -1.0, 1.0))
                time_diff = EM_time - GW_time[i]
                condition = (~em_matched_mask) & (EM_D >= D1) & (EM_D <= D2) & (time_diff >= 0) & (time_diff <= 10080) & (ang_sep <= area_thresh)
                matched_indices = np.where(condition)[0]
                if len(matched_indices) > 0:
                    selected_em = np.random.choice(matched_indices)
                    em_matched_mask[selected_em] = True
                    N_gw_em += 1
                    if GW_z[i] != df_kilo['z'].iloc[selected_em]:
                        mis += 1
            MISID.append([N_gw_em, mis, len(df_gwall), np.sum(~em_matched_mask), len(df_gw)])
        COLUMNS = ['Ngwemc', 'N_mis', 'Ngw', 'Nemc', 'Ngwast']
        df_result = pd.DataFrame(MISID, columns=COLUMNS)
        if pathname:
            df_result.to_csv(f'{pathname}.csv', index=False)
        return df_result

    def sgrb(self, pathname=None):
        """Simulate GW-EM coincidence for sGRB."""
        if not self.network:
            raise ValueError('Only network detectors can perform GW-EM coincidence.')
        MISID = []
        rho_list = np.linspace(8.0, 9.3, 14)
        df_sgrb = self.create_sgrbcatalog(flim=2e-8, lam=1)
        EM_theta = df_sgrb['theta'].values
        EM_varphi = df_sgrb['varphi'].values
        EM_time = df_sgrb['time'].values
        for rho in rho_list:
            df_gw = self.create_GWcatalog(rho)
            df_back = self.create_backgroundcatalog(rho)
            df_gwall = pd.concat([df_gw, df_back]).reset_index(drop=True)
            GW_theta = df_gwall['theta'].values
            GW_varphi = df_gwall['varphi'].values
            GW_time = df_gwall['time'].values
            GW_area = df_gwall['area'].values
            GW_z = df_gwall['z'].values
            N_gw_em = 0
            mis = 0
            em_matched_mask = np.zeros(len(df_sgrb), dtype=bool)
            for i in range(len(df_gwall)):
                area_thresh = np.sqrt(GW_area[i] * (np.pi / 180)**2 / np.pi) + 5 * np.pi / 180
                time_diff = EM_time - GW_time[i]
                cos_ang = np.cos(GW_theta[i]) * np.cos(EM_theta) + np.sin(GW_theta[i]) * np.sin(EM_theta) * np.cos(GW_varphi[i] - EM_varphi)
                ang_sep = np.arccos(np.clip(cos_ang, -1.0, 1.0))
                condition = (~em_matched_mask) & (time_diff >= -2) & (time_diff <= 10) & (ang_sep <= area_thresh)
                matched_indices = np.where(condition)[0]
                if len(matched_indices) > 0:
                    selected_em = np.random.choice(matched_indices)
                    em_matched_mask[selected_em] = True
                    N_gw_em += 1
                    if GW_z[i] != df_sgrb['z'].iloc[selected_em]:
                        mis += 1
            MISID.append([N_gw_em, mis, len(df_gwall), np.sum(~em_matched_mask), len(df_gw)])
        COLUMNS = ['Ngwemc', 'N_mis', 'Ngw', 'Nemc', 'Ngwast']
        df_result = pd.DataFrame(MISID, columns=COLUMNS)
        if pathname:
            df_result.to_csv(f'{pathname}.csv', index=False)
        return df_result
