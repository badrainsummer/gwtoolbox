from .constants import H0_1,Om0_1
from .parameters import cosmology_dict
import astropy.cosmology


class Cosmology:
  """
  A class to define cosmological model.
  """
  
  def __init__(self):
     self.cosmos = astropy.cosmology.FlatLambdaCDM(H0=H0_1, Om0=Om0_1)
  
  def set_cosmo(self, cosmoID=None, H0=None, Om0=None, Tcmb=None):
    """
    Set cosmological model.
    
    Parameters:
      cosmoID (Optional[string]): name of cosmological model, if None function set the initial model
      H0 (Optional[float]): value of Hubble constant
      Om0 (Optional[float]): value of Omega matter
    
    Returns:
      (class): cosmological model
    """
    if cosmoID == None or cosmoID == 'FlatLambdaCDM':
      self.cosmos = astropy.cosmology.FlatLambdaCDM(H0=H0, Om0=Om0,Tcmb0=Tcmb)
    #elif cosmoID == 'myownLCDM':
    #  self.cosmos = astropy.cosmology.FlatLambdaCDM(H0=H0, Om0=Om0,)
    else:
      self.cosmos = cosmology_dict[cosmoID]
    return self.cosmos
    

