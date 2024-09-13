import numpy as np 
import math
from gwtoolbox.sources_mHz import *
from gwtoolbox.detectors_space import *
from gwtoolbox.cosmology import Cosmology
from gwtoolbox.constants import *
from gwtoolbox.parameters import *

def set_cosmology(cosmoID=None, H0=None, Om0=None,Tcmb=None):
    cosmos_class = Cosmology()
    cosmos = cosmos_class.set_cosmo(cosmoID,H0,Om0,Tcmb)
    return cosmos

class Tools:
    def __init__(self,Tobs,popID, cosmos, det_setup=[8.3391,2.0,0.3]):
        if len(det_setup)==3:
            lisaLT, lisaP, lisaD=det_setup
            Sacc0, Sopo0, Sops0=[3.9e-44,2.81e-38,5.3e-38]
        elif len(det_setup)==6:
            lisaLT, lisaP, lisaD, Sacc0, Sopo0, Sops0=det_setup
        if not 0.83391<lisaLT<83.391:
            raise ValueError("we limit the light travel time across the arm in beween 0.83391 second to 83.391 second")
        if not lisaP>0:
            raise ValueError("The laser power should be positive.")
        if not lisaD>0:
            raise ValueError("The diameter of the telescope should be positive")
        if not 3.9e-49<Sacc0<3.9e-39:
            raise ValueError("We limit Sacc0 in between 3.9e-49,3.9e-39")
        if not 2.81e-43<Sopo0<2.81e-33:
            raise ValueError("We limit Sopo0 in between 2.81e-43,2.81e-33")
        if not 5.3e-43<Sops0<5.3e-33:
            raise ValueError("We limit Sops0 in beween 5.3e-43,5.3e-33")
        self.Tobs=Tobs
        self.cosmos=cosmos
        self.pop=mHz_pops[math.floor(popID)] # in parameters
        self.model=mHz_models[popID] # in parameters
        self.det=LisaLike('SciRDv1', lisaLT,lisaD, lisaP, Sacc0, Sopo0, Sops0)
        if self.pop=='MBHB':
            if not 0.5<self.Tobs<10:
                raise ValueError("We limit Tobs in between 0.5-10 years")
            self.pop_class=SMBHB(self.model, self.cosmos)
        elif self.pop=='GB':
            if not 0<self.Tobs:
                raise ValueError("Tobs needs to >0")
            self.pop_class=GB(self.model, self.cosmos)
        elif self.pop=='EMRI':
            if not 0.5<self.Tobs<10:
                raise ValueError("We limit Tobs in between 0.5-10 years")
            self.pop_class=EMRI(self.model,self.cosmos)
    
    def total(self,rho_cri):
        if self.pop=='MBHB':
            cata, tot=self.pop_class.givelist(self.Tobs, rho_cri,10, self.det)
        elif self.pop=='GB':
            tot=self.pop_class.tot_num(self.Tobs, self.det,10, rho_cri)
        elif self.pop=='EMRI':
            tot, cata=self.pop_class.givelist(self.Tobs, self.det,10, rho_cri)
        return tot

    def list(self, rho_cri, size):
        if self.pop=='MBHB':
#            kys=['Mass1', 'Mass2', 'Redshift', 'Spin1', 'Spin2', 'AzimuthalAngleOfSpin1', 'AzimuthalAngleOfSpin2', 'CoalescenceTime', 'Distance', 'EclipticLatitude', 'EclipticLongitude', 'PhaseAtCoalescence']
            cata, tot_num=self.pop_class.givelist(self.Tobs, rho_cri, self.det)
        elif self.pop=='GB':
            cata=self.pop_class.givelist(self.Tobs, self.det, rho_cri)
        elif self.pop=='EMRI':
            tot, cata=self.pop_class.givelist(self.Tobs, self.det, rho_cri)
            cata=self.pop_class.translate(cata)
        if size<=len(cata):
            cata=cata[:size]
        return cata
    def errorlist(self,rho_cri, size):
        if self.pop=='MBHB':
            cata=self.pop_class.givelisterrors(self.Tobs, rho_cri, self.det)
        elif self.pop=='GB':
            cata=self.pop_class.givelisterrors(self.Tobs, self.det, rho_cri)
        elif self.pop=='EMRI':
            cata=self.pop_class.givelisterrors(self.Tobs, self.det, rho_cri)
        if size<=len(cata):
            cata=cata[:size]
        return cata
    def noisecurve_X(self):
        freq, Stdix = [self.det.fq, self.det.Sx]
        return [freq, Stdix]
    def dataframe(self, rho_cri, size):
        if self.pop=='MBHB':
            df ,tot =self.pop_class.givedataframe(self.Tobs, rho_cri, self.det,size)
        elif self.pop=='GB':
            df, tot=self.pop_class.givedataframe(self.Tobs, self.det, rho_cri, size)  
        elif self.pop=='EMRI':
            df, tot=self.pop_class.givedataframe(self.Tobs, self.det, rho_cri,size, fast=True)
            #tot=None
        return [df,tot]              
    def errordataframe(self,rho_cri, size):
        if self.pop=='MBHB':
            df, tot = self.pop_class.giveerrordataframe(self.Tobs, rho_cri, self.det, size)
        elif self.pop=='GB':
            df, tot=self.pop_class.giveerrordataframe(self.Tobs, self.det, rho_cri, size)
        elif self.pop=='EMRI':
            df, tot=self.pop_class.giveerrordataframe(self.Tobs, self.det, rho_cri, size, fast=True) 
        return [df,tot]
