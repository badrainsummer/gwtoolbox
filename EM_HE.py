from pkg_resources import resource_filename
from gwtoolbox.HE_detectors import *
from gwtoolbox.HE_models import *
from gwtoolbox.functions_EMC import mathcalI
import numpy as np
import random
from scipy.signal.windows import gaussian
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats as stats
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
file_dir = os.path.join(parent_dir, 'random_number')

if os.path.exists(os.path.join(file_dir, 'random.csv')):
    df_read = pd.read_csv(os.path.join(file_dir, 'random.csv'))
else:
    1
if os.path.exists(os.path.join(file_dir, 'gauss_gamma.csv')):
    df_read_2 = pd.read_csv(os.path.join(file_dir, 'gauss_gamma.csv'))
else:
    1
if os.path.exists(os.path.join(file_dir, 'gauss_theta.csv')):
    df_read_3 = pd.read_csv(os.path.join(file_dir, 'gauss_theta.csv'))
else:
    1
if os.path.exists(os.path.join(file_dir, 'gauss_r0.csv')):
    df_read_4 = pd.read_csv(os.path.join(file_dir, 'gauss_r0.csv'))
else:
    1
if os.path.exists(os.path.join(file_dir, 'gauss_egrb.csv')):
    df_read_5 = pd.read_csv(os.path.join(file_dir, 'gauss_egrb.csv'))
else:
    1
if os.path.exists(os.path.join(file_dir, 'gauss_nup0.csv')):
    df_read_6 = pd.read_csv(os.path.join(file_dir, 'gauss_nup0.csv'))
else:
    1

class sGRB_DNS:
# short Gamma-ray Bursts, corresponding to DNS mergers, or BH-NS mergers/
    def __init__(self, cosmos, half_open_mean=half_open_mean_, half_open_std=half_open_std_, lg_r_grb_mean=lg_r_grb_mean_, lg_r_grb_std=lg_r_grb_std_, lgEgrb_mean=lgEgrb_mean_, lgEgrb_std=lgEgrb_std_, nup0_mean=nup0_mean_, nup0_std=nup0_std_, jet_Gamma_mean=jet_Gamma_mean_, jet_Gamma_std=jet_Gamma_std_,detID='Fermi_GBM', sub=False):
        self.Erange=HE_detectors[detID]['Erange']
        self.sens=HE_detectors[detID]['sens']
        self.flux_unit=HE_detectors[detID]['flux_unit']
        self.window=HE_detectors[detID]['window']
        self.bkg=HE_detectors[detID]['bkg']
        self.FoV=HE_detectors[detID]['FoV']
        self.correction=HE_detectors[detID]['corr']
        if sub!=False:
            self.sens=self.sens*sub
        self.Zs=np.linspace(0,30, 100)
        self.Ds=cosmos.luminosity_distance(self.Zs).value
        self.half_open_mean=half_open_mean
        self.half_open_std=half_open_std
        self.lg_r_grb_mean=lg_r_grb_mean
        self.lg_r_grb_std=lg_r_grb_std
        self.lgEgrb_mean=lgEgrb_mean
        self.lgEgrb_std=lgEgrb_std
        self.nup0_mean=nup0_mean
        self.nup0_std=nup0_std
        self.jet_Gamma_mean=jet_Gamma_mean
        self.jet_Gamma_std=jet_Gamma_std
        #print(self.jet_Gamma_mean)
        self.Band_alpha=Band_alpha
        self.Band_beta=Band_beta
        self.Band_s=Band_s
    def Band(self, nup, nup0, alphaB, betaB, s):
        '''
        band spectrum of GRB
        inputs:
        nup, the frequency in rest frame, can be np_array 
        nup0 is the reference frequency, put it at 1 MeV
        alphaB, betaB, and s are parameters for the band spectrum, the typical values are:
        '''
    # alphaB=-1, betaB=-2.5; s=1
        try:
            #print(nup.shape)
            return (nup/nup0)**(1+alphaB)*(1+(nup/nup0)**s)**((betaB-alphaB)/s)
        except:
            return np.zeros(nup.shape)
    def DfT(self, T, dtheta, thetav, r0):
        '''
           In Kai s thesis, Delta Phi(T)
           T can be a np_array, T is the time of the first photon, IF the viewing on-axis AND on-axis! 
        '''
        c=3e10 # cm/s, speed of light
    #print(1/beta-c/r0*T)
    #thetaT=np.arccos(1/beta-c/r0*T)
        thetaT=np.arccos(np.where(1-c/r0*T>=-1,1-c/r0*T, -1))
        costhetaT=np.maximum(1-c/r0*T,-1)
        sinthetaT=np.sqrt(1-costhetaT**2)
        #print(thetaT.shape, end='\n')
        #print(T.shape, end='\n')
        #if np.sin(thetav)*sinthetaT==0:
        #    result=np.pi
        #else:
        result=np.where((dtheta*np.ones(T.shape)>=thetav*np.ones(T.shape)) * (0<thetaT)*(thetaT<=dtheta-thetav), np.pi, np.where(thetaT==0.0, 0.0, np.arccos(np.maximum(np.minimum((np.cos(dtheta)-costhetaT*np.cos(thetav))/(np.sin(thetav)*sinthetaT),1),-1)))) # copied from Kai ...
        #result=np.where((dtheta*np.ones(T.shape)>=thetav*np.ones(T.shape)) * (thetaT<=dtheta-thetav), np.pi, np.arccos(np.maximum(np.minimum((np.cos(dtheta)-costhetaT*np.cos(thetav))/(1e-4+np.sin(thetav)*sinthetaT),1),-1))) # shuxu's original
        #if dtheta>=thetav and 0<=thetaT<=dtheta-thetav:
        #    result=np.pi
        #else:
        #    result1=(np.cos(dtheta)-np.cos(thetaT)*np.cos(thetav))/(np.sin(thetav)*np.sin(thetaT))
        #    if abs(result1)>=1:
        #        result=0
        #    else:
        #        result=np.arccos(result1)
        return result
# dtheta is the opening angle
# thetav is the vewing angle
# r0 is the place of inner shock
# beta is the v/c of the jet clums
    def FvT(self, nu_here, T_here, Gamma, dtheta, thetav, r0, D, Egrb=None, nup0=None, alphaB=None, betaB=None,s=None):
        '''
        inputs:
        nu_here: the frequency in the observer frame (with cosmological redshift), in MeV
        T_here: observer's time (on Earth, with already the cosmological redshift), can also be np_array? not yet 
        Gamma: Lorentz factor of the jet
        dtheta: half opening angle of the jet
        thetav: viewing angle 
        r0: the location of the GRB dissipation site, in cm
        D: distance to the sGRB, in Mpc
        nu0p: reference rest frame frequency, should in MeV, can be np_array
        Egrb: ergs, the total energy releasing of the GRB event.
        nup0, alphaB, beteB, s are the parameters for the Band spectrum. nup0 is in Hz for 1 MeV
        
        return: differential flux in unit of ergs/cm2/s, with cosmological redshift considered.
        '''
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        z=np.interp(D, self.Ds, self.Zs)
        c=3e10 # cm/s speed of light
        nu=nu_here*(1+z) # np_array, length of nu # increased ...
        T=T_here/(1+z)
        #thetaT=np.arccos(np.maximum(np.minimum(1-c/r0*T,1),-1)) # np_array, length of T
        #gamma=1./(1-beta**2)**0.5
        part1=Gamma**2*self.DfT(T, dtheta, thetav, r0) # np_array, length of T
        beta=np.sqrt(abs(1-1./Gamma**2))
        costhetaT=1-c/r0*T
        nup=np.outer(nu,Gamma*(1-beta*costhetaT)) # 2-D array, axis0:nu. axis1: Time
        spec=self.Band(nup, nup0=nup0, alphaB=alphaB, betaB=betaB, s=s) # 2-D array, axis0: nu. axis1: Time
        nomi=(Gamma**2*(1-beta*costhetaT))**2 # 1-D, time
        nomi_matrix=np.outer(np.ones(nu.shape),part1/nomi) # 2 D matrix, nu-time
        
        result=spec*nomi_matrix # 2D, nu-time matrix  
        #result=part1*spec/nomi
        I=1-np.cos(dtheta)
        II=mathcalI(alphaB, betaB, s)
        nu0p_Hz=2.418e20*nup0
        A0=Egrb/(Gamma*8*np.pi**2*r0**2*nu0p_Hz*I*II)/(1+z) # considering the cosmological redshift of the differential flux
        Dcm=3.086e24*D # Mpc to cm
        factor=2*c*A0*r0/Dcm**2
        return result*factor    
    def spectra(self,t_afterGW, Gamma, dtheta, thetav, r0, D, Egrb=None, nup0=None, alphaB=None, betaB=None, s=None):
        '''
        inputs: t_afterGW, the instance after GW chirp, as observed ON EARTH. (So z needs to be considered)
        Gamma: Lorentz factor of the jet
        dtheta: half opening angle of the jet
        thetav: viewing angle 
        r0: the location of the GRB dissipation site, in cm
        D: distance to the sGRB, in Mpc
        nu0p: reference rest frame frequency, should in MeV
        Egrb: ergs, the total energy releasing of the GRB event.
        nu0p: reference rest frame frequency, should in MeV
        Egrb: ergs, the total energy releasing of the GRB event.
        nup0, alphaB, beteB, s are the parameters for the Band spectrum. 
        return [nu (keV), vFv (ergs/s/cm2)]
        the nu and vFv are all considered z.
        '''
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        c=3e10 # speed of light, in unit of cm/s
        beta=np.sqrt(1-1./Gamma**2)
        #t_start=r0/(c*beta)*(1-beta*np.cos(max(0,thetav-dtheta)))*(1+z)
        #T_here=np.maximum(t_afterGW-r0/c*(1/beta-1),1e-6) # observer frame T.
        T_here=t_afterGW-r0/c*(1/beta-1)
        #T_here=np.maximum(t_afterGW-t_start,0)  # observer frame T on the Earth
        #T=np.linspace(0,10,10)
        #nus_here=np.logspace(-4,2,20) # still MeV now # observed nu on the Earth
        nus_here=np.logspace(-3,np.log10(25),50)
        #FvTs=[nu*2.418e20*self.FvT(nu, T, Gamma, dtheta, thetav, r0, D, Egrb, nup0, alphaB, betaB,s) for nu in nus]
        nus_matrix=np.outer(nus_here, np.ones(T_here.shape)) # 2-D matrix in nus-time shape
        #vFv=2.418e20*np.matmul(nus_matrix,self.FvT(nus,T, Gamma, dtheta, thetav, r0, D, Egrb, nup0, alphaB, betaB,s)) # 2-D array, nu-T
        vFv=2.418e20*self.FvT(nus_here,T_here, Gamma, dtheta, thetav, r0, D, Egrb, nup0, alphaB, betaB,s)*nus_matrix
        return [1000*nus_here, vFv]
    def spectra_ave(self,t_range, Gamma, dtheta, thetav, r0, D, Egrb=None, nup0=None, alphaB=None, betaB=None,s=None):
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        t0=t_range[0]
        tf=t_range[1]
        t=np.linspace(t0,tf,20) #Change back to 20?
        E,vFv=self.spectra(t_afterGW=t, Gamma=Gamma, dtheta=dtheta, thetav=thetav, r0=r0, D=D, Egrb=Egrb, nup0=nup0, alphaB=alphaB, betaB=betaB,s=s)
        
        #plt.figure()
        #plt.plot(E,vFv[:,1:5])
        #plt.xscale('log')
        #plt.yscale('log')

        #CONVOLUTION
        z=np.interp(D, self.Ds, self.Zs)
        sigma_t=(1+z)*t_d_sigma
        mean_t=sigma_t
        len_kernel=len(t)
        sigma=len_kernel/6
        kernel=gaussian(len_kernel,sigma)
        td_range = np.linspace(mean_t-3*sigma_t, mean_t+3*sigma_t, len_kernel)
        idx_start=np.where(td_range>0)[0][0]
        kernel=kernel[idx_start:]
        kernel/=np.trapz(kernel,td_range[idx_start:])
        s=mean_t-td_range[idx_start]
        filtered_t_afterGW=np.linspace(t0+mean_t-s,tf+mean_t+3*sigma_t, len(t)+len(kernel)-1)

        filtered_vFv=np.zeros((len(E),len(filtered_t_afterGW)))
        for i in np.arange(len(E)):
            vFv_step = np.convolve(vFv[i,:], kernel, 'full')
            vFv_step = vFv_step*np.trapz(vFv[i,:],t)/np.trapz(vFv_step,filtered_t_afterGW)
            filtered_vFv[i,:]=vFv_step #rows are time, columns are energy
        #flux=[np.trapz(filtered_vFv[:,i]/E,E) for i in np.arange(len(filtered_t_afterGW))] #to test if the lightcurves are the same
        #print("fluence: ",np.trapz(flux,filtered_t_afterGW))
        #vFv_ave=np.average(vFv,axis=1)
        vFv_ave=np.average(filtered_vFv,axis=1)

        #plt.figure()
        #plt.plot(filtered_t_afterGW,flux)
        #t_test,flux_test=self.light_curve(Gamma=Gamma, dtheta=dtheta, thetav=thetav, r0=r0, D=D, Egrb=Egrb, Erange=None, nup0=nup0)
        #plt.plot(t_test,flux_test)
        #plt.yscale('log')
        

        return [E,vFv_ave]
    def E_peak(self, t_range, Gamma, dtheta, thetav, r0, D, Egrb=None, nup0=None, alphaB=None, betaB=None,s=None):
        '''
        inputs: t_afterGW, the instance after GW chirp
        Gamma: Lorentz factor of the jet
        dtheta: half opening angle of the jet
        thetav: viewing angle 
        r0: the location of the GRB dissipation site, in cm
        D: distance to the sGRB, in Mpc
        nu0p: reference rest frame frequency, should in MeV
        Egrb: ergs, the total energy releasing of the GRB event.
        nup0, alphaB, beteB, s are the parameters for the Band spectrum. nup0 is in Hz for 1 MeV
        return: E_peak at instance t in keV
        '''
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        EkeV, vFv=self.spectra_ave(t_range, Gamma, dtheta, thetav, r0, D, Egrb, nup0, alphaB, betaB,s)
        #plt.yscale('log')
        #plt.xscale('log')
        #plt.plot(EkeV, vFv)
        Max=np.average(np.amax(np.array(vFv)))
        try:
            index=np.asscalar(np.arange(0,len(EkeV),1)[np.where(vFv==Max)])
        except:
            index=np.arange(0,len(EkeV),1)[np.where(vFv==Max)]
        #print(index)
        u=np.random.uniform(low=0,high=1,size=None)
        #if np.isscalar(index):
        #print(Max)
        if np.isscalar(index):
            E_peak=EkeV[index-1]*u+EkeV[min(index+1,len(EkeV)-1)]*(1-u)
        elif len(index)==2:
            E_peak=EkeV[index[0]]*u+EkeV[index[-1]]*(1-u)
        else:
            E_peak=0
        #    index=index[0]
        #    E_peak=EkeV[index-1]*u+EkeV[min(index+1,len(EkeV))]*(1-u)
        #    try:
        #        E_peak=EkeV[index[0]]*u+EkeV[index[1]]*(1-u)
        #    except:
        #        E_peak=np.average(EkeV[index])
        #if np.isscalar(E_peak):
        #    return E_peak
        #else:
        #    try:
        #        return E_peak[0]
        #    except:
                #print(E_peak)
        #        return 0
        return E_peak

    def flux(self, T, Gamma, dtheta, thetav, r0, D, Egrb, Erange=None, nup0=None, alphaB=None, betaB=None,s=None):
        '''
        inputs:
        T is the observer time (on Earth), and T=0 is the time of arrival of the first photon (IF on-axis AND jet speed=c)
        Gamma is the Lorentz factor of the jet
        dtheta is the opening angle, in radiance
        thetav is the viewing angle, also in radiance
        r0 is the GRB inner dissipative location, in unit of cm
        D is the distance to the event, in unit of Mpc
        Egrb is the total releasing energy of the GRB event, in unit of ergs
        Erange is the Energy range of the HE detector, in Unit of MeV, in a list of [lower, upper]
        
        output: FLUX the integrated flux in given Erange. 
        '''
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        #T=np.linpsace(0,10,50)
        if Erange==None:
            Erange=self.Erange
        Elow,Eup=Erange
        Esteps=np.linspace(Elow,Eup,50)
        Fvs=self.FvT(Esteps, T, Gamma, dtheta, thetav, r0, D, Egrb,nup0, alphaB, betaB,s) #photon flux
        # 2D matrix, nu-T
        FLUX=np.trapz(Fvs, Esteps, axis=0)*2.418e20
        #c=3e10 # speed of light, in unit of cm/s
        # FLUX has the same shape as T
        #t_delay=T+r0/c*(1/beta-1)
        return FLUX

    def photon_flux(self, T, Gamma, dtheta, thetav, r0, D, Egrb, Erange=None, nup0=None, alphaB=None, betaB=None,s=None):
        '''
        inputs:
        T is the observer time (on Earth), and T=0 is the time of arrival of the first photon (IF on-axis AND jet speed=c)
        Gamma is the Lorentz factor of the jet
        dtheta is the opening angle, in radiance
        thetav is the viewing angle, also in radiance
        r0 is the GRB inner dissipative location, in unit of cm
        D is the distance to the event, in unit of Mpc
        Egrb is the total releasing energy of the GRB event, in unit of ergs
        Erange is the Energy range of the HE detector, in Unit of MeV, in a list of [lower, upper]
        
        output: FLUX the integrated photon flux in given Erange. 
        '''
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        #T=np.linpsace(0,10,50)
        if Erange==None:
            Erange=self.Erange
        Elow,Eup=Erange
        Esteps=np.linspace(Elow,Eup,50)
        Fvs=self.FvT(Esteps, T, Gamma, dtheta, thetav, r0, D, Egrb,nup0, alphaB, betaB,s)/(Esteps[:,None]*1.6022e-6) #photon flux
        # 2D matrix, nu-T
        FLUX=np.trapz(Fvs, Esteps, axis=0)*2.418e20
        #c=3e10 # speed of light, in unit of cm/s
        # FLUX has the same shape as T
        #t_delay=T+r0/c*(1/beta-1)
        return FLUX

    def light_curve(self, Gamma, dtheta, thetav, r0, D, Egrb, Erange=None, nup0=None, alphaB=None, betaB=None,s=None, flux_unit=None, tend=10, tstep=100):
        '''
        inputs:
        Gamma is the Lorentz factor of the jet
        dtheta is the opening angle, in radiance
        thetav is the viewing angle, also in radiance
        r0 is the GRB inner dissipative location, in unit of cm
        D is the distance to the event, in unit of Mpc
        Egrb is the total releasing energy of the GRB event, in unit of ergs
        Erange is the Energy range of the HE detector, in Unit of MeV, in a list of [lower, upper]
        output [t_afterGW, fluxes]
        '''   
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        if flux_unit==None:
            flux_unit=self.flux_unit
        z=np.interp(D, self.Ds, self.Zs)
        c=3e10 # speed of light, in unit of cm/s
        beta=np.sqrt(abs(1-1./Gamma**2))
        t_start = max(r0/c * (1-np.cos(max(0, thetav-dtheta))),1e-6)*(1+z) # the time lag due to off-axis/on-axis difference, trust me! (25th Mar, 2022)
        #t_start=r0/(c*beta)*(1-beta*np.cos(max(0,thetav-dtheta)))*(1+z)
        if thetav<=dtheta:
            dT=10*r0/(2*c*beta*Gamma**2)
        elif dtheta<thetav<=2*dtheta:
            dT=5*(t_start + r0/c*(1/beta-1))
        elif 2*dtheta<=thetav:
            dT=r0/c*(1-np.cos(thetav+dtheta)) - t_start
        t_end=t_start+dT*(1+z)
       #  T=np.linspace(t_start,tend,tstep)
        #tstart=min(tstart,t_start)
        #tend=max(tend, t_end)
        if Erange==None:
            Erange=self.Erange
        c=3e10 # speed of light, in unit of cm/s
        T=np.linspace(t_start,t_end,tstep)
        #T=np.linspace(1e-6,tend,tstep) # this is time step on Earth
        if flux_unit=='ergs':
            fluxes=self.flux(T, Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s)
        elif flux_unit=='ph':
            fluxes=self.photon_flux(T, Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s)
        t_afterGW=T+r0/c*(1/beta-1)#t_start # and here we add the second contribution of time-lag due to beta*c and c
        
        # SMOOTHING
        sigma_t=(1+z)*t_d_sigma
        mean_t=sigma_t
        len_kernel=len(T)
        sigma=len_kernel/6
        kernel=gaussian(len_kernel,sigma)
        td_range = np.linspace(mean_t-3*sigma_t, mean_t+3*sigma_t, len_kernel)
        idx_start=np.where(td_range>0)[0][0]
        kernel=kernel[idx_start:]
        kernel/=np.trapz(kernel,td_range[idx_start:])
         
        filtered_flux=np.convolve(fluxes,kernel,'full')
        s=mean_t-td_range[idx_start]
        filtered_t_afterGW=np.linspace(t_afterGW[0]+mean_t-s,t_afterGW[len(t_afterGW)-1]+mean_t+3*sigma_t, len(filtered_flux))
        filtered_flux=filtered_flux*np.trapz(fluxes,t_afterGW)/np.trapz(filtered_flux, filtered_t_afterGW)
        return [filtered_t_afterGW, filtered_flux] 
        
        
        
#        #t_smear=(1+z)*t_smear_intrinsic
#        kernel= gaussian(len(fluxes), len(fluxes)/10.*t_smear)
#        t=np.linspace(0,10,len(fluxes))*(1+z)
#        #kernel=np.exp(-np.abs(t-t_smear)/t_smear) 
#        filtered_flux=np.convolve(fluxes, kernel, "same")/sum(kernel)
#        return [t_afterGW, filtered_flux]  
        #return [t_afterGW, fluxes] 

    def F_peak(self, Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s, flux_unit, window, bkg):
        '''
        return: Fpeak: if flux_unit is 'ph', this is the photon flux in ph/cm2/s accumulated over the x-s window.
                If flux_unit is 'ergs', this is the energy flux of the lightcurve in ergs/cm2/s over the x-s window.
                Added to the peak is the background signal, either in ph/cm2/s or ergs/cm2/s.
                t_peak: the instance where the max flux appears.
        '''
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        if Erange==None:
            Erange=self.Erange
        if flux_unit==None:
            flux_unit=self.flux_unit
        if window==None:
            window=self.window
        if bkg==None:
            bkg=self.bkg
        t, fluxs=self.light_curve(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0=nup0, alphaB=alphaB, betaB=betaB,s=s, flux_unit=flux_unit)
        Max=np.amax(np.array(fluxs))
        temp=t[np.where(fluxs==Max)]
        if np.isscalar(temp):
            t_peak=temp
        else:
            try:
                t_peak=temp[0]
            except:
                t_peak=0
                Fpeak=0
                return Fpeak, t_peak
        
        if window>(t[len(t)-1]-t[0]):
            Fpeak=np.trapz(fluxs, t)/window
        else:
            #window starts at Tstart but the curve itself is longer than the window width
            f = interp1d(t, fluxs) #20ms is typically smaller than the step size so we need to interpolate
            t1=max(t[0], t_peak-.5*window)#t1 & t2 are the start and end time of the accumulation window
            t2=t1+window
            t_interp=np.linspace(t1,t2,100)
            Fpeak=np.trapz(f(t_interp),t_interp)/window
        return Fpeak+bkg, t_peak 
   
    def Fluence(self, Gamma, dtheta, thetav, r0, D, Egrb, Erange=None, nup0=None, alphaB=None, betaB=None,s=None):
        '''
        input:
        sens: sensitivity of the HE detector, above whicih the fluence is calculated.
        return: Fluence in unit of ergs/cm2
        '''
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        if Erange==None:
            Erange=self.Erange
        
        #if flux_unit=='ph':
        #    t, ph_fluxs=self.light_curve(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s,flux_unit)
        #    t, fluxs=self.light_curve(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s,flux_unit='ergs')
            #ice_burg=np.where(np.array(ph_fluxs)>sens,np.array(fluxs)-sens*fluxs/ph_fluxs,0)
        #    ice_burg=fluxs
        #elif flux_unit=='ergs':
        #    t, fluxs=self.light_curve(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s,flux_unit)
        #    ice_burg=np.where(np.array(fluxs)>sens,np.array(fluxs),0)
        #return np.trapz(ice_burg, t)
        t, fluxs=self.light_curve(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s,flux_unit='ergs')
        return np.trapz(fluxs,t)

    def duration(self, Gamma, dtheta, thetav, r0, D, Egrb, Erange=None, nup0=None, alphaB=None, betaB=None,s=None, percent=0.9):
        '''
        input:
        percent=0.9 correspond to T90, 0.95 to T95, 0.5 to T50.
        return:
        tdu=pulse duration
        '''
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        if Erange==None:
            Erange=self.Erange

        total_fluence=self.Fluence(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s)
        c=3e10 # speed of light, in unit of cm/s
        beta=np.sqrt(1-1./Gamma**2)
        #t_start=r0/(c*beta)*(1-beta*np.cos(max(0,thetav-dtheta)))  # equation 22 in Kai's thesis
        #t_start=F_peak(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s)[1]
        if total_fluence==0:
            return 0
        else:
            t, fluxs=self.light_curve(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s,flux_unit='ergs')
            part_fluence=[np.trapz(fluxs[:i],t[:i])/total_fluence for i in range(0,len(t))]
            t_end=np.interp(percent,part_fluence,t[0:])
            tdu=t_end-t[0]
            return tdu
    def detected_intrinsic(self, Gamma, dtheta, thetav, r0, D, Egrb, Erange=None, nup0=None, alphaB=None, betaB=None,s=None, sens=None, flux_unit=None, window=None, bkg=None):
        if nup0==None:
            nup0=self.nup0_mean;
        if alphaB==None:
            alphaB=self.Band_alpha
        if betaB==None:
            betaB=self.Band_beta
        if s==None:
            s=self.Band_s
        if flux_unit==None:
            flux_unit=self.flux_unit
        if window==None:
            window=self.window
        if bkg==None:
            bkg=self.bkg

        flux_max,t_max=self.F_peak(Gamma, dtheta, thetav, r0, D, Egrb, Erange, nup0, alphaB, betaB,s,flux_unit,window,bkg)
        if Erange==None:
            Erange=self.Erange
        if sens==None:
            sens=self.sens
        if flux_max>=sens:
            return True
        else:
            return False

    def Give_GRB(self, thetav, D, full=True):
        '''
        inputs:
        thetav is the inclination angle 
        D is the luminosity distance
        returns:
        -Gamma, dtheta, r0, Egrb, nup0
        if full=True, also give:
        -T90, F_max, E_peak(t_peak)/t,t_start ...
        also the bool of whether detected
        '''
        #random.seed(1)
        #np.random.seed(1)
        c=3e10
        Gamma=max(random.gauss(self.jet_Gamma_mean, self.jet_Gamma_std),10)
        beta=np.sqrt(abs(1-1./Gamma**2))
        z=np.interp(D, self.Ds, self.Zs)
        dtheta=random.gauss(self.half_open_mean, self.half_open_std)
        r0=10**random.gauss(self.lg_r_grb_mean, self.lg_r_grb_std)
        Egrb=10**random.gauss(self.lgEgrb_mean, self.lgEgrb_std)
        nup0=abs(random.gauss(self.nup0_mean, self.nup0_std))
        u=np.random.uniform(low=0,high=1.0, size=None)
        if u>self.FoV/self.correction:
            detected=False
            T90=F_max=t_max=Epeak=fluence=0
        elif self.detected_intrinsic(Gamma, dtheta, thetav, r0, D, Egrb, Erange=self.Erange, nup0=nup0, alphaB=self.Band_alpha, betaB=self.Band_beta,s=self.Band_s, sens=self.sens, flux_unit=self.flux_unit, window=self.window, bkg=self.bkg):
            detected=True
            if not full:
                return Gamma,dtheta,r0,Egrb,nup0,detected   
            else:
                T90=self.duration(Gamma, dtheta, thetav, r0, D, Egrb, Erange=self.Erange, nup0=nup0, alphaB=self.Band_alpha, betaB=self.Band_beta, s=self.Band_s, percent=0.9) 
                F_max, t_max=self.F_peak(Gamma, dtheta, thetav, r0, D, Egrb, Erange=self.Erange, nup0=nup0, alphaB=self.Band_alpha, betaB=self.Band_beta ,s=self.Band_s, flux_unit=self.flux_unit, window=self.window, bkg=self.bkg)
                t_start = max(r0/c * (1-np.cos(max(0, thetav-dtheta))),1e-6)*(1+z)
                if thetav<=dtheta:
                    dT=10*r0/(2*c*beta*Gamma**2)
                elif dtheta<thetav<=2*dtheta:
                    dT=5*(t_start + r0/c*(1/beta-1))
                elif 2*dtheta<=thetav:
                    dT=r0/c*(1-np.cos(thetav+dtheta)) - t_start
                t_range=[t_start+r0/c*(1/beta-1)*(1+z),t_start+r0/c*(1/beta-1)*(1+z)+dT*(1+z)]
                Epeak=self.E_peak(t_range, Gamma, dtheta, thetav, r0, D, Egrb, nup0, alphaB=self.Band_alpha, betaB=self.Band_beta,s=self.Band_s)
                fluence=self.Fluence(Gamma, dtheta, thetav, r0, D, Egrb, Erange=self.Erange, nup0=nup0, alphaB=self.Band_alpha, betaB=self.Band_beta,s=self.Band_s)
        else: 
            T90=F_max=t_max=Epeak=fluence=0
            detected=False
        if not full:
            return Gamma,dtheta,r0,Egrb,nup0,detected
        else:
            return Gamma,dtheta,r0,Egrb,nup0, T90, F_max, t_max, Epeak,fluence, detected
   
    def Give_GRB_new(self, thetav, D, i, full=True):
        '''
        inputs:
        thetav is the inclination angle 
        D is the luminosity distance
        returns:
        -Gamma, dtheta, r0, Egrb, nup0
        if full=True, also give:
        -T90, F_max, E_peak(t_peak)/t,t_start ...
        also the bool of whether detected
        '''
        #df_read = pd.read_csv('../random.csv')
        #random_value = df_read.loc[99, 'Random Number']

        #random.seed(1)
        #np.random.seed(1)
        c=3e10
        #Gamma=stats.truncnorm((10-self.jet_Gamma_mean)/self.jet_Gamma_std, (np.inf-self.jet_Gamma_mean)/self.jet_Gamma_std, loc=self.jet_Gamma_mean, scale=self.jet_Gamma_std).rvs(1)
        Gamma=max(df_read_2.loc[i, 'Random Number'],10)
        #Gamma=df_read_2.loc[i, 'Random Number']
        #Gamma=max(random.gauss(self.jet_Gamma_mean, self.jet_Gamma_std),10)
        #beta=np.sqrt(abs(1-1./Gamma**2))
        #z=np.interp(D, self.Ds, self.Zs)
        #dtheta=random.gauss(self.half_open_mean, self.half_open_std)
        #r0=10**random.gauss(self.lg_r_grb_mean, self.lg_r_grb_std)
        #Egrb=10**random.gauss(self.lgEgrb_mean, self.lgEgrb_std)
        #nup0=abs(random.gauss(self.nup0_mean, self.nup0_std))
        #u=np.random.uniform(low=0,high=1.0, size=None)
        beta=np.sqrt(abs(1-1./Gamma**2))
        z=np.interp(D, self.Ds, self.Zs)
        dtheta=df_read_3.loc[i, 'Random Number']
        r0=10**df_read_4.loc[i, 'Random Number']
        Egrb=10**df_read_5.loc[i, 'Random Number']
        nup0=abs(df_read_6.loc[i, 'Random Number'])
        u=df_read.loc[i, 'Random Number']
        if u>self.FoV/self.correction:
            detected=False
            T90=F_max=t_max=Epeak=fluence=0
        elif self.detected_intrinsic(Gamma, dtheta, thetav, r0, D, Egrb, Erange=self.Erange, nup0=nup0, alphaB=self.Band_alpha, betaB=self.Band_beta,s=self.Band_s, sens=self.sens, flux_unit=self.flux_unit, window=self.window, bkg=self.bkg):
            detected=True
            if not full:
                return Gamma,dtheta,r0,Egrb,nup0,detected
            else:
                T90=self.duration(Gamma, dtheta, thetav, r0, D, Egrb, Erange=self.Erange, nup0=nup0, alphaB=self.Band_alpha, betaB=self.Band_beta, s=self.Band_s, percent=0.9)
                F_max, t_max=self.F_peak(Gamma, dtheta, thetav, r0, D, Egrb, Erange=self.Erange, nup0=nup0, alphaB=self.Band_alpha, betaB=self.Band_beta ,s=self.Band_s, flux_unit=self.flux_unit, window=self.window, bkg=self.bkg)
                t_start = max(r0/c * (1-np.cos(max(0, thetav-dtheta))),1e-6)*(1+z)
                if thetav<=dtheta:
                    dT=10*r0/(2*c*beta*Gamma**2)
                elif dtheta<thetav<=2*dtheta:
                    dT=5*(t_start + r0/c*(1/beta-1))
                elif 2*dtheta<=thetav:
                    dT=r0/c*(1-np.cos(thetav+dtheta)) - t_start
                t_range=[t_start+r0/c*(1/beta-1)*(1+z),t_start+r0/c*(1/beta-1)*(1+z)+dT*(1+z)]
                Epeak=self.E_peak(t_range, Gamma, dtheta, thetav, r0, D, Egrb, nup0, alphaB=self.Band_alpha, betaB=self.Band_beta,s=self.Band_s)
                fluence=self.Fluence(Gamma, dtheta, thetav, r0, D, Egrb, Erange=self.Erange, nup0=nup0, alphaB=self.Band_alpha, betaB=self.Band_beta,s=self.Band_s)
        else:
            T90=F_max=t_max=Epeak=fluence=0
            detected=False
        if not full:
            return Gamma,dtheta,r0,Egrb,nup0,detected
        else:
            return Gamma,dtheta,r0,Egrb,nup0, T90, F_max, t_max, Epeak,fluence, detected

class sGRB_BH_NS(sGRB_DNS):
    def R_NS(self, M, label):
        # input: 
        # M is the mass of the NS, in unit of solar masses
        # the mass-radius relationship of Neutron Star
        # the allowed labels are now: MPA1, PAL1, MS2, MS0
        # output: R_NS is in unit of km
        if label==None:
            label='MPA1'
        filename=label+'.txt'
        file_fullname='NS_MRR/'+filename
        filepath=resource_filename('gwtoolbox', file_fullname)
        data=np.loadtxt(filepath,skiprows=1, delimiter=',')
        Rs=data[:,0]
        Ms=data[:,1]
        R_NS=np.interp(M,Ms,Rs)
        return R_NS 
    def R_ISCO(self, M_BH, chi):
        '''
        input: M_BH is the mass of the BH, in unit of solar masses
               chi is the effective spin of the binary system
        based on Kai's thesis Equation (1)
        output: ISCO radius in unit of km
        '''
        solar_gravR=1.477 # in unit of km
        Z1=1+(1-chi**2)**(1./3.)*((1+chi)**(1./3.)+(1-chi)**(1./3.))
        Z2=np.sqrt(3*chi**2+Z1**2)
        #print('Z1=',Z1, 'Z2=', Z2)
        retro=np.random.choice([-1,1])
        Risco=M_BH*solar_gravR*(3+Z2+retro*np.sqrt((3-Z1)*(3+Z1+2*Z2)))
        return Risco
    def R_tid(self, M_BH, M_NS, label):
        '''
        input M_BH, M_NS: masses of BH and Neutron Stars respectively
        output: Tidal radius
        '''
        Rns=self.R_NS(M_NS, label)
        Rtid=(M_BH/M_NS)**(1./3.)*Rns
        return Rtid
    def Tidal_disrupted(self, M_BH, M_NS, chi, label):
        Rtid=self.R_tid(M_BH, M_NS, label)
        Risco=self.R_ISCO(M_BH, chi)
        if Rtid>=Risco:
            return True
        else:
            #print('Rtid=',Rtid)
            #print('Risco=',Risco)
            return False
    def Give_GRB_BHNS(self, M_BH, M_NS, chi, label, thetav, D, full=True):
        if self.Tidal_disrupted(M_BH, M_NS, chi, label):
            results=self.Give_GRB(thetav, D, full)
        else:
            if full:
                Gamma='-'
                dtheta='-'
                r0='-'
                Egrb='-'
                nup0='-'
                T90='-'
                F_max='-'
                t_max='-'
                Epeak='-'
                fluence='-'
                detected=False
                results=[Gamma, dtheta, r0, Egrb, nup0, T90, F_max, t_max, Epeak, fluence, detected]
            else:
                Gamma='-'
                dtheta='-'
                r0='-'
                Egrb='-'
                nup0='-'
                detected=False
                results=[Gamma,dtheta, r0, Egrb, nup0, detected]
        return results

        
