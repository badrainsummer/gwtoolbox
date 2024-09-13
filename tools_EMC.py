import numpy as np
import pandas as pd
from gwtoolbox.HE_models import *
from gwtoolbox.EM_HE import sGRB_DNS, sGRB_BH_NS
import astropy.units as u
from gwtoolbox.sources_kHz import DNS

class High_Energy:
    '''
    This class if for X-ray/Gamma-ray counter parts
    '''
    def __init__(self, event_type, signal_type, detector, model_para=None, MRR_label=None, cosmos=None, sub=False):
        '''
        input event_type: nsns, bhns, bhbh, etc.
              signal_type: corresponding to the event_type, includes: sGRB_p, after_glow_X, after_glow_opt, after_glow_radio, Kilonova, FRB, etc.
              detector: should be corresponding to the signal_type
        '''
        event_type_range=['nsns', 'bhns', 'bhbh']
        detector_range=['Fermi_GBM', 'insight_HE', 'GECAM_GRM', 'Swift_BAT', 'Konus']
        self.sub=sub
        self.event_type=event_type
        self.signal_type=signal_type
        self.NoWay=False
        self.detector=detector
        self.model_para=model_para
        self.MRR_label=MRR_label
        self.cosmos=cosmos
        if not self.model_para==None:
            self.half_open_mean, self.half_open_std, self.lg_r_grb_mean, self.lg_r_grb_std, self.lgEgrb_mean, self.lgEgrb_std, self.nup0_mean, self.nup0_std, self.jet_Gamma_mean, self.jet_Gamma_std=self.model_para
        else:
            self.half_open_mean, self.half_open_std, self.lg_r_grb_mean, self.lg_r_grb_std, self.lgEgrb_mean, self.lgEgrb_std, self.nup0_mean, self.nup0_std, self.jet_Gamma_mean, self.jet_Gamma_std= half_open_mean_,half_open_std_, lg_r_grb_mean_, lg_r_grb_std_,lgEgrb_mean_, lgEgrb_std_,nup0_mean_, nup0_std_, jet_Gamma_mean_, jet_Gamma_std_
        if not event_type in event_type_range:
            raise ValueError('donnot have this event type, event_type should be in %s' % event_type_range)
        elif event_type=='nsns':
            if not signal_type in ['sGRB_p']: # for now I only implement the sGRB_p YiShuxu 7th Jan, 2022. 
                self.NoWay=True
            else:
                if not detector in detector_range:
                    raise ValueError('donnot have this detector for %s' % signal_type )
                else:
                    pass                
        elif event_type=='bhns':
            if not signal_type in ['sGRB_p']: # for now I only implement the sGRB_p YiShuxu 7th Jan, 2022. 
                self.NoWay=True
            else:
                if not detector in detector_range:
                    raise ValueError('donnot have this detector for %s' % signal_type )
                else:
                    pass                
    def Search_EMC(self, df, full, verbose=False):
        '''
        input df: is the dataframe containning the GW catalogue, the column 'inc' should be there!
        full=True or False, determine whether to return full list of parameters
        '''      
        tot=len(df)
        
        if not 'inc' in df.columns:
            raise ValueError("no inclination parameter found in the catalogue")
        else: 
            if self.signal_type=='sGRB_p':
                Gamma_s=[]
                dtheta_s=[]
                r0_s=[]
                Egrbs=[]
                nup0s=[]
                T90s=[]
                F_maxs=[]
                t_maxs=[]
                Epeaks=[]
                fluences=[]
                detecteds=[]
                if self.event_type=='nsns':
                    i=0
                    j=0
                    GRB_class=sGRB_DNS(self.cosmos, self.half_open_mean, self.half_open_std, self.lg_r_grb_mean, self.lg_r_grb_std, self.lgEgrb_mean, self.lgEgrb_std, self.nup0_mean, self.nup0_std, self.jet_Gamma_mean, self.jet_Gamma_std, detID=self.detector,sub=self.sub)
                    for thetav, D in zip(df['inc'],df['D']):
                        if verbose:
                            i+=1;
                            print("checking the %d -th source, total number: %d" % (i, tot), end="\r")
                        if full:
                            #j+=1
                            Gamma,dtheta,r0,Egrb,nup0, T90, F_max, t_max, Epeak,fluence, detected=GRB_class.Give_GRB(thetav, D, full=full)
                            Gamma_s.append(Gamma)
                            dtheta_s.append(dtheta)
                            r0_s.append(r0)
                            Egrbs.append(Egrb)
                            nup0s.append(nup0)
                            T90s.append(T90)
                            F_maxs.append(F_max)
                            t_maxs.append(t_max)
                            Epeaks.append(Epeak)
                            fluences.append(fluence)
                            detecteds.append(detected)
                            
                        else:
                            j+=1
                            Gamma,dtheta,r0,Egrb,nup0,detected=GRB_class.Give_GRB_new(thetav, D, j, full=full)
                            #print("Gamma=",Gamma) 
                            Gamma_s.append(Gamma)
                            dtheta_s.append(dtheta)
                            r0_s.append(r0)
                            Egrbs.append(Egrb)
                            nup0s.append(nup0)
                            detecteds.append(detected)
                            #print("\n", len(Gamma_s))
                  
                elif self.event_type=='bhns':
                    i=0
                    GRB_class=sGRB_BH_NS(self.cosmos, self.half_open_mean, self.half_open_std, self.lg_r_grb_mean, self.lg_r_grb_std, self.lgEgrb_mean, self.lgEgrb_std, self.nup0_mean, self.nup0_std, self.jet_Gamma_mean, self.jet_Gamma_std,detID=self.detector,sub=self.sub)
                    for m1, m2, chi, thetav, D in zip(df['m1'],df['m2'],df['inc'],df['inc'],df['D']):
                        if verbose:
                            i+=1;
                            print("checking the %d -th source, total number: %d" % (i, tot), end="\r")
                        if full:
                            Gamma,dtheta,r0,Egrb,nup0, T90, F_max, t_max, Epeak,fluence, detected=GRB_class.Give_GRB_BHNS( m1, m2, chi, self.MRR_label, thetav, D, full=full)
                            Gamma_s.append(Gamma)
                            dtheta_s.append(dtheta)
                            r0_s.append(r0)
                            Egrbs.append(Egrb)
                            nup0s.append(nup0)
                            T90s.append(T90)
                            F_maxs.append(F_max)
                            t_maxs.append(t_max)
                            Epeaks.append(Epeak)
                            fluences.append(fluence)
                            detecteds.append(detected)
                        else: 
                            Gamma,dtheta,r0,Egrb,nup0,detected=GRB_class.Give_GRB_BHNS(m1, m2, chi, self.MRR_label, thetav, D, full=full)
                            Gamma_s.append(Gamma)
                            dtheta_s.append(dtheta)
                            r0_s.append(r0)
                            Egrbs.append(Egrb)
                            nup0s.append(nup0)
                            detecteds.append(detected)
        
                df['Gamma']=Gamma_s
                df['dtheta']=dtheta_s
                df['r0']=r0_s
                df['Egrb']=Egrbs
                df['nup0']=nup0s
                df['detected']=detecteds
                if full:
                    df['T90']=T90s
                    df['F_max']=F_maxs
                    df['t_max']=t_maxs
                    df['Epeak']=Epeaks
                    df['fluence']=fluences
            else: 
                pass   
        return df   

    def Num_joint_det(self, df):
        dataframe=self.Search_EMC(df, full=False)   
        result=len(dataframe[dataframe['detected']==True])
        return result
    def det_density(self, theta_j, z):
        n=0
        for i in range(2000):
            thetav = np.arccos(np.random.rand())
            GRB_class=sGRB_DNS(self.cosmos, theta_j, 0, self.lg_r_grb_mean, self.lg_r_grb_std, self.lgEgrb_mean, self.lgEgrb_std, self.nup0_mean, self.nup0_std, self.jet_Gamma_mean, self.jet_Gamma_std, detID=self.detector,sub=self.sub)
            Gamma,dtheta,r0,Egrb,nup0, T90, F_max, t_max, Epeak,fluence, detected=GRB_class.Give_GRB(thetav, self.cosmos.luminosity_distance(z)/u.Mpc)
            if detected==True:
                n+=1
            else:
                pass
        return n/2000
