import numpy as np
import os.path
from gwtoolbox.functions_pta import * 

my_path = os.path.abspath(os.path.dirname(__file__))
PPTA_path = os.path.join(my_path, "PPTA.dat")
EPTA_path = os.path.join(my_path, "EPTA.dat")
Nano_path = os.path.join(my_path, "NanoGrav.dat")
data_PPTA=np.loadtxt(PPTA_path, dtype=str,skiprows=1, max_rows=26)
data_EPTA=np.loadtxt(EPTA_path, dtype=str,skiprows=1, max_rows=42)
data_nano=np.loadtxt(Nano_path, dtype=str,skiprows=1, max_rows=47)
def Existing_PTA(which='PPTA'):
    # creat the PTA
    PulsarsArray=[]
    if which=='PPTA':
        for i in range(len(data_PPTA)):
            psrname=data_PPTA[i,0]
            Ntoa=int(data_PPTA[i,1])
            rms=float(data_PPTA[i,2]) # root-mean-square in micro-second
            log10Ared=data_PPTA[i,3]
            if log10Ared=='inf':
                #log10Ared=float(log10Ared)
                Ared=0
            else: 
                Ared=10**float(log10Ared) # this one is dimensionless!! I should convert it to year^3/2!! But if I redefine gamma as gamma+3, then Ared is in unit of year^3/2
            gamma=float(data_PPTA[i,4])+3 # positive
            T_year=float(data_PPTA[i,5])
            RA=data_PPTA[i,6]
            DEC=data_PPTA[i,7]
            sig_w=data_PPTA[i,8]
            f_low=1./T_year # in unit of year**-1
            f_high=Ntoa/2.*f_low # in unit of year**-1
            if sig_w!='nan':
                sig_w=float(sig_w)
            #f_high=1./T_year # in unit of year**-1
            #f_low=Ntoa/2.*f_high # in unit of year**-1
            elif sig_w=='nan' or log10Ared!='inf':
                delta=max(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high**(1-gamma)-f_low**(1-gamma))*1e27,1e-3)
                #print("delta=",delta)
                #sig_w=np.sqrt(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high-f_low)*1e27) # 1 year^2=1e27 microseconds^2
                sig_w=np.sqrt(delta)
            #elif sig_w=='nan' and log10Ared=='inf':
            #    sig_w=rms
            #if sig_w=='nan':
            #    sig_w=float(rms) # ysx: 3rd 11, 2020.
            #else: 
            #    sig_w=float(sig_w)
            deltat=T_year*365./Ntoa
            psrdict={"name":psrname,"deltat":deltat, "Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"PPTA"}
            PulsarsArray.append(psrdict)
    elif which=='EPTA':
        for i in range(len(data_EPTA)):
            psrname=data_EPTA[i,0]
            Ntoa=int(data_EPTA[i,1])
            rms=float(data_EPTA[i,2]) # micro-second
            log10Ared=float(data_EPTA[i,3])
            Ared=10**log10Ared ## dimensionless!
            gamma=float(data_EPTA[i,4])+3 # positive
            T_year=float(data_EPTA[i,5])
            RA=data_EPTA[i,6]
            DEC=data_EPTA[i,7]
            f_low=1./T_year # in unit of year**-1
            f_high=Ntoa/2.*f_low # in unit of year**-1
            delta=max(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high**(1-gamma)-f_low**(1-gamma))*1e27,1e-3)
            #print("delta=",delta)
            sig_w=np.sqrt(delta)
            #sig_w=rms
            deltat=T_year*365./Ntoa
            psrdict={"name":psrname,"deltat":deltat, "Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"EPTA"}
            PulsarsArray.append(psrdict)

    elif which=='NanoGrav':
        for i in range(len(data_nano)):
            psrname=data_nano[i,0]
            #Ntoa=int(data_nano[i,1]) # this number for NanoGrav is problematic

            sig_w=float(data_nano[i,2]) # unit : us
            Ared=max(float(data_nano[i,3]),0.1)*3.169e-14 # in year^3/2 yes! convert from us*yr^0.5 to days^3/2
            gamma=max(float(data_nano[i,4]),0.5)+3 # for this your don't need to substract 3. ysx, 5th Oct, sorry but you still need
            T_year=float(data_nano[i,5])
            Ntoa=T_year*365.2/7.
            RA=data_nano[i,6]
            DEC=data_nano[i,7]
            deltat=T_year*365./Ntoa
            psrdict={"name":psrname,"deltat":deltat, "Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"NanoGrav"}
            PulsarsArray.append(psrdict)

    elif which=='IPTA':
        for i in range(len(data_PPTA)):
            psrname=data_PPTA[i,0]
            Ntoa=int(data_PPTA[i,1])
            rms=float(data_PPTA[i,2]) # root-mean-square in micro-second
            log10Ared=data_PPTA[i,3]
            if log10Ared!="inf":
                Ared=10**float(log10Ared)
            gamma=float(data_PPTA[i,4])+3
            T_year=float(data_PPTA[i,5])
            RA=data_PPTA[i,6]
            DEC=data_PPTA[i,7]
            f_low=1./T_year # in unit of year**-1
            f_high=Ntoa/2.*f_low # in unit of year**-1
            sig_w=data_PPTA[i,8]
            if sig_w!='nan':
                sig_w=float(sig_w)
            #f_high=1./T_year # in unit of year**-1
            #f_low=Ntoa/2.*f_high # in unit of year**-1
            elif sig_w=='nan' or log10Ared!="inf":
                sig_w=np.sqrt(max(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high**(1-gamma)-f_low**(1-gamma))*1e27,1e-3)) # 1 year^2=1e27 microseconds^2
            #elif sig_w=='nan' and log10Ared=="inf":
            #    sig_w=rms
            #if sig_w=='nan':
            #    sig_w=float(rms)
            #else:
            #    sig_w=float(rms)
            deltat=T_year*365./Ntoa
            psrdict={"name":psrname,"deltat":deltat,"Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"PPTA"}
            PulsarsArray.append(psrdict) 
        Previous_names=[x["name"] for x in PulsarsArray]
        for i in range(len(data_EPTA)):
            psrname=data_EPTA[i,0]
            #Previous_names=[x["name"] for x in PulsarsArray]
            if psrname in Previous_names:
                index=Previous_names.index(psrname)
                T_previous=PulsarsArray[index]["Tyear"]
                #w_previous=PulsarsArray[index]["sigw"]
            #Ntoa=data_EPTA[i,1]
            #rms=data_EPTA[i,2] # micro-second
            #log10Ared=data_EPTA[i,3]
            #Ared=10**log10Ared
            #gamma=data_EPTA[i,4]
                T_year=float(data_EPTA[i,5])
                # a old pulsar 
                if T_year>=T_previous: # substitute the pulsar psrname with the longer data
                    Ntoa=int(data_EPTA[i,1])
                    rms=float(data_EPTA[i,2]) # micro-second
                    log10Ared=float(data_EPTA[i,3])
                    Ared=10**log10Ared
                    gamma=float(data_EPTA[i,4])+3    
                    RA=data_EPTA[i,6]
                    DEC=data_EPTA[i,7]
                    f_low=1./T_year # in unit of year**-1
                    f_high=Ntoa/2.*f_low # in unit of year**-1
                    sig_w=np.sqrt(max(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high**(1-gamma)-f_low**(1-gamma))*1e27,1e-3))
                    #sig_w=rms
                    deltat=T_year*365./Ntoa
                    psrdict={"name":psrname,"deltat":deltat, "Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"EPTA"}
                    PulsarsArray[index]=psrdict
                else : pass
            else :
                # a new pulsar
                Ntoa=int(data_EPTA[i,1])
                rms=float(data_EPTA[i,2]) # micro-second
                log10Ared=float(data_EPTA[i,3])
                Ared=10**log10Ared
                gamma=float(data_EPTA[i,4])+3
                    #T_year=data_EPTA[i,5]
                RA=data_EPTA[i,6]
                DEC=data_EPTA[i,7]
                f_low=1./T_year # in unit of year**-1
                f_high=Ntoa/2.*f_low # in unit of year**-1
                sig_w=np.sqrt(max(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high**(1-gamma)-f_low**(1-gamma))*1e27,1e-3))
                #sig_w=rms
                deltat=T_year*365./Ntoa
                psrdict={"name":psrname,"deltat":deltat, "Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"EPTA"}
                PulsarsArray.append(psrdict)
        Previous_names=[x["name"] for x in PulsarsArray]
        for i in range(len(data_nano)):
            psrname=data_nano[i,0]
            T_year = float(data_nano[i,5])
            if psrname in Previous_names:
                index=Previous_names.index(psrname)
                T_previous=PulsarsArray[index]["Tyear"]
                # a old pulsar 
                if T_year>=T_previous: # substitute the pulsar psrname with the longer data
                    #Ntoa=int(data_nano[i,1]) # this number in the catalogue of Nanogra is problematic
                    Ntoa=T_year*365.2/7.
                    rms=float(data_nano[i,2]) # micro-second
                        #log10Ared=data_nano[i,3]
                    Ared=max(float(data_nano[i,3]),0.1)*3.169e-14 # in year^3/2 yes! convert from us*yr^0.5
                    gamma=max(float(data_nano[i,4]),0.5)+3    
                    RA=data_nano[i,6]
                    DEC=data_nano[i,7]
                    f_low=1./T_year # in unit of year**-1
                    f_high=Ntoa/2.*f_low # in unit of year**-1
                    #sig_w=np.sqrt(max(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high**(1-gamma)-f_low**(1-gamma))*1e27,0))
                    sig_w=rms
                    deltat=T_year*365./Ntoa
                    psrdict={"name":psrname,"deltat":deltat, "Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"NanoGrav"}
                    PulsarsArray[index]=psrdict
                else : pass
            else :
                # a new pulsar
                #Ntoa=int(data_nano[i,1])
                Ntoa=T_year*365.2/7.
                rms=float(data_nano[i,2]) # micro-second
                    #log10Ared=data_EPTA[i,3]
                Ared=max(float(data_nano[i,3]),0.1)*3.169e-14 # in year^3/2 yes! convert from us*yr^0.5#
                gamma=max(float(data_nano[i,4]),0.5)+3
                    #T_year=data_EPTA[i,5]
                RA=data_nano[i,6]
                DEC=data_nano[i,7]
                f_low=1./T_year # in unit of year**-1
                f_high=Ntoa/2.*f_low # in unit of year**-1
                sig_w=rms
                #sig_w=np.sqrt(max(rms**2-Ared**2/12./np.pi**2/(1-gamma)*(f_high**(1-gamma)-f_low**(1-gamma))*1e27,0))
                deltat=T_year*365./Ntoa
                psrdict={"name":psrname,"deltat":deltat, "Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"NanoGrav"}
                PulsarsArray.append(psrdict)
    return PulsarsArray  # tested and debuged 

def newPulsars(meal, deltat=1, T=10, Ndot=10, which='IPTA'):
#	"""
#    	meal ='A'/'B'or'C'
#        delta is the interval of days, 'cadence'
#        T in years, future obervaton durations
#        Ndot is how many new pulsars will be discovered per future year
#    """
    if not meal in ['A','B','C']:
        raise ValueError("We only have meal A,B,C.")
    if not T>=1:
        raise ValueError("We want the future observation duration larger than 1 year.")
    if not (deltat>=1 and deltat<=T*365):
        raise ValueError("We limit the  between deltat 1 day and T")
    if not Ndot>=0:
        raise ValueError("Ndot should be non-negative")
    if not which in ['IPTA','EPTA','PPTA','NanoGrav']:
        raise ValueError("which should be in 'IPTA','EPTA','PPTA','NanoGrav'");
    if meal=='A':
        PulsarsArray=Existing_PTA(which)
    elif meal=='B':  
    # more observation with the exsisting PTAs
        PulsarsArray=Existing_PTA(which)
        for pulsar in PulsarsArray: 
            #pulsar["deltat"]=deltat
            Ntoa_old=pulsar["Tyear"]*365./pulsar["deltat"]
            Ntoa_new=Ntoa_old+T*365./deltat
            pulsar["Tyear"]+=T
            pulsar["deltat"]=pulsar["Tyear"]*365./Ntoa_new
    elif meal=='C':
    # increasing number of pulsars into the current IPTA
        PulsarsArray=Existing_PTA(which)
        Areds=[pulsar["Ared"] for pulsar in PulsarsArray]
        gammas=[pulsar["gamma"] for pulsar in PulsarsArray]
        RAs=[pulsar["RA"] for pulsar in PulsarsArray]
        DECs=[pulsar["DEC"] for pulsar in PulsarsArray]
        sig_ws=[pulsar["sigw"] for pulsar in PulsarsArray]
        for pulsar in PulsarsArray: # old pulsars, new observation.
            Ntoa_old=pulsar["Tyear"]*365./pulsar["deltat"]
            Ntoa_new=Ntoa_old+T*365./deltat
            pulsar["Tyear"]+=T
            pulsar["deltat"]=pulsar["Tyear"]*365./Ntoa_new
            #pulsar["deltat"]=deltat
            #pulsar["Tyear"]+=T
        for year in range(T):
            # this is the year-th year:
            # how many new pulsars will be added to the original PTA?               
            for i in range(Ndot+1):
                psrname="newpsr"+str(year)+"-th year-"+str(i)
                Ared=GetOne(Areds)
                gamma=GetOne(gammas)
                RA=GetOne(RAs, sty='hms')
                DEC=GetOne(DECs, sty='dms')
                sig_w=GetOne(sig_ws)  
                T_year=T-year   
                psrdict={"name":psrname,"deltat":deltat, "Ared":Ared, "gamma": gamma, "Tyear": T_year, "RA": RA, "DEC": DEC, "sigw":sig_w,"PTA":"NEW"}
                PulsarsArray.append(psrdict)
    return PulsarsArray        




