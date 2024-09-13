import sys
import numpy as np
from .cosmology import Cosmology
from scipy import special
from scipy.stats import skewnorm
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import random
#sys.path.append('/Users/yishuxu/Downloads/itsample')
#from itsample import sample
#mcut=55
#import zeus
from .ESS import EllipticalSliceSampler
def sample1D(pdf, ranges, size):
    """
    sample from 1D distribution using 2-D MC method.
    """
    xs=np.random.uniform(low=ranges[0], high=ranges[1], size=1000)
    ys=pdf(xs)
    peak_y=max(ys)
    scores=np.random.uniform(low=0, high=peak_y, size=1000)
    if len(xs[scores<ys])>=1:
        result=np.random.choice(xs[scores<ys])
    else:
        result=None
        print('xs=',xs,'ys=',ys)
    return result

def lognm(x,mu,sig):
    """
    x can be array-like
    """
    return 1./(x*sig*np.sqrt(2.*np.pi))*np.exp(-(np.log(x)-mu)**2/(2.*sig**2))
# the normalization is right, for integrate over dx

def pzm(zm, zf, tau, cosmos):
    # zm is a float, and zf can be a array.
    # zm should not be an array! otherwise it'll mess up with zf
    # NOW zm CAN BE an ARRAY too
    #tm=cosmos.lookback_time(zm).value
    #tf=cosmos.lookback_time(zf).value
    tm=lookback(zm, cosmos)
    tf=lookback(zf, cosmos)
    #TM=np.einsum('i,j->ij',tm,np.ones(len(tf)))
    TM=np.outer(tm, np.ones(len(tf))) # np.outer is equivalent with einsum i,j-->ij, but three times faster
    #TF=np.einsum('i,j->ij',np.ones(len(tm)),tf)
    if np.isscalar(tm):
        TF=tf
    else:
        TF=np.outer(np.ones(len(tm)), tf)
    return 1/tau*np.exp(-(TF-TM)/tau)*np.heaviside(TF-TM,1)#/(1.-np.exp(-tm/tau))

def lookback(z, cosmos):
    if np.isscalar(z)==True:
        zf=np.linspace(0,z,50)
    else:
        zf=np.linspace(0,max(z), 50)
    tf=cosmos.lookback_time(zf).value
    return np.interp(z, zf, tf)


def zofD(D, cosmos):
    # D is in Mpc
    zs=np.logspace(-4,1,1000)
    Ds=cosmos.luminosity_distance(zs).value
    return np.interp(D, Ds, zs)

def dtovdz(z,cosmos):
    #t=cosmos.lookback_time(z).value
    t=lookback(z, cosmos)
    dz=1e-10*np.ones(len(z))
    t_right=lookback(z+dz, cosmos)
    t_left=lookback(z-dz, cosmos)
    #t_right=cosmos.lookback_time(z+dz).value
    #t_left=cosmos.lookback_time(z-dz).value
    return (t_right-t_left)/dz/2.
def dDovdz(z,cosmos):
    D=cosmos.luminosity_distance(z).value # Mpc
    if np.isscalar(z):
        dz=1e-10
    else:
        dz=1e-10*np.ones(len(z))
    D_right=cosmos.luminosity_distance(z+dz).value #Mpc
    D_left=cosmos.luminosity_distance(z-dz).value
    return (D_right-D_left)/dz/2.

def Psi(z, alpha, beta, C):
    return (1+z)**alpha/(1+((1+z)/C)**beta)

def R(zm, alpha, beta, C, tau, cosmos):
    #ZF=np.logspace(-2, 1., 1000)
    ZF=np.linspace(1e-2,20,1000)
    #ZF=np.linspace(1e-8,max(zm),5000)
    YM=pzm(zm, ZF, tau, cosmos)*Psi(ZF, alpha, beta, C)*dtovdz(ZF,cosmos)
    result=np.trapz(YM,ZF)
    if np.isscalar(zm):
        result=result[0]
    else:
        pass
    return result
def nor(c, gamma, mu, mcut):
    # normalization of m1 distributioin
    #mcut=70 # the m1_cutoff in the catalogue
    try:
        ycut=c/(mcut-mu)
    #cutterm=np.exp(-c/(mcut-mu))*(mcut-mu)**(1-gamma)/(gamma-1)
    # cut-off term:
        return c**(1-gamma)*special.gamma(gamma-1)*special.gammaincc(gamma-1,ycut)
    except:
        return None
#def PM1(m,skewness,loc,scale):

#    return skewnorm.pdf(m,a=skewness,loc=loc,scale=scale)
def PM1(m1, mu, c, gamma,mcut):
    """
    m1 can be array-like! proud of it.
    """
    #if m1<=mu:
    #    return 0
    #else:
    #    return np.exp(-c/(m1-mu))*(m1-mu)**(-gamma)/nor(c,gamma)
    #mcut=70 # the m1_cutoff in the catalogue
    return np.exp(-c/np.abs(m1-mu))*np.abs(m1-mu)**(-gamma)/nor(c,gamma,mu,mcut)*np.where(m1<=mcut,1,0)*np.where(m1>mu,1,0)
    #return np.where(m1<mcut, np.exp(-c/np.abs(m1-mu))*np.abs(m1-mu)**(-gamma)/nor(c,gamma,mu),0)
def PM2(m1, mu, c, gamma, mcut, m_peak, mpeak_scale, m_peak_sigma):
    component1=np.exp(-c/np.abs(m1-mu))*np.abs(m1-mu)**(-gamma)*np.where(m1<=mcut,1,0)*np.where(m1>mu,1,0)
    component2=mpeak_scale*np.exp(-0.5*((m1-m_peak)/m_peak_sigma)**2)
    norm1=nor(c, gamma, mu, mcut)
    norm2=mpeak_scale*m_peak_sigma*np.sqrt(2*np.pi)
    return (component1+component2)/(norm1+norm2)

#def Pm(m2, mcut, m1, ):
#    # mass function of the secondary BH, truncated normal



def PQ(q, q_low, mcut):
    #q_cut=np.where(mcut/m1>q_low, mcut/m1, q_low)
    return np.where(q>=q_low, 1./(1-q_low), 0)*np.where(q<=1, 1, 0)
    #return np.where(q>=q_low, 1./(1-q_low+0.05), 1/(1-q_low+0.05)*np.exp(q-q_low))*np.where(q<=1, 1, 1e-6)

def ind_masses(Mch, eta):
    """
    Calculating the individual masses of binary from given chirp mass M and symmetric mass ratio eta.

    Parameters:
      Mch (float): chirp mass
      eta (float): mass ratio

    Returns:
      (float): the mass of the primary object (heavier)
      (float): the mass of the secondary object (lighter)
    """
    Mtot = Mch*eta**-0.6
    product = eta*Mtot**2.0
    M = 0.5*(Mtot+np.sqrt(Mtot**2.0-4.*product));
    m = 0.5*(Mtot-np.sqrt(Mtot**2.0-4.*product));
    return [M,m]

def sym_ratio(m1, m2):
    """
    Calculating the symnetric mass ratio eta from given m1 and m2.

    Parameters:
      m1 (float or array like floats): mass m1
      m2 (float or array like floats): mass m2

    Returns:
      (float or array of dtype float): mass ratio
    """
    product = m1*m2
    Mtot = m1+m2
    return product/Mtot**2.0

def chirp_mass(m1, m2):
    """
    Calculating the chirp mass from given m1 and m2.

    Parameters:
      m1 (float or array of dtype float): mass m1
      m2 (float or array of dtype float): mass m2

    Returns:
      (float or array of dtype float): chirp mass
    """
    m1=np.abs(m1)
    m2=np.abs(m2)
    return (m1*m2)**0.6/(m1+m2)**0.2

def rho_sq_core(noise_tab, mod_shape, f_up=None):
    """
    Calculating rho sq.

    Parameters:
      noise_tab (array of dtype float): The noise power spectrum
      mod_shape (function): The (normalized) shape of modulus of waveform
      f_up (Optional[1-D ArrayLike float]): the upper limit of frequency


    Returns:
      (1-D ArrayLike float): rho sq
    """
    freq = noise_tab[0]
    # all frequency in the noise_tab curve
    nois = noise_tab[1]
#    # all noise power in the frequency range
#    if f_up<=freq[0]:
#        freq_cut=np.array([1e-10,1e-10,1e-10])
#        nois_cut=np.array([1e-10,1e-10,1e-10])
#    if f_up.any() != None:
#        freq_cut = freq #freq[freq<=f_up]
#    # cut frequency beyond f_up
#        nois_cut = np.where(np.less(np.einsum('i,j->ij',np.ones(len(freq_cut)),freq),np.einsum('i,j->ij',freq_cut,np.ones(len(freq)))),noise,0)
#    else:
#        freq_cut = freq
#        nois_cut = nois
#    # cut frequency beyond f_up
    temp = np.diff(freq)
    if  len(temp)>0:
        freq_steps = np.append(temp,temp[-1])
    # to make the length of freq_step the same as freq_cut.
        hsq = mod_shape(freq)**2
        if np.isscalar(f_up)==True:
                hsq=np.where(freq<f_up,hsq,0)
                return np.sum(hsq/nois*freq_steps)
        elif np.array(f_up).any()!=None:
                hsq=np.where(np.less(np.einsum('i,j->ij',np.ones(len(f_up)),freq),np.einsum('i,j->ij',f_up,np.ones(len(freq)))),hsq,0)
                return np.sum(hsq/nois*freq_steps,axis=1)
        else:
                return np.sum(hsq/nois*freq_steps)
    else: return 0

def rho_sq_core_merger(noise_tab, mod_shape, f1, f2):
    """
    Calculating rho sq from the merger stage.

    Parameters:
      noise_tab (array of dtype float): The noise power spectrum
      mod_shape (function): The (normalized) shape of modulus of waveform
      f_1,2 (Optional[1-D ArrayLike float]): the upper limit of frequency


    Returns:
      (1-D ArrayLike float): rho sq, dim(f1)
    """
    freq = noise_tab[0]
    # all frequency in the noise_tab curve
    nois = noise_tab[1]
#    # all noise power in the frequency range
    temp = np.diff(freq)
    if  len(temp)>0:
        freq_steps = np.append(temp,temp[-1])
    # to make the length of freq_step the same as freq_cut.
        hsq = mod_shape(freq)**2
        if np.isscalar(f1)==True:
                hsq=np.where(f1<freq,hsq,0)*np.where(f2>freq,1,0)
                return np.sum(hsq/nois*freq_steps)*f1**(-1)
        elif np.array(f1).any()!=None:
                hsq_1=np.where(np.less(np.einsum('i,j->ij',np.ones(len(f1)),freq),np.einsum('i,j->ij',f2,np.ones(len(freq)))),hsq,0)
                hsq_2=np.where(np.greater(np.einsum('i,j->ij',np.ones(len(f1)),freq),np.einsum('i,j->ij',f1,np.ones(len(freq)))),1,0)
                hsq=hsq_1*hsq_2
                return np.sum(hsq/nois*freq_steps,axis=1)*f1**(-1)
        else:
                return np.sum(hsq/nois*freq_steps)*f1**(-1)
    else: return 0

def rho_sq_core_ringdown(noise_tab, mod_shape,f1, f2, freq_sig, f3):
    """
    Calculating rho sq from the merger stage.

    Parameters:
      noise_tab (array of dtype float): The noise power spectrum
      mod_shape (function): The (normalized) shape of modulus of waveform
      f_2,3 (Optional[1-D ArrayLike float]): the lower and upper frequency limits of the ringdown waveform.
      freq_sig, frequency width of the Lorentz function.

    Returns:
      (1-D ArrayLike float): rho sq
    """
    freq = noise_tab[0]
    # all frequency in the noise_tab curve
    nois = noise_tab[1]
#    # all noise power in the frequency range
    temp = np.diff(freq)
    wr=f1**-0.5*f2**-0.66667*freq_sig*np.pi*0.5
    if  len(temp)>0:
        freq_steps = np.append(temp,temp[-1])
    # to make the length of freq_step the same as freq_cut.
        hsq = mod_shape(freq,f2,freq_sig)**2
        if np.isscalar(f2)==True:
                hsq=np.where(f2<freq,hsq,0)*np.where(f3>freq,1,0)
                return np.sum(hsq/nois*freq_steps)*wr**2
        elif np.array(f2).any()!=None:
                hsq_1=np.where(np.less(np.einsum('i,j->ij',np.ones(len(f2)),freq),np.einsum('i,j->ij',f3,np.ones(len(freq)))),hsq,0)
                hsq_2=np.where(np.greater(np.einsum('i,j->ij',np.ones(len(f2)),freq),np.einsum('i,j->ij',f2,np.ones(len(freq)))),1,0)
                hsq=hsq_1*hsq_2
                return np.sum(hsq/nois*freq_steps,axis=1)*wr**2
        else:
                return np.sum(hsq/nois*freq_steps)*wr**2
    else: return 0
def fis_inf_matr(part_mat, noise_tab, f_up=None, numpar=5):
    """
    The function calculates the Fisher information matrix.

    Parameters:
      part_mat (a function of frequency, whose return value is a numpar*numpar numpy array)
      noise_tab (array of dtype float): The noise power spectrum
      f_up (Optional[float]): the upper limit of frequency
      numpar (int): the number of parameters that determine the waveform (phase)

    Returns:
      (array of dtype float): The Fisher Information Matrix
    """
    freq = noise_tab[0]
    # all frequency in the noise_tab curve
    nois = noise_tab[1]
    # all noise power in the frequency range

    if f_up!=None:
        freq_cut = freq[freq<f_up]
        # cut frequency beyond f_up
        nois_cut = nois[freq<f_up]
    else:
        freq_cut = freq
        nois_cut = nois
    # cut frequency beyond f_up
    temp = np.diff(freq_cut)
    freq_steps = np.append(temp,temp[-1])

    #FIM = np.zeros((numpar,numpar))
    #FIM_integrand=[part_mat(f)/nois*freq_step for (f,nois,freq_step) in zip(freq_cut,nois_cut,freq_steps)]
    FIM_integrand=part_mat(freq_cut)/np.array(nois_cut)*np.array(freq_steps)
    # the shape of FIM_integrand is (len(freq_cut), numpar, numpar)
    #print(part_mat(freq_cut).shape)
    #Pij_list=[part_mat(f) for f in freq_cut]
    FIM= 4.*np.sum(FIM_integrand,axis=2)
    # sum along axis 0 makes FIM the shape of (numpar, numpar)

    #for i in range(0,numpar):
    #    for j in range(i,numpar):
            #Pij=lambda f: PartMat(f,i,j)
    #        Pij_list = [part_mat(f,i,j) for f in freq_cut]
    #        FIM[(i,j)] = 4.*np.sum(Pij_list/nois_cut*freq_steps)
    #return np.matrix(FIM)+np.matrix(FIM).H-np.diag(np.diag(FIM))
    return np.matrix(FIM)

def errors(FIM):
    """
    To return errors from Fisher Information Matrix.

    Parameters:
      FIM (array of dtype float): FIM

    Returns:
      (array of dtype float): errors
    """
    a = np.linalg.inv(FIM)
    return a

def FIM_conv_matr(m1, m2, dimension):
    """
    To convert the FIM from base (M,eta) to (m1,m2). This matrix is needed for calculating the uncertainty of chirp mass.

    Parameters:
      XXXXXXXXX
      XXXXXXXXX
      XXXXXXXXX

    Returns:
      (array of dtype float): a
    """
    a = np.zeros((5,5))
    product = m1*m2
    sum_m1m2 = m1+m2
    a[(0,0)] = 3./5*m2*product**(-2./5)/sum_m1m2**(1./5)-1./5*product**(3./5)/sum_m1m2**(6./5);
    a[(1,0)] = 3./5*m1*product**(-2./5)/sum_m1m2**(1./5)-1./5*product**(3./5)/sum_m1m2**(6./5);
    a[(0,1)] = m2/sum_m1m2**2-2.*product/sum_m1m2**3
    a[(1,1)] = m1/sum_m1m2**2-2.*product/sum_m1m2**3
    a[(2,2)] = 1
    a[(3,3)] = 1
    a[(4,4)] = 1
    return a
#
#def conv_err(cov, convt)
#    a = np.linalg.inv(np.transpose(convt))
#    b = np.linalg.inv(convt)
#    c = np.dot(a,cov)
#    d = np.dot(c,b)
#    dm1,dm2 = np.sqrt(np.diag(d))
#    # cov:  covriance matrix of Mch and eta
#    # convt: the coordinate converting matrix from (Mch, eta) to (m1, m2).
#    return [dm1,dm2]
#
def mcmcunit(p):
    func, nsample, inits, ranges, T, rho_cri, ant_fun, noise_tab, cosmos, seed, chi_sig = p
    """
    func is the target distribution
    nsample is the number of accepted steps
    cosmos is the cosmology model
    """
    # initializing:
    burnin_steps=500 # For Science Paper I: 1000
    skipping=1 # For Science Paper I: 15
    np.random.seed(seed)
    z0, m10, m20, chi0 = inits
    z,m1,m2 = [z0,m10,m20]
    length = 1
    samples_z = np.array([z])
    samples_m1 = np.array([m1])
    samples_m2 = np.array([m2])
    z_range, m1_range, m2_range, chi_range = ranges
    while length < nsample*skipping+burnin_steps:
        #if length<burnin_steps:
        #    print('burning in')
        #elif length%50==0:
        #    print('sampling: %d' % (length/skipping))
        # sampling along z-dimension:
        pdf = lambda z: func(T,z,m1*np.ones(len(z)),m2*np.ones(len(z)),rho_cri,ant_fun,noise_tab,accurate=False)
        z=sample1D(pdf,z_range,1)
        # sampling along m1-dimension:
        pdf = lambda m1:func(T,z*np.ones(len(m1)),m1,m2*np.ones(len(m1)),rho_cri,ant_fun,noise_tab,accurate=False)
        m1=sample1D(pdf,[m2,m1_range[1]],1)
        # sampling along m2-dimension:
        pdf = lambda m2:func(T,z*np.ones(len(m2)),m1*np.ones(len(m2)),m2,rho_cri,ant_fun,noise_tab,accurate=False)
        m2=sample1D(pdf,[m2_range[0],m1],1)
        samples_z = np.append(samples_z,[z])
        samples_m1 = np.append(samples_m1,[m1])
        samples_m2 = np.append(samples_m2,[m2])
        length += 1
    samples_D = cosmos.luminosity_distance(samples_z).value
    samples_Mc = chirp_mass(samples_m1,samples_m2)
    samples_Mc_short=samples_Mc[burnin_steps::skipping]
    samples_z_short=samples_z[burnin_steps::skipping]
    samples_m1_short=samples_m1[burnin_steps::skipping]
    samples_m2_short=samples_m2[burnin_steps::skipping]
    samples_chi_short=np.random.normal(loc=0, scale=chi_sig, size=nsample)
    #samples_chi_short=samples_chi[burnin_steps::]
    samples_D_short=samples_D[burnin_steps::skipping]
    cat=[samples_Mc_short,samples_z_short,samples_m1_short,samples_m2_short,samples_chi_short,samples_D_short]
    return cat

def mcmcunit_old(p):
    func, nsample, inits, T, rho_cri, ant_fun, noise_tab, cosmos, seed, chi_sig = p
    """
    func is the target distribution
    nsample is the number of accepted steps
    cosmos is the cosmology model
    """
    np.random.seed(seed)
    #if nsample<=1000:
    #    burnin_steps=1000
    #    skipping=20
    #else:
    #    burnin_steps=1000
    #    skipping=10
    burnin_steps=500
    skipping=max(int(100/nsample),1) # should be 10000 in you want ensure convergence
    z0, m10, m20, chi0 = inits
    step_z=min(0.5*z0, 0.02)
    step_m1=min(0.1*m10,1)
    step_m2=min(0.1*m20,1)
    #step_m1=0.3*m10
    #step_m2=0.3*m20
    step_chi=0.1
    z_ranges = [0,20]
    m1_ranges = [0,100]
    m2_ranges = [0,100]
    chi_ranges = [-1,1]
    # MCMC sampling according to distribution function(z,m1,m2)
    z,m1,m2 = [z0,m10,m20]
    # inital parameters
    length = 1
    samples_z = np.array([z])
    samples_m1 = np.array([m1])
    samples_m2 = np.array([m2])
    #samples_chi = np.array([chi])
    #i=0;
    while length < nsample*skipping+burnin_steps:
        #step_m1=0.3*m1
        #step_m2=0.3*m2
        #step_m1=.5
        #step_m2=.5
        print(i);
        #i+=1;
        #step_z=min(z,1)
        z_next,m1_next,m2_next = [z,m1,m2]+np.random.normal(scale=[step_z,step_m1, step_m2],size=3)

        #m2_next=np.random.uniform(low=0,high=1,size=None)*m1_next
        temp_max=max(m1_next, m2_next)
        temp_min=min(m1_next, m2_next)
        m1_next=temp_max
        m2_next=temp_min
        if z_ranges[0]<z_next<z_ranges[1] and m1_ranges[0]<m1_next<m1_ranges[1] and m2_ranges[0]<m2_next<m2_ranges[1]:
            #print("above ")
            p0 =func (T,z,m1,m2,rho_cri,ant_fun,noise_tab, accurate=False)
            #print("below")
            if length<burnin_steps:
                temperature=1. # low temperature make it goes to peak quicker
            else:
                temperature=1. # high temperautre make it jumps around easier
            pac = min(1,(func(T,z_next,m1_next,m2_next,rho_cri,ant_fun,noise_tab,accurate=False)/(p0+1e-3))**(1./temperature)) # accept probability

            if pac==1 or np.random.uniform(low=0.0, high=1.0, size=None) <= pac:
                z,m1,m2 = [z_next,m1_next,m2_next]
                samples_z = np.append(samples_z,[z])
                samples_m1 = np.append(samples_m1,[max(m1,m2)])
                samples_m2 = np.append(samples_m2,[min(m1,m2)])
                #samples_chi = np.append(samples_chi,[chi])
                length += 1

    samples_D = cosmos.luminosity_distance(samples_z).value
    samples_Mc = chirp_mass(samples_m1,samples_m2)

    samples_Mc_short=samples_Mc[burnin_steps::skipping]
    samples_z_short=samples_z[burnin_steps::skipping]
    samples_m1_short=samples_m1[burnin_steps::skipping]
    samples_m2_short=samples_m2[burnin_steps::skipping]
    samples_chi_short=np.random.normal(loc=0, scale=chi_sig, size=nsample)
    #samples_chi_short=samples_chi[burnin_steps::]
    samples_D_short=samples_D[burnin_steps::skipping]
    cat=[samples_Mc_short,samples_z_short,samples_m1_short,samples_m2_short,samples_chi_short,samples_D_short]
    return cat
def mcmcunit_scout_bhbh_zeus(p):
    func, nsample, inits, T, rho_cri, ant_fun, noise_tab, cosmos, seed, chi_sig = p
    #ivar=[T, rho_cri, ant_fun, noise_tab]
    z0, m10, m20, chi0 = inits
    mean=np.array([np.log10(z0),np.log10(m10),np.log10(m20)])
    covariance=np.array([[2,0,0],[0,1,0], [0,0,1]])
    loglike= lambda f: np.log(1e-8+func(T,10**f[0],10**f[1],10**f[2],rho_cri, ant_fun, noise_tab, accurate=False))
    #loglike=lambda f: np.log(multivariate_normal.pdf(f, mean=mean, cov=covariance))
    start_1=np.array([np.log10(z0),np.log10(m10),np.log10(m20)])
    #start=np.tile(start_1, (6,1))
    start=np.random.normal(loc=start_1,scale=np.ones(3)*0.001,size=(6,3))
    sampler=zeus.EnsembleSampler(6, 3, loglike, args=[])
    sampler.run_mcmc(start, nsample)
    samples=sampler.get_chain(flat=True)
    samples_z, samples_m1, samples_m2=[10**samples[:,0],10**samples[:,1],10**samples[:,2]]
    samples_D = cosmos.luminosity_distance(samples_z).value
    samples_Mc = chirp_mass(samples_m1,samples_m2)
    samples_chi_short=np.random.normal(loc=0, scale=chi_sig, size=nsample)
    cat=[samples_Mc,samples_z,samples_m1,samples_m2,samples_chi_short,samples_D]
    return cat
def mcmcunit_scout_bhbh(p):
    burnin=1000 # that's in my laptop
    func, nsample, inits, T, rho_cri, ant_fun, noise_tab, cosmos, seed, chi_sig = p
    nskip=max(int(3000/nsample),1)
    loglike= lambda f: f[0]+f[1]+f[2]+np.log(1e-100+func (T,np.exp(f[0]),np.exp(f[1]),np.exp(f[2]),rho_cri,ant_fun,noise_tab, accurate=False))
    z0, m10, m20, chi0 = inits
    mean=np.array([np.log(z0),np.log(m10),np.log(m20)])
    covariance=np.array([[1,0,0],[0,1,0], [0,0,1]])
    logpri=lambda f: np.log(multivariate_normal.pdf(f, mean=mean, cov=covariance))
    logtarg=lambda f: loglike(f)-logpri(f)
    ess_sampler=EllipticalSliceSampler(mean=mean, covariance=covariance,loglik=logtarg)
    samples=ess_sampler.sample(n_samples=nsample, burnin=burnin, nskip=nskip, seed=seed)
    samples_z, samples_m1, samples_m2=[np.exp(samples[:,0]),np.exp(samples[:,1]),np.exp(samples[:,2])]
    samples_D = cosmos.luminosity_distance(samples_z).value
    samples_Mc = chirp_mass(samples_m1,samples_m2)
    #samples_chi_short=np.random.normal(loc=0, scale=chi_sig, size=nsample)
    # the above normal distribution is replaced with truncated normal between -1,1.
    a,b=-1/chi_sig,1/chi_sig
    samples_chi_short=truncnorm.rvs(a,b,loc=0,scale=chi_sig, size=nsample)

    cat=[samples_Mc,samples_z,samples_m1,samples_m2,samples_chi_short,samples_D]
    return cat

def mcmcunit_scout_bhbh_orign(p):
    func, nsample, inits, T, rho_cri, ant_fun, noise_tab, cosmos, seed, chi_sig = p
    """
    func is the target distribution
    nsample is the number of accepted steps
    cosmos is the cosmology model
    """
    np.random.seed(seed)
    #if nsample<=1000:
    #    burnin_steps=1000
    #    skipping=20
    #else:
    #    burnin_steps=1000
    #    skipping=10
    z0, m10, m20, chi0 = inits
    if z0<=5: # 2G
        burnin_steps=1000
        skipping=max(int(10000/nsample),1) # should be 10000 in you want ensure convergence
    else:
        burnin_steps=1000
        skipping=max(int(6000/nsample),1) # should be 10000 in you want ensure convergence
    #step_z=min(0.5*z0, 0.02)
    step_m1=min(0.2*m10,1.)
    step_m2=min(0.2*m20,1.)
    #step_m1=0.3*m10
    #step_m2=0.3*m20
    step_chi=0.1
    z_ranges = [0,20]
    m1_ranges = [0,100]
    m2_ranges = [0,100]
    chi_ranges = [-1,1]
    # MCMC sampling according to distribution function(z,m1,m2)
    z,m1,m2 = [z0,m10,m20]
    # inital parameters
    length = 1
    samples_z = np.array([z])
    samples_m1 = np.array([m1])
    samples_m2 = np.array([m2])
    #samples_chi = np.array([chi])

    while length < nsample*skipping+burnin_steps:
        #step_m1=0.3*m1
        #step_m2=0.3*m2
        #step_m1=max(0.1*m1,1.1)
        #step_m2=max(0.1*m2,1.1)
        n_scout=10
        step_z=min(2*abs(z),1)
        z_scout=np.abs(np.random.normal(loc=z, scale=step_z,size=n_scout))
        m1_scout = np.random.normal(loc=m1, scale=step_m1,size=n_scout)
        m2_scout= np.random.normal(loc=m2, scale=step_m2,size=n_scout)
        #temp1=np.where(m1_scout>m2_scout, m1_scout,m2_scout)
        #temp2=np.where(m1_scout<=m2_scout, m1_scout, m2_scout)
        #m1_scout=temp1
        #m2_scout=temp2
        p0 =func (T,z,m1,m2,rho_cri,ant_fun,noise_tab, accurate=False)
        if length<burnin_steps:
            temperature=0.5 # low temperature make it goes to peak quicker
        else:
            temperature=1. # high temperautre make it jumps around easier
        p_scout = func(T,z_scout,m1_scout,m2_scout,rho_cri,ant_fun,noise_tab,accurate=False)
        pac_scout = np.where(p_scout>=p0, 1, p_scout/p0) # nparray len=n_scout
        dice=np.random.uniform(low=0.0, high=1.0, size=n_scout)
        indice_accept=np.arange(n_scout)[pac_scout>=dice]
        if len(indice_accept)>0: # accpeted
            index=np.random.choice(indice_accept, size=1)
            z,m1,m2=[z_scout[index], m1_scout[index], m2_scout[index]]
            samples_z = np.append(samples_z,[z])
            samples_m1 = np.append(samples_m1,[max(m1,m2)])
            samples_m2 = np.append(samples_m2,[min(m1,m2)])
            #samples_chi = np.append(samples_chi,[chi])
            length += 1
        else: # rejected
            pass
    samples_D = cosmos.luminosity_distance(samples_z).value
    samples_Mc = chirp_mass(samples_m1,samples_m2)
    samples_Mc_short=samples_Mc[burnin_steps::skipping]
    samples_z_short=samples_z[burnin_steps::skipping]
    samples_m1_short=samples_m1[burnin_steps::skipping]
    samples_m2_short=samples_m2[burnin_steps::skipping]
    samples_chi_short=np.random.normal(loc=0, scale=chi_sig, size=nsample)
    samples_D_short=samples_D[burnin_steps::skipping]
    cat=[samples_Mc_short,samples_z_short,samples_m1_short,samples_m2_short,samples_chi_short,samples_D_short]
    return cat

def give_iota(det, pop, dtb):
    '''
    inputs:
    det: the detector class
    pop: the population class
    dtb: the detectability, can be float or list or np.array
    '''
    N=10000
    theta=np.arccos(np.random.uniform(0,1,size=N))
    varphi=np.random.uniform(0,2*np.pi,size=N)
    psi=np.random.uniform(0,np.pi,size=N)
    iota=np.arccos(np.random.uniform(0,1,size=N))
    F=det.ante_pattern(theta=theta,varphi=varphi,psi=psi)
    scal_factors=pop.mod_norm(1,F, iota, 1)
    indice=np.argsort(scal_factors)
    if np.isscalar(dtb):
        index=np.random.choice(indice[1-int(len(indice)*dtb):])
        result=iota[index]
    else:
       result=np.array([iota[np.random.choice(indice[1-int(len(indice)*dtb_):])] for dtb_ in dtb])
    return result

def give_angles(det, pop, dtb):
    '''
    inputs:
    det: the detector class
    pop: the population class
    dtb: the detectability, can be float or list or np.array
    '''
    N=10000
    theta=np.arccos(np.random.uniform(0,1,size=N))
    varphi=np.random.uniform(0,2*np.pi,size=N)
    psi=np.random.uniform(0,np.pi,size=N)
    iota=np.arccos(np.random.uniform(0,1,size=N))
    F=det.ante_pattern(theta=theta,varphi=varphi,psi=psi)
    scal_factors=pop.mod_norm(1,F, iota, 1)
    indice=np.argsort(scal_factors)
    if np.isscalar(dtb):
        index=np.random.choice(indice[1-int(len(indice)*dtb):])
        result=[theta[index], varphi[index], iota[index], psi[index]]
    else:
        THETA=np.array([theta[np.random.choice(indice[1-int(len(indice)*dtb_):])] for dtb_ in dtb])
        VARPHI=np.array([varphi[np.random.choice(indice[1-int(len(indice)*dtb_):])] for dtb_ in dtb])
        IOTA=np.array([iota[np.random.choice(indice[1-int(len(indice)*dtb_):])] for dtb_ in dtb])
        PSI=np.array([psi[np.random.choice(indice[1-int(len(indice)*dtb_):])] for dtb_ in dtb])
        result=[THETA,VARPHI,IOTA,PSI]
    return result
