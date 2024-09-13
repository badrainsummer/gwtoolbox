Z_HIGH = 10.

#thetaBHB = [1.4,0.07,12,1.3,12.5,-0.1,2,-0.1,10,0.1]
# The parameters for the "default model" of Double Black Holes.

BBH_Pop1 = [13, 3, 3, 6, 2.5, 95,0.4, 0.1]
# R0, tau, mu, c, gamma, mcut, ql, sig_x
BBH_Pop2=  [13, 3, 3, 6, 2.5, 95, 40, .002, 1, 0.4, 0.1]
# R0, tau, mu, c, gamma, mcut, m_peak, m_peak_scale, m_peak_sig, ql, sig_x
BBH_Pop3 = [13, 3, 2.7, 5.6, 2.9, 3,0.1, 6 ,0.1, 2.5, 0.1, 95, 0.4, 0, 0.1]
# R0, tau, alpha, beta, C, mu_0, mu_1, c_0,c_1,gamma_0,gamma_1,mcut,ql_0,ql_1,sig_x
# The parameters for the model "Yi et al. (2020)" of Double BHs.

DNS_Pop1 = [300,3,1.4,0.5,1.1,2.5, 0.1]
# R0, tau, m_mean, m_sclae, m_low, m_high, chi_sigma
# The parameters for the "default model" of Double neutron stars.

BHNS_Pop1 = [45, 3, 1.4, 0.5, 1.1, 2.5, 3, 15, 2.5, 95, 0.1]
# The parameters for the "default model" of black hole - neutron star.
# R0, tau, m_mean, m_sclae, m_low, m_high, mu, c, gamma, mcut, chi_sigma
BHNS_Pop2 = [45, 3, 1.4, 0.5, 1.1, 2.5, 3, 15, 2.5, 95, 40, .002, 0.4, 0.1]
# R0, tau, m_mean, m_sclae, m_low, m_high, mu, c, gamma, mcut, m_peak, m_peak_scale, m_peak_sig, chi_sigma

BG_DNS=[1e30, 0.1, 6e3]
BG_BBH=[5500, 0.18, 1]
# Rn, alpha

detector_dict = {0:'virgo',1:'ligo',2:'et',3:'lisa', 4:'kagra',-1:'ligo-like',-2:'et-like',-3:'lisa-like'}

param_dict = {0:'Mc',1:'z',2:'m1',3:'m2',4:'chi',5:'D'}

param_labels = {0:'$M_c$',1:'$z$',2:'$m_1$',3:'$m_2$',4:'$\chi$',5:'$D$'}

param_error_dict = {0:'dm1',1:'dm2',2:'dchi'}

param_error_labels = {0:'$\Delta m_1$',1:'$\Delta m_2$',2:'$\Delta \chi$'}

populations = {0:'bhbh', 0.1: 'bhbh_yi' ,1:'nsns',2:'bhns',3:'smbh',4:'emri',5:'insp'}

TIME_OBS_EARTH = 43829.0639
# in minutes

RHO_CRIT_EARTH = 8.

SAMPLE_SIZE = 10

import astropy.cosmology
cosmology_dict = {'Planck13':astropy.cosmology.Planck13,'Planck15':astropy.cosmology.Planck15,'WMAP5':astropy.cosmology.WMAP5,'WMAP7':astropy.cosmology.WMAP7,'WMAP9':astropy.cosmology.WMAP9}

MBHBunits = {'EclipticLatitude':                 'Radian',\
         'EclipticLongitude':                'Radian',\
         'PolarAngleOfSpin1':                'Radian',\
         'PolarAngleOfSpin2':                'Radian',\
         'AzimuthalAngleOfSpin1':            'Radian',\
         'AzimuthalAngleOfSpin2':            'Radian',\
         'Spin1':                            'MassSquared',\
         'Spin2':                            'MassSquared',\
         'Mass1':                            'SolarMass',\
         'Mass2':                            'SolarMass',\
         'CoalescenceTime':                  'Second',\
         'PhaseAtCoalescence':               'Radian',\
         'InitialPolarAngleL':               'Radian',\
         'InitialAzimuthalAngleL':           'Radian',\
         'Approximant':                                      'ModelName',\
         'Cadence':                          'Seconds',\
         'Redshift':                         'dimensionless',\
         'Distance':                         'Gpc',
         'ObservationDuration':              'Seconds',
         'SourceType':                       'name'}

mHz_pops={0:'MBHB', 1:'GB', 2:'EMRI'}
mHz_models={0.1:'pop3', 0.2:'Q3_delays', 0.3:'Q3_nodelays', 1.1:'Gijs',1.2:'VBs', 2.01:'M1', 2.02:'M2', 2.03:'M3', 2.04:'M4', 2.05:'M5', 2.06:'M6', 2.07:'M7', 2.08:'M8', 2.09:'M9', 2.1:'M10', 2.11:'M11'}
MBHBKys=['Mass1', 'Mass2', 'Redshift', 'Spin1', 'Spin2', 'AzimuthalAngleOfSpin1', 'AzimuthalAngleOfSpin2', 'CoalescenceTime', 'Distance', 'EclipticLatitude', 'EclipticLongitude', 'PhaseAtCoalescence']
MBHBerkys=['dm1','dm2','ds1','ds2','dD','dOme']
GBKys=['Frequency', 'FrequencyDerivative', 'EclipticLatitude', 'EclipticLongitude', 'Amplitude', 'Inclination', 'Polarization', 'InitialPhase']
GBerkys=['deltaF', 'deltaAmp', 'deltaOmega']
EMRIKys=['mu','M','s','e','iota','gamma','psi','theta_S','phi_S','theta_K','phi_K','alpha','D']
EMRIerkys=['deltamu','deltaM','deltaD','deltaOmega']
VBsNames=['J0806', 'V407 Vul',
 'ES Cet', 'SDSS J135154.46--064309.0',
 'AM CVn', 'SDSS J190817.07+394036.4 ',
 'HP Lib                   ', 'PTF1 J191905.19+481506.2 ',
 'ASASSN-14cc              ', 'CXOGBS J175107.6--294037 ',
 'CR Boo                   ', 'KL Dra                   ',
 'V803 Cen                 ', 'PTF1 J071912.13+485834.0 ',
 'SDSS J092638.71+362402.4 ', 'CP Eri                   ',
 'SDSS J104325.08+563258.1 ', 'CRTS J0910-2008          ',
 'CRTS J0105+1903          ', 'V406 Hya/2003aw          ',
 'SDSS J173047.59+554518.5 ', '2QZ J142701.6--012310    ',
 'SDSS J124058.03--015919.2', 'NSV1440                  ',
 'SDSS J012940.05+384210.4 ', 'SDSS J172102.48+273301.2 ',
 'ASASSN-14mv              ', 'ASASSN-14ei              ',
 'SDSS J152509.57+360054.5 ', 'SDSS J080449.49+161624.8 ',
 'SDSS J141118.31+481257.6 ', 'GP Com                   ',
 'SDSS J090221.35+381941.9 ', 'ASASSN-14cn              ',
 'SDSS J120841.96+355025.2 ', 'SDSS J164228.06+193410.0 ',
 'SDSS J155252.48+320150.9 ', 'SDSS J113732.32+405458.3 ',
 'V396 Hya/CE 315          ', 'SDSS J1319+5915          ',
 'ZTF J153932.16+502738.8  ', 'SDSS J065133.34+284423.4 ',
 'SDSS J093506.92+441107.0 ', 'SDSS J232230.20+050942.06',
 'PTF J053332.05+020911.6  ', 'SDSS J010657.39--100003.3',
 'SDSS J163030.58+423305.7 ', 'SDSS J082239.54+304857.2 ',
 'ZTF J190125.42+530929.5  ', 'SDSS J104336.27+055149.9 ',
 'SDSS J105353.89+520031.0 ', 'SDSS J005648.23--061141.5',
 'SDSS J105611.02+653631.5 ', 'SDSS J092345.59+302805.0 ',
 'SDSS J143633.28+501026.9 ', 'SDSS J082511.90+115236.4 ',
 'WD 0957--666             ', 'SDSS J174140.49+652638.7 ',
 'SDSS J075552.40+490627.9 ', 'SDSS J233821.51--205222.8',
 'SDSS J230919.90+260346.7 ', 'SDSS J084910.13+044528.7 ',
 'SDSS J002207.65--101423.5', 'SDSS J075141.18--014120.9',
 'SDSS J211921.96--001825.8', 'SDSS J123410.36--022802.8',
 'SDSS J100559.10+224932.2 ', 'SDSS J115219.99+024814.4 ',
 'SDSS J105435.78--212155.9', 'SDSS J074511.56+194926.5 ',
 'WD 1242--105             ', 'SDSS J110815.50+151246.6 ',
 'WD 1101+364         ', 'WD 1704+4807BC           ',
 'SDSS J011210.25+183503.7 ', 'SDSS J123316.20+160204.6 ',
 'SDSS J113017.42+385549.9 ', 'SDSS J111215.82+111745.0 ',
 'SDSS J100554.05+355014.2 ', 'SDSS J144342.74+150938.6 ',
 'SDSS J184037.78+642312.3 ']
