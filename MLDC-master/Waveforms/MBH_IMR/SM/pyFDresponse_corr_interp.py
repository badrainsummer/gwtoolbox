import sys
import numpy as np
from numpy import pi, conjugate, dot, sqrt, cos, sin, tan, exp, real, imag, arccos, arcsin, arctan, arctan2
import scipy
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline

import pyIMRPhenomD
import LISAConstants as LC

# Constants - from LALConstants.h
# Gravitational constant, N m^2 kg^-2
#G_SI = 6.67384e-11
G_SI = LC.G
# Speed of light in vacuo, m s^-1
#C_SI = 299792458e0
C_SI = LC.clight
# Parsec, m
#PC_SI = 3.085677581491367278913937957796471611e16
PC_SI = LC.pc
# Astronomical unit, m
#AU_SI = 149597870700e0
AU_SI = LC.ua
# Sidereal year (2000), s
# see http://hpiers.obspm.fr/eop-pc/models/constants.html
#YRSID_SI = 31558149.763545600
YRSID_SI = LC.YRSID_SI
# Solar mass, kg
# MSUN_SI = LAL_GMSUN_SI / LAL_G_SI
MSUN_SI = LC.MsunKG
#MSUN_SI = 1.988546954961461467461011951140572744e30


# Geometrized solar mass, s
# MTSUN_SI = LAL_GMSUN_SI / (LAL_C_SI * LAL_C_SI * LAL_C_SI)
#MTSUN_SI = 4.925491025543575903411922162094833998e-6
MTSUN_SI = LC.MTsun

# Armlength in m
L_SI = 2.5e9
# Orbit around the Sun
R_SI = AU_SI
# LISA orbital frequency
f0 = 1./YRSID_SI
Omega0 = 2*pi*f0

# Numerical constants
sqrt2 = sqrt(2)
sqrt3 = sqrt(3)
sqrt6 = sqrt(6)
invsqrt2 = 1./sqrt(2)
invsqrt3 = 1./sqrt(3)
invsqrt6 = 1./sqrt(6)
invpi = 1./pi

# Constants for the orbits
aorbit = R_SI
eorbit = L_SI/(2*sqrt3*aorbit)

# Wave-frame matrices for plus and cross - in that frame, propagation along z
HSplus = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]])
HScross = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])

# Functions for higher-order corrections of Fresnel type
# Stencil coefficients - only supports N<=20
coeffs_fresnel_stencil = [[1.],
[1.+1.j, 0.-1.j],
[0.25+1.25j, 1.-1.3333333333333333j, -0.25+0.08333333333333333j],
[-0.16666666666666666+0.9444444444444444j, 1.625-0.875j, -0.5-0.1j, 0.041666666666666664+0.030555555555555555j],
[-0.23958333333333334+0.6423611111111112j, 1.7416666666666667-0.39166666666666666j, -0.5583333333333333-0.3416666666666667j, 0.058333333333333334+0.0996031746031746j, -0.0020833333333333333-0.008630952380952382j],
[-0.19166666666666668+0.4636111111111111j, 1.6618055555555555-0.09375j, -0.5126984126984127-0.5119047619047619j, 0.041220238095238095+0.16344246031746032j, 0.0017195767195767196-0.022817460317460316j, -0.0003802910052910053+0.0014186507936507936j],
[-0.12648148148148147+0.3704398148148148j, 1.5500595238095238+0.06597222222222222j, -0.44285714285714284-0.6117311507936508j, 0.010179673721340389+0.2078097442680776j, 0.011031746031746031-0.0361276455026455j, -0.0020734126984126985+0.003838684463684464j, 0.00014109347442680775-0.00020166947250280584j],
[-0.07119708994708995+0.32227985638699924j, 1.4533118386243387+0.15025214947089946j, -0.3783586860670194-0.6679177689594357j, -0.02206955467372134+0.23590305335097j, 0.02275873817540484-0.04634339426006093j, -0.005005160734327401+0.00639262165303832j, 0.0005921316337983005-0.0005945828862495529j, -0.00003221701138367805+0.000028065243839053363j],
[-0.02864440133692366+0.29656112705498866j, 1.3776626144284874+0.1959743349500294j, -0.32540422912992356-0.6999232987948266j, -0.05095380391213725+0.2533606150793651j, 0.0347938420247448-0.05361737831355887j, -0.00870826961104739+0.008630770592576147j, 0.0013856549645240121-0.0010741862304362304j, -0.00013802012214710627+0.00009201235639727703j, 6.612694422714264e-6-3.996694534888979e-6j],
[0.0036205013769911257+0.2822347010220412j, 1.3195857895434409+0.22176190180933483j, -0.2831665383044351-0.7186778928743215j, -0.07559245689367217+0.26430079495907044j, 0.04616552801622246-0.058666692104192106j, -0.012769586036575124+0.010434096946373731j, 0.0024686726779980747-0.0015550732581155861j, -0.000341085943423493+0.00018217867408715623j, 0.000030502791043465646-0.000014604496616051238j, -1.3272275900417436e-6+5.893223378423477e-7j],
[0.028337604386104578+0.273935169693306j, 1.2746456022541437+0.23685195877067156j, -0.24946139783746224-0.729995435595324j, -0.09633408179642472+0.27126543663353353j, 0.056536340467598735-0.06214901294142366j, -0.016917911017125635+0.011827025281266352j, 0.0037650242344201093-0.00199036336276953j, -0.000646109839052207+0.00028459987518220185j, 0.00008134010698158463-0.00003167469679855884j, -6.678524004580584e-6+2.3861855149484112e-6j, 2.6756482072694205e-7-8.984315885530319e-8j],
[0.04761385435144253+0.2689727890020743j, 1.239305810651024+0.24594965670459634j, -0.22227694275813947-0.7369936647752662j, -0.11380980291884649+0.2757642982492106j, 0.06585672506622368-0.0645484058031181j, -0.020995579279024047+0.012876759658257674j, 0.005204201268031314-0.002360857848766467j, -0.0010458812372775416+0.00038751501018135105j, 0.0001655025066079709-0.00005334104100890604j, -0.00001930288394853852+5.636137146500491e-6j, 1.4698848153896027e-6-3.9936236186026306e-7j, -5.465090884830276e-8+1.406905468204363e-8j],
[0.06293752354757395+0.26593884298306175j, 1.2110159598273968+0.2515507878166195j, -0.2000492028252895-0.7413945535061415j, -0.12862829620741315+0.27869822406979416j, 0.07419212754104242-0.06619873907719635j, -0.024918121620115218+0.01365338708135332j, 0.00672963440067788-0.0026628796244147737j, -0.0015275969633764572+0.0004828903077545005j, 0.0002859314381326998-0.0000771848654021934j, -0.000042241728048486886+0.000010177817983317132j, 4.5979090108371065e-6-1.0186824759716234e-6j, -3.266530128002596e-7+6.792297764824887e-8j, 1.1333420997998203e-8-2.2439134569252184e-9j],
[0.07534246952390897+0.264061777336553j, 1.1879782030142032+0.2550367668744215j, -0.18161899737473464-0.744183336752383j, -0.1412990624546696+0.2806155125515853j, 0.08164551945119329-0.06732655583119111j, -0.02864481757519065+0.0142172954583507j, 0.008298769539657011-0.002900314730518934j, -0.0020767942620191528+0.0005659925948909567j, 0.00044284495203061284-0.00010092837601260946j, -0.00007790389029801257+0.000015574070394775326j, 0.00001080002418466766-1.957161156225222e-6j, -1.101917409529079e-6+1.8523281267994874e-7j, 7.335457273630375e-8-1.1628700259461208e-8j, -2.385428913011752e-9+3.609533385590765e-10j],
[0.08555152227917957+0.2629010680885456j, 1.168921304537698+0.2572034241373687j, -0.16613526736257422-0.7459437457785277j, -0.15222875422795934+0.28185815421709914j, 0.08832477553487034-0.06808594796011624j, -0.03216021551396804+0.01461697552620603j, 0.009880698612106838-0.003080170761053832j, -0.002679433908666706+0.00063450917795187j, 0.0006345939305093796-0.00012272910698653646j, -0.0001279253629446474+0.000021261217605364973j, 0.00002122116431938325-3.1419834917647324e-6j, -2.769299831083573e-6+3.748043863662703e-7j, 2.6574485214643767e-7-3.3502343377113704e-8j, -1.663656072116982e-8+1.9812231991259274e-9j, 5.089689931485025e-10-5.78667807345304e-11j],
[0.09407765476569514+0.26219568206416816j, 1.1529348061254814+0.25852602293307636j, -0.152969915728984-0.7470329447867575j, -0.1617370637411078+0.2826447979452651j, 0.0943300236484378-0.06858277557790528j, -0.03546310197643015+0.01489023071599j, 0.011453501689469745-0.0032102922799985795j, -0.0033228533494060776+0.0006877407084292669j, 0.0008583919968535089-0.0001412444219351962j, -0.0001931997989616851+0.000026661517798724075j, 0.0000368870289634723-4.438055538170916e-6j, -5.781966108793006e-6+6.240490106751518e-7j, 7.120657821774647e-7-7.04274729043554e-8j, -6.44566603673513e-8+5.937487077044683e-9j, 3.806906899781708e-9-3.307125654185825e-10j, -1.0993126355444017e-10+9.094859489468401e-12j],
[0.10129140663349086+0.2617858314080466j, 1.1393559790802188+0.2592975065210698j, -0.14165422652459855-0.747675847776752j, -0.17007493999697076+0.28311851593789267j, 0.09974964321474873-0.0688906922731132j, -0.03856002744289354+0.015066183113251674j, 0.013001964422701438-0.0032982684786294165j, -0.003996098016028552+0.0007259912295731092j, 0.0011108587468369369-0.00015558836736413706j, -0.00027398915895638206+0.00003125158033598514j, 0.00005863801050050609-5.67384160589505e-6j, -0.000010615517561467182+8.986681368360704e-7j, 1.575199970154996e-6-1.19466602575948e-7j, -1.8350965181252804e-7+1.2701504962781594e-8j, 1.571220604429938e-8-1.0071143539922735e-9j, -8.780150793297739e-10+5.27336845587388e-11j, 2.4002619242979178e-11-1.3637132834146997e-12j],
[0.10746530164195334+0.2615709071250154j, 1.1276941773975677+0.2597034746112399j, -0.13183376194973426-0.7480177156421585j, -0.17744028842811901+0.2833749168369475j, 0.10465987550218087-0.06906162620581639j, -0.041461528340012536+0.015167189528030835j, 0.014515790977720046-0.003350967477644631j, -0.004689935187078748+0.0007501449374550825j, 0.0013883936152570154-0.0001652498505169264j, -0.00037005892110179384+0.00003459593988887377j, 0.00008710312521025774-6.664762954899088e-6j, -0.00001773179623890509+1.14639847408708e-6j, 3.047533489624908e-6-1.7072115511063965e-7j, -4.288985717241801e-7+2.1243930385230196e-8j, 4.737529248451255e-8-2.109362795598545e-9j, -3.8464294330997585e-9+1.5606947595932672e-10j, 2.039065194714631e-10-7.626488519813969e-12j, -5.291291183190704e-12+1.8419927165880205e-13j],
[0.11280327904767755+0.2614860044564951j, 1.1175801149446165+0.2598643428252783j, -0.1232368088647258-0.7481544536240909j, -0.18399034792145877+0.28347909815651523j, 0.10912582515673072-0.06913265892370349j, -0.04417993247756461+0.0152104268345708j, 0.015988259885560754-0.00337438768535378j, -0.0053967202628422885+0.0007613866371554739j, 0.001687418070387744-0.00017000595423632275j, -0.00048080871929835996+0.00003635745978494649j, 0.00012270127463058256-7.230965778636748e-6j, -0.000027551975389339526+1.3025923564974686e-6j, 5.338908624726277e-6-2.0716639433973035e-7j, -8.723905333567029e-7+2.8297847655376785e-8j, 1.1667091148959425e-7-3.2115373690589492e-9j, -1.2245898403412694e-8+2.896663939545272e-10j, 9.45036134499075e-10-1.941445187233166e-11j, -4.7641554899054234e-11+8.577971775169558e-13j, 1.1763962143295425e-12-1.8711052940504268e-14j],
[0.11746044935942912+0.2614883728878653j, 1.1087314913522883+0.2598598428056749j, -0.11565227435701607-0.7481505964644312j, -0.18985112458650721+0.2834761176240506j, 0.11320288718459051-0.06913058550981509j, -0.04672809624497698+0.015209130950890555j, 0.017415231595311678-0.003373661990492842j, -0.0061102061177177504+0.0007610237897250047j, 0.002004522894776838-0.00016984468871166977j, -0.0006053856145940755+0.00003629410547168995j, 0.00016565882473255341-7.209119463720701e-6j, -0.00004043924041993078+1.2960384620226549e-6j, 8.664654439072408e-6-2.0547506673332675e-7j, -1.599897430244919e-6+2.7927869741475997e-8j, 2.4894489274199717e-7-3.144268657440625e-9j, -3.169795446994254e-8+2.7977393636359727e-10j, 3.1681282563882e-9-1.8283885290511092e-11j, -2.32899231723148e-10+7.635832956985752e-13j, 1.1190324691307583e-11-1.3618410680051262e-14j, -2.635244336046853e-13-1.3401690159086854e-16j],
[0.1215565660394987+0.2615493533684122j, 1.100929364342632+0.25974368950939525j, -0.10891407375776743-0.7480502822540077j, -0.19512449896852788+0.2833976108506758j, 0.1169381940385218-0.06907497654534125j, -0.04911869263149301+0.0151735412136273j, 0.0187944218183017-0.0033531294497640405j, -0.006825341788897761+0.0007503772871248855j, 0.002336550170681843-0.00016490166964732876j, -0.0007427762115202844+0.000034248718272652295j, 0.00021603537693883003-6.459144157406893e-6j, -0.000056689741131632924+1.0541109438569102e-6j, 0.000013235107764238632-1.374329522492111e-7j, -2.707886115133701e-6+1.1432811684720687e-8j, 4.770602102190994e-7+2.51772707185469e-10j, -7.080343746601718e-8-3.024045832865903e-10j, 8.599445339176345e-9+6.257424243868163e-11j, -8.200686460786233e-10-7.977835918268205e-12j, 5.7545804772003e-11+6.764936325278525e-13j, -2.6407285403070145e-12-3.552437809173978e-14j, 5.943010266755823e-14+8.847590297537227e-16j]]

def ComputehphcFromAmpPhase(wf, freq, iota, phi):
    f, amp, phase = wf
    amp_Int = spline(f, amp)
    phase_Int = spline(f, phase)
    indices = (f[0]<=freq) & (freq<=f[-1])
    factorp = 1/2. * (SpinWeightedSphericalHarmonic(-2, 2, 2, iota, phi) + np.conj(SpinWeightedSphericalHarmonic(-2, 2, -2, iota, phi)))
    factorc = 1j/2. * (SpinWeightedSphericalHarmonic(-2, 2, 2, iota, phi) - np.conj(SpinWeightedSphericalHarmonic(-2, 2, -2, iota, phi)))
    h22tilde = np.zeros_like(freq, dtype=np.complex128)
    h22tilde[indices] = amp_Int(freq[indices]) * np.exp(1j*phase_Int(freq[indices]))
    hptilde = factorp * h22tilde
    hctilde = factorc * h22tilde
    return freq, hptilde, hctilde

# Compute hp, hc on input frequency grid by interpolating amp/phase
def Computeh22FromAmpPhase(wf, freq):
    f, amp, phase = wf
    amp_Int = spline(f, amp)
    phase_Int = spline(f, phase)
    indices = (f[0]<=freq) & (freq<=f[-1])
    h22tilde = np.zeros_like(freq, dtype=np.complex128)
    h22tilde[indices] = amp_Int(freq[indices]) * np.exp(1j*phase_Int(freq[indices]))
    return freq, h22tilde

# Compute Tf and its sign
def func_compute_Tf(freq, wf_freq, wf_phase):
    dffphasespline = spline(wf_freq, wf_phase).derivative(2)
    epsTf2vec = 1./(4*pi*pi)*dffphasespline(freq)
    Tfvec = np.sqrt(np.abs(epsTf2vec))
    epsTfvec = np.sign(epsTf2vec)
    return Tfvec, epsTfvec
# Compute transfer function from stencil formula
def func_fresnel_stencil(func, t, Tf, epsTf, coeffs_stencil):
    if epsTf==1. or epsTf==0.: # sign can be 0 if Tf was exactly 0
        coeffs = coeffs_stencil
    else:
        coeffs = np.conj(coeffs_stencil)
    res = 0.
    for k in range(len(coeffs)):
        res += 0.5 * coeffs[k] * (func(t+k*Tf) + func(t-k*Tf))
    return res

# NOTE: numpy defines sinc(x) as sin(pi*x)/(pi*x)
def sinc(x):
    return np.sinc(invpi*x)

# Logarithmically spaced samples
def logsampling(start, stop, num=50.):
    res = np.exp(np.linspace(np.log(start), np.log(stop), num=num))
    res[0] = start # Eliminate possible rounding error
    res[-1] = stop # Eliminate possible rounding error
    return res

# Quadratic Lagrange interpolation polynomial
def quad_lagrange(x, y):
    res = np.zeros(3, dtype=y.dtype)
    if (not len(x)==3) or (not len(y)==3):
        raise ValueError('Only allows an input length of 3 for x and y.')
    c0 = y[0] / ((x[0]-x[1]) * (x[0]-x[2]))
    c1 = y[1] / ((x[1]-x[0]) * (x[1]-x[2]))
    c2 = y[2] / ((x[2]-x[0]) * (x[2]-x[1]))
    res[0] = c0*x[1]*x[2] + c1*x[2]*x[0] + c2*x[0]*x[1]
    res[1] = -c0*(x[1]+x[2]) - c1*(x[2]+x[0]) - c2*(x[0]+x[1])
    res[2] = c0 + c1 + c2
    return res

# Compute symmetric mass ratio from m1 and m2
def funcetaofm1m2(m1, m2):
    return m1*m2 / (m1+m2)**2
# Compute chirp mass from m1 and m2 - same units as input m1,m2
def funcMchirpofm1m2(m1, m2):
    return pow(m1*m2, 3./5) / pow(m1+m2, 1./5)
# Newtonian estimate of the relation f(deltat) (for the 22 mode freq) - gives the starting geometric frequency for a given time to merger and chirp mass - output in Hz
# Input chirp mass (solar masses), time in years
def funcNewtonianfoft(Mchirp, t):
    if(t<=0.):
        return 0.
    return 1./pi * pow(Mchirp*MTSUN_SI, -5./8) * pow(256.*t*YRSID_SI/5, -3./8)
# Newtonian estimate of the relation deltat(f) (for the 22 mode freq) - gives the time to merger from a starting frequency for a given chirp mass - output in years
# Input chirp mass (solar masses), frequency in Hz
def funcNewtoniantoffchirp(Mchirp, f):
    return 5./256 * pow(Mchirp*MTSUN_SI, -5./3) * pow(pi*f, -8./3) / YRSID_SI

# Vectorial functions: can take a vector t as input
# Cartesian SSB components of the center of the constellation
def funcp0_MLDC(t):
    alpha = Omega0*t
    c = cos(alpha); s = sin(alpha);
    a = aorbit
    return np.array([a*c, a*s, 0.*t]) # The 0.*t is to ensure broadcasting when t is a vector
# Cartesian components of the spacecraft positions, measured from constellation center
def funcp1L_MLDC(t):
    alpha = Omega0*t
    c = cos(alpha); s = sin(alpha);
    a, e = aorbit, eorbit
    return np.array([- a*e*(1 + s**2), a*e*c*s, -a*e*sqrt3*c])
def funcp2L_MLDC(t):
    alpha = Omega0*t
    c = cos(alpha); s = sin(alpha);
    a, e = aorbit, eorbit
    return np.array([a*e/2*(sqrt3*c*s + (1 + s**2)), a*e/2*(-c*s - sqrt3*(1 + c**2)), -a*e*sqrt3/2*(sqrt3*s - c)])
def funcp3L_MLDC(t):
    alpha = Omega0*t
    c = cos(alpha); s = sin(alpha);
    a, e = aorbit, eorbit
    return np.array([a*e/2*(-sqrt3*c*s + (1 + s**2)), a*e/2*(-c*s + sqrt3*(1 + c**2)), -a*e*sqrt3/2*(-sqrt3*s - c)])
# Cartesian SSB components of spacecraft positions
def funcp1_MLDC(t):
    return funcp0_MLDC(t) + funcp1L_MLDC(t)
def funcp2_MLDC(t):
    return funcp0_MLDC(t) + funcp2L_MLDC(t)
def funcp3_MLDC(t):
    return funcp0_MLDC(t) + funcp3L_MLDC(t)

# Cartesian SSB components of unit vectors - in accordance with MLDC2 orbits, except that spacecraft indices go from 1 to 3 (not 0 to 2)
def funcn1_MLDC(t):
    alpha = Omega0*t
    c = cos(alpha); s = sin(alpha);
    return np.array([-1./2*c*s, 1./2*(1 + c**2), sqrt3/2*s])
def funcn2_MLDC(t):
    alpha = Omega0*t
    c = cos(alpha); s = sin(alpha);
    return 1./4 * np.array([c*s - sqrt3*(1 + s**2), sqrt3*c*s - (1 + c**2), -sqrt3*s - 3*c])
def funcn3_MLDC(t):
    alpha = Omega0*t
    c = cos(alpha); s = sin(alpha);
    return 1./4 * np.array([c*s + sqrt3*(1 + s**2), -sqrt3*c*s - (1 + c**2), -sqrt3*s + 3*c])

# Propagation vector of the incoming wave - lambd, beta ecliptic longitude and latitude
def funck(lambd, beta):
    return np.array([-cos(beta)*cos(lambd), -cos(beta)*sin(lambd), -sin(beta)])

# Rotation matrices used to compute n.H.n
def funcO2(t):
    eta = Omega0*t
    xi = -Omega0*t + 3*pi/2
    zeta = -pi/6
    ceta = cos(eta); seta = sin(eta);
    cxi = cos(xi); sxi = sin(xi);
    czeta = cos(zeta); szeta = sin(zeta);
    return np.array([[cxi*seta-ceta*szeta*sxi,-ceta*cxi*szeta-seta*sxi,-czeta*ceta],[-ceta*cxi-szeta*seta*sxi,-cxi*szeta*seta+ceta*sxi,-czeta*seta],[czeta*sxi,czeta*cxi,-szeta]])
def funcinverseO2(t):
    eta = Omega0*t
    xi = -Omega0*t + 3*pi/2
    zeta = -pi/6
    ceta = cos(eta); seta = sin(eta);
    cxi = cos(xi); sxi = sin(xi);
    czeta = cos(zeta); szeta = sin(zeta);
    return np.array([[cxi*seta-ceta*szeta*sxi,-ceta*cxi-szeta*seta*sxi,czeta*sxi],[-ceta*cxi*szeta-seta*sxi,-cxi*szeta*seta+ceta*sxi,czeta*cxi],[-czeta*ceta,-czeta*seta,-szeta]])
def funcO1(lambd, beta, psi):
    clambd = cos(lambd); slambd = sin(lambd);
    cbeta = cos(beta); sbeta = sin(beta);
    cpsi = cos(psi); spsi = sin(psi);
    return np.array([[cpsi*slambd-clambd*sbeta*spsi,-clambd*cpsi*sbeta-slambd*spsi,-cbeta*clambd],[-clambd*cpsi-sbeta*slambd*spsi,-cpsi*sbeta*slambd+clambd*spsi,-cbeta*slambd],[cbeta*spsi,cbeta*cpsi,-sbeta]])
def funcinverseO1(lambd, beta, psi):
    clambd = cos(lambd); slambd = sin(lambd);
    cbeta = cos(beta); sbeta = sin(beta);
    cpsi = cos(psi); spsi = sin(psi);
    return np.array([[cpsi*slambd-clambd*sbeta*spsi,-clambd*cpsi-sbeta*slambd*spsi,cbeta*spsi],[-clambd*cpsi*sbeta-slambd*spsi,-cpsi*sbeta*slambd+clambd*spsi,cbeta*cpsi],[-cbeta*clambd,-cbeta*slambd,-sbeta]])

trajdict_MLDC = {
    'funcp0': funcp0_MLDC,
    'funcp1L': funcp1L_MLDC,
    'funcp2L': funcp2L_MLDC,
    'funcp3L': funcp3L_MLDC,
    'funcn1': funcn1_MLDC,
    'funcn2': funcn2_MLDC,
    'funcn3': funcn3_MLDC,
    }

# Functions to build a grid of frequencies allowing an accurate representation of a waveform amplitude/phase using cubic splines
# Here, geometric frequencies Mf
# For low-f, sampling from power-law Newtonian phase (a tuned factor times theoretical spline error)
# For high-f, logarithmic sampling with minimal number of points and maximal deltalnf
# Accuracy of spline should be 1e-4, with locally errors reaching 1e-3 (near merger, high aligned spins)
def funcDeltaMfPowerLaw(eta, Mf, acc):
    return 3.8 * np.power(eta * acc, 1./4) * Mf * np.power(pi*Mf, 5./12)
def funcDeltaMfLog(Mfmin, Mfmax, npt, Mf):
    return Mf * (np.power(Mfmax/Mfmin, 1/(npt - 1)) - 1)
def WaveformFrequencyGridGeom(eta, Mfmin, Mfmax, acc=1e-4):
    # Get parameters for log-sampling
    nptlogmin = 50
    deltalnMfmax = 0.025
    deltalnMf = np.fmin(deltalnMfmax, np.log(Mfmax/Mfmin)/(nptlogmin - 1))
    nptlog = 1 + np.floor(np.log(Mfmax/Mfmin) / deltalnMf)

    # Will be used to ensure the last point is not too close to fmax, which poses problems for splines (notably, derivatives become meaningless)
    minsteptoMfmax = 1e-5*Mfmax

    # Build mixed frequency grid iteratively
    Mfreq = [Mfmin]
    while Mfreq[-1] < Mfmax:
        Mf = Mfreq[-1]
        DeltaMf = np.fmin(funcDeltaMfPowerLaw(eta, Mf, acc), funcDeltaMfLog(Mfmin, Mfmax, nptlog, Mf))
        if Mf+DeltaMf < Mfmax and Mf+DeltaMf > Mfmax-minsteptoMfmax:
            Mf_append = Mfmax
        else:
            Mf_append = np.fmin(Mfmax, Mf+DeltaMf)
        Mfreq.append(Mf_append)

    # Output
    return np.array(Mfreq)


# We take the densest sampling as from three conditions:
# - logarithmic sampling with at least nptlogmin points
# - resampling at low-f such that Deltat<=1./24 yr (as measured from tf)
# - resampling at high-f such that Deltaf<=0.002 Hz
def ResponseFrequencyGrid(wf, nptlogmin=100, Deltat_max=0.0416667, Deltaf_max=0.002):
    # Min and max physical frequencies (Hz)
    freq_PhD, amp_PhD, phase_PhD = wf
    fmin, fmax = freq_PhD[0], freq_PhD[-1]

    # Sampling target for log
    Deltalnf_max = np.log(fmax/fmin)/(nptlogmin - 1)

    # Build tf function and f(t) by interpolation of the monotonous part
    tfspline = spline(freq_PhD, 1/(2*pi) * phase_PhD).derivative()
    tf = tfspline(freq_PhD)
    index_cuttf = 0
    tfdiff = np.diff(tf)
    while index_cuttf<len(tfdiff)-1 and tfdiff[index_cuttf]>0:
        index_cuttf += 1
    tfr = tf[:index_cuttf+1]
    freq_PhDr = freq_PhD[:index_cuttf+1]
    ftspline = spline(tfr, freq_PhDr)

    # Will be used to ensure the last point is not too close to fmax, which poses problems for splines (notably, derivatives become meaningless)
    minsteptofmax = 1e-5*fmax

    # Build mixed frequency grid iteratively
    freq = [fmin]
    while freq[-1] < fmax:
        f = freq[-1]
        t_new = tfspline(f) + Deltat_max * YRSID_SI
        if tfr[0]<=t_new and t_new<=tfr[-1]:
            Deltaf_from_time = ftspline(t_new) - f
        else:
            Deltaf_from_time = fmax - f
        Deltaf = np.min([Deltaf_max, f * Deltalnf_max, Deltaf_from_time])
        if f+Deltaf < fmax and f+Deltaf > fmax-minsteptofmax:
            f_append = fmax
        else:
            f_append = np.fmin(fmax, f+Deltaf)
        freq.append(f_append)

    # Output
    return np.array(freq)

# Generate waveform amplitude and phase on a grid adapted to yield a bounded interpolation error
# Starting frequency takes into account both minf and tobs (tobs is the duration from start to ~merger)
# NOTE: fRef will be set to fstart if zero, in keeping with Phenom code, *after* possibly cutting by duration tobs
def GenerateResamplePhenomD(phi0, fRef, m1, m2, chi1, chi2, dist, inc, tobs=1., minf=1e-5, maxf=1., settRefAtfRef=False, tRef=0., acc_sampling=1e-4):
    # Maximum frequency covered by PhenomD
    # NOTE: we take some small margin of error here. Before this was interpreted as outside range by the C code and the last sample was not computed (was given a phase of 0, which messed up the interpolation afterwards)
    MfCUT_PhenomD = 0.2 - 1e-7

    # Check args
    if minf>=maxf:
        raise ValueError("Error in GenerateResamplePhenomDStartTime: incompatible minf and maxf.")
    if minf<=0 and tobs<=0:
        raise ValueError("Error in GenerateResamplePhenomDStartTime: both minf and tobs set to 0 and ignored, does not know where to start !")

    # Parameters
    m1_SI = m1*MSUN_SI
    m2_SI = m2*MSUN_SI
    dist_SI = dist*1e6*PC_SI
    Ms = (m1 + m2) * MTSUN_SI
    Mchirp = funcMchirpofm1m2(m1, m2) # in solar masses
    eta = funcetaofm1m2(m1, m2)

    # Starting frequency estimated from Newtonian order, with a margin of 2 in duration since the estimate is approximate
    if tobs>0:
        fstartN_twiceduration = funcNewtonianfoft(Mchirp, 2*tobs) # margin of a factor of 2 - Newtonian estimate is not very accurate
    else:
        fstartN_twiceduration = 0. # if tobs is zero, ignore it
    # Minimal starting frequency Mf=0.003 - Newtonian estimate is not reliable beyond it, notably because of extreme cases of high antialigned spins and high mass ratio
    fstart_safe_duration = 0.003 / Ms
    # Starting frequency, taking into account also minf
    fstart_byduration = np.fmin(fstart_safe_duration, fstartN_twiceduration)
    if minf < fstart_byduration:
        fstart = fstart_byduration
        cut_f_by_minf = False
    else:
        fstart = minf
        cut_f_by_minf = True

    # Ending frequency - use the specified maxf or use the range covered by PhenomD
    fCUT_PhD = MfCUT_PhenomD / Ms
    fend = np.fmin(maxf, fCUT_PhD)

    # Construct frequency grid with sampling adapted to waveform - converting between f and Mf
    freq_PhD = 1/Ms * WaveformFrequencyGridGeom(eta, Ms*fstart, Ms*fend, acc=acc_sampling)

    # Generate the waveform
    wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq_PhD, phi0, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
    freq_PhD, amp_PhD, phase_PhD = wf_PhD_class.GetWaveform()

    # Build tf function and f(t) by interpolation of the monotonous part
    tfspline = spline(freq_PhD, 1/(2*pi) * phase_PhD).derivative()
    tf = tfspline(freq_PhD)
    index_cuttf = 0
    tfdiff = np.diff(tf)
    while index_cuttf<len(tfdiff)-1 and tfdiff[index_cuttf]>0:
        index_cuttf += 1
    tfr = tf[:index_cuttf+1]
    freq_PhDr = freq_PhD[:index_cuttf+1]
    ftspline = spline(tfr, freq_PhDr)

    # If tobs is within the range covered by f(t), use it to determine minf_rs
    minf_rs = minf
    if tobs>0:
        tobs_s = tobs * YRSID_SI
        if tfr[0] <= -tobs_s and -tobs_s < tfr[-1]:
            minf_rs = np.fmax(freq_PhD[0], ftspline(-tobs_s))
        elif not cut_f_by_minf:
            raise ValueError("Error in GenerateResamplePhenomD: tf for interpolation of ft (from monotonous part of tf) does not cover tobs.")
    else:
        minf_rs = freq_PhD[0]

    # If tobs causes a cut of the waveform, build a new frequency grid and resample - note conversion between f and Mf
    if minf_rs > freq_PhD[0]:
        freq_rs = 1/Ms * WaveformFrequencyGridGeom(eta, Ms*minf_rs, Ms*freq_PhD[-1], acc=acc_sampling)
        amp_int = spline(freq_PhD, amp_PhD)
        phase_int = spline(freq_PhD, phase_PhD)
        amp_rs = amp_int(freq_rs)
        # If fRef=0, in Phenom fRef is set to fstart. We have to propagate a phase shift due to the change of fstart coming from cutting with tobs
        if fRef <= 0.:
            phase_shift = phase_int(freq_PhD[0]) - phase_int(freq_rs[0])
        else:
            phase_shift = 0.
        phase_rs = phase_int(freq_rs) + phase_shift
    else:
        freq_rs, amp_rs, phase_rs = np.copy(freq_PhD), np.copy(amp_PhD), np.copy(phase_PhD)

    # If settRefAtfRef, arrange so that t=tRef at fRef
    # FIXME: improve the check that fRef is in range - here simply forced back in
    if settRefAtfRef:
        fRef_inrange = min(max(freq_rs[0], fRef), freq_rs[-1])
        phase_rs += 2*pi * (freq_rs - fRef_inrange) * (tRef - tfspline(fRef_inrange))

    # Output
    return freq_rs, amp_rs, phase_rs

# Functions to generate PhenomD waveform and resample it in preparation for the FD response
# We try to ensure three conditions:
# - logarithmic sampling with Deltalnf=0.02 (natural log)
# - resampling at low-f such that Deltat<=1./24 yr (as measured from tf)
# - resampling at high-f such that Deltaf<=0.002 Hz
# Waveform generated for a given time to coalescence tobs (in years) (approximate, using tf)
# Also admits args for min, max frequency (in Hz)
# m1, m2 in solar masses, dist in Mpc
# Set minf and tobs to 0 to ignore (but not both !)
# If settRefAtfRef, arrange so that t=tRef at fRef FIXME: check fRef is in range
# def GenerateResamplePhenomD(phi0, fRef, m1, m2, chi1, chi2, dist, inc, tobs=1., minf=1e-5, maxf=1., settRefAtfRef=False, tRef=0., nptmin=100):
#     # Sampling targets
#     Delta_lnf = 0.02
#     Delta_t = 1./24 # yrs
#     Delta_f = 0.002 # Hz
#     # Maximum frequency covered by PhenomD
#     # NOTE: we take some small margin of error here. Before this was interpreted as outside range by the C code and the last sample was not computed (was given a phase of 0, which messed up the interpolation afterwards)
#     MfCUT_PhenomD = 0.2 - 1e-7
#
#     # Check args
#     if minf>=maxf:
#         raise ValueError("Error in GenerateResamplePhenomDStartTime: incompatible minf and maxf.")
#     if minf<=0 and tobs<=0:
#         raise ValueError("Error in GenerateResamplePhenomDStartTime: both minf and tobs set to 0 and ignored, does not know where to start !")
#
#     # Parameters
#     m1_SI = m1*MSUN_SI
#     m2_SI = m2*MSUN_SI
#     dist_SI = dist*1e6*PC_SI
#     Ms = (m1 + m2) * MTSUN_SI
#     Mchirp = funcMchirpofm1m2(m1, m2) # in solar masses
#
#     # First generate PhenomD waveform using log sampling
#     # Here starting frequency estimated from Newtonian order, with a margin of 2 in duration since the estimate is approximate - also take minf into account
#     if tobs>0:
#         fstartN_twiceduration = funcNewtonianfoft(Mchirp, 2*tobs) # margin of a factor of 2 - Newtonian estimate is not very accurate
#     else:
#         fstartN_twiceduration = 0. # if tobs is zero, ignore it
#     minf_PhD = max(minf, fstartN_twiceduration)
#     cut_by_minf = (minf>fstartN_twiceduration) # Keep track of wether we cut by frequency or duration here
#     fCUT_PhD = MfCUT_PhenomD/Ms
#     maxf_PhD = min(maxf, fCUT_PhD)
#     cut_by_maxf = (maxf<fCUT_PhD) # Keep track of wether we cut by argument maxf here
#     # Here, ensure a minimum of 200 pts to do the first generation
#     npt_PhD = max(int(np.ceil(np.log(maxf_PhD/minf_PhD) / Delta_lnf)), 200)
#     freq_PhD = logsampling(minf_PhD, maxf_PhD, num=npt_PhD)
#     wf_PhD_class = pyIMRPhenomD.IMRPhenomDh22AmpPhase(freq_PhD, phi0, fRef, m1_SI, m2_SI, chi1, chi2, dist_SI)
#     wf_PhD = wf_PhD_class.GetWaveform() # freq, amp, phase
#
#     # Build tf function and f(t) by interpolation of the monotonous part
#     # FIXME: can go wrong if the waveform is too short, add safety checks
#     tfspline = spline(freq_PhD, 1/(2*pi)*wf_PhD[2]).derivative()
#     tf = tfspline(freq_PhD)
#     index_cuttf = 0
#     tfdiff = np.diff(tf)
#     while index_cuttf<len(tfdiff) and tfdiff[index_cuttf]>0:
#         index_cuttf += 1
#     nocuttf = (index_cuttf==len(freq_PhD)-1) # keep track of the case where the last tfr corresponds to maxf_rs=maxf_PhD
#     tfr = tf[:index_cuttf+1]
#     freq_PhDr = freq_PhD[:index_cuttf+1]
#     ftspline = spline(tfr, freq_PhDr)
#
#     # Build frequencies for resampling
#     # Note: tobs is positive, duration of signal, hence -tobs - tobs in yrs, tf in SI (s)
#     # If tobs=0., ignored
#     if tobs>0:
#         tstartobs = -tobs*YRSID_SI
#         if (not cut_by_minf) and tfr[0]>tstartobs:
#             raise ValueError("Error in GenerateResamplePhenomDStartTime: tf from initial waveform generation (with fstart decided by Newtonian estimate for duration of 2*tobs) does not include tobs.")
#         if tfr[-1]<tstartobs:
#             raise ValueError("Error in GenerateResamplePhenomDStartTime: tf for interpolation of ft (from monotonous part of tf) stops before tobs - indicates that tobs is too short.")
#         if tfr[0]>tstartobs: # in this case, we can't evaluate ftspline for comparison as it is out of range
#             minf_rs = minf_PhD
#         else:
#             minf_rs = ftspline(tstartobs)
#     else:
#         minf_rs = minf_PhD
#     maxf_rs = maxf_PhD
#     tstart = tfspline(minf_rs) # is either -tobs or tf(minf)
#     # Original ln-sampling for comparison - ensure it is at least nptmin
#     npt_lnsampling = max(nptmin, int(np.ceil(np.log(maxf_rs/minf_rs) / Delta_lnf))) # npt if all the freq interval was covered with ln-sampling
#     Delta_lnf_mod = np.log(maxf_rs/minf_rs) / npt_lnsampling # possibly lower than Delta_lnf, to ensure at least 100 ln-pts on [minf_rs, maxf_rs]
#     # Deltat-resampling at low-f - limit the time range to where ftspline is defined, tfr[-1] should be close to 0
#     times_deltatresampling = np.linspace(tstart, tfr[-1], int(np.ceil(abs(tstart - tfr[-1])/(Delta_t*YRSID_SI))))
#     if len(times_deltatresampling)==0:
#         freq_rsdeltat = np.array([])
#     else:
#         freqs_deltatresampling = ftspline(times_deltatresampling)
#         index_samplingdeltat = 0
#         while index_samplingdeltat<len(freqs_deltatresampling)-1 and  np.log(freqs_deltatresampling[index_samplingdeltat+1]/freqs_deltatresampling[index_samplingdeltat])<Delta_lnf_mod:
#             index_samplingdeltat += 1
#         if index_samplingdeltat==0:
#             freq_rsdeltat = np.array([])
#         else:
#             freq_rsdeltat = freqs_deltatresampling[:index_samplingdeltat+1]
#             # Re-adjust extremity -- this ensures the start point is minf_rs, without numerical rounding error
#             freq_rsdeltat[0] = minf_rs
#             # If last time not excluded and corresponds to maxf_rs, then ensure the end point is exactly maxf_rs without numerical rounding error
#             if nocuttf and index_samplingdeltat==len(freqs_deltatresampling)-1:
#                 freq_rsdeltat[-1] = maxf_rs
#     # Deltaf-resampling at high-f
#     # Ensure f_startdeltafsampling in [minf_rs, maxf_rs]
#     f_startdeltafsampling = max(minf_rs, Delta_f/((maxf_rs/minf_rs)**(1./(npt_lnsampling-1)) - 1))
#     npt_deltafsampling = int(np.ceil((maxf_rs-f_startdeltafsampling)/Delta_f))
#     if npt_deltafsampling<=0: # if f_startdeltafsampling>=maxfrs, do not do deltaf-sampling
#         freq_rsdeltaf = np.array([])
#     else:
#         freq_rsdeltaf = np.linspace(f_startdeltafsampling, maxf_rs, num=npt_deltafsampling)
#     # Original ln-sampling in the middle
#     # FIXME: assumes there is always ln-sampling in the middle, or reduced to one extremity (no contact between deltat-sampling and deltaf-sampling) - will always be the case in practice, but to be made more robust
#     if len(freq_rsdeltat)>0:
#         minf_rsln = freq_rsdeltat[-1]
#         freq_rsdeltat = freq_rsdeltat[:-1] # last freq will be included in the ln-sampling part
#     else:
#         minf_rsln = minf_rs
#     if len(freq_rsdeltaf)>0:
#         maxf_rsln = freq_rsdeltaf[0]
#         freq_rsdeltaf = freq_rsdeltaf[1:] # first freq will be included in the ln-sampling part
#     else:
#         maxf_rsln = maxf_rs
#     if minf_rsln<maxf_rsln:
#         npt_rsln = int(np.ceil(np.log(maxf_rsln/minf_rsln) / Delta_lnf_mod)) + 1
#         freq_rsln = logsampling(minf_rsln, maxf_rsln, num=npt_rsln)
#     else:
#         freq_rsln = np.array([maxf_rsln])
#     freq_rs = np.concatenate((freq_rsdeltat, freq_rsln, freq_rsdeltaf))
#
#     # Resampling
#     amp_PhD_Int = spline(wf_PhD[0], wf_PhD[1])
#     phase_PhD_Int = spline(wf_PhD[0], wf_PhD[2])
#     amp_rs = amp_PhD_Int(freq_rs)
#     phase_rs = phase_PhD_Int(freq_rs)
#
#     # If settRefAtfRef, arrange so that t=tRef at fRef
#     # FIXME: improve the check that fRef is in range - here simply forced back in
#     if settRefAtfRef:
#         fRef_inrange = min(max(freq_rs[0], fRef), freq_rs[-1])
#         phase_rs += 2*pi * (freq_rs - fRef_inrange) * (tRef - tfspline(fRef_inrange))
#
#     # Output
#     return freq_rs, amp_rs, phase_rs



def ComputehphcFromAmpPhase(wf, freq, iota, phi):
    f, amp, phase = wf
    amp_Int = spline(f, amp)
    phase_Int = spline(f, phase)
    indices = (f[0]<=freq) & (freq<=f[-1])
    factorp = 1/2. * (SpinWeightedSphericalHarmonic(-2, 2, 2, iota, phi) + \
                        np.conj(SpinWeightedSphericalHarmonic(-2, 2, -2, iota, phi)))
    factorc = 1j/2. * (SpinWeightedSphericalHarmonic(-2, 2, 2, iota, phi) - \
                        np.conj(SpinWeightedSphericalHarmonic(-2, 2, -2, iota, phi)))
    h22tilde = np.zeros_like(freq, dtype=np.complex128)
    h22tilde[indices] = amp_Int(freq[indices]) * np.exp(1j*phase_Int(freq[indices]))
    hptilde = factorp * h22tilde
    hctilde = factorc * h22tilde
    #print "compare phases"
    #tmph = np.angle(hptilde)
    # tmph = np.angle(np.exp(1j*phase_Int(freq[indices])))
    # tmph = np.unwrap(tmph)
    # plt.plot(f, phase)
    # plt.plot(freq[indices], phase_Int(freq[indices]), '--')
    # plt.plot(freq[indices], tmph, 'b.')
    # plt.show()
    # sys.exit(0)
    return freq, hptilde, hctilde


# Fourier-domain LISA response, perturbative formalism
# t0 in yr
def LISAFDresponseTDI(freq, wf, inc, lambd, beta, psi, t0=0., trajdict=trajdict_MLDC, TDItag='TDIXYZ', order_fresnel_stencil=0):
    # Waveform given in the form of downsampled amplitude and phase, ready to be interpolated
    # The frequencies are assumed to have been chosen to ensure sufficient sampling at low and high frequencies
    # For now, we consider single-mode h22 waveforms only
    wf_freq = wf[0]
    wf_amp = wf[1]
    wf_phase = wf[2]
    n = len(freq)

    # Check frequency bounds
    if freq[0]<wf_freq[0] or freq[-1]>wf_freq[-1]:
        raise ValueError("Error: the range of frequencies for the response exceeds the one of the waveform.")

    # Wave unit vector
    kvec = funck(lambd, beta)

    # Trajectories
    funcp0 = trajdict['funcp0']

    # Compute constant matrices Hplus and Hcross in the SSB frame
    O1 = funcO1(lambd, beta, psi)
    invO1 = funcinverseO1(lambd, beta, psi)
    Hplus = np.dot(O1, np.dot(HSplus, invO1))
    Hcross = np.dot(O1, np.dot(HScross, invO1))

    # Build function tf by interpolating FD phase
    tfspline = spline(wf_freq, 1/(2*pi)*(wf_phase - wf_phase[0])).derivative() # get rid of possibly huge constant in the phase before interpolating
    tfvec = tfspline(freq)

    # Spin-weighted spherical harmonics prefactors for plus and cross FD
    Y22 = SpinWeightedSphericalHarmonic(-2, 2, 2, inc, 0.)
    Y2m2 = SpinWeightedSphericalHarmonic(-2, 2, -2, inc, 0.)
    Yfactorplus = 1./2 * (Y22 + Y2m2)
    Yfactorcross = 1j/2 * (Y22 - Y2m2)
    # The matrix H is now complex
    H = Yfactorplus*Hplus + Yfactorcross*Hcross

    # Initialize output
    wfTDI = {}
    wfTDI['TDItag'] = TDItag
    wfTDI['t0'] = t0
    wfTDI['freq'] = np.copy(wf_freq)
    wfTDI['amp'] = np.copy(wf_amp)
    wfTDI['phase'] = np.copy(wf_phase)
    wfTDI['freq_response'] = np.copy(freq)
    wfTDI['phaseRdelay'] = np.zeros(n, dtype=np.float64)
    wfTDI['transferL1'] = np.zeros(n, dtype=np.complex128)
    wfTDI['transferL2'] = np.zeros(n, dtype=np.complex128)
    wfTDI['transferL3'] = np.zeros(n, dtype=np.complex128)

    # For higher-order corrections: Fresnel tranform stencil
    if order_fresnel_stencil>20:
        raise ValueError('Only order_fresnel_stencil<=20 is supported for now.')
    if order_fresnel_stencil>=1:
        coeffs = np.array(coeffs_fresnel_stencil[order_fresnel_stencil])
        coeffs_array = np.concatenate((coeffs[1:][::-1], 2*coeffs[0:1], coeffs[1:])) / 2
        # Values and signs of Tf
        Tfvec, epsTfvec = func_compute_Tf(freq, wf_freq, wf_phase)
    # Main loop over frequencies
    for i in range(n):
        f = freq[i]
        t = tfvec[i] + t0*YRSID_SI # t0 is the orbital time at merger - t is the SSB time
        # Phase of transfer at leading order in the delays and Fresnel correction - we keep the phase separate because it is easy to interpolate as a phase but can be less so in Re/Im form
        p0 = funcp0(t)
        kR = np.dot(kvec, p0)
        phaseRdelay = 2*pi/C_SI*f*kR
        # Transfer function
        if order_fresnel_stencil>=1:
            Gslr = {}
            Tf = Tfvec[i] # used for higher-order correction of Fresnel type
            epsTf = epsTfvec[i] # keeping track of the sign associated with Tf
            if epsTf==1. or epsTf==0.:
                coeffs_array_signeps = coeffs_array
            else:
                coeffs_array_signeps = np.conj(coeffs_array)
            tvec = t + Tf * np.arange(-order_fresnel_stencil, order_fresnel_stencil+1)
            Gslrvec = EvaluateGslr(tvec, f, H, kvec, trajdict=trajdict, response='full')
            for key in Gslrvec:
                Gslr[key] = np.dot(coeffs_array_signeps, Gslrvec[key])
        else:
            Gslr = EvaluateGslr(t, f, H, kvec, trajdict=trajdict, response='full')
        # Scale out the leading-order correction from the orbital delay term, that we keep separate
        Tslr = {}
        for key in Gslr:
            Tslr[key] = Gslr[key] * exp(-1j*phaseRdelay)
        # Build TDI combinations
        tdi = TDICombinationFD(Tslr, f, TDItag=TDItag, rescaled=False)
        wfTDI['phaseRdelay'][i] = phaseRdelay
        wfTDI['transferL1'][i] = tdi['transferL1']
        wfTDI['transferL2'][i] = tdi['transferL2']
        wfTDI['transferL3'][i] = tdi['transferL3']

    # Output
    return wfTDI

# Fourier-domain LISA response for the observable y12, perturbative formalism
# t0 in yr
def LISAFDresponsey12(freq, wf, inc, lambd, beta, psi, t0=0., trajdict=trajdict_MLDC, order_fresnel_stencil=0):
    # Waveform given in the form of downsampled amplitude and phase, ready to be interpolated
    # The frequencies are assumed to have been chosen to ensure sufficient sampling at low and high frequencies
    # For now, we consider single-mode h22 waveforms only
    wf_freq = wf[0]
    wf_amp = wf[1]
    wf_phase = wf[2]
    n = len(freq)

    # Check frequency bounds
    if freq[0]<wf_freq[0] or freq[-1]>wf_freq[-1]:
        raise ValueError("Error: the range of frequencies for the response exceeds the one of the waveform.")

    # Wave unit vector
    kvec = funck(lambd, beta)

    # Trajectories
    funcp0 = trajdict['funcp0']

    # Compute constant matrices Hplus and Hcross in the SSB frame
    O1 = funcO1(lambd, beta, psi)
    invO1 = funcinverseO1(lambd, beta, psi)
    Hplus = np.dot(O1, np.dot(HSplus, invO1))
    Hcross = np.dot(O1, np.dot(HScross, invO1))

    # Build function tf by interpolating FD phase
    tfspline = spline(wf_freq, 1/(2*pi)*(wf_phase - wf_phase[0])).derivative() # get rid of possibly huge constant in the phase before interpolating
    tfvec = tfspline(freq)

    # Spin-weighted spherical harmonics prefactors for plus and cross FD
    Y22 = SpinWeightedSphericalHarmonic(-2, 2, 2, inc, 0.)
    Y2m2 = SpinWeightedSphericalHarmonic(-2, 2, -2, inc, 0.)
    Yfactorplus = 1./2 * (Y22 + Y2m2)
    Yfactorcross = 1j/2 * (Y22 - Y2m2)
    # The matrix H is now complex
    H = Yfactorplus*Hplus + Yfactorcross*Hcross

    # Initialize output
    wfTDI = {}
    wfTDI['TDItag'] = 'y12'
    wfTDI['t0'] = t0
    wfTDI['freq'] = np.copy(wf_freq)
    wfTDI['amp'] = np.copy(wf_amp)
    wfTDI['phase'] = np.copy(wf_phase)
    wfTDI['freq_response'] = np.copy(freq)
    wfTDI['phaseRdelay'] = np.zeros(n, dtype=np.float64)
    wfTDI['transferL'] = np.zeros(n, dtype=np.complex128)

    # For higher-order corrections: Fresnel tranform stencil
    if order_fresnel_stencil>20:
        raise ValueError('Only order_fresnel_stencil<=20 is supported for now.')
    if order_fresnel_stencil>=1:
        coeffs = np.array(coeffs_fresnel_stencil[order_fresnel_stencil])
        coeffs_array = np.concatenate((coeffs[1:][::-1], 2*coeffs[0:1], coeffs[1:])) / 2
        # Values and signs of Tf
        Tfvec, epsTfvec = func_compute_Tf(freq, wf_freq, wf_phase)

    # Main loop over frequencies
    for i in range(n):
        f = freq[i]
        t = tfvec[i] + t0*YRSID_SI # t0 is the orbital time at merger - t is the SSB time
        # Phase of transfer at leading order in the delays and Fresnel correction - we keep the phase separate because it is easy to interpolate as a phase but can be less so in Re/Im form
        p0 = funcp0(t)
        kR = np.dot(kvec, p0)
        phaseRdelay = 2*pi/C_SI*f*kR
        # Transfer function
        if order_fresnel_stencil>=1:
            Gslr = {}
            Tf = Tfvec[i] # used for higher-order correction of Fresnel type
            epsTf = epsTfvec[i] # keeping track of the sign associated with Tf
            if epsTf==1. or epsTf==0.:
                coeffs_array_signeps = coeffs_array
            else:
                coeffs_array_signeps = np.conj(coeffs_array)
            tvec = t + Tf * np.arange(-order_fresnel_stencil, order_fresnel_stencil+1)
            Gslrvec = EvaluateGslr(tvec, f, H, kvec, trajdict=trajdict, response='full')
            for key in Gslrvec:
                Gslr[key] = np.dot(coeffs_array_signeps, Gslrvec[key])
        else:
            Gslr = EvaluateGslr(t, f, H, kvec, trajdict=trajdict, response='full')
        # Scale out the leading-order correction from the orbital delay term, that we keep separate
        Tslr = {}
        Tslr[(1,2)] = Gslr[(1,2)] * exp(-1j*phaseRdelay)
        # Output
        wfTDI['phaseRdelay'][i] = phaseRdelay
        wfTDI['transferL'][i] = Tslr[(1,2)]

    # Output
    return wfTDI

# Data for the comb extent of the LISA response
data_combextent_f = np.array([0.00001,0.0000126486,0.0000159986,0.0000202359,0.0000255955,0.0000323746,0.0000409492,0.0000517947,0.0000655129,0.0000828643,0.000104811,0.000132571,0.000167683,0.000212095,0.00026827,0.000339322,0.000429193,0.000542868,0.000686649,0.000868511,0.00109854,0.0013895,0.00175751,0.002223,0.00281177,0.00355648,0.00449843,0.00568987,0.00719686,0.00910298,0.011514,0.0145635,0.0184207,0.0232995,0.0294705,0.0372759,0.0471487,0.0596362,0.0754312,0.0954095,0.120679,0.152642,0.19307,0.244205,0.308884,0.390694,0.494171,0.625055,0.790604,1.])
data_combextent_orb = np.array([6.,6.,6.,7.,7.,7.,8.,8.,8.,9.,9.,10.,10.,11.,12.,13.,13.,15.,16.,17.,19.,21.,23.,25.,29.,32.,37.,42.,48.,56.,66.,76.,86.,98.,111.,129.,149.,174.,206.,243.,289.,347.,419.,508.,619.,759.,932.,1150.,1422.,1765.])
data_combextent_const = np.array([39.,30.,7.,7.,7.,7.,8.,8.,9.,9.,9.,10.,10.,10.,11.,11.,11.,11.,11.,11.,11.,12.,12.,13.,13.,13.,13.,13.,14.,14.,15.,16.,16.,17.,17.,18.,20.,21.,23.,24.,25.,27.,30.,33.,36.,40.,44.,50.,57.,65.])
spline_combextent_orb = spline(data_combextent_f, data_combextent_orb)
spline_combextent_const = spline(data_combextent_f, data_combextent_const)
# Function to compute the comb extent for a given frequency - aims at a comb truncation error of 1e-12
def LISACombExtent(f):
    return int(np.ceil(spline_combextent_orb(f) + spline_combextent_const(f)))

# Function to compute the comb coefficients for the full response (orbital and constellation) -- returns a dict for the 6 slr links
def LISACombCoeffs(f, M, inc, lambd, beta, psi, t0=0., trajdict=trajdict_MLDC):
    # Wave unit vector
    kvec = funck(lambd, beta)

    # Trajectories
    funcp0 = trajdict['funcp0']

    # Compute constant matrices Hplus and Hcross in the SSB frame
    O1 = funcO1(lambd, beta, psi)
    invO1 = funcinverseO1(lambd, beta, psi)
    Hplus = np.dot(O1, np.dot(HSplus, invO1))
    Hcross = np.dot(O1, np.dot(HScross, invO1))

    # Spin-weighted spherical harmonics prefactors for plus and cross FD
    Y22 = SpinWeightedSphericalHarmonic(-2, 2, 2, inc, 0.)
    Y2m2 = SpinWeightedSphericalHarmonic(-2, 2, -2, inc, 0.)
    Yfactorplus = 1./2 * (Y22 + Y2m2)
    Yfactorcross = 1j/2 * (Y22 - Y2m2)
    # The matrix H is now complex
    H = Yfactorplus*Hplus + Yfactorcross*Hcross

    # Function returning orbital delay factor in the kernel G(f,t)
    # def func_G_Rdelay(t):
    #     p0 = funcp0(t)
    #     kR = np.dot(kvec, p0)
    #     phaseRdelay = 2*pi/C_SI*f*kR
    #     return exp(1j*phaseRdelay)

    # Comb extent
    N = 2*M

    # Times
    deltat = 1./N * YRSID_SI
    times = t0 * YRSID_SI + deltat * np.arange(N)

    # Links
    links = [(1,2), (2,1), (2,3), (3,2), (3,1), (1,3)]

    # Values of G(f,t)
    Gslrvals = EvaluateGslr(times, f, H, kvec, trajdict=trajdict, response='full')

    # FFT and set the cn
    cn = {}
    for link in links:
        # Vector c_-M,...,c0,...c_M of length N+1
        cn[link] = np.zeros(N+1, dtype=complex)
        # FFT with convention 1/N, + sign
        Y = np.fft.ifft(Gslrvals[link])
        # Set cn
        cn[link][M:2*M] = Y[:M]
        cn[link][1:M] = Y[M+1:N]
        cn[link][0] = Y[M]/2
        cn[link][N] = Y[M]/2

    return cn

# Fourier-domain LISA response, comb
# t0 in yr
# TODO: add range checking (with comb, need to evaluate splines a bit outside the range of freq, wf must be wider)
def LISAFDresponseCombTDI(freq, wf, inc, lambd, beta, psi, t0=0., trajdict=trajdict_MLDC, TDItag='TDIXYZ'):
    # Waveform given in the form of downsampled amplitude and phase, ready to be interpolated
    # The frequencies are assumed to have been chosen to ensure sufficient sampling at low and high frequencies
    # For now, we consider single-mode h22 waveforms only
    wf_freq = wf[0]
    wf_amp = wf[1]
    wf_phase = wf[2]
    n = len(freq)

    # Check frequency bounds
    if freq[0]<wf_freq[0] or freq[-1]>wf_freq[-1]:
        raise ValueError("Error: the range of frequencies for the response exceeds the one of the waveform.")

    # Links
    links = [(1,2), (2,1), (2,3), (3,2), (3,1), (1,3)]

    # Initialize output
    # In keeping with format of perturbative response, phaseRdelay is kept separated, but we will compute directly the full response with the comb
    wfTDI = {}
    wfTDI['TDItag'] = TDItag
    wfTDI['t0'] = t0
    wfTDI['freq'] = np.copy(wf_freq)
    wfTDI['amp'] = np.copy(wf_amp)
    wfTDI['phase'] = np.copy(wf_phase)
    wfTDI['freq_response'] = np.copy(freq)
    wfTDI['phaseRdelay'] = np.zeros(n, dtype=np.float64)
    wfTDI['transferL1'] = np.zeros(n, dtype=np.complex128)
    wfTDI['transferL2'] = np.zeros(n, dtype=np.complex128)
    wfTDI['transferL3'] = np.zeros(n, dtype=np.complex128)

    # Build function tf by interpolating FD phase - used to factor out the orbital delay term
    tfspline = spline(wf_freq, 1/(2*pi)*(wf_phase - wf_phase[0])).derivative() # get rid of possibly huge constant in the phase before interpolating
    tfvec = tfspline(freq)

    # Function returning orbital delay factor in the kernel G(f,t)
    funcp0 = trajdict['funcp0']
    kvec = funck(lambd, beta)
    def func_orb_delay(f, t):
        p0 = funcp0(t)
        kR = np.dot(kvec, p0)
        return 2*pi/C_SI*f*kR

    # Orbital delay phase, factored out in keeping with conventions of the perturbative response
    phaseRdelay = np.array([func_orb_delay(freq[i], tfvec[i] + t0 * YRSID_SI) for i in range(n)])
    wfTDI['phaseRdelay'][:] = phaseRdelay

    # Get comb extent
    M = LISACombExtent(freq[-1])

    # Three frequencies, assumed to be enough to interpolate frequency-dependence in the cn(f)
    # Take into account the extended range required by the convolution
    n_freq_cn = 3
    freq_cn = np.array([freq[0] - M*f0, (freq[0]+freq[-1])/2, freq[-1] + M*f0])

    # Compute cn on three frequencies
    listcn = list(map(lambda x: LISACombCoeffs(x, M, inc, lambd, beta, psi, t0=t0, trajdict=trajdict), freq_cn))
    # cnvals_matrix: rows are [cn[f0], cn[f1], cn[f2]]
    cnvals_matrix = {}
    for link in links:
        cnvals_matrix[link] = np.zeros((2*M+1, 3), dtype='complex')
        for i in range(len(listcn)):
            cnvals_matrix[link][:,i] = listcn[i][link]

    # Build cn as quadratic interpolating polynomials on the three frequencies
    cnpoly_matrix = {}
    for link in links:
        cnpoly_matrix[link] = np.zeros((2*M+1, 3), dtype='complex')
        for i in range(2*M+1):
            cnpoly_matrix[link][i,:] = quad_lagrange(freq_cn, cnvals_matrix[link][i,:])

    # To evaluate the cn from their quadratic polynomial, build matrix [[1], [f], [f**2]]
    f_matrix = np.zeros((3, n), dtype='float')
    f_matrix[0,:] = np.ones(n)
    f_matrix[1,:] = freq
    f_matrix[2,:] = freq*freq

    # Matrix of values of htilde(f-n*f0)
    htildevals_matrix = np.zeros((n,2*M+1), dtype='complex')
    for i, f in enumerate(freq):
        # Vector of values htilde(f-n*f0)
        fs = f - f0*np.arange(-M, M+1)
        htildevals_matrix[i,:] = spline_amp(fs) * np.exp(1j*spline_phase(fs))

    # htilde*exp(i*phasedelayR) values to be divided with to extract transfer function without phase delay term
    htilde_withRdelay = amp * exp(1j*(phase + phaseRdelay))

    # Main linear algebra step to compute Tslr transfer functions
    TslrL = {}
    for link in links:
        TslrL[link] = np.zeros(n, dtype='complex')
        cnf_matrix = np.dot(cnpoly_matrix[link], f_matrix).T
        for i, f in enumerate(freq):
            TslrL[link][i] = np.dot(cnf_matrix[i], htildevals_matrix[i]) / htilde_withRdelay[i]

    # Build TDI combinations for the constellation response - TDICombinationFD can be vectorial
    tdi = TDICombinationFD(TslrL, freq, TDItag=TDItag, rescaled=False)
    wfTDI['transferL1'] = tdi['transferL1']
    wfTDI['transferL2'] = tdi['transferL2']
    wfTDI['transferL3'] = tdi['transferL3']

    return wfTDI

# Fourier-domain LISA response for the y12 observable, comb
# t0 in yr
# TODO: add range checking (with comb, need to evaluate splines a bit outside the range of freq, wf must be wider)
def LISAFDresponseComby12(freq, wf, inc, lambd, beta, psi, t0=0., trajdict=trajdict_MLDC):
    # Waveform given in the form of downsampled amplitude and phase, ready to be interpolated
    # The frequencies are assumed to have been chosen to ensure sufficient sampling at low and high frequencies
    # For now, we consider single-mode h22 waveforms only
    wf_freq = wf[0]
    wf_amp = wf[1]
    wf_phase = wf[2]
    n = len(freq)

    # Check frequency bounds
    if freq[0]<wf_freq[0] or freq[-1]>wf_freq[-1]:
        raise ValueError("Error: the range of frequencies for the response exceeds the one of the waveform.")

    # Links
    link = (1,2)

    # Initialize output
    # In keeping with format of perturbative response, phaseRdelay is kept separated, but we will compute directly the full response with the comb
    wfTDI = {}
    wfTDI['TDItag'] = 'y12'
    wfTDI['t0'] = t0
    wfTDI['freq'] = np.copy(wf_freq)
    wfTDI['amp'] = np.copy(wf_amp)
    wfTDI['phase'] = np.copy(wf_phase)
    wfTDI['freq_response'] = np.copy(freq)
    wfTDI['phaseRdelay'] = np.zeros(n, dtype=np.float64)
    wfTDI['transferL'] = np.zeros(n, dtype=np.complex128)

    # Build function tf by interpolating FD phase - used to factor out the orbital delay term
    tfspline = spline(wf_freq, 1/(2*pi)*(wf_phase - wf_phase[0])).derivative() # get rid of possibly huge constant in the phase before interpolating
    tfvec = tfspline(freq)

    # Function returning orbital delay factor in the kernel G(f,t)
    funcp0 = trajdict['funcp0']
    kvec = funck(lambd, beta)
    def func_orb_delay(f, t):
        p0 = funcp0(t)
        kR = np.dot(kvec, p0)
        return 2*pi/C_SI*f*kR

    # Orbital delay phase, factored out in keeping with conventions of the perturbative response
    phaseRdelay = np.array([func_orb_delay(freq[i], tfvec[i] + t0 * YRSID_SI) for i in range(n)])
    wfTDI['phaseRdelay'][:] = phaseRdelay

    # Get comb extent
    M = LISACombExtent(freq[-1])

    # Three frequencies, assumed to be enough to interpolate frequency-dependence in the cn(f)
    # Take into account the extended range required by the convolution
    n_freq_cn = 3
    freq_cn = np.array([freq[0] - M*f0, (freq[0]+freq[-1])/2, freq[-1] + M*f0])

    # Compute cn on three frequencies
    listcn = list(map(lambda x: LISACombCoeffs(x, M, inc, lambd, beta, psi, t0=t0, trajdict=trajdict), freq_cn))
    # cnvals_matrix: rows are [cn[f0], cn[f1], cn[f2]]
    cnvals_matrix = np.zeros((2*M+1, 3), dtype='complex')
    for i in range(len(listcn)):
        cnvals_matrix[:,i] = listcn[i][link]

    # Build cn as quadratic interpolating polynomials on the three frequencies
    cnpoly_matrix = np.zeros((2*M+1, 3), dtype='complex')
    for i in range(2*M+1):
        cnpoly_matrix[i,:] = quad_lagrange(freq_cn, cnvals_matrix[i,:])

    # To evaluate the cn from their quadratic polynomial, build matrix [[1], [f], [f**2]]
    f_matrix = np.zeros((3, n), dtype='float')
    f_matrix[0,:] = np.ones(n)
    f_matrix[1,:] = freq
    f_matrix[2,:] = freq*freq

    # Matrix of values of htilde(f-n*f0)
    htildevals_matrix = np.zeros((n,2*M+1), dtype='complex')
    for i, f in enumerate(freq):
        # Vector of values htilde(f-n*f0)
        fs = f - f0*np.arange(-M, M+1)
        htildevals_matrix[i,:] = spline_amp(fs) * np.exp(1j*spline_phase(fs))

    # htilde*exp(i*phasedelayR) values to be divided with to extract transfer function without phase delay term
    htilde_withRdelay = amp * exp(1j*(phase + phaseRdelay))

    # Main linear algebra step to compute Tslr transfer functions
    TslrL = np.zeros(n, dtype='complex')
    cnf_matrix = np.dot(cnpoly_matrix, f_matrix).T
    for i, f in enumerate(freq):
        TslrL[i] = np.dot(cnf_matrix[i], htildevals_matrix[i]) / htilde_withRdelay[i]

    # Transfer for the constellation response for a single link
    wfTDI['transferL'] = TslrL

    return wfTDI

# Fourier-domain TDI combinations
def TDICombinationFD(Gslr, f, TDItag='TDIXYZ', rescaled=False):
    x = pi*f*L_SI/C_SI;
    z = exp(2*1j*x);
    TDI = {}
    if TDItag=='TDIXYZ':
        # First-generation TDI XYZ
        # With x=pifL, factor scaled out: 2I*sin2x*e2ix
        if rescaled:
            factor = 1.
        else:
            factor = 2*1j*sin(2.*x)*z
        Xraw = Gslr[(2,1)] + z*Gslr[(1,2)] - Gslr[(3,1)] - z*Gslr[(1,3)]
        Yraw = Gslr[(3,2)] + z*Gslr[(2,3)] - Gslr[(1,2)] - z*Gslr[(2,1)]
        Zraw = Gslr[(1,3)] + z*Gslr[(3,1)] - Gslr[(2,3)] - z*Gslr[(3,2)]
        TDI['transferL1'] = factor * Xraw
        TDI['transferL2'] = factor * Yraw
        TDI['transferL3'] = factor * Zraw
        return TDI
    elif TDItag=='TDIAET':
        # First-generation TDI AET from X,Y,Z
        # With x=pifL, factors scaled out: A,E:I*sqrt2*sin2x*e2ix T:2*sqrt2*sin2x*sinx*e3ix
        # Here we include a factor 2, because the code was first written using the definitions (2) of McWilliams&al_0911 where A,E,T are 1/2 of their LDC definitions
        factor_convention = 2.
        if rescaled:
            factorAE = 1.
            factorT = 1.
        else:
            factorAE = 1j*sqrt2*sin(2.*x)*z
            factorT = 2.*sqrt2*sin(2.*x)*sin(x)*exp(1j*3.*x)
        Araw = 0.5 * ( (1.+z)*(Gslr[(3,1)] + Gslr[(1,3)]) - Gslr[(2,3)] - z*Gslr[(3,2)] - Gslr[(2,1)] - z*Gslr[(1,2)] )
        Eraw = 0.5*invsqrt3 * ( (1.-z)*(Gslr[(1,3)] - Gslr[(3,1)]) + (2.+z)*(Gslr[(1,2)] - Gslr[(3,2)]) + (1.+2*z)*(Gslr[(2,1)] - Gslr[(2,3)]) )
        Traw = invsqrt6 * ( Gslr[(2,1)] - Gslr[(1,2)] + Gslr[(3,2)] - Gslr[(2,3)] + Gslr[(1,3)] - Gslr[(3,1)])
        TDI['transferL1'] = factor_convention * factorAE * Araw
        TDI['transferL2'] = factor_convention * factorAE * Eraw
        TDI['transferL3'] = factor_convention * factorT * Traw
        return TDI
    else:
        raise ValueError("Error in TDICombinationFD: TDItag not recognized.")

# Single-link response
# 'full' does include the orbital-delay term, 'constellation' does not
# t can be a scalar or a 1D vector
def EvaluateGslr(t, f, H, k, trajdict=trajdict_MLDC, response='full'):

    # Trajectories, p0 used only for the full response
    funcp0 = trajdict['funcp0']
    funcn1 = trajdict['funcn1']
    funcn2 = trajdict['funcn2']
    funcn3 = trajdict['funcn3']
    funcp1L = trajdict['funcp1L']
    funcp2L = trajdict['funcp2L']
    funcp3L = trajdict['funcp3L']
    funcp1L = trajdict['funcp1L']
    funcp2L = trajdict['funcp2L']
    funcp3L = trajdict['funcp3L']

    # Here t is a scalar or a vector - the trajectory functions are vectorial
    # If t is a N-vector, then these are 3-vectors with each component a N-vector (shape (3,N))
    p0 = funcp0(t)
    p1L = funcp1L(t)
    p2L = funcp2L(t)
    p3L = funcp3L(t)
    n1 = funcn1(t)
    n2 = funcn2(t)
    n3 = funcn3(t)

    # Compute intermediate scalar products
    # Distinguishing wether t is a scalar or a vector
    if isinstance(t, np.ndarray): # t vector case
        # Note: np.einsum comes with a startup cost that is excessive here, as we do an external loop on frequencies - revert to ugly for loop with dot
        n = len(t)
        # Transpose vectors to shape (N,3)
        n1T = n1.T
        n2T = n2.T
        n3T = n3.T
        p1LT = p1L.T
        p2LT = p2L.T
        p3LT = p3L.T
        p0T = p0.T
        # H is a fixed 3*3 matrix, not vectorialized
        # n1Hn1 = np.einsum('ik,ij,jk->k', n1, H, n1)
        # n2Hn2 = np.einsum('ik,ij,jk->k', n2, H, n2)
        # n3Hn3 = np.einsum('ik,ij,jk->k', n3, H, n3)
        n1Hn1 = np.zeros(n, dtype='complex')
        n2Hn2 = np.zeros(n, dtype='complex')
        n3Hn3 = np.zeros(n, dtype='complex')
        for i in range(n):
            n1Hn1[i] = np.dot(n1T[i], np.dot(H, n1T[i]))
            n2Hn2[i] = np.dot(n2T[i], np.dot(H, n2T[i]))
            n3Hn3[i] = np.dot(n3T[i], np.dot(H, n3T[i]))
            # if (i<10):
            #     print (i, n1Hn1[i], n2Hn2[i], n3Hn3[i])
        # k is a fixed 3-vector, not vectorialized
        # kn1 = np.einsum('i,ij->j', k, n1)
        # kn2 = np.einsum('i,ij->j', k, n2)
        # kn3 = np.einsum('i,ij->j', k, n3)
        kn1 = np.zeros(n, dtype='float')
        kn2 = np.zeros(n, dtype='float')
        kn3 = np.zeros(n, dtype='float')
        for i in range(n):
            kn1[i] = np.dot(k, n1T[i])
            kn2[i] = np.dot(k, n2T[i])
            kn3[i] = np.dot(k, n3T[i])
            # if (i<10):
            #     print (i, kn1[i], kn2[i], kn3[i])
        # kp1Lp2L = np.einsum('i,ij->j', k, (p1L+p2L))
        # kp2Lp3L = np.einsum('i,ij->j', k, (p2L+p3L))
        # kp3Lp1L = np.einsum('i,ij->j', k, (p3L+p1L))
        # kp0 = np.einsum('i,ij->j', k, p0)
        kp1Lp2L = np.zeros(n, dtype='float')
        kp2Lp3L = np.zeros(n, dtype='float')
        kp3Lp1L = np.zeros(n, dtype='float')
        kp0 = np.zeros(n, dtype='float')
        for i in range(n):
            kp1Lp2L[i] = np.dot(k, p1LT[i] + p2LT[i])
            kp2Lp3L[i] = np.dot(k, p2LT[i] + p3LT[i])
            kp3Lp1L[i] = np.dot(k, p3LT[i] + p1LT[i])
            kp0[i] = np.dot(k, p0T[i])
            # if (i<10):
            #     print (i, kp1Lp2L[i]/L_SI, kp2Lp3L[i]/L_SI, kp3Lp1L[i]/L_SI, kp0[i]/C_SI)
    else: # t scalar case
        kn1 = np.dot(k, n1)
        kn2 = np.dot(k, n2)
        kn3 = np.dot(k, n3)
        n1Hn1 = np.dot(n1, np.dot(H, n1))
        n2Hn2 = np.dot(n2, np.dot(H, n2))
        n3Hn3 = np.dot(n3, np.dot(H, n3))
        kp1Lp2L = np.dot(k, (p1L+p2L))
        kp2Lp3L = np.dot(k, (p2L+p3L))
        kp3Lp1L = np.dot(k, (p3L+p1L))
        kp0 = np.dot(k, p0)

    # Prefactors - projections are either scalars or vectors
    if response=='full':
        factorcexp0 = exp(1j*2*pi*f/C_SI * kp0)
    elif response=='constellation':
        factorcexp0 = 1.
    else:
        raise Exception('Keyword response not recognized.')
    prefactor = pi*f*L_SI/C_SI
    factorcexp12 = exp(1j*prefactor * (1.+kp1Lp2L/L_SI))
    factorcexp23 = exp(1j*prefactor * (1.+kp2Lp3L/L_SI))
    factorcexp31 = exp(1j*prefactor * (1.+kp3Lp1L/L_SI))
    factorsinc12 = sinc( prefactor * (1.-kn3))
    factorsinc21 = sinc( prefactor * (1.+kn3))
    factorsinc23 = sinc( prefactor * (1.-kn1))
    factorsinc32 = sinc( prefactor * (1.+kn1))
    factorsinc31 = sinc( prefactor * (1.-kn2))
    factorsinc13 = sinc( prefactor * (1.+kn2))

    # Compute the Gslr - either scalars or vectors
    commonfac = 1j * prefactor * factorcexp0
    G12 = commonfac * n3Hn3 * factorsinc12 * factorcexp12
    G21 = commonfac * n3Hn3 * factorsinc21 * factorcexp12
    G23 = commonfac * n1Hn1 * factorsinc23 * factorcexp23
    G32 = commonfac * n1Hn1 * factorsinc32 * factorcexp23
    G31 = commonfac * n2Hn2 * factorsinc31 * factorcexp31
    G13 = commonfac * n2Hn2 * factorsinc13 * factorcexp31

    ### FIXME
    # G13 = -1j * prefactor * n2Hn2 * factorsinc31 * np.conjugate(factorcexp31)

    # if isinstance(t, np.ndarray):
    #     print (np.shape(prefactor), np.shape(factorcexp31), np.shape(factorsinc31))
    #     for i in range(10):
    #         print (i, prefactor[i], factorcexp31[i], factorsinc31[i], factorsinc13[i])

    # Output dictionary - either scalars or vectors
    Gslr = {}
    Gslr[(1,2)] = G12
    Gslr[(2,1)] = G21
    Gslr[(2,3)] = G23
    Gslr[(3,2)] = G32
    Gslr[(3,1)] = G31
    Gslr[(1,3)] = G13
    return Gslr

# Wrapper function for waveform generation, resampling, and FD response processing in the perturbative formalism
def LISAGenerateTDI(phi0, fRef, m1, m2, chi1, chi2, dist, inc, lambd, beta, psi, tobs=1., minf=1e-5, maxf=1., t0=0., settRefAtfRef=False, tRef=0., trajdict=trajdict_MLDC, TDItag='TDIXYZ', order_fresnel_stencil=0, nptmin=100):
    wf = GenerateResamplePhenomD(phi0, fRef, m1, m2, chi1, chi2, dist, inc, tobs=tobs, minf=minf, maxf=maxf, settRefAtfRef=settRefAtfRef, tRef=tRef)
    freq_response = ResponseFrequencyGrid(wf)
    return LISAFDresponseTDI(freq_response, wf, inc, lambd, beta, psi, t0=t0, trajdict=trajdict, TDItag=TDItag, order_fresnel_stencil=order_fresnel_stencil)
# Wrapper function for waveform generation, resampling, and FD response processing in the perturbative formalism
def LISAGeneratey12(phi0, fRef, m1, m2, chi1, chi2, dist, inc, lambd, beta, psi, tobs=1., minf=1e-5, maxf=1., t0=0., settRefAtfRef=False, tRef=0., trajdict=trajdict_MLDC, order_fresnel_stencil=0, nptmin=100):
    wf = GenerateResamplePhenomD(phi0, fRef, m1, m2, chi1, chi2, dist, inc, tobs=tobs, minf=minf, maxf=maxf, settRefAtfRef=settRefAtfRef, tRef=tRef)
    freq_response = ResponseFrequencyGrid(wf)
    return LISAFDresponsey12(freq_response, wf, inc, lambd, beta, psi, t0=t0, trajdict=trajdict, order_fresnel_stencil=order_fresnel_stencil)

# Wrapper function for waveform generation, resampling, and FD response processing in the comb formalism
#TODO: make sure waveform is wider than output freq, for comb overflow
def LISAGenerateCombTDI(phi0, fRef, m1, m2, chi1, chi2, dist, inc, lambd, beta, psi, tobs=1., minf=1e-5, maxf=1., t0=0., settRefAtfRef=False, tRef=0., trajdict=trajdict_MLDC, TDItag='TDIXYZ', nptmin=400):
    wf = GenerateResamplePhenomD(phi0, fRef, m1, m2, chi1, chi2, dist, inc, tobs=tobs, minf=minf, maxf=maxf, settRefAtfRef=settRefAtfRef, tRef=tRef)
    freq_response = ResponseFrequencyGrid(wf)
    return LISAFDresponseCombTDI(freq_response, wf, inc, lambd, beta, psi, t0=t0, trajdict=trajdict, TDItag=TDItag)
# Wrapper function for waveform generation, resampling, and FD response processing in the comb formalism
#TODO: make sure waveform is wider than output freq, for comb overflow
def LISAGenerateComby12(phi0, fRef, m1, m2, chi1, chi2, dist, inc, lambd, beta, psi, tobs=1., minf=1e-5, maxf=1., t0=0., settRefAtfRef=False, tRef=0., trajdict=trajdict_MLDC, nptmin=400):
    wf = GenerateResamplePhenomD(phi0, fRef, m1, m2, chi1, chi2, dist, inc, tobs=tobs, minf=minf, maxf=maxf, settRefAtfRef=settRefAtfRef, tRef=tRef)
    freq_response = ResponseFrequencyGrid(wf)
    return LISAFDresponseComby12(freq_response, wf, inc, lambd, beta, psi, t0=t0, trajdict=trajdict)


# Given a waveform TDI channel, evaluate on a given frequency grid by interpolating amplitude, phases and transfer functions
def LISAEvaluateTDIFreqGrid(freqs, wfTDI, chan=1):
    res = np.zeros(len(freqs), dtype=complex)
    fbeg = max(freqs[0], wfTDI['freq'][0])
    fend = min(freqs[-1], wfTDI['freq'][-1])
    ibeg = np.where(freqs>=fbeg)[0][0]
    iend = np.where(freqs<=fend)[0][-1]
    fs = freqs[ibeg:iend+1]
    t0 = wfTDI['t0'] # will be used to apply timeshift
    ampspline = spline(wfTDI['freq'], wfTDI['amp'])
    phasespline = spline(wfTDI['freq'], wfTDI['phase'])
    phaseRdelayspline = spline(wfTDI['freq_response'], wfTDI['phaseRdelay'])
    keytransfer = 'transferL'+str(chan)
    transferLRespline = spline(wfTDI['freq_response'], np.real(wfTDI[keytransfer]))
    transferLImspline = spline(wfTDI['freq_response'], np.imag(wfTDI[keytransfer]))
    amp = ampspline(fs)
    phase = phasespline(fs)
    phaseRdelay = phaseRdelayspline(fs)
    transferLRe = transferLRespline(fs)
    transferLIm = transferLImspline(fs)
    phasetimeshift = 2*pi*t0*fs
    vals = (transferLRe+1j*transferLIm) * amp * np.exp(1j*(phase+phaseRdelay+phasetimeshift))
    res[ibeg:iend+1] = vals
    return res

# Individual yslr observables for the TD response
# hp,hc input as functions of time
def Evaluateyslr(hp, hc, t, Hplus, Hcross, k, t0=0., trajdict=trajdict_MLDC):
    # Trajectories
    funcp0 = trajdict['funcp0']
    funcp1L = trajdict['funcp1L']
    funcp2L = trajdict['funcp2L']
    funcp3L = trajdict['funcp3L']
    funcn1 = trajdict['funcn1']
    funcn2 = trajdict['funcn2']
    funcn3 = trajdict['funcn3']

    # Compute spacecraft positions
    torb = t + t0 # t0 is the orbital time at merger - t is the SSB time, arg of hp, hc
    # Compute positions and unit vectors
    # In our approximation, all positions are evaluated at torb - neglecting the effect of the interspacecraft delays on p,n, while keeping them in the waveform
    p0 = funcp0(torb)
    p1L = funcp1L(torb)
    p2L = funcp2L(torb)
    p3L = funcp3L(torb)
    n1 = funcn1(torb)
    n2 = funcn2(torb)
    n3 = funcn3(torb)

    # Compute intermediate scalar products
    kn1 = np.dot(k, n1)
    kn2 = np.dot(k, n2)
    kn3 = np.dot(k, n3)
    n1Hn1plus = np.dot(n1, np.dot(Hplus, n1))
    n2Hn2plus = np.dot(n2, np.dot(Hplus, n2))
    n3Hn3plus = np.dot(n3, np.dot(Hplus, n3))
    n1Hn1cross = np.dot(n1, np.dot(Hcross, n1))
    n2Hn2cross = np.dot(n2, np.dot(Hcross, n2))
    n3Hn3cross = np.dot(n3, np.dot(Hcross, n3))
    kp0 = np.dot(k, p0)
    kp1L = np.dot(k, p1L)
    kp2L = np.dot(k, p2L)
    kp3L = np.dot(k, p3L)

    # Prefactors and delays
    factorp = {}; factorc = {}; firstdelay = {}; seconddelay = {};
    # y12
    factorp[(1,2)] = (1./(1.-kn3)) * 0.5*n2Hn2plus;
    factorc[(1,2)] = (1./(1.-kn3)) * 0.5*n2Hn2cross;
    firstdelay[(1,2)] = -(kp0 + (kp1L + L_SI))/C_SI;
    seconddelay[(1,2)] = -(kp0 + kp2L)/C_SI;
    # y21
    factorp[(2,1)] = (1./(1.+kn3)) * 0.5*n2Hn2plus;
    factorc[(2,1)] = (1./(1.+kn3)) * 0.5*n2Hn2cross;
    firstdelay[(2,1)] = -(kp0 + (kp2L + L_SI))/C_SI;
    seconddelay[(2,1)] = -(kp0 + kp1L)/C_SI;
    # y23
    factorp[(2,3)] = (1./(1.-kn1)) * 0.5*n2Hn2plus;
    factorc[(2,3)] = (1./(1.-kn1)) * 0.5*n2Hn2cross;
    firstdelay[(2,3)] = -(kp0 + (kp2L + L_SI))/C_SI;
    seconddelay[(2,3)] = -(kp0 + kp3L)/C_SI;
    # y32
    factorp[(3,2)] = (1./(1.+kn1)) * 0.5*n2Hn2plus;
    factorc[(3,2)] = (1./(1.+kn1)) * 0.5*n2Hn2cross;
    firstdelay[(3,2)] = -(kp0 + (kp3L + L_SI))/C_SI;
    seconddelay[(3,2)] = -(kp0 + kp2L)/C_SI;
    # y31
    factorp[(3,1)] = (1./(1.-kn2)) * 0.5*n2Hn2plus;
    factorc[(3,1)] = (1./(1.-kn2)) * 0.5*n2Hn2cross;
    firstdelay[(3,1)] = -(kp0 + (kp3L + L_SI))/C_SI;
    seconddelay[(3,1)] = -(kp0 + kp1L)/C_SI;
    # y13
    factorp[(1,3)] = (1./(1.+kn2)) * 0.5*n2Hn2plus;
    factorc[(1,3)] = (1./(1.+kn2)) * 0.5*n2Hn2cross;
    firstdelay[(1,3)] = -(kp0 + (kp1L + L_SI))/C_SI;
    seconddelay[(1,3)] = -(kp0 + kp3L)/C_SI;

    # yslr
    keys = [(1,2), (2,1), (2,3), (3,2), (3,1), (1,3)]
    yslr = {}
    for k in keys:
        yslr[k] = factorp[k] * (hp(t+firstdelay) - hp(t+seconddelay)) + factorc[k] * (hc(t+firstdelay) - hc(t+seconddelay))

    return yslr

def TDICombinationTD(yslr, yslrd, yslrdd, yslrddd, t, TDItag='TDIXYZ'):
    # X,Y,Z combinations
    X = (yslr[(3,1)] + yslrd[(1,3)]) + (yslrdd[(2,1)] + yslrddd[(1,2)]) - (yslrdd[(3,1)] + yslrddd[(1,3)]) - (yslr[(2,1)] + yslrd[(1,2)])
    Y = (yslr[(1,2)] + yslrd[(2,1)]) + (yslrdd[(3,2)] + yslrddd[(2,3)]) - (yslrdd[(1,2)] + yslrddd[(2,1)]) - (yslr[(3,2)] + yslrd[(2,3)])
    Z = (yslr[(2,3)] + yslrd[(3,2)]) + (yslrdd[(1,3)] + yslrddd[(3,1)]) - (yslrdd[(2,3)] + yslrddd[(3,2)]) - (yslr[(1,3)] + yslrd[(3,1)])
    tdi = {}
    if TDItag=='TDIXYZ':
        tdi['TDI1'] = X
        tdi['TDI2'] = Y
        tdi['TDI3'] = Z
    elif TDItag=='TDIAET': # Note that the LDC convention here is 2* the convention of McWilliams&al_0911
        tdi['TDI1'] = 1/(sqrt2) * (Z - X)
        tdi['TDI2'] = 1/(sqrt6) * (Z + X - 2*Y)
        tdi['TDI3'] = 1/(sqrt3) * (Z + X + Y)
    else:
        raise ValueError("Error in TDICombinationTD: TDItag not recognized.")

    return tdi

# Time-domain LISA response
# hp,hc input as functions of time
def LISATDresponseTDI(hp, hc, times, lambd, beta, psi, t0=0., trajdict=trajdict_MLDC, TDItag='TDIXYZ'):
    # Waveform given in the form of pure functions for hplus and hcross
    # The times are assumed to fall within the range of definition of hp, hc
    # NOTE: since TDI essentially takes discrete derivatives, accuracy matters close to merger - make sure there is enough sampling there
    # For now, we consider single-mode h22 waveforms only
    n = len(times)
    armd = L_SI/C_SI

    # Wave unit vector
    k = funck(lambd, beta)

    # Compute constant matrices Hplus and Hcross in the SSB frame
    O1 = funcO1(lambd, beta, psi)
    invO1 = funcinverseO1(lambd, beta, psi)
    Hplus = np.dot(O1, np.dot(HSplus, invO1))
    Hcross = np.dot(O1, np.dot(HScross, invO1))

    # Initialize output
    wfTDI = {}
    wfTDI['TDItag'] = TDItag
    wfTDI['t'] = np.copy(times)
    wfTDI['TDI1'] = np.zeros(n, dtype=np.float64)
    wfTDI['TDI2'] = np.zeros(n, dtype=np.float64)
    wfTDI['TDI3'] = np.zeros(n, dtype=np.float64)

    # Main loop over frequencies
    for i in range(n):
        t = times[i]
        yslr = Evaluateyslr(hp, hc, t, Hplus, Hcross, k, t0=t0, trajdict=trajdict)
        yslrd = Evaluateyslr(hp, hc, t-armd, Hplus, Hcross, k, t0=t0, trajdict=trajdict)
        yslrdd = Evaluateyslr(hp, hc, t-2*armd, Hplus, Hcross, k, t0=t0, trajdict=trajdict)
        yslrddd = Evaluateyslr(hp, hc, t-3*armd, Hplus, Hcross, k, t0=t0, trajdict=trajdict)
        tdi = TDICombinationTD(yslr, yslrd, yslrdd, yslrddd, t, TDItag=TDItag)
        wfTDI['TDI1'][i] = tdi['TDI1']
        wfTDI['TDI2'][i] = tdi['TDI2']
        wfTDI['TDI3'][i] = tdi['TDI3']

    return wfTDI

# Function reproducing XLALSpinWeightedSphericalHarmonic
# - Currently only supports s=-2, l=2,3,4,5 modes
def SpinWeightedSphericalHarmonic(s, l, m, theta, phi):
    func = "SpinWeightedSphericalHarmonic"
    # Sanity checks
    if ( l < abs(s) ):
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |s| <= l\n' % (func, s, l, m))
    if ( l < abs(m) ):
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |m| <= l\n' % (func, s, l, m))
    if not ( s == -2 ):
        raise ValueError('Error - %s: Invalid mode s=%d - only s=-2 implemented\n' % (func, s))

    fac = {
        # l=2
        (2,-2) : sqrt( 5.0 / ( 64.0 * pi ) ) * ( 1.0 - cos( theta ))*( 1.0 - cos( theta )),
        (2,1) : sqrt( 5.0 / ( 16.0 * pi ) ) * sin( theta )*( 1.0 - cos( theta )),
        (2,0) : sqrt( 15.0 / ( 32.0 * pi ) ) * sin( theta )*sin( theta ),
        (2,1) : sqrt( 5.0 / ( 16.0 * pi ) ) * sin( theta )*( 1.0 + cos( theta )),
        (2,2) : sqrt( 5.0 / ( 64.0 * pi ) ) * ( 1.0 + cos( theta ))*( 1.0 + cos( theta )),
        # l=3
        (3,-3) : sqrt(21.0/(2.0*pi))*cos(theta/2.0)*pow(sin(theta/2.0),5.0),
        (3,-2) : sqrt(7.0/4.0*pi)*(2.0 + 3.0*cos(theta))*pow(sin(theta/2.0),4.0),
        (3,-1) : sqrt(35.0/(2.0*pi))*(sin(theta) + 4.0*sin(2.0*theta) - 3.0*sin(3.0*theta))/32.0,
        (3,0) : (sqrt(105.0/(2.0*pi))*cos(theta)*pow(sin(theta),2.0))/4.0,
        (3,1) : -sqrt(35.0/(2.0*pi))*(sin(theta) - 4.0*sin(2.0*theta) - 3.0*sin(3.0*theta))/32.0,
        (3,2) : sqrt(7.0/pi)*pow(cos(theta/2.0),4.0)*(-2.0 + 3.0*cos(theta))/2.0,
        (3,3) : -sqrt(21.0/(2.0*pi))*pow(cos(theta/2.0),5.0)*sin(theta/2.0),
        # l=4
        (4,-4) : 3.0*sqrt(7.0/pi)*pow(cos(theta/2.0),2.0)*pow(sin(theta/2.0),6.0),
        (4,-3) : 3.0*sqrt(7.0/(2.0*pi))*cos(theta/2.0)*(1.0 + 2.0*cos(theta))*pow(sin(theta/2.0),5.0),
        (4,-2) : (3.0*(9.0 + 14.0*cos(theta) + 7.0*cos(2.0*theta))*pow(sin(theta/2.0),4.0))/(4.0*sqrt(pi)),
        (4,-1) : (3.0*(3.0*sin(theta) + 2.0*sin(2.0*theta) + 7.0*sin(3.0*theta) - 7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*pi)),
        (4,0) : (3.0*sqrt(5.0/(2.0*pi))*(5.0 + 7.0*cos(2.0*theta))*pow(sin(theta),2.0))/16.0,
        (4,1) : (3.0*(3.0*sin(theta) - 2.0*sin(2.0*theta) + 7.0*sin(3.0*theta) + 7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*pi)),
        (4,2) : (3.0*pow(cos(theta/2.0),4.0)*(9.0 - 14.0*cos(theta) + 7.0*cos(2.0*theta)))/(4.0*sqrt(pi)),
        (4,3) : -3.0*sqrt(7.0/(2.0*pi))*pow(cos(theta/2.0),5.0)*(-1.0 + 2.0*cos(theta))*sin(theta/2.0),
        (4,4) : 3.0*sqrt(7.0/pi)*pow(cos(theta/2.0),6.0)*pow(sin(theta/2.0),2.0),
        # l= 5
        (5,-5) : sqrt(330.0/pi)*pow(cos(theta/2.0),3.0)*pow(sin(theta/2.0),7.0),
        (5,-4) : sqrt(33.0/pi)*pow(cos(theta/2.0),2.0)*(2.0 + 5.0*cos(theta))*pow(sin(theta/2.0),6.0),
        (5,-3) : (sqrt(33.0/(2.0*pi))*cos(theta/2.0)*(17.0 + 24.0*cos(theta) + 15.0*cos(2.0*theta))*pow(sin(theta/2.0),5.0))/4.0,
        (5,-2) : (sqrt(11.0/pi)*(32.0 + 57.0*cos(theta) + 36.0*cos(2.0*theta) + 15.0*cos(3.0*theta))*pow(sin(theta/2.0),4.0))/8.0,
        (5,-1) : (sqrt(77.0/pi)*(2.0*sin(theta) + 8.0*sin(2.0*theta) + 3.0*sin(3.0*theta) + 12.0*sin(4.0*theta) - 15.0*sin(5.0*theta)))/256.0,
        (5,0) : (sqrt(1155.0/(2.0*pi))*(5.0*cos(theta) + 3.0*cos(3.0*theta))*pow(sin(theta),2.0))/32.0,
        (5,1) : sqrt(77.0/pi)*(-2.0*sin(theta) + 8.0*sin(2.0*theta) - 3.0*sin(3.0*theta) + 12.0*sin(4.0*theta) + 15.0*sin(5.0*theta))/256.0,
        (5,2) : sqrt(11.0/pi)*pow(cos(theta/2.0),4.0)*(-32.0 + 57.0*cos(theta) - 36.0*cos(2.0*theta) + 15.0*cos(3.0*theta))/8.0,
        (5,3) : -sqrt(33.0/(2.0*pi))*pow(cos(theta/2.0),5.0)*(17.0 - 24.0*cos(theta) + 15.0*cos(2.0*theta))*sin(theta/2.0)/4.0,
        (5,4) : sqrt(33.0/pi)*pow(cos(theta/2.0),6.0)*(-2.0 + 5.0*cos(theta))*pow(sin(theta/2.0),2.0),
        (5,5) : -sqrt(330.0/pi)*pow(cos(theta/2.0),7.0)*pow(sin(theta/2.0),3.0)
        }.get((l,m), None)
    if fac==None:
        raise ValueError('Error - %s: Invalid mode s=%d, l=%d, m=%d - require |m| <= l\n' % (func, s, l, m))

    # Result
    if m==0:
        return fac
    else:
        return fac * exp(1j*m*phi)
