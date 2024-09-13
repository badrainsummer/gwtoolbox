import matplotlib.pyplot as plt
import numpy as np

from lal import MSUN_SI, PC_SI, CreateDict
import lalsimulation as LS

phi0 = 0.0
fRef = 100.0
deltaF = 1.0/32.0
m1_SI = 50*MSUN_SI
m2_SI = 30*MSUN_SI
chi1 = 0.85
chi2 = 0.42
f_min = 10.0
f_max = 0.0
distance = 500.0e6 * PC_SI
inclination = 0.5

LALpars = CreateDict()
longAscNodes, eccentricity, meanPerAno = 0, 0, 0
S1x, S1y, S1z = 0, 0, chi1
S2x, S2y, S2z = 0, 0, chi2

hptildeFS, hctildeFS = LS.SimInspiralChooseFDWaveform(m1_SI, m2_SI,
    S1x, S1y, S1z, S2x, S2y, S2z,
    distance, inclination, phi0, longAscNodes, eccentricity, meanPerAno,
    deltaF, f_min, f_max, fRef,
    LALpars, LS.IMRPhenomD)

fHz = np.arange(hptildeFS.data.length)*deltaF
hptilde = hptildeFS.data.data
hctilde = hctildeFS.data.data

np.save("test_lalsimulation.npy", np.vstack([fHz, hptilde, hctilde]))

plt.semilogx(fHz, hptilde.real)
plt.semilogx(fHz, hctilde.real, 'r')
plt.xlim([f_min, 1e3])
plt.xlabel(r'$f \, \mathrm{[Hz]}$')
plt.ylabel(r'$\tilde h_+, h_\times$')
plt.show()
