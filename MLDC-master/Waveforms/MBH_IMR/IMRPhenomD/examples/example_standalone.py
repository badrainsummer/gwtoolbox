import matplotlib.pyplot as plt
import numpy as np

from pyIMRPhenomD import IMRPhenomD, MSUN_SI, PC_SI

#MSUN_SI = 1.98854695496e+30

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

wf = IMRPhenomD(phi0, fRef, deltaF, m1_SI, m2_SI, chi1, chi2, f_min, f_max, distance, inclination)

fHz, hptilde, hctilde = wf.GetWaveform()

np.save("test_standalone.npy", np.vstack([fHz, hptilde, hctilde]))

plt.semilogx(fHz, hptilde.real)
plt.semilogx(fHz, hctilde.real, 'r')
plt.xlim([f_min, 1e3])
plt.xlabel(r'$f \, \mathrm{[Hz]}$')
plt.ylabel(r'$\tilde h_+, h_\times$')
plt.show()
