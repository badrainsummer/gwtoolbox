import countdown
import FrequencyArray
import LISAConstants as LC
import tdi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


### Testing some functionality of tdi module

df = 1.e-6
N = int(1.e-1/df)
fr = np.arange(1,N+1)*df

SA = tdi.noisepsd_AE(fr,includewd=None)
SAtot = tdi.noisepsd_AE(fr,includewd=2.0)
plt.loglog(fr, SA)
plt.loglog(fr, SAtot)
plt.grid(True)
plt.show()


Sinst = tdi.lisasens(fr, noisemodel=None, includewd=None)

Stot = tdi.lisasens(fr, noisemodel=None, includewd=2.0)

plt.loglog(fr, np.sqrt(Sinst))
plt.loglog(fr, np.sqrt(Stot))
plt.grid(True)
plt.show()

#### reading the data and computing the PSD
noise = np.genfromtxt("/Users/stas/Projects/LISA/Gaps/Code/l3gapsXAE-TDI.txt")

print np.shape(noise)
tm = noise[:, 0]
A = noise[:, 2]

### removing the glitch in the begining and estimating PSD
ind = np.argwhere(tm>=500)[0][0]
tm = tm[ind:]
A = A[ind:]
f, psd_A = signal.welch(A, fs=0.1, nperseg=16384)

f = np.clip(f, 6.1e-6, 5.e-2)


#### Reading Antoine's files

datA1 = np.genfromtxt("/Users/stas/Downloads/Baseline_PSDrf_XAE_AnalyticHR.txt")
datA2 = np.genfromtxt("/Users/stas/Downloads/IMS15pm_PSDrf_XAE_AnalyticHR.txt")


plt.loglog(f, psd_A, label="PSD")
plt.loglog(fr, SA, label="SA")
plt.loglog(datA1[:, 0], datA1[:, 2], label="Baseline")
plt.loglog(datA2[:, 0], datA2[:, 2], label="IMS15pm")
plt.legend()
plt.grid(True)
plt.show()
