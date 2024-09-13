import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('/Users/yishuxu/Downloads/MLDC-master/software/Packages/common/')
import FastBinary as FB
year = 31558149.8
fm  = 3.168753575e-8
print("Let's roll")

ks = ['Frequency', 'FrequencyDerivative', 'EclipticLatitude', 'EclipticLongitude', 'Amplitude', 'Inclination', 'Polarization', 'InitialPhase']
par1 = [1e-3, 0.0, 0.25*np.pi, 0.25*np.pi, 8.e-22, 0.45*np.pi, 0.35*np.pi, 1.75*np.pi]
bnr = {}
for i in  range(len(ks)):
    bnr[ks[i]] = par1[i]

print(bnr)
dt = 15.0
Nt = 2**21
#print Nt*dt, year
#sys.exit(0)



#Tobs = 2*4194304*dt
Tobs=2*year
fastB = FB.FastGalacticBinary("Stas", bnr)
Xf, Yf, Zf = fastB.onefourier(simulator='synthlisa', vector=par1, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
#Xf, Yf, Zf = fastB.onefourier(simulator='synthlisa', vector=par1, buffer=None, T=Tobs, dt=dt, algorithm='mldc', oversample=2)
Xf2, Yf2, Zf2 = fastB.onefourier(simulator='synthlisa', vector=par1, buffer=None, T=Tobs*0.5, dt=dt, algorithm='Michele', oversample=2)

#print Xf
#print Xf.df
#print Xf.kmin, len(Xf)

#print int(1.0/(dt*Xf.df)), 2*4194304

frq = np.arange(Xf.kmin, Xf.kmin + len(Xf))*Xf.df
frq2= np.arange(Xf2.kmin, Xf2.kmin +len(Xf2))*Xf2.df
#xf = np.zeros(Xf.kmin + len(Xf), dtype='complex128')
#xf[Xf.kmin : Xf.kmin+len(Xf)] = Xf
#plt.yscale('log')
#plt.plot(frq, np.absolute(Xf))
#plt.plot(frq2, np.absolute(Xf2))

Xt = Xf.ifft(dt)
Xt2= Xf2.ifft(dt)

tm = np.arange(len(Xt))*dt
tm2= np.arange(len(Xt2))*dt

fig1, ax1 = plt.subplots(2, sharex=True)
fig2, ax2 = plt.subplots(2, sharex=True)
ax1[0].plot((tm)/year, Xt)
#ax[0].plot((tm[Nyear:N3years]-one_year)/year, Xt[Nyear:N3years])
#ax[0].plot((tm[Nyear:N2years]-one_year)/year, Xt[Nyear:N2years])
ax1[0].grid(True)
ax1[1].plot((tm2)/year, Xt2)
#ax[1].plot((tm[Nyear:N3years]-one_year)/year, Yt[Nyear:N3years])
#ax[1].plot((tm[Nyear:N2years]-one_year)/year, Yt[Nyear:N2years])
ax1[1].grid(True)
ax2[0].plot(frq, np.absolute(Xf))
ax2[0].grid(True)
ax2[1].plot(frq2, np.absolute(Xf2))
ax2[1].grid(True)
fig1.savefig("figure1.png")
fig2.savefig("figure2.png")
