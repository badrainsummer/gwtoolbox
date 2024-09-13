import numpy as np
import sys
sys.path.append('/Users/yishuxu/Downloads/MLDC-master/software/Packages/common/')
import FastBinary as FB
import matplotlib.pyplot as plt

year = 31558149.8
fm  = 3.168753575e-8

def LISAmotion(tm):
    light =  299792458.0
    AU = 149597870700.0/light  ## AU in sec.
    lamb = 0.0
    kappa = 0.0
    arm = 2.5e9 ### armlength in m
    arm_t = arm/light

    N = len(tm)

    a = AU
    e = arm_t/(2.0*np.sqrt(3.0)*a)
    nn = np.array([1,2,3])

    Beta = (nn-1)*2.0*np.pi/3.0 + lamb
    alpha = 2.0*np.pi*tm/year + kappa

    x = np.zeros((3, N))
    y = np.zeros((3, N))
    z = np.zeros((3, N))
    for i in range(3):
        x[i, :] = a*np.cos(alpha) + a*e*(np.sin(alpha)*np.cos(alpha)*np.sin(Beta[i]) - (1.0 + (np.sin(alpha))**2)*np.cos(Beta[i]))
        y[i, :] = a*np.sin(alpha) + a*e*(np.sin(alpha)*np.cos(alpha)*np.cos(Beta[i]) - (1.0 + (np.cos(alpha))**2)*np.sin(Beta[i]))
        z[i, :] = -np.sqrt(3.0)*a*e*np.cos(alpha - Beta[i])

    return (x, y, z, alpha)




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



Tobs = 2*4194304*dt
fastB = FB.FastGalacticBinary("Stas", bnr)
Xf, Yf, Zf = fastB.onefourier(simulator='synthlisa', vector=par1, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
#Xf, Yf, Zf = fastB.onefourier(simulator='synthlisa', vector=par1, buffer=None, T=Tobs, dt=dt, algorithm='mldc', oversample=2)

#print Xf
#print Xf.df
#print Xf.kmin, len(Xf)

#print int(1.0/(dt*Xf.df)), 2*4194304

frq = np.arange(1, Xf.kmin + len(Xf)+1)*Xf.df
#xf = np.zeros(Xf.kmin + len(Xf), dtype='complex128')
#xf[Xf.kmin : Xf.kmin+len(Xf)] = Xf

#plt.plot(frq, np.absolute(xf))
#plt.grid(True)
#plt.show()

Xt = Xf.ifft(dt)
Yt = Yf.ifft(dt)
Zt = Zf.ifft(dt)

#print len(Xt), year/dt
tm = np.arange(len(Xt))*dt

#Nyear = 4194304
Nyear = Nt
N3years = 3*Nt
N2years = 2*Nt
one_year = Nyear*dt



f, ax = plt.subplots(3, sharex=True)
ax[0].plot((tm)/year, Xt)
#ax[0].plot((tm[Nyear:N3years]-one_year)/year, Xt[Nyear:N3years])
#ax[0].plot((tm[Nyear:N2years]-one_year)/year, Xt[Nyear:N2years])
ax[0].grid(True)
ax[1].plot((tm)/year, Yt)
#ax[1].plot((tm[Nyear:N3years]-one_year)/year, Yt[Nyear:N3years])
#ax[1].plot((tm[Nyear:N2years]-one_year)/year, Yt[Nyear:N2years])
ax[1].grid(True)
ax[2].plot((tm)/year, Zt)
#ax[2].plot((tm[Nyear:N3years]-one_year)/year, Zt[Nyear:N3years])
#ax[2].plot((tm[Nyear:N2years]-one_year)/year, Zt[Nyear:N2years])

plt.grid(True)
plt.show()


Tobs = 4.*one_year

GW = np.genfromtxt("Galaxy_Bright_2.dat", skip_header=1)#, names=True)

table = []
#table.append(par1)

#print len(GW)
#print GW
#print np.array(GW[0])
for i in range(len(GW)):
    #table.append(np.array(GW[i]))
    table.append(GW[i])

print(table)

#sys.exit(0)

fastB = FB.FastGalacticBinary("Stas", bnr)
Xt, Yt, Zt = fastB.TDI(T=Tobs, dt=dt ,simulator='synthlisa', table=table, algorithm='Michele',oversample=4)

#Xf, Yf, Zf = fastB.onefourier(simulator='synthlisa', vector=par1, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)

tm = np.arange(len(Xt))*dt
Nyear = int(year/dt)+1

trd = tm[Nyear:Nyear + 2*Nt]
Xtrd = Xt[Nyear:Nyear + 2*Nt]
Ytrd = Yt[Nyear:Nyear + 2*Nt]
Ztrd = Zt[Nyear:Nyear + 2*Nt]


f, ax = plt.subplots(3, sharex=True)
ax[0].plot((tm)/year, Xt)
ax[0].plot((trd)/year, Xtrd)

#ax[0].plot((tm[Nyear:N3years]-one_year)/year, Xt[Nyear:N3years])
#ax[0].plot((tm[Nyear:N2years]-one_year)/year, Xt[Nyear:N2years])
ax[0].grid(True)
ax[1].plot((tm)/year, Yt)
ax[1].plot((trd)/year, Ytrd)
#ax[1].plot((tm[Nyear:N3years]-one_year)/year, Yt[Nyear:N3years])
#ax[1].plot((tm[Nyear:N2years]-one_year)/year, Yt[Nyear:N2years])
ax[1].grid(True)
ax[2].plot((tm)/year, Zt)
ax[2].plot((trd)/year, Ztrd)
#ax[2].plot((tm[Nyear:N3years]-one_year)/year, Zt[Nyear:N3years])
#ax[2].plot((tm[Nyear:N2years]-one_year)/year, Zt[Nyear:N2years])

plt.grid(True)
plt.show()

x,y,z,alp = LISAmotion(tm)
xr, yr, zr, alpr = LISAmotion(trd)

#print np.sin(alp[0]), np.sin(alpr[0]), np.arcsin(np.sin(alpr[0]))

#plt.plot(tm, np.sin(alp))
#plt.plot(trd, np.sin(alpr))
plt.plot(tm, z[2, :])
plt.plot(trd-trd[0], zr[2, :], '--')
plt.grid(True)
plt.show()

#ks = ['Frequency', 'FrequencyDerivative', 'EclipticLatitude', 'EclipticLongitude', 'Amplitude', 'Inclination', 'Polarization', 'InitialPhase']

fout = open("Params.dat", 'w')
for par in table:
    fr0 = par[0]
    fdot = par[1]
    amp = par[4]
    phi0 = par[7]
    del_tm = trd[0]
    fr = fr0 + fdot*del_tm
    amp = amp *(fr/fr0)**(2./3.)
    phi = phi0 + fr0*(2.0*np.pi*del_tm) + fdot*(np.pi*del_tm**2)
    fout.write(str(fr) + "   " + str(fdot) + "   " + str(par[2]) + "   " + str(par[3]) + "   " +\
                str(amp) + "   " + str(par[5]) + "   " + str(par[6]) + "   " + str(phi) + " \n")
fout.close()


trd = trd-trd[0]
fout = open("CheckGB.dat", 'w')
for i in range(len(trd)):
    fout.write(str(trd[i]) + "   " + str(Xtrd[i]) + "   " + str(Ytrd[i]) + "   " + str(Ztrd[i]) + " \n")
fout.close()


nm = "src_"
for i, pr in enumerate(table):
    print(pr)
    Xf, Yf, Zf = fastB.onefourier(simulator='synthlisa', vector=pr, buffer=None, T=Tobs, dt=dt, algorithm='Michele', oversample=4)
    Xt = Xf.ifft(dt)
    Yt = Yf.ifft(dt)
    Zt = Zf.ifft(dt)

    Xtrd= Xt[Nyear:Nyear + 2*Nt]
    Ytrd = Yt[Nyear:Nyear + 2*Nt]
    Ztrd = Zt[Nyear:Nyear + 2*Nt]

    f, ax = plt.subplots(3, sharex=True)
    ax[0].plot((tm)/year, Xt)
    ax[0].plot((trd)/year, Xtrd)
    ax[0].grid(True)

    ax[1].plot((tm)/year, Yt)
    ax[1].plot((trd)/year, Ytrd)
    ax[1].grid(True)

    ax[2].plot((tm)/year, Zt)
    ax[2].plot((trd)/year, Ztrd)
    ax[2].grid(True)
    plt.show()
    fout = open(nm + str(i) +".dat", 'w')
    for i in range(len(trd)):
        fout.write(str(trd[i]) + "   " + str(Xtrd[i]) + "   " + str(Ytrd[i]) + "   " + str(Ztrd[i]) + " \n")
    fout.close()


sys.exit(0)


ks = ['Frequency', 'EclipticLatitude', 'EclipticLongitude', 'Amplitude', 'Inclination', 'Polarization', 'InitialPhase']

GBs = {}
for ty in ks:
    GBs[ty] = GW[ty]
