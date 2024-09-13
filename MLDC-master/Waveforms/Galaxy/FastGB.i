%{
#define SWIG_FILE_WITH_INIT
%}


/* name of module to use*/
%module FastGB
%include "numpy.i"



%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* XLS, int nXLS),
                                      (double* YLS, int nYLS),
                                      (double* ZLS, int nZLS),
                                      (double* XSL, int nXSL),
                                      (double* YSL, int nYSL),
                                      (double* ZSL, int nZSL)}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* params, int NP)}


/* void spacecraft(double t, double *x, double *y, double *z); */

/* void Fast_GB(double *pars, long N, double Tobs, double dt, double *XLS, double *ALS, double *ELS, int NP); */



%inline %{

#include "Constants.h"
#include "GB.h"

void ComputeXYZ_FD(double* params, int NP, long N, double Tobs, double dt, double* XLS, int nXLS,
            double* YLS, int nYLS, double* ZLS, int nZLS, double* XSL, int nXSL,
                        double* YSL, int nYSL, double* ZSL, int nZSL){
    if (NP != 8){
        printf("only 8 parameters are accepted: %d",  NP);
        exit(1);
    }
    // Params passed and what will be used in C-function do not match, the input is:
    // Frequency FrequencyDerivative  EclipticLatitude EclipticLongitude Amplitude Inclination Polarization InitialPhase
    double f0 = params[0];
    double df0 = params[1];
    double lat = params[2];
    double lng = params[3];
    double Amp = params[4];
    double incl = params[5];
    double psi = params[6];
    double phi0 = params[7];

    // Transform:
    double* pars = NULL;
    pars = malloc(NP*sizeof(double));

    pars[0] = f0*Tobs;
    pars[1] = cos(PIon2 - lat); // convert to spherical polar
    pars[2] = lng;
    pars[3] = log(Amp);
    pars[4] = cos(incl);
    pars[5] = psi;
    pars[6] = phi0;
    pars[7] = df0*Tobs*Tobs;
    //pars[8] = 11./3.*df0*df0/f0*Tobs*Tobs*Tobs;

    //printf("Stas in wrapper: f0 = %f, q = %f, Tobs = %f \n", f0, pars[0], Tobs);

    Fast_GB(pars, N, Tobs, dt, XLS, YLS, ZLS, XSL, YSL, ZSL, NP);

    free(pars);

};

void TestMe(double* XLS, int nXLS){
    printf("Stas: %d \n", nXLS);
    int i;
    for (i=0; i<nXLS; i++){
        printf("%d   %f\n", i, XLS[i]);
    }
};

%}


%pythoncode %{

import numpy as np
import math
import FrequencyArray
import LISAConstants as LC
import tdi
from countdown import countdown

class FastGB:
  #FastBinaryCache = {}
  def __init__(self, name=" ", dt=15.0, Tobs=6.2914560e7, orbit="analytic"):
        self.outputlist = ( ('Frequency',                  'Hertz',     'GW frequency'),\
                            ('FrequencyDerivative',        'Hertz/s',    'GW frequency derivative'),\
                            ('EclipticLatitude',           'Radian',     'standard ecliptic latitude'),\
                            ('EclipticLongitude',          'Radian',     'standard ecliptic longitude'),\
                            ('Amplitude',                  'strain',     'dimensionless GW amplitude'),\
                            ('Inclination',                'Radian',     'standard source inclination'),\
                            ('Polarization',               'Radian',     'standard source polarization'),\
                            ('InitialPhase',               'Radian',     'GW phase at t = 0') )
        self.Tobs = Tobs
        self.dt = dt



  def fourier(self,simulator='synthlisa',table=None,T=6.2914560e7,dt=15,algorithm='mldc',oversample=1,kmin=0,length=None,status=True):
        if (np.any(table) == None):
            return self.onefourier(simulator=simulator,T=T,dt=dt,algorithm=algorithm,oversample=oversample)
        else:
            if length == None:
                length = int(T/dt)/2 + 1    # was "NFFT = int(T/dt)", and "NFFT/2+1" passed to np.zeros

            buf = tuple(FrequencyArray.FrequencyArray(np.zeros(length,dtype=np.complex128),kmin=kmin,df=1.0/T) for i in range(3))

            if status: c = countdown(len(table),10000)
            for line in table:
                self.onefourier(simulator=simulator,vector=line,buffer=buf,T=T,dt=dt,algorithm=algorithm,oversample=oversample)
                if status: c.status()
            if status: c.end()

            return buf

  def buffersize(self,f,A,algorithm='ldc',oversample=1):


      YEAR = LC.YRSID_SI
      Acut = tdi.simplesnr(f,A,years=self.Tobs/YEAR)
      mult = 8
      if((self.Tobs/YEAR) <= 8.0): mult = 8
      if((self.Tobs/YEAR) <= 4.0): mult = 4
      if((self.Tobs/YEAR) <= 2.0): mult = 2
      if((self.Tobs/YEAR) <= 1.0): mult = 1
      N = 32*mult
      if(f > 0.001): N = 64*mult
      if(f > 0.01):  N = 256*mult
      if(f > 0.03):  N = 512*mult
      if(f > 0.1):   N = 1024*mult

      M = int(math.pow(2.0,1 + int(np.log(Acut)/np.log(2.0))))

      if(M > 8192):
        M = 8192

      M = N = max(M,N)

      N *= oversample

      return(N)

  def onefourier(self,simulator='synthlisa',params=None,buffer=None,T=6.2914560e7,dt=15,algorithm='mldc',oversample=1):
      if np.any(params) != None:
          self.Frequency,self.FrequencyDerivative,self.Amplitude = params[0],params[1],params[4]

      ##FIXME I assume that T=Tobs below
      if (T != self.Tobs):
          print ("times do not match:", T, self.Tobs)
          raise NotImplementedError
      N = self.buffersize(self.Frequency,self.Amplitude,algorithm,oversample)
      M = N

      XLS = np.zeros(2*M,'d')
      YLS = np.zeros(2*M,'d')
      ZLS = np.zeros(2*M,'d')

      XSL = np.zeros(2*M,'d')
      YSL = np.zeros(2*M,'d')
      ZSL = np.zeros(2*M,'d')

      #TestMe(XLS)

      if np.any(params) != None:
          NP = 8
          # vector must be ordered as required by Fast_GB
          #Fast_GB(double *params, long N, double *XLS, double *ALS, double *ELS, int NP)

          ComputeXYZ_FD(params, N, self.Tobs, self.dt, XLS, YLS, ZLS, XSL, YSL, ZSL)
          ### TODO Need to transform to SL if required
          Xf = XLS
          Yf = YLS
          Zf = ZLS
          if (simulator == 'synthlisa'):
              Xf = XSL
              Yf = YSL
              Zf = ZSL
      else:
          #FIXME
          pass

      f0 = self.Frequency
      # f0 = self.Frequency + 0.5 * self.FrequencyDerivative * T
      if buffer == None:
          retX, retY, retZ = map(lambda a: FrequencyArray.FrequencyArray(a[::2] + 1.j* a[1::2], \
                                dtype = np.complex128, kmin = int(f0*T) - M/2, df= 1.0/T), (Xf, Yf, Zf))
          return (retX,retY,retZ)
      else:
          kmin, blen, alen = buffer[0].kmin, len(buffer[0]), 2*M
          #print ("herak", kmin, blen, alen, len(Xf))

          beg, end = int(int(f0*T) - M/2), int(f0*T  + M/2)                             # for a full buffer, "a" begins and ends at these indices
          begb, bega = (beg - kmin, 0) if beg >= kmin else (0, 2*(kmin - beg))    # left-side alignment of partial buffer with "a"
          endb, enda = (end - kmin, alen) if end - kmin <= blen else (blen, alen - 2*(end - kmin - blen))
                                                                          # the corresponding part of "a" that should be assigned to the partial buffer
                                                                          # ...remember "a" is doubled up
                                                                          # check: if kmin = 0, then begb = beg, endb = end, bega = 0, enda = alen
          bega = int(bega); begb = int(begb)
          enda = int(enda); endb = int(endb)
          for i,a in enumerate((Xf, Yf, Zf)):
              buffer[i][begb:endb] += a[bega:enda:2] + 1j * a[(bega+1):enda:2]



  def fourier(self,simulator='synthlisa',table=None,T=6.2914560e7,dt=15,algorithm='mldc',oversample=1,kmin=0,length=None,status=True):
        if (np.any(table) == None):
            return self.onefourier(simulator=simulator,T=T,dt=dt,algorithm=algorithm,oversample=oversample)
        else:
            if length == None:
                length = int(0.5*T/dt) + 1    # was "NFFT = int(T/dt)", and "NFFT/2+1" passed to numpy.zeros

            #length = int(length)
            print ("Stas", type(length), type(kmin), type(1.0/T))
            buf = tuple(FrequencyArray.FrequencyArray(np.zeros(length,dtype=np.complex128),kmin=kmin,df=1.0/T) for i in range(3))

            if status: c = countdown(len(table),10000)
            for line in table:
                self.onefourier(simulator=simulator,params=line,buffer=buf,T=T,dt=dt,algorithm=algorithm,oversample=oversample)
                if status: c.status()
            if status: c.end()

            return buf
  def TDI(self,T=6.2914560e7,dt=15.0,simulator='synthlisa',table=None,algorithm='mldc',oversample=1):
          X, Y, Z = self.fourier(simulator,table,T=T,dt=dt,algorithm=algorithm,oversample=oversample)

          return X.ifft(dt), Y.ifft(dt), Z.ifft(dt)

%}
