%module FastBinary
%{
#include "FastBinary.h"
%}

// the following three lines allow the automatic conversion of a numpy array
// into a pointer to an array of doubles and the long length of the array
// each pair of these that appears in the library header file (e.g., BBHChallenge1.hh)
// needs to be named explicitly

%include numpy_typemaps.i
%typemap(in) (double *XLS,long XLSlen) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *XSL,long XSLlen) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *YLS,long YLSlen) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *YSL,long YSLlen) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *ZLS,long ZLSlen) = (double *numpyarray,long numpyarraysize);
%typemap(in) (double *ZSL,long ZSLlen) = (double *numpyarray,long numpyarraysize);

// the following include defines all the calls that need to be wrapped by SWIG
// the include at the top serves only to provide definitions for this one;
// since it is between braces %{ %}, its content goes straight into
// the CPP wrapper file

%include "FastBinary.h"

// and here we can add some Python to the interface code

%pythoncode %{
import sys, math, time

if 'lisaxml2' in sys.modules:
	import lisaxml2 as lisaxml
else:
	import lisaxml

import numpy
import mc3lisa, FrequencyArray
from countdown import countdown

lisaxml.SourceClassModules['FastGalacticBinary'] = 'FastBinary'

# define the interface class

class FastGalacticBinary(lisaxml.Source):
    outputlist = ( ('Frequency',                  'Hertz',     None, 'GW frequency'),
                   ('FrequencyDerivative',        'Hertz/s',   0,    'GW frequency derivative'),
                   ('EclipticLatitude',           'Radian',    None, 'standard ecliptic latitude'),
                   ('EclipticLongitude',          'Radian',    None, 'standard ecliptic longitude'),
                   ('Amplitude',                  '1',         None, 'dimensionless GW amplitude'),
                   ('Inclination',                'Radian',    None, 'standard source inclination'),
                   ('Polarization',               'Radian',    None, 'standard source polarization'),
                   ('InitialPhase',               'Radian',    None, 'GW phase at t = 0') )
    
    FastBinaryCache = {}
        
    # TO DO: the initialization may be a useful extension of lisaxml.Source
    def __init__(self,name='',init=None):
        super(FastGalacticBinary, self).__init__('FastGalacticBinary',name)
        
        # try to initialize the parameters from "init"
        if isinstance(init,(tuple,list)) and len(init) == len(self.outputlist):
            for i,value in enumerate(init):
                setattr(self,self.outputlist[i][0],value)
        elif isinstance(init,dict):
            for par in self.outputlist:
                if par[0] in init:
                    setattr(self,par[0],init[par[0]])
                elif par[2] != None:
                    setattr(self,par[0],par[2])
    
    def fourier(self,simulator='synthlisa',table=None,T=6.2914560e7,dt=15,algorithm='mldc',oversample=1,kmin=0,length=None,mpi=False,status=True):
        if table == None:
            return self.onefourier(simulator=simulator,T=T,dt=dt,algorithm=algorithm,oversample=oversample)
        elif mpi == False:
            if length == None:
                length = int(T/dt)/2 + 1    # was "NFFT = int(T/dt)", and "NFFT/2+1" passed to numpy.zeros
            
            buf = tuple(FrequencyArray.FrequencyArray(numpy.zeros(length,dtype=numpy.complex128),kmin=kmin,df=1.0/T) for i in range(3))
            
            if status: c = countdown(len(table),10000)
            for line in table:
                self.onefourier(simulator=simulator,vector=line,buffer=buf,T=T,dt=dt,algorithm=algorithm,oversample=oversample)                
                if status: c.status()
            if status: c.end()
            
            return buf
        else:
            from mpi4py import MPI
            
            size, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
            slaves = size - 1
            
            send_bufs = [numpy.zeros(8,'d') for s in range(slaves)]
            send_reqs = [MPI.COMM_WORLD.Send_init((send_bufs[s], MPI.DOUBLE), s+1, 0) for s in range(slaves)]
            
            recv_bufs = [numpy.zeros(1,'d') for s in range(slaves)]
            recv_reqs = [MPI.COMM_WORLD.Recv_init((recv_bufs[s], MPI.DOUBLE), s+1, 1) for s in range(slaves)]
            
            busy = [-1 for s in range(slaves)]
            
            ln = len(table); cnt = 0; c = countdown(ln,10000)
            while cnt < ln:
                for s in range(slaves):                 
                    if busy[s] == -1 and cnt < ln:
                        send_bufs[s][:] = table[cnt][:]
                        send_reqs[s].Start()
                        recv_reqs[s].Start()
                        
                        busy[s] = cnt
                        cnt = cnt + 1
                
                if sum(x == -1 for x in busy) != len(busy):
                    f = MPI.Prequest.Waitany(recv_reqs)
                    
                    busy[f] = -1
                    c.status()
            c.end()
            
            for s in range(0,slaves):
                send_bufs[s][0] = 0.0   # terminate by sending a zero as first parameter
                send_reqs[s].Start()
            
            MPI.Prequest.Waitall(send_reqs)
            
            for r in recv_reqs: r.Free()
            for r in send_reqs: r.Free()
            
            if length == None:
                length = int(T/dt)/2 + 1
            
            buf = tuple(FrequencyArray.FrequencyArray(numpy.zeros(length,dtype=numpy.complex128),kmin=kmin,df=1.0/T) for i in range(3))
            rbuf = numpy.zeros(length,dtype=numpy.complex128)
            
            for s in range(slaves):
                for i in range(3):
                    MPI.COMM_WORLD.Recv((rbuf,MPI.DOUBLE_COMPLEX),s+1,i)
                    buf[i][:] += rbuf[:]
            
            # something strange going on with mpi4py's Reduce. Will use simple communication.
            # for i in range(3):
            #     MPI.COMM_WORLD.Reduce(None,(buf[i],MPI.DOUBLE_COMPLEX),MPI.SUM,0)
            
            return buf
    
    def slave(self,simulator='synthlisa',T=6.2914560e7,dt=15,algorithm='mldc',oversample=1,kmin=0,length=None):
        from mpi4py import MPI
        
        if length == None:
            length = int(T/dt)/2 + 1
        
        buf = tuple(FrequencyArray.FrequencyArray(numpy.zeros(length,dtype=numpy.complex128),kmin=kmin,df=1.0/T) for i in range(3))
        
        recv_buf, send_buf = numpy.zeros(8,'d'), numpy.zeros(1,'d')
        recv_req, send_req = MPI.COMM_WORLD.Recv_init((recv_buf, MPI.DOUBLE), 0, 0), MPI.COMM_WORLD.Send_init((send_buf, MPI.DOUBLE), 0, 1)
        
        while True:
            recv_req.Start()
            MPI.Prequest.Wait(recv_req)
            
            if recv_buf[0] == 0.0:
                break
            else:
                self.onefourier(simulator,vector=recv_buf,buffer=buf,T=T,dt=dt,algorithm=algorithm,oversample=oversample)
                send_req.Start()
        
        recv_req.Free(); send_req.Free()
        
        for i in range(3):
            MPI.COMM_WORLD.Send((buf[i],MPI.DOUBLE_COMPLEX),0,i)
        
        # for i in range(3):
        #     MPI.COMM_WORLD.Reduce((buf[i],MPI.DOUBLE_COMPLEX),None,MPI.SUM,0)
    
    def buffersize(self,T,f,fdot,A,algorithm='mldc',oversample=1):
        if algorithm == 'legacy' or algorithm == 'mldc':
            # bins are smaller for multiple years
            if T/year <= 1.0:
                mult = 1
            elif T/year <= 2.0:
                mult = 2
            elif T/year <= 4.0:
                mult = 4
            else:
                mult = 8
            
            # this has to do with Doppler modulation, which is of order 1e-4 * f0
            # by contrast df = pi x 10^-8 Hz
            if f > 0.1:     # 631 bins/year
                N = 1024*mult
            elif f > 0.03:  # 189 bins/year
                N = 512*mult
            elif f > 0.01:  # 63 bins/year
                N = 256*mult
            elif f > 0.001: # 3 bins/year
                N = 64*mult
            else:
                N = 32*mult
            
            # new logic for high-frequency chirping binaries (r1164 lisatools:Fast_Response.c)
            if algorithm == 'mldc':
                try:
                    chirp = int(math.pow(2.0,math.ceil(math.log(fdot*T*T)/math.log(2.0))))
                except ValueError:
                    chirp = 0
                
                if chirp > N:
                    N = 2*chirp
                elif 4*chirp > N:
                    N = N * 2
            
            # TDI noise, normalized by TDI response function
            Sm = AEnoise(f) / (4.0 * math.sin(f/cvar.fstar) * math.sin(f/cvar.fstar))
            
            # approximate SNR
            Acut = A * math.sqrt(T/Sm)
            
            # this is [2^(1+[log_2 SNR])]; remember that the response of the sinc is bound by 1/(delta bin #)
            # therefore we're using a number of bins proportional to SNR, rounded to the next power of two
            # according to my tests, this is consistent with accepting +-0.4 in the detection SNR
            M = int(math.pow(2.0,1 + int(math.log(Acut)/math.log(2.0))));
            
            # corrected per Neil's 2009-07-28 changes to Fast_Response3.c
            # use the larger of the two buffer sizes, but not above 8192
            # -- further changed for new chirping logic (r1164)
            if algorithm == 'mldc':
                M = N = max(M,N)
            else:
                M = N = min(8192,max(M,N))
        else:   # we should call this new algorithm 'lisasolve'
            # LISA response bandwidth for Doppler v/c = 1e-4 on both sides, and frequency evolution
            deltaf = fdot * T + 2.0e-4 * f
            # bins, rounded to next-highest power of two; make sure we have enough for sidebands
            bins = 8 + deltaf*T
            N = int(2**math.ceil(math.log(bins,2)))
            
            # approximate source SNR
            f0 = f + 0.5 * fdot * T
            SNR = mc3lisa.tdi.simplesnr(f0,A,years=T/year)
            
            # bandwidth for carrier frequency, off bin, worst case dx = 0.5 (accept 0.1 +- SNR)
            bins = max(1,2*2*SNR)
            M = int(2**math.ceil(math.log(bins,2)))
            
            # more aggressive: adjusted for better case dx
            # dx = abs(f0*T - math.floor(f0*T + 0.5))
            # bins = max(1,2*(4*dx)*SNR)
            # M = int(2**math.ceil(math.log(bins,2)))
            
            M = N = max(M,N)
        
        M *= oversample; N *= oversample
        
        return M,N
    
    def onefourier(self,simulator='synthlisa',vector=None,buffer=None,T=6.2914560e7,dt=15,algorithm='mldc',oversample=1):
        if vector != None:
            self.Frequency,self.FrequencyDerivative,self.Amplitude = vector[0],vector[1],vector[4]
        
        M, N = self.buffersize(T,self.Frequency,self.FrequencyDerivative,self.Amplitude,algorithm,oversample)
        
        # cache FastResponse objects
        if (N,T,dt) not in FastGalacticBinary.FastBinaryCache:
            fastbin = FastResponse(N,T,dt)
            fastbin.XLS = numpy.zeros(2*M,'d'); fastbin.XSL = numpy.zeros(2*M,'d')
            fastbin.YLS = numpy.zeros(2*M,'d'); fastbin.YSL = numpy.zeros(2*M,'d')
            fastbin.ZLS = numpy.zeros(2*M,'d'); fastbin.ZSL = numpy.zeros(2*M,'d')
            
            FastGalacticBinary.FastBinaryCache[(N,T,dt)] = fastbin
        else:
            fastbin = FastGalacticBinary.FastBinaryCache[(N,T,dt)]
        
        method = 0 if algorithm == 'legacy' else 1
        
        # TO DO: pass a vector of parameters directly
        if vector != None:
            fastbin.Response(vector[0],vector[1],0.5*math.pi - vector[2],vector[3],
                             vector[4],vector[5],vector[6],vector[7],
                             fastbin.XLS,fastbin.XSL,fastbin.YLS,fastbin.YSL,fastbin.ZLS,fastbin.ZSL,
                             method)
        else:
            fastbin.Response(self.Frequency,self.FrequencyDerivative,0.5*math.pi - self.EclipticLatitude,self.EclipticLongitude,
                             self.Amplitude,self.Inclination,self.Polarization,self.InitialPhase,
                             fastbin.XLS,fastbin.XSL,fastbin.YLS,fastbin.YSL,fastbin.ZLS,fastbin.ZSL,
                             method)
        
        f0 = self.Frequency if (algorithm == 'legacy') else (self.Frequency + 0.5 * self.FrequencyDerivative * T)
        
        if buffer == None:
            retX, retY, retZ = map(lambda a: FrequencyArray.FrequencyArray(a[::2] + 1j * a[1::2],
                                                                           dtype = numpy.complex128,            # note the first frequency bin is f0 - (M/2) df,
                                                                           kmin = int(f0*T) - M/2,df = 1.0/T),  # with df = 1/T
                                   (fastbin.XSL, fastbin.YSL, fastbin.ZSL) if (simulator == 'synthlisa') else (fastbin.XLS, fastbin.YLS, fastbin.ZLS))
        
            return (retX,retY,retZ)
        else:
            kmin, blen, alen = buffer[0].kmin, len(buffer[0]), 2*M
            
            beg, end = int(f0*T) - M/2, int(f0*T) + M/2                             # for a full buffer, "a" begins and ends at these indices
            begb, bega = (beg - kmin, 0) if beg >= kmin else (0, 2*(kmin - beg))    # left-side alignment of partial buffer with "a"
            endb, enda = (end - kmin, alen) if end - kmin <= blen else (blen, alen - 2*(end - kmin - blen))
                                                                                    # the corresponding part of "a" that should be assigned to the partial buffer
                                                                                    # ...remember "a" is doubled up
                                                                                    # check: if kmin = 0, then begb = beg, endb = end, bega = 0, enda = alen
            
            for i,a in enumerate((fastbin.XSL, fastbin.YSL, fastbin.ZSL) if (simulator == 'synthlisa') else (fastbin.XLS, fastbin.YLS, fastbin.ZLS)):
                buffer[i][begb:endb] += a[bega:enda:2] + 1j * a[(bega+1):enda:2]
    
    def TDI(self,T=6.2914560e7,dt=15.0,simulator='synthlisa',table=None,algorithm='mldc',oversample=1):
        X, Y, Z = self.fourier(simulator,table,T=T,dt=dt,algorithm=algorithm,oversample=oversample)
        
        return X.ifft(dt), Y.ifft(dt), Z.ifft(dt)
    

%}
