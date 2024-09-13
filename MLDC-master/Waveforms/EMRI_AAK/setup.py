import numpy as np
import glob
import sys

modulename  = 'EMRI_AAK'
version     = '1.0'
description = 'LDC plugin to create EMRI waveforms'
author      = 'Alvin Chua, M. Vallisneri, Stas Babak'
email       = 'mldc-all@in2p3.fr'

header_dir ='./include/'
source_prefix = './src/'

src = glob.glob(source_prefix+'*/*.cc')
src.append('EMRI_AAK.i')



#headers = ['AAK.h', 'CKR.h', 'GKG.h', 'GKTrajFast.h', 'IEKG.h', 'KSParMap.h', 'NRCKG.h', 'NRLB.h', 'NRUtil.h', 'SWSH.h', 'AK.h', 'Cosmology.h', 'GKInsp.h', 'GKTraj.h', 'Inspiral.h',  'KSTools.h', 'NRCosmology.h',  'NRMM.h', 'RRGW.h', 'Waveform.h', 'CKG.h', 'DopplerShiftedWaveform.h',  'GKR.h', 'Globals.h', 'Integral.h', 'LB.h', 'NRIEKG.h', 'NRRoot.h',  'SNT.h', 'WDBinary.h']
headers = glob.glob(header_dir+'*.h')

libs = ['m', 'gsl', 'gslcblas', 'fftw3']
#sourcefiles = src.tolist()
sourcefiles = src
src = np.setdiff1d(src, glob.glob(source_prefix+'exec/*.cc'))
src = np.setdiff1d(src, glob.glob(source_prefix+'*/GKR_GG*.cc'))
src = np.setdiff1d(src, glob.glob(source_prefix+'*/AK.cc'))
src = np.setdiff1d(src, glob.glob(source_prefix+'*/CKR.cc'))
sourcefiles = src.tolist()

print (sourcefiles)
print (headers)
#sys.exit(0)

from distutils.core import setup, Extension
from distutils.command.build import build
from distutils.command.install_lib import install_lib
from distutils.spawn import spawn
from distutils.file_util import copy_file
from distutils.util import get_platform

clibrary = []
from numpy import __path__ as numpypath
numpyinclude = numpypath[0] + '/core/include'

setup(name = modulename,
      version = version,
      description = description,
      author = author,
      author_email = email,

      py_modules = [modulename],

      ext_modules = [Extension('_' + modulename,
                               sources=sourcefiles,
                               libraries=['m', 'gsl', 'gslcblas', 'fftw3'],
                               include_dirs = [numpyinclude, header_dir],
                               depends = headers,
                               language="c++",
                               swig_opts = ['-c++'])]
      )
