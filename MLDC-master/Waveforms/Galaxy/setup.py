#!/usr/bin/env python

modulename  = 'FastGB'
version     = '1.0'
description = 'A MLDC plugin to create Galactic-binary waveforms'
author      = 'Travis Robson, Stas Babak,  based on work by Neil Cornish, Tyson Littenberg, ...'
email       = 'mldc-all@in2p3.fr'

csourcefiles = ['GB.c','LISA.c']
sourcefiles  = csourcefiles + ['FastGB.i']
headers      = ['GB.h', 'LISA.h']


from distutils.core import setup, Extension
from distutils.command.build import build
from distutils.command.install_lib import install_lib
from distutils.spawn import spawn
from distutils.file_util import copy_file
from distutils.util import get_platform
import sys

clibrary = []
from numpy import __path__ as numpypath
numpyinclude = numpypath[0] + '/core/include'


argv_replace = []
for arg in sys.argv:
    if arg.startswith('--with-gsl='):
        gsl_prefix = arg.split('=', 1)[1]
    else:
        argv_replace.append(arg)
sys.argv = argv_replace
lib_gsl_dir = gsl_prefix+"/lib"
include_gsl_dir = gsl_prefix+"/include"


setup(name = modulename,
      version = version,
      description = description,
      author = author,
      author_email = email,

      py_modules = [modulename],

      ext_modules = [Extension('_' + modulename,
                               sourcefiles,
                               libraries=['m', 'gsl', 'gslcblas'],
                               include_dirs = [numpyinclude,include_gsl_dir],
                               library_dirs=[lib_gsl_dir],
                               depends = headers)],

      libraries = clibrary

      )
