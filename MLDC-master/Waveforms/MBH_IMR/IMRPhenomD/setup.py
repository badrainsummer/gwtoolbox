# coding: utf-8
#! /usr/local/bin/python3
#  Copyright (C) 2017, Michael Pürrer.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with with program; see the file COPYING. If not, write to the
#  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#  MA  02111-1307  USA
#

import numpy
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_lib
import sys

argv_replace = []
for arg in sys.argv:
    if arg.startswith('--with-gsl='):
        gsl_prefix = arg.split('=', 1)[1]
    else:
        argv_replace.append(arg)
sys.argv = argv_replace

gsl_prefix='/usr/local/include/gsl'

try:
    from Cython.Build import cythonize
    print('Using Cython')
    USE_CYTHON = True
except:
    print('Not using Cython')
    USE_CYTHON = False



VERSION = '0.1'

NUMPY_DEP = 'numpy>=1.11'

SETUP_REQUIRES = [NUMPY_DEP]

ext = '.pyx' if USE_CYTHON else '.c'


lib_gsl_dir = gsl_prefix+"/lib"
include_gsl_dir = gsl_prefix+"/include"

extensions=[
    Extension("pyIMRPhenomD",
              sources=["pyIMRPhenomD"+ext, "IMRPhenomD.c", "IMRPhenomD_internals.c"],
              include_dirs = [numpy.get_include(),include_gsl_dir],
              language="c",
              extra_compile_args = ["-std=c99", "-O3"],
              libraries=["gsl", "gslcblas"],
              library_dirs=[lib_gsl_dir],
    )
]


cls_txt = \
"""
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Cython
Programming Language :: Python
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
"""

short_desc = "IMRPhenomD inspiral-merger-ringdown GW waveform model"

long_desc = \
"""
    Standalone IMRPhenomD inspiral-merger-ringdown GW waveform model
    for binary black hole coalescences.
"""


if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name="pyIMRPhenomD",
    version=VERSION,
    ext_modules=extensions,
    author="Michael Pürrer, Sylvain Marsat",
    author_email="Michael.Puerrer@aei.mpg.de",
    description=short_desc,
    long_description=long_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],
    install_requires=SETUP_REQUIRES,
    py_modules = ['pyFDresponse'],
    url=""
)
