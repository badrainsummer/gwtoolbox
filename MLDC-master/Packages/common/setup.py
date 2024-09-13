#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

import sys

install_prefix = None

for arg in sys.argv:
    if arg.startswith('--prefix='):
        install_prefix = arg.split('=', 1)[1]

# if install_prefix and sys.platform == 'darwin':
#     # this hack needed because of a bug in OS X Leopard's stock Python 2.5.1; fixed in 2.6.1...
#     install_path = get_python_lib(standard_lib=True,prefix=install_prefix) + '/site-packages'
# else:
#     install_path = get_python_lib(prefix=install_prefix)
install_path = get_python_lib(prefix=install_prefix)

# now run the setup

setup(name = 'common',
      version = '$Id: m.vallis@gmail.com $',
      description = 'lisasolve common: sundry utilities',
      author = 'M. Vallisneri',
      author_email = 'vallis@vallis.org',
      url = 'http://lisasolve.googlecode.com',
      py_modules = ['countdown','FrequencyArray','tdi', 'LISAConstants', 'LISAParameters', 'LISAhdf5', 'Cosmology','LISACommonFunctions', 'GenerateFD_SignalTDIs', 'LW_simple']
      )
