import numpy
from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    print 'Using Cython'
    USE_CYTHON = True
except:
    print 'Not using Cython'
    USE_CYTHON = False



VERSION = '0.1'

NUMPY_DEP = 'numpy>=1.11'

SETUP_REQUIRES = [NUMPY_DEP]


ext = '.pyx' if USE_CYTHON else '.c'

extensions=[
    Extension("pyInsp2PN_cy",
              sources=["pyInsp2PN_cy"+ext],
              include_dirs = [numpy.get_include()],
              language="c",
              extra_compile_args = ["-std=c99", "-O3"],
              libraries=["gsl", "gslcblas"]
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

short_desc = "2PN non-spinning inspiral GW waveform model"

long_desc = \
"""
    Standalone 2PN non-spinning inspiral GW waveform model
    for binary black hole coalescences.
    Same model as in Arnaud&al, An overview of the second round of the Mock LISA Data Challenges.
"""


if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name="pyInsp2PN_cy",
    version=VERSION,
    ext_modules=extensions,
    author="Sylvain Marsat",
    author_email="sylvain.marsat@aei.mpg.de",
    description=short_desc,
    long_description=long_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],
    install_requires=SETUP_REQUIRES,
    url=""
)
