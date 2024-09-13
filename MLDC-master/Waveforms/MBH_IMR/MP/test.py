# coding: utf-8

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

# A py.test compliant unit test

#!/usr/bin/env python

"""
Designed to be run-able with py.test
MP 06/2017
"""

import pytest
import numpy as np
from pyIMRPhenomD import IMRPhenomD, MSUN_SI, PC_SI

def test_IMRPhenomD_against_stored_data():
    phi0 = 0.0
    fRef = 100.0
    deltaF = 1.0/32.0
    m1_SI = 50*MSUN_SI
    m2_SI = 30*MSUN_SI
    chi1 = 0.85
    chi2 = 0.42
    f_min = 10.0
    f_max = 0.0
    distance = 500.0e6 * PC_SI
    inclination = 0.5

    # First create a waveform with the standalone Cython-wrapped code
    wf = IMRPhenomD(phi0, fRef, deltaF, m1_SI, m2_SI, chi1, chi2, f_min, f_max, distance, inclination)
    fHz_Cy, hptilde_Cy, hctilde_Cy = wf.GetWaveform()

    # Load pregenerated lalsimulation test data
    # Produced by example_lalsimulation.py
    fHz_LAL, hptilde_LAL, hctilde_LAL = np.load('test_data/test_lalsimulation.npy')

    # Compare the FD polarizations
    assert np.array_equal(fHz_Cy, fHz_LAL)
    assert np.array_equal(hptilde_Cy, hptilde_LAL)
    assert np.array_equal(hctilde_Cy, hctilde_LAL)

def test_IMRPhenomD_against_lalsimulation_on_the_fly():
    try:
        from lal import MSUN_SI, PC_SI, CreateDict
        import lalsimulation as LS
    except Exception as e:
        print '****************************************************************'
        print '*** Could not import LAL'
        print '*** Error %s' % e
        print '****************************************************************'


    phi0 = 0.0
    fRef = 100.0
    deltaF = 1.0/128.0
    m1_SI = 10*MSUN_SI
    m2_SI = 30*MSUN_SI
    chi1 = 0.85
    chi2 = 0.42
    f_min = 10.0
    f_max = 0.0
    distance = 500.0e6 * PC_SI
    inclination = 0.5

    # First create a waveform with the standalone Cython-wrapped code
    wf = IMRPhenomD(phi0, fRef, deltaF, m1_SI, m2_SI, chi1, chi2, f_min, f_max, distance, inclination)
    fHz_Cy, hptilde_Cy, hctilde_Cy = wf.GetWaveform()

    # Now create a waveform with SWIG-wrapped lalsimulation
    LALpars = CreateDict()
    longAscNodes, eccentricity, meanPerAno = 0, 0, 0
    S1x, S1y, S1z = 0, 0, chi1
    S2x, S2y, S2z = 0, 0, chi2

    hptildeFS, hctildeFS = LS.SimInspiralChooseFDWaveform(m1_SI, m2_SI,
        S1x, S1y, S1z, S2x, S2y, S2z,
        distance, inclination, phi0, longAscNodes, eccentricity, meanPerAno,
        deltaF, f_min, f_max, fRef,
        LALpars, LS.IMRPhenomD)

    fHz_LAL = np.arange(hptildeFS.data.length)*deltaF
    hptilde_LAL = hptildeFS.data.data
    hctilde_LAL = hctildeFS.data.data

    # Compare the FD polarizations
    assert np.array_equal(fHz_Cy, fHz_LAL)
    assert np.array_equal(hptilde_Cy, hptilde_LAL)
    assert np.array_equal(hctilde_Cy, hctilde_LAL)
