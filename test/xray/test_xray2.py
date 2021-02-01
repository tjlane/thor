
"""
Some extras on test_xray. Split up so Travis can parallelize them.
"""

import os
import sys
import warnings
import tables
from nose import SkipTest

import numpy as np
from scipy import stats

from thor import utils
from thor import math2
from thor import utils
from thor import xray
from thor import parse
from thor import structure
from thor.testing import (skip, ref_file, expected_failure, 
                          brute_force_masked_correlation)
from thor.refdata import cromer_mann_params

from mdtraj import Trajectory, io

from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_equal)

import test_xray


class TestShotsetFromDisk(test_xray.TestShotset):
    """
    Test all the same shotset functionality, but working from the intensities
    on disk via pytables. Works by subclassing TestShotset but overwriting the
    setup() method.
    """
    
    def setup(self):
        
        self.q_values = np.array([1.0, 2.0])
        self.num_phi  = 360
        self.l = 50.0
        self.d = xray.Detector.generic(spacing=0.4, l=self.l)
        self.t = Trajectory.load(ref_file('ala2.pdb'))
        
        self.num_shots = 2
        intensities = np.abs(np.random.randn(self.num_shots, self.d.num_pixels))
        io.saveh('tmp_tables.h5', data=intensities)
        
        self.tables_file = tables.File('tmp_tables.h5')
        self.i = self.tables_file.root.data
        
        self.shot = xray.Shotset(self.i, self.d)
        
        return
        
    def test_append(self):
        # we have to make the shotset use an EArray type instead of a CArray
        if os.path.exists('tmp.shot'): os.remove('tmp.shot')
        self.shot.save('tmp.shot')
        self.shot = xray.Shotset.load('tmp.shot')
        super(TestShotsetFromDisk, self).test_append()
        if os.path.exists('tmp.shot'): os.remove('tmp.shot')
        
    def test_average_intensity(self):
        assert_array_almost_equal(self.i.read().mean(0), self.shot.average_intensity)
        
    def teardown(self):
        self.tables_file.close()
        os.remove('tmp_tables.h5')
        return

class TestRingsFromDisk(test_xray.TestRings):
    """
    Test the rings class when the handle is a tables array.
    """

    def setup(self):

        self.q_values  = np.array([1.0, 2.0])
        self.num_phi   = 360
        self.traj      = Trajectory.load(ref_file('ala2.pdb'))
        self.num_shots = 3

        # generate the tables file on disk, then re-open it
        intensities = np.abs( np.random.randn(self.num_shots, len(self.q_values),
                                              self.num_phi) / 100.0 + \
                              np.cos( np.linspace(0.0, 4.0*np.pi, self.num_phi) ) )

        if os.path.exists('tmp_tables.h5'):
            os.remove('tmp_tables.h5')
            
        hdf = tables.File('tmp_tables.h5', 'w')
        a = tables.Atom.from_dtype(np.dtype(np.float64))
        node = hdf.create_earray(where='/', name='data',
                                shape=(0, len(self.q_values), self.num_phi), 
                                atom=a, filters=io.COMPRESSION)
        node.append(intensities)
        hdf.close()

        self.tables_file = tables.File('tmp_tables.h5', 'r+')
        pi = self.tables_file.root.data
        pm = np.random.binomial(1, 0.9, size=(len(self.q_values), self.num_phi))
        k = 1.0
        
        self.rings = xray.Rings(self.q_values, pi, k, pm)

        return
        
    def test_conversion_to_nparray(self):
        pitx = self.rings.polar_intensities
        for i,ref_i in enumerate(self.rings.polar_intensities_iter):
            assert_allclose(ref_i, pitx[i])
        
    def test_polar_intensities_type(self):
        assert self.rings._polar_intensities_type == 'tables'
       
    @skip # SKIPPING -- UNKNOWN ISSUES
    def test_correlate_intra(self):
        # because we have noise in these sims, the error tol needs to be higher
        # this is uncomfortably high right now, but better to have a basic
        # sanity check than no test at all...
        super(TestRingsFromDisk, self).test_correlate_intra(rtol=0.2, atol=0.1)
        
    def test_correlate_inter(self):
        # see comment above
        super(TestRingsFromDisk, self).test_correlate_inter(rtol=0.2, atol=0.1)
        
    def test_correlate_inter_mean_only(self):
        # see comment above
        super(TestRingsFromDisk, self).test_correlate_inter_mean_only(rtol=0.2, atol=0.1)

    def teardown(self):
        self.tables_file.close()
        if os.path.exists('tmp_tables.h5'):
            os.remove('tmp_tables.h5')
        return
    
        
class TestRingsFFTPack(test_xray.TestRings):
    """
    Test the rings class when pyfftw is not available
    """

    def setup(self):
        xray.FORCE_NO_FFTW = True
        super(TestRingsFFTPack, self).setup()
        return
        
    def teardow(self):
        xray.FORCE_NO_FFTW = False
        return
