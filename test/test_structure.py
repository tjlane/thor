
"""
Tests: src/python/structure.py
"""

import mdtraj
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_equal)
                           
from scipy import ndimage

from thor import structure
from thor import math2
from thor.testing import skip, ref_file


def test_pad_grid_to_square():
    
    tg = np.random.rand(5,4,3)
    padded = structure.pad_grid_to_square(tg)
    assert padded.shape == (5,5,5), padded.shape
    
    padded2 = structure.pad_grid_to_square(tg, min_pad_size=1)
    assert padded2.shape == (7,7,7), padded2.shape
    
    padded3 = structure.pad_grid_to_square(tg, min_pad_size=2)
    assert padded3.shape == (9,9,9), padded3.shape


class TestAtomicToDensity(object):

    def test_atomic_to_density_positions(self):
        # ensure the atoms are in the right spot
    
        traj = mdtraj.load(ref_file('pentagon.pdb'))
        atomic_positions = traj.xyz[0] * 10.0
        #print atomic_positions
        
        # permutation that matches the results, manually coded = :'(
        pi = np.array([4, 3, 1, 0, 2]) 
    
        for spcg in [0.25, 0.5, 1.0]:
            
            grid = structure.atomic_to_density(traj, (51,)*3, spcg)
            
            maxima = math2.find_local_maxima(grid)
            grid_maxima  = np.array(maxima).astype(np.float).T
            grid_maxima -= (np.array(grid.shape) / 2 + 1.0)[None,:] # center around origin
            grid_maxima *= spcg                                     # apply correct scale
            
            grid_maxima = grid_maxima[pi]
            
            #print grid_maxima - atomic_positions
            err = np.sum(np.abs(grid_maxima - atomic_positions)) / spcg
            assert err < 5.0, 'maxima found in estimated density map do not ' \
                              'match atomic positions in original model'
    

    def test_atomic_to_density_consistency(self):
        # Make sure changing the grid spacing parameters doesnt change the results
    
        traj = mdtraj.load(ref_file('pentagon.pdb'))
    
        tst1 = structure.atomic_to_density(traj, (25,) *3, 0.2)
        tst2 = structure.atomic_to_density(traj, (51,)*3, 0.1)
    
        R = np.corrcoef(tst1.flatten(), tst2[1::2,1::2,1::2].flatten())[0,1]
        assert R > 0.9
    
    
    