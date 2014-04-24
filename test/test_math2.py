
"""
tests: src/python/math2.py
"""

from thor import math2
from thor import structure
from thor.testing import ref_file

import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_equal)
                           
from scipy import special

def test_kabsch():
    
    fn = ref_file('gold1k.coor')

    obj = structure.load_coor(fn).xyz[0,:,:]
    rot_obj = structure.rand_rotate_molecule2(obj)

    U = math2.kabsch(rot_obj, obj)
    obj2 = np.dot(rot_obj, U)
    
    assert_almost_equal(obj, obj2)
    
def test_sph_hrm():
    
    phi = np.linspace(0.01, 2*np.pi-.01, 50)
    theta = np.linspace(0.01, np.pi-.01, 50)
    
    for l in range(2,5):
        for m in range(-l, l+1):
    
            ref = special.sph_harm(m, l, phi, theta)
            Ylm = math2.sph_harm(l, m, theta, phi)
            
            ref /= ref[0]
            Ylm /= Ylm[0]
            
            # print l, m, (ref-Ylm) / ref
            
            assert_allclose(np.real(ref), np.real(Ylm), atol=1e-6, rtol=1e-4)
            assert_allclose(np.imag(ref), np.imag(Ylm), atol=1e-6, rtol=1e-4)