
"""
tests: src/python/math2.py
"""

from thor import math2
from thor import structure
from thor.testing import ref_file

import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_equal)

def test_kabsch():
    
    fn = ref_file('gold1k.coor')

    obj = structure.load_coor(fn).xyz[0,:,:]
    rot_obj = structure.rand_rotate_molecule2(obj)

    U = math2.kabsch(rot_obj, obj)
    obj2 = np.dot(rot_obj, U)
    
    assert_almost_equal(obj, obj2)