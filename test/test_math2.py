
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

def test_local_maxima():
    
    tarr = np.zeros((15,)*3)
    maxima = [ (1,1,1),
               (2,3,5),
               (7,7,7) ]
               
    for m in maxima:
        tarr[m] += 1.0
        
    found_maxima = math2.find_local_maxima(tarr)
    assert np.all( np.array(np.where(tarr)) == np.array(found_maxima) )
    

def test_kabsch():
    
    fn = ref_file('gold1k.coor')

    obj = structure.load_coor(fn).xyz[0,:,:]
    rot_obj = structure.rand_rotate_molecule(obj)

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
            
            #ref /= ref[0]
            #Ylm /= Ylm[0]
            
            # print l, m, (ref-Ylm) / ref
            
            assert_allclose(np.real(ref), np.real(Ylm), atol=1e-6, rtol=1e-4)
            assert_allclose(np.imag(ref), np.imag(Ylm), atol=1e-6, rtol=1e-4)        
            
def test_interp_grid_to_spherical():
    
    # test by generating a function on a grid that just increases (as sqr) w
    # radius, test to make sure radial interps match, theta/phi are equal
    # inside each radial band
    
    pts = 25
    
    origin = np.array( [float(pts) / 2.0]*3 )
    grid_coords = np.meshgrid( *[np.arange(pts)]*3 ) - origin[:,None,None,None]
    
    grid = np.sum( np.square(grid_coords), axis=0 )
    assert grid.shape == (pts,)*3
    
    radii = np.arange(1.0, 6.0)
    y = math2.interp_grid_to_spherical(grid, radii, 36, 36, grid_origin=origin)
    
    # test to make sure radial values are decent
    interp_radii = y.mean(axis=2).mean(axis=1)
    radii_diff = np.abs(interp_radii - np.square(radii) - 0.5) # WHY 0.5???
    #print radii_diff
    assert np.all(radii_diff < 0.1)
    
    # make sure polar values are all the same
    for i in range(len(radii)):
        vs = y[i,:,:].flatten()
        vs = vs[ np.logical_not( np.isnan(vs) ) ] # discard NaNs
        sigma = np.std(vs)
        assert sigma < 0.25, 'polar values significantly different! ' \
                             'r=%f, std: %f' % (radii[i], sigma)
    
    
    

    