
"""
cython wrapper for misc functions, including:
-- solid angle correction
"""

import numpy as np
cimport numpy as np

cdef extern from "solidangle.hh":
    void fastSAC(int num_pixels, double * theta, double * correction_factor)
    void rigorousGridSAC(int num_pixels_s,
                         int num_pixels_f,
                         double * s,
                         double * f,
                         double * p,
                         double * correction_factor)
    void rigorousExplicitSAC(double * pixel_xyz,
                             double * correction_factor)

                             
class SolidAngle(object):
    """
    Python interface to a solid angle correction.
    """
    
    def __init__(self, detector, use_fast_approximation=False):
        """
        Initialize a SolidAngle object.
        
        Parameters
        ----------
        detector : odin.xray.Detector
            A detector object specifying the experimental geometry.
            
        fast_approximation: bool
            Whether or not to use a fast approximate version of the correction.
        """
        
        self._use_fast_approximation = use_fast_approximation
        
        # initialize output array
        cdef double[:] correction = np.zeros(detector.num_pixels, dtype=np.float64)
        cdef double[:] theta = np.ascontiguousarray(detector.polar[:,1], dtype=np.float64)            
        
        cdef double[:] cs = np.zeros(3, dtype=np.float64)
        cdef double[:] cf = np.zeros(3, dtype=np.float64)
        cdef double[:] cp = np.zeros(3, dtype=np.float64)
        
        if use_fast_approximation:
            fastSAC(detector.num_pixels, &theta[0], &correction[0])
            
        else: # use rigorous
        
            if detector._xyz_type == 'implicit':
                bg = detector._basis_grid
                
                start = 0
                end   = 0
                for i in range(bg.num_grids):
                    
                    p, s, f, shp = bg.get_grid(i)
                    num_grid_pixels = np.product(shp)
                    
                    cs = s.astype(np.float64)
                    cf = f.astype(np.float64)
                    cp = p.astype(np.float64)
                    
                    end += np.product(num_grid_pixels)
                    
                    rigorousGridSAC(shp[0], shp[1],
                                    &cs[0], # s
                                    &cf[0], # f
                                    &cp[0], # p
                                    &correction[start])
                    
                    start = end
                    
                assert end == detector.num_pixels # sanity check
                
            elif detector._xyz_type == 'explicit':
                raise NotImplementedError('SAC for explicit detectors')
                
            else:
                raise RuntimeError('invalid detector object')
            
        self._correction = correction
        
        return
        
        
    def __call__(self, intensities):
        """
        Apply the correction to an array of intensities
        """
        if not intensities.shape == self._correction.shape:
            raise ValueError('`intensities` must be a one-D array with length '
                             'equal to the number of pixels in the detector '
                             'originally registered with the SolidAngle '
                             'instance. Passed `intensities` shape: '
                             '%s' % str(intensities.shape))
        return intensities * self._correction
    
        
    @property
    def use_fast_approximation(self):
        return self._use_fast_approximation
