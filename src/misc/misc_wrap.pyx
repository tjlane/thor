
"""
cython wrapper for misc functions, including:
-- solid angle correction
"""

import numpy as np
cimport numpy as np

cdef extern from "solidangle.hh":
    void fastSAC(int num_pixels, float * theta, float * correction_factor)
    void rigorousGridSAC(int num_pixels_s,
                         int num_pixels_f,
                         float * s,
                         float * f,
                         float * p,
                         float * correction_factor)
    void rigorousExplicitSAC(float * pixel_xyz,
                             float * correction_factor)

                             
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
        cdef float[::1] correction = np.zeros(detector.num_pixels, dtype=np.float64)
        
        if use_fast_approximation:
            cdef float[::1] theta = np.ascontiguousarray(detector.polar[:,1], dtype=np.float64)            
            fastSAC(detector.num_pixels, &theta[0], &correction[0])
            
        else: # use rigorous
        
            if self.detector._xyz_type == 'implicit':
                bg = detector._basis_grid
                
                start = 0
                for i in range(bg.num_grids):
                    
                    p, s, f, shp = bg.get_grid(i)
                    num_grid_pixels = np.product(shp)
                    cdef float[::1] grid_correction = np.zeros(num_grid_pixels, dtype=np.float64)
                    end += np.product(num_grid_pixels)
                    
                    rigorousGridSAC(shp[0], shp[1],
                                    &s[0], &f[0], &p[0],
                                    &grid_correction[0])
                    
                    correction[start:end] = grid_correction
                    
                    start = end
                    
                assert end == detector.num_pixels # sanity check
                
            elif self.detector._xyz_type == 'explicit':
                raise NotImplementedError('SAC for explicit detectors')
                
            else:
                raise RuntimeError('invalid detector object')
            
        self._correction = correction
        
        return
        
        
    def __call__(self, intensities):
        """
        Apply the correction to an array of intensities
        """
        if not intensities.shape == self._correlation.shape:
            raise ValueError('`intensities` must be a one-D array with length '
                             'equal to the number of pixels in the detector '
                             'originally registered with the SolidAngle '
                             'instance. Passed `intensities` shape: '
                             '%s' % str(intensities.shape))
        return intensities * self._correction
    
        
    @property
    def use_fast_approximation(self):
        return self._use_fast_approximation
