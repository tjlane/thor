
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
    
    def __init__(self, detector, fast_approximation=False):
        """
        Initialize a SolidAngle object.
        
        Parameters
        ----------
        detector : odin.xray.Detector
            A detector object specifying the experimental geometry.
            
        fast_approximation: bool
            Whether or not to use a fast approximate version of the correction.
        """
        
        
        self._fast_approximation = fast_approximation
        
        if self.detector._xyz_type == 'implicit':
            pass
        elif self.detector._xyz_type == 'explicit':
            pass
        
        
        return
        
    @property
    def fast_approximation(self):
        return self._fast_approximation
