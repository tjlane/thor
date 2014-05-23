
"""
Cython wrapper for GPU scattering.
"""

import numpy as np
cimport numpy as np

from time import time
import os

from thor.refdata import get_cromermann_parameters

    
def output_sanity_check(intensities):
    """
    Perform a basic sanity check on the intensity array we're about to return.
    """

    # check for NaNs in output
    if np.isnan(np.sum(intensities)):
        raise RuntimeError('Fatal error, NaNs detected in scattering output!')

    # check for negative values in output
    if len(intensities[intensities < 0.0]) != 0:
        raise RuntimeError('Fatal error, negative intensities detected in scattering output!')

    return


cdef extern from "gpuscatter.hh":
    cdef cppclass C_GPUScatter "GPUScatter":
        C_GPUScatter(int    device_id_,
                     int    bpg_,
                     int    nQ_,
                     float* h_qx_,
                     float* h_qy_,
                     float* h_qz_,
                     int    nAtoms_,
                     float* h_rx_,
                     float* h_ry_,
                     float* h_rz_,
                     int*   h_id_,
                     int    nCM_,
                     float* h_cm_,
                     int    nRot_,
                     float* h_rand1_,
                     float* h_rand2_,
                     float* h_rand3_,
                     float* h_outQ_ ) except +


cdef C_GPUScatter * gpu_scatter_obj


def simulate(n_molecules, qxyz, rxyz, atomic_numbers, device_id=0, rfloats=None):
    """
    Parameters
    ----------
    n_molecules : int
        The number of molecules to include in the simulation.

    qxyz : ndarray, float
        An n x 3 array of the (x,y,z) positions of each q-vector describing
        the detector.

    rxyz : ndarray, float
        An m x 3 array of the (x,y,z) positions of each atom in the molecule.

    atomic_numbers : ndarray, int
        A 1d array of the atomic numbers of each atom (same len as `rxyz`).
        
    Optional Parameters
    -------------------
    device_id : int
        Which GPU device to run on.
        
    rfloats : ndarray, float
        An `n_molecules` x 3 array of random floats uniform on [0,1]. If passed,
        these are used to randomly rotate the molecules. If not, new rands
        are generated. This is for debugging only.

    Returns
    -------
    self.intensities : ndarray, float
        A flat array of the simulated intensities, each position corresponding
        to a scattering vector from `qxyz`.
    """
    
    if not type(device_id) == int:
        raise TypeError('`device_id` must be type int')
        
    if not n_molecules % 32 == 0:
        raise ValueError('`n_rotations` must be a multiple of 32')
    bpg = int(n_molecules) / 32 # blocks-per-grid

    if bpg <= 0:
        print "Error, bpg = %d" % bpg 
        raise RuntimeError('bpg <= 0; invalid number of molecules passed: %d' % n_molecules)
    
    # A NOTE ABOUT ARRAY ORDERING
    # In what follows, for multi-dimensional arrays I often take the transpose
    # somewhat mysteriously. This is because the way C++ will loop over arrays
    # in c-order, but ODIN's arrays in python land are in "fortran" order
    
    # extract arrays from input  
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_qxyz
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rxyz
    c_qxyz = np.ascontiguousarray(qxyz.T, dtype=np.float32)
    c_rxyz = np.ascontiguousarray(rxyz.T, dtype=np.float32)
    
    
    # generate random numbers
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rfloats
    if rfloats == None:
        np.random.seed( int(time() + os.getpid()) )
        c_rfloats = np.ascontiguousarray( np.random.rand(3, n_molecules), dtype=np.float32)
    else:
        c_rfloats = np.ascontiguousarray(rfloats.T, dtype=np.float32)
        print "WARNING: employing fed random numbers -- this should be a test"
    

    # get the Cromer-Mann parameters
    py_cromermann, py_aid = get_cromermann_parameters(atomic_numbers)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] c_cromermann
    c_cromermann = np.ascontiguousarray(py_cromermann, dtype=np.float32)
    
    cdef int[::1] c_aid = np.ascontiguousarray(py_aid, dtype=np.int32) # memory-view contiguous "C" array
    
    
    # initialize output array
    cdef np.ndarray[ndim=1, dtype=np.float32_t] h_outQ
    h_outQ = np.zeros(qxyz.shape[0], dtype=np.float32)
    
    
    # call the actual C++ code
    gpu_scatter_obj = new C_GPUScatter(device_id, bpg, qxyz.shape[0],
                               &c_qxyz[0,0], &c_qxyz[1,0], &c_qxyz[2,0],
                               rxyz.shape[0], &c_rxyz[0,0], &c_rxyz[1,0], &c_rxyz[2,0], 
                               &c_aid[0], len(c_cromermann), &c_cromermann[0],
                               n_molecules, &c_rfloats[0,0], &c_rfloats[1,0], &c_rfloats[2,0],
                               &h_outQ[0])
    del gpu_scatter_obj
                                   
    # deal with the output
    output_sanity_check(h_outQ)
    
    return h_outQ

