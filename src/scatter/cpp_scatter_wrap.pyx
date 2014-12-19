
"""
Cython wrapper for C++ based scattering.
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
    
                    
cdef extern from "cpp_scatter.hh":
    void CPUScatter(int    nQ,
                    float* h_qx,
                    float* h_qy,
                    float* h_qz,
                    int    nAtoms,
                    float* h_rx,
                    float* h_ry,
                    float* h_rz,
                    float* h_ff,
                    int    nRot,
                    float* h_rand1,
                    float* h_rand2,
                    float* h_rand3,
                    float* h_outQ ) except +


         
def cpp_scatter(n_molecules, 
                np.ndarray qxyz,
                np.ndarray rxyz,
                np.ndarray atomic_formfactors,
                device_id='CPU',
                random_state=np.random.RandomState()):
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
    
    atomic_formfactors : ndarray, float
        A 1d array of the atomic form factors of each atom (same len as `rxyz`).
    
    Optional Parameters
    -------------------
    device_id : str or int
        Either 'CPU' or the GPU device ID (int) specifying the compute hardware.
        
    random_state : np.random.RandomState
    
    Returns
    -------
    self.intensities : ndarray, float
        A flat array of the simulated intensities, each position corresponding
        to a scattering vector from `qxyz`.
    """
    
    # A NOTE ABOUT ARRAY ORDERING
    # In what follows, for multi-dimensional arrays I often take the transpose
    # somewhat mysteriously. This is because the way C++ will loop over arrays
    # in c-order, but Thor's arrays in python land are in "fortran" order
    
    # extract arrays from input  
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_qxyz
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rxyz
    cdef np.ndarray[ndim=1, dtype=np.float32_t] c_formfactors
    
    c_qxyz = np.ascontiguousarray(qxyz.T, dtype=np.float32)
    c_rxyz = np.ascontiguousarray(rxyz.T, dtype=np.float32)
    c_formfactors = np.ascontiguousarray(atomic_formfactors, dtype=np.float32)
    
    # generate random numbers
    if not isinstance(random_state, np.random.RandomState):
        raise TypeError('`random_state` must be instance of np.random.RandomState')
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rfloats
    c_rfloats = np.ascontiguousarray( random_state.rand(3, n_molecules), dtype=np.float32)

    # initialize output array
    cdef np.ndarray[ndim=1, dtype=np.float32_t] h_outQ
    h_outQ = np.zeros(qxyz.shape[0], dtype=np.float32)
    
    
    # --- call the actual C++ code
    
    if device_id == 'CPU':
    
        CPUScatter(qxyz.shape[0], &c_qxyz[0,0], &c_qxyz[1,0], &c_qxyz[2,0],
                     rxyz.shape[0], &c_rxyz[0,0], &c_rxyz[1,0], &c_rxyz[2,0], 
                     &c_formfactors[0],
                     n_molecules, &c_rfloats[0,0], &c_rfloats[1,0], &c_rfloats[2,0],
                     &h_outQ[0])
    
    elif (type(device_id) == int) or (device_id == 'GPU'): # run on GPU

        raise NotImplementedError()
        
        # if device_id == 'GPU':
        #     device_id = 0
        # if not type(device_id) == int:
        #     raise TypeError('`device_id` must be type int')
        # 
        # if not n_molecules % 512 == 0:
        #     raise ValueError('`n_rotations` must be a multiple of 512')
        # bpg = int(n_molecules) / 512 # blocks-per-grid
        # 
        # if bpg <= 0:
        #     print "Error, bpg = %d" % bpg 
        #     raise RuntimeError('bpg <= 0; invalid number of molecules passed: %d' % n_molecules)
        # 
        # # call the actual C++ code
        # gpu_scatter_obj = new C_GPUScatter(device_id, bpg, qxyz.shape[0],
        #                            &c_qxyz[0,0], &c_qxyz[1,0], &c_qxyz[2,0],
        #                            rxyz.shape[0], &c_rxyz[0,0], &c_rxyz[1,0], &c_rxyz[2,0], 
        #                            &c_aid[0], len(c_cromermann), &c_cromermann[0],
        #                            n_molecules, &c_rfloats[0,0], &c_rfloats[1,0], &c_rfloats[2,0],
        #                            &h_outQ[0])
        # del gpu_scatter_obj
        
    else:
        raise ValueError('`device_id` must be one of {CPU, GPU, int}, got: '
                         '%s' % str(device_id))
                                   
    # deal with the output
    output_sanity_check(h_outQ)
    
    return h_outQ

