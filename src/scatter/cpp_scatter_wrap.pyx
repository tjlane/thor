
"""
Cython wrapper for C++ based scattering.
"""

import numpy as np
cimport numpy as np

from time import time
import os

def output_sanity_check(output):
    """
    Perform a basic sanity check on the intensity array we're about to return.
    """
    
    # check for NaNs in output
    if np.isnan(np.sum(output)):
        raise RuntimeError('Fatal error, NaNs detected in scattering output!')
        
    if np.any(np.isinf(output)):
        raise RuntimeError('Fatal error, Infs detected in scattering output!')
        
    return
    
                    
cdef extern from "cpp_scatter.hh":
    void cpuscatter(int  n_q,
                    float * q_x, 
                    float * q_y, 
                    float * q_z, 

                    int    n_atoms,
                    float * r_x, 
                    float * r_y, 
                    float * r_z,

                    int   n_atom_types,
                    int   * atom_types,
                    float * cromermann,

                    int   n_rotations,
                    float * randN1, 
                    float * randN2, 
                    float * randN3,

                    float * q_out_real,
                    float * q_out_imag ) except +


         
def cpp_scatter(n_molecules, 
                np.ndarray qxyz,
                np.ndarray rxyz,
                np.ndarray atom_types,
                np.ndarray cromermann_parameters,
                device_id='CPU',
                random_state=None):
    """
    A python interface to the C++ and CUDA scattering code. The idea here is to
    mirror that interface closely, but in a pythonic fashion.
    
    Parameters
    ----------
    n_molecules : int
        The number of molecules to include in the simulation.
    
    qxyz : ndarray, float
        An n x 3 array of the (x,y,z) positions of each q-vector describing
        the detector.
    
    rxyz : ndarray, float
        An m x 3 array of the (x,y,z) positions of each atom in the molecule.
    
    atom_types : ndarray, int
        A length-m one dim array of the ID of each atom telling the code which
        Cromer-Mann parameter to use. See below.
        
    cromermann_parameters : ndarray, float
        A one-d array of length 9 * the number of unique `atom_types`,
        specifying the 9 Cromer-Mann parameters for each atom type.
    
    
    Optional Parameters
    -------------------
    device_id : str or int
        Either 'CPU' or the GPU device ID (int) specifying the compute hardware.
        
    random_state : np.random.RandomState
    
    Returns
    -------
    amplitudes : ndarray, complex128
        A flat array of the simulated amplitudes, each position corresponding
        to a scattering vector from `qxyz`.
    """
    
    # check input sanity
    num_atom_types = len(np.unique(atom_types))
    if not len(cromermann_parameters) == num_atom_types * 9:
        raise ValueError('input array `cromermann_parameters` should be len '
                         '9 * num of unique `atom_types`')
    
    # A NOTE ABOUT ARRAY ORDERING
    # In what follows, for multi-dimensional arrays I often take the transpose
    # somewhat mysteriously. This is because the way C++ will loop over arrays
    # in c-order, but Thor's arrays in python land are in "fortran" order
    
    # extract arrays from input  
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_qxyz
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rxyz
    cdef np.ndarray[ndim=1, dtype=np.float32_t] c_cromermann
    
    c_qxyz = np.ascontiguousarray(qxyz.T, dtype=np.float32)
    c_rxyz = np.ascontiguousarray(rxyz.T, dtype=np.float32)
    c_cromermann = np.ascontiguousarray(cromermann_parameters, dtype=np.float32)
    
    # for some reason "memoryview" syntax necessary for int arrays...
    cdef int[::1] c_atom_types = np.ascontiguousarray(atom_types, dtype=np.int32)
    
    # generate random numbers
    if random_state is None:
        s = int(time.time() * 1e6) + os.getpid()
        random_state = np.random.RandomState(s)
    
    if not isinstance(random_state, np.random.RandomState):
        raise TypeError('`random_state` must be instance of np.random.RandomState')
        
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rfloats
    
    # TEMPORARY FOR DEBUGGING
    # c_rfloats = np.ascontiguousarray( random_state.rand(3, n_molecules), 
    #                                   dtype=np.float32 )
    c_rfloats = np.zeros((3, n_molecules), dtype=np.float32)

    # initialize output arrays
    cdef np.ndarray[ndim=1, dtype=np.float32_t] real_amplitudes
    cdef np.ndarray[ndim=1, dtype=np.float32_t] imag_amplitudes
    real_amplitudes = np.zeros(qxyz.shape[0], dtype=np.float32)
    imag_amplitudes = np.zeros(qxyz.shape[0], dtype=np.float32)
    
    
    # --- call the actual C++ code
    
    if device_id == 'CPU':
        cpuscatter(qxyz.shape[0], &c_qxyz[0,0], &c_qxyz[1,0], &c_qxyz[2,0],
                   rxyz.shape[0], &c_rxyz[0,0], &c_rxyz[1,0], &c_rxyz[2,0], 
                   num_atom_types, &c_atom_types[0], &c_cromermann[0],
                   n_molecules, &c_rfloats[0,0], &c_rfloats[1,0], &c_rfloats[2,0],
                   &real_amplitudes[0], &imag_amplitudes[0])
    
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
    output_sanity_check(real_amplitudes)
    output_sanity_check(imag_amplitudes)
    
    #amplitudes = np.empty(qxyz.shape[0], dtype=np.complex128)
    amplitudes = real_amplitudes + 1j * imag_amplitudes
    
    return amplitudes
    

