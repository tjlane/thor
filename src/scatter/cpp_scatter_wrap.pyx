
"""
Cython wrapper for C++/CUDA implementation of scattering simulations.
"""

import numpy as np
cimport numpy as np

import os
import re
import subprocess
from threading import Thread
from time import time

try:
    from mpi4py import MPI
    MPI_ENABLED = True
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except ImportError as e:
    MPI_ENABLED = False


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
    
    void gpuscatter(int   device_id,

                    int   n_q,
                    float * q_x, 
                    float * q_y, 
                    float * q_z, 

                    int   n_atoms,
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

    void cpuscatter(int   n_q,
                    float * q_x, 
                    float * q_y, 
                    float * q_z, 

                    int   n_atoms,
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

    void cpudiffuse(int   n_q,
                    float * q_x, 
                    float * q_y, 
                    float * q_z, 

                    int   n_atoms,
                    float * r_x, 
                    float * r_y, 
                    float * r_z,

                    int   n_atom_types,
                    int   * atom_types,
                    float * cromermann,

                    float * V,

                    float * q_out_bragg,
                    float * q_out_diffuse ) except +                    

    void gpudiffuse(int device_id,

                    int   n_q,
                    float * q_x,
                    float * q_y,
                    float * q_z,

                    int   n_atoms,
                    float * r_x,
                    float * r_y,
                    float * r_z,

                    int   n_atom_types,
                    int   * atom_types,
                    float * cromermann,

                    float * V,

                    float * q_out_bragg,
                    float * q_out_diffuse ) except +



def _detect_gpus():
    try:
        gpu_list = subprocess.check_output(['nvidia-smi', '-L']).strip().split('\n')
    except OSError as e:
        gpu_list = []

    devices = {}
    for gpu in gpu_list:

        s = re.match('GPU (\d+): (.+) \(UUID: (\S+)\)', gpu)
        if s is None:
            raise RuntimeError('regex error parsing: %s' % gpu)

        # key : (name, UUID)
        devices[ int(s.group(1)) ] = (s.group(2), s.group(3))

    return devices


def _evenly_distribute_jobs(n_jobs, n_workers):
    
    assert n_jobs > 0, '`n_jobs` must be >0'
    assert n_workers > 0, '`n_workers` must be >0'
    
    nj = n_jobs
    nt = n_workers
    jobs_per_worker = [((nj / nt) + int((nj%nt - n) > 0)) for n in range(nt)]
    
    assert len(jobs_per_worker) == n_workers, 'num workers off'
    assert np.sum(jobs_per_worker) == n_jobs, 'num jobs off'

    return jobs_per_worker

    
# TODO:
#   -- make this MPI aware

def parallel_cpp_scatter(n_molecules, 
                         rxyz,
                         qxyz,
                         atom_types,
                         cromermann_parameters,
                         procs_per_node=1,
                         nodes=[],
                         devices=[],
                         random_state=None,
                         ignore_gpu_check=False):
    """
    Multi-threaded interface to `cpp_scatter`. Specify the devices the
    simulation should run on and the code will split the simulation between
    them in an efficient manner.

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
        
    Returns
    -------
    amplitudes : ndarray, complex128
        A flat array of the simulated amplitudes, each position corresponding
        to a scattering vector from `qxyz`.
    """

    # check devices
    gpus_available = _detect_gpus()
    for d in devices:
        if (d not in gpus_available.keys()) and (not ignore_gpu_check):
            print 'Device requested: %d' % d
            print 'Available:', gpus_available
            raise RuntimeError('Requested GPU Device %d not found. Ensure'
                               ' this GPU is online and visible to the os using'
                               ' nvidia-smi. Overwrite this error using the'
                               '`ignore_gpu_check` flag')

    

    # divide work between CPUs & GPUs
    # TODO -- right now use as much GPU as possible
    if len(devices) > 0:
        num_per_gpu = _evenly_distribute_jobs(n_molecules, len(devices))
        num_per_cpu = [0,] * procs_per_node
    else:
        num_per_gpu = []
        num_per_cpu = _evenly_distribute_jobs(n_molecules, procs_per_node)


    # multiprocessing cannot return values, so generate a helper function
    # that will dump returned values into a shared array
    
    amplitudes = np.zeros(qxyz.shape[0], dtype=np.complex128)
    threads = []
    
    def t_fxn(*fargs):
        a = cpp_scatter(*fargs)
        amplitudes[:] += a

    # run dat shit
    for cpu_thread in range(procs_per_node):
        num = num_per_cpu[cpu_thread]
        if num == 0:
            continue
        print('CPU Thread %d :: %d shots' % (cpu_thread, num))
        cpu_args = (num, rxyz, qxyz, atom_types, cromermann_parameters, 
                    'CPU', random_state)
        t_cpu = Thread(target=t_fxn, args=cpu_args)
        t_cpu.start()
        threads.append(t_cpu)                

    for gpu_device in range(len(devices)):
        num = num_per_gpu[gpu_device]
        if num == 0:
            continue
        print('GPU Device %d :: %d shots' % (gpu_device, num))
        gpu_args = (num, rxyz, qxyz, atom_types, cromermann_parameters, 
                    gpu_device, random_state)
        t_gpu = Thread(target=t_fxn, args=gpu_args)
        t_gpu.start()
        threads.append(t_gpu)

    # ensure child processes have finished
    for t in threads:
        t.join()

    return amplitudes

         
def cpp_scatter(n_molecules, 
                np.ndarray rxyz,
                np.ndarray qxyz,
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
        Seed the random state. For testing only.
    
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
                         
    n_molecules = int(n_molecules)
    if n_molecules <= 0:
        raise ValueError('`num` must be 1 or greater, got: %d' % n_molecules)
        
    # for s in [rxyz.shape, qxyz.shape]:
    #     if (not len(s) == 2) or (s[1] == 3):
    #         raise ValueError('`rxyz` and `qxyz` must be (N, 3) shape arrays')
    
    
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
        s = (int(time() * 1e6) + os.getpid()) % 4294967294
        random_state = np.random.RandomState(s)
    
    if not isinstance(random_state, np.random.RandomState):
        raise TypeError('`random_state` must be instance of np.random.RandomState')
        
    cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rfloats
    c_rfloats = np.ascontiguousarray( random_state.rand(3, n_molecules), 
                                      dtype=np.float32 )

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
    
        if device_id == 'GPU':
            device_id = 0 # default GPU
        
        gpuscatter(device_id,
                   qxyz.shape[0], &c_qxyz[0,0], &c_qxyz[1,0], &c_qxyz[2,0],
                   rxyz.shape[0], &c_rxyz[0,0], &c_rxyz[1,0], &c_rxyz[2,0], 
                   num_atom_types, &c_atom_types[0], &c_cromermann[0],
                   n_molecules, &c_rfloats[0,0], &c_rfloats[1,0], &c_rfloats[2,0],
                   &real_amplitudes[0], &imag_amplitudes[0])
        
    else:
        raise ValueError('`device_id` must be one of {CPU, GPU, int}, got: '
                         '%s' % str(device_id))
                                   
    # deal with the output
    output_sanity_check(real_amplitudes)
    output_sanity_check(imag_amplitudes)
    
    #amplitudes = np.empty(qxyz.shape[0], dtype=np.complex128)
    amplitudes = real_amplitudes + 1j * imag_amplitudes
    
    return amplitudes

    
def cpp_scatter_diffuse(np.ndarray rxyz,
                        np.ndarray qxyz,
                        np.ndarray atom_types,
                        np.ndarray cromermann_parameters,
                        np.ndarray V,
                        device_id='CPU'):
        """
        A python interface to the C++ code implementing the MVN diffuse scatter
        model.
    
        Parameters
        ----------
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
            
        V : ndarray, float
            A 4-d array of shape (n_atoms, n_atoms, 3, 3) representing the
            anisotropic Gaussian correlation between atoms. The value
            V[i,j,a,b] is the correlation between atom i in the a direction with
            j in the b direction, where a & b are one of {x/y/z}.
    
        Returns
        -------
        intensities : ndarray, float64
            A flat array of the simulated intensities, each position corresponding
            to a scattering vector from `qxyz`.
        """
    
        # check input sanity
        num_atom_types = len(np.unique(atom_types))
        if not len(cromermann_parameters) == num_atom_types * 9:
            raise ValueError('input array `cromermann_parameters` should be len '
                             '9 * num of unique `atom_types`')
                             
        num_atoms = rxyz.shape[0]
        if not (V.shape[0] == num_atoms) and (V.shape[1] == num_atoms):
            raise ValueError('shape of V incorrect, first 2 dims should be '
                             'number of atoms -- got shape:')
        if not (V.shape[2] == 3) and (V.shape[3] == 3):
            raise ValueError('shape of V incorrect, last 2 dims should be '
                             'size 3 -- got shape:')
    
    
        # A NOTE ABOUT ARRAY ORDERING
        # In what follows, for multi-dimensional arrays I often take the transpose
        # somewhat mysteriously. This is because the way C++ will loop over arrays
        # in c-order, but Thor's arrays in python land are in "fortran" order
    
        # extract arrays from input  
        cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_qxyz
        cdef np.ndarray[ndim=2, dtype=np.float32_t, mode="c"] c_rxyz
        cdef np.ndarray[ndim=1, dtype=np.float32_t] c_cromermann
        cdef np.ndarray[ndim=1, dtype=np.float32_t, mode="c"] c_V
    
        c_qxyz = np.ascontiguousarray(qxyz.T, dtype=np.float32)
        c_rxyz = np.ascontiguousarray(rxyz.T, dtype=np.float32)
        c_cromermann = np.ascontiguousarray(cromermann_parameters, dtype=np.float32)
        c_V = np.ascontiguousarray(V.flatten(), dtype=np.float32)
    
        # for some reason "memoryview" syntax necessary for int arrays...
        cdef int[::1] c_atom_types = np.ascontiguousarray(atom_types, dtype=np.int32)

        # initialize output arrays
        cdef np.ndarray[ndim=1, dtype=np.float32_t] bragg_intensities
        cdef np.ndarray[ndim=1, dtype=np.float32_t] diffuse_intensities
        bragg_intensities = np.zeros(qxyz.shape[0], dtype=np.float32)
        diffuse_intensities = np.zeros(qxyz.shape[0], dtype=np.float32)
    
    
        # --- call the actual C++ code
        if device_id == 'GPU':
            device_id = 0 # default GPU

        if device_id == 'CPU':
            cpudiffuse(qxyz.shape[0], &c_qxyz[0,0], &c_qxyz[1,0], &c_qxyz[2,0],
                       rxyz.shape[0], &c_rxyz[0,0], &c_rxyz[1,0], &c_rxyz[2,0], 
                       num_atom_types, &c_atom_types[0], &c_cromermann[0], &c_V[0],
                       &bragg_intensities[0], &diffuse_intensities[0])
        elif type(device_id) is int:
            gpudiffuse(device_id,
                       qxyz.shape[0], &c_qxyz[0,0], &c_qxyz[1,0], &c_qxyz[2,0],
                       rxyz.shape[0], &c_rxyz[0,0], &c_rxyz[1,0], &c_rxyz[2,0],
                       num_atom_types, &c_atom_types[0], &c_cromermann[0], &c_V[0],
                       &bragg_intensities[0], &diffuse_intensities[0])
        else:
            raise ValueError('uninterpretable device_id: %s' % str(device_id))
                                   
        # deal with the output
        output_sanity_check(bragg_intensities)
        output_sanity_check(diffuse_intensities)
    
        return bragg_intensities, diffuse_intensities
        
        

# FIND OUT IF A GPU IS AVAILABLE
global GPU_ENABLED
if len(_detect_gpus()) > 0:
    GPU_ENABLED = True
else:
    GPU_ENABLED = False

