
"""
Library for performing simulations of x-ray scattering experiments.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

import numpy as np
from scipy import misc
from scipy import special
from threading import Thread

from thor import _cpuscatter
from thor.math2 import arctan3
from thor.refdata import cromer_mann_params

try:
    from thor import _gpuscatter
    GPU = True
except ImportError as e:
    GPU = False


def simulate_shot(traj, num_molecules, detector, traj_weights=None,
                  finite_photon=False, force_no_gpu=False, device_id=0):
    """
    Simulate a scattering 'shot', i.e. one exposure of x-rays to a sample.
    
    Assumes we have a Boltzmann distribution of `num_molecules` identical  
    molecules (`trajectory`), exposed to a beam defined by `beam` and projected
    onto `detector`.
    
    Each conformation is randomly rotated before the scattering simulation is
    performed. Atomic form factors from X, finite photon statistics, and the 
    dilute-sample (no scattering interference from adjacent molecules) 
    approximation are employed.
    
    Parameters
    ----------
    traj : mdtraj.trajectory
        A trajectory object that contains a set of structures, representing
        the Boltzmann ensemble of the sample. If len(traj) == 1, then we assume
        the sample consists of a single homogenous structure, replecated 
        `num_molecules` times.
        
    detector : thor.xray.Detector OR ndarray, float
        A detector object the shot will be projected onto. Can alternatively be
        just an n x 3 array of q-vectors to project onto.
        
    num_molecules : int
        The number of molecules estimated to be in the `beam`'s focus.
        
    Optional Parameters
    -------------------
    traj_weights : ndarray, float
        If `traj` contains many structures, an array that provides the Boltzmann
        weight of each structure. Default: if traj_weights == None, weights
        each structure equally.
        
    finite_photon : bool OR float
        Whether or not to employ finite photon statistics in the simulation. If
        this is "False", run a continuous simulation (infinite photons). If
        "True", use the finite photon parameters from a xray.Detector object. If
        a float, then that float specifies the mean photons scattered per shot.
        
    force_no_gpu : bool
        Run the (slow) CPU version of this function.
        
    device_id : int
        The index of the GPU device to run on.
        
    Returns
    -------
    intensities : ndarray, float
        An array of the intensities at each pixel of the detector.
        
    See Also
    --------
    thor.xray.Shotset.simulate()
        These are factory functions that call this function, and wrap the
        results into the Shotset class.
    """
    
    
    logger.debug('Performing scattering simulation...')
    logger.debug('Simulating %d copies in the dilute limit' % num_molecules)

    if traj_weights == None:
        traj_weights = np.ones( traj.n_frames )
    traj_weights /= traj_weights.sum()
        
    num_per_shapshot = np.random.multinomial(num_molecules, traj_weights)
    
        
    # get the scattering vectors
    if str(type(detector)).find('Detector') > -1:    
        qxyz = detector.reciprocal
        assert detector.num_pixels == qxyz.shape[0]
    elif isinstance(detector, np.ndarray):
        qxyz = detector
    else:
        raise ValueError('`detector` must be {thor.xray.Detector, np.ndarray}')
    num_q = qxyz.shape[0]
    
    
    # figure out finite photon statistics
    if finite_photon in [None, False]:
        poisson_parameter = 0.0 # flag to downstream code to not use stats
    elif finite_photon == True:
        try:
            poisson_parameter = float(detector.beam.photons_scattered_per_shot)
        except:
            raise RuntimeError('`detector` object must have a beam attribute if'
                               ' finite photon statistics are to be computed')
    elif type(finite_photon) == float:
        poisson_parameter = finite_photon
    else:
        raise TypeError('Finite photon must be one of {True, False, float},'
                        ' got: %s' % str(finite_photon))
        
    
    # extract the atomic numbers
    atomic_numbers = np.array([ a.element.atomic_number for a in traj.topology.atoms ])
    
    
    # iterate over snapshots in the trajectory
    intensities = np.zeros(num_q)
    for i,num in enumerate(num_per_shapshot):
        num = int(num)
        if num > 0: # else, we can just skip...
        
            # pull xyz coords
            rxyz = traj.xyz[i,:,:] * 10.0 # convert nm -> ang.

            # choose the number of molecules & divide work between CPU & GPU
            # GPU is fast but can only do multiples of 512 molecules - run
            # the remainder on the CPU
            if force_no_gpu or (not GPU):
                num_cpu = num
                num_gpu = 0
                bpg = 0
                logger.debug('Forced "no GPU": running CPU-only computation')
            else:
                num_cpu = num % 512
                num_gpu = num - num_cpu
            
            logger.info('Running %d molc, snapshot %d, dev %d: %d CPU / %d GPU.' % (num, i, device_id, num_cpu, num_gpu))  

            # multiprocessing cannot return values, so generate a helper function
            # that will dump returned values into a shared array
            threads = []
            def multi_helper(compute_device, fargs):
                """ a helper function that performs either CPU or GPU calcs """
                if compute_device == 'cpu':
                    func = _cpuscatter.simulate
                elif compute_device == 'gpu':
                    func = _gpuscatter.simulate
                else:
                    raise ValueError('`compute_device` should be one of {"cpu",\
                     "gpu"}, was: %s' % compute_device)
                intensities[:] += func(*fargs)
                return

            # run dat shit
            if num_cpu > 0:
                logger.debug('Running CPU scattering code (%d/%d)...' % (num_cpu, num))
                cpu_args = (num_cpu, qxyz, rxyz, atomic_numbers, None)
                t_cpu = Thread(target=multi_helper, args=('cpu', cpu_args))
                t_cpu.start()
                threads.append(t_cpu)                

            if num_gpu > 0:
                logger.debug('Sending calc to GPU dev: %d' % device_id)
                gpu_args = (num_gpu, qxyz, rxyz, atomic_numbers, device_id, None)
                t_gpu = Thread(target=multi_helper, args=('gpu', gpu_args))
                t_gpu.start()
                threads.append(t_gpu)
                
            # ensure child processes have finished
            for t in threads:
                t.join()
        
        
    # if we're using finite photons, sample those stats
    if poisson_parameter > 0.0:
        n = np.random.poisson(poisson_parameter)
        p = intensities / intensities.sum()
        intensities = np.random.multinomial(n, p)
        
        
    return intensities
        
        
def atomic_formfactor(atomic_Z, q_mag):
    """
    Compute the (real part of the) atomic form factor.
    
    Parameters
    ----------
    atomic_Z : int
        The atomic number of the atom to compute the form factor for.
        
    q_mag : float
        The magnitude of the q-vector at which to evaluate the form factor.
        
    Returns
    -------
    fi : float
        The real part of the atomic form factor.
    """
        
    qo = np.power( q_mag / (4. * np.pi), 2)
    cromermann = cromer_mann_params[(atomic_Z,0)]
        
    fi = cromermann[8]
    for i in range(4):
        fi += cromermann[i] * np.exp( - cromermann[i+4] * qo)
        
    return fi


def sph_hrm_coefficients(trajectory, q_values, weights=None,
                         num_coefficients=10):
    """
    Numerically evaluates the coefficients of the projection of a structure's
    fourier transform onto the three-dimensional spherical harmonic basis. Can
    be used to directly compare a proposed structure to the same coefficients
    computed from correlations in experimentally observed scattering profiles.
    
    Parameters
    ----------
    trajectory : mdtraj.trajectory
        A trajectory object representing a Boltzmann ensemble.
        
    q_values : ndarray, float
        A list of the reciprocal space magnitudes at which to evaluate 
        coefficients.
        
    weights : ndarray, float
        A list of weights, for how to weight each snapshot in the trajectory.
        If not provided, treats each snapshot with equal weight.

    num_coefficients : int
        The order at which to truncate the spherical harmonic expansion
    
    Returns
    -------
    sph_coefficients : ndarray, float
        A 3-dimensional array of coefficients. The first dimension indexes the
        order of the spherical harmonic. The second two dimensions index the
        array `q_values`.
        
    References
    ----------
    .[1] Kam, Z. Determination of macromolecular structure in solution by 
    spatial correlation of scattering fluctuations. Macromolecules 10, 927–934 
    (1977).
    """
    
    logger.debug('Projecting image into spherical harmonic basis...')
    
    # first, deal with weights
    if weights == None:
        weights = np.ones(trajectory.n_frames)
    else:
        if not len(weights) == trajectory.n_frames:
            raise ValueError('length of `weights` array must be the same as the'
                             'number of snapshots in `trajectory`')
        weights /= weights.sum()
    
    # initialize the q_values array
    q_values = np.array(q_values).flatten()
    num_q_mags = len(q_values)
    
    # don't do odd values of ell
    l_vals = range(0, 2*num_coefficients, 2)
    
    # initialize spherical harmonic coefficient array
    # note that it's 4* num_coeff - 3 b/c we're skipping odd l's -- (2l+1)
    Slm = np.zeros(( num_coefficients, 4*num_coefficients-3, num_q_mags), 
                     dtype=np.complex128 )
    
    # get the quadrature vectors we'll use, a 900 x 4 array : [q_x, q_y, q_z, w]
    from thor.refdata import sph_quad_900
    q_phi = arctan3(sph_quad_900[:,1], sph_quad_900[:,0])
        
    # iterate over all snapshots in the trajectory
    for i in range(trajectory.n_frames):

        for iq,q in enumerate(q_values):
            logger.info('Computing coefficients for q=%f\t(%d/%d)' % (q, iq+1, num_q_mags))
            
            # compute S, the single molecule scattering intensity
            S_q = simulate_shot(trajectory[i], 1, q * sph_quad_900[:,:3],
                                force_no_gpu=True)

            # project S onto the spherical harmonics using spherical quadrature
            for il,l in enumerate(l_vals):                
                for m in range(-l, l+1):
                    
                    N = np.sqrt( 2. * l * misc.factorial(l-m) / \
                                ( 4. * np.pi * misc.factorial(l+m) ) )
                    Plm = special.lpmv(m, l, sph_quad_900[:,2])
                    Ylm = N * np.exp( 1j * m * q_phi ) * Plm

                    # NOTE: we're going to use the fact that negative array
                    #       indices wrap around here -- the value of m can be
                    #       negative, but those values just end up at the *end*
                    #       of the array 
                    Slm[il, m, iq] = np.sum( S_q * Ylm * sph_quad_900[:,3] )

        # now, reduce the Slm solution to C_l(q1, q2)
        sph_coefficients = np.zeros((num_coefficients, num_q_mags, num_q_mags))
        for iq1, q1 in enumerate(q_values):
            for iq2, q2 in enumerate(q_values):
                for il, l in enumerate(l_vals):
                    ip = np.sum( Slm[il,:,iq1] * np.conjugate(Slm[il,:,iq2]) )
                    if not np.imag(ip) < 1e-6: 
                        logger.warning('C_l coefficient has non-zero imaginary'
                                       ' component (%f) -- this is theoretically'
                                       ' forbidden and usually a sign something '
                                       'went wrong numerically' % np.imag(ip))
                    sph_coefficients[il, iq1, iq2] += weights[i] * np.real(ip)
    
    return sph_coefficients

        
        
