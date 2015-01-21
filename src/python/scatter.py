
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

from thor import _cppscatter
from thor.math2 import arctan3, sph_harm
from thor.refdata import cromer_mann_params
from thor.refdata import get_cromermann_parameters
from thor.refdata import sph_quad_900



def _qxyz_from_detector(detector):
    """
    Attempt to interpret many possible detectory geometries & return a
    2d array of the {xyz} coordinates for each q-vector on the detector
    (in units of inverse angstroms).
    """

    # get the scattering vectors
    if str(type(detector)).find('Detector') > -1:    
        qxyz = detector.reciprocal
        assert detector.num_pixels == qxyz.shape[0]
    elif isinstance(detector, np.ndarray):
        qxyz = detector
    else:
        raise ValueError('`detector` must be {thor.xray.Detector, np.ndarray}')

    return qxyz


def _sample_finite_photon_statistics(intensities, poisson_parameter):
    
    if not poisson_parameter > 0.0:
        raise ValueError('`poisson_parameter` <= 0 (got: %f)' % poisson_parameter)
        
    n = np.random.poisson(poisson_parameter) # total scattered photons
    p = intensities / intensities.sum()      # prob of going to each pixel
    if not np.all( intensities > 0.0 ):
        raise ValueError('negative intensities found')
    
    photons = np.random.multinomial(n, p)
    assert np.all(photons >= 0), 'negative sample from np.random.multinomial'
    
    return photons
    
    
class _NonRandomState(np.random.RandomState):
    """
    A mimic of np.random.RandomState that returns zeros, so that molecules
    wont rotate
    """
        
    def rand(self, *args):
        return np.zeros(args)
    

def simulate_atomic(traj, num_molecules, detector, traj_weights=None,
                    finite_photon=False, ignore_hydrogens=False, 
                    dont_rotate=False, procs_per_node=1, 
                    nodes=[], devices=[], random_state=None):
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
        
    ignore_hydrogens : bool
        Ignore hydrogens in calculation. Hydrogens are often costly to compute
        and don't change the scattering appreciably.
        
    dont_rotate : bool
        Don't apply a random rotation to the molecule before computing the
        scattering. Good for debugging and the like.
        
    Returns
    -------
    amplitudes : ndarray, complex128
        A flat array of the simulated amplitudes, each position corresponding
        to a scattering vector from `qxyz`.
        
    See Also
    --------
    thor.xray.Shotset.simulate(), thor.xray.Rings.simulate()
        These are factory functions that call this function, and wrap the
        results into the Shotset class.
    """
    
    
    logger.debug('Performing scattering simulation...')
    logger.debug('Simulating %d copies in the dilute limit' % num_molecules)
    
    
    if dont_rotate:
        random_state = _NonRandomState()
    
    
    # sampling statistics for the trajectory
    if traj_weights == None:
        traj_weights = np.ones( traj.n_frames )
    traj_weights /= traj_weights.sum()    
    num_per_shapshot = np.random.multinomial(num_molecules, traj_weights)
    
    
    # extract the atomic numbers & CM parameters
    atomic_numbers = np.array([ a.element.atomic_number for a in traj.topology.atoms ])
    cromermann_parameters, atom_types = get_cromermann_parameters(atomic_numbers)
    
    
    # decide what the poisson parameter is
    if finite_photon is True:
        try:
            finite_photon = float(detector.beam.photons_scattered_per_shot)
        except:
            raise RuntimeError('`detector` object must have a beam attribute if'
                               ' finite photon statistics are to be computed')
        logger.debug('Finite photon stats from detector')
    elif type(finite_photon) == float:
        logger.debug('Finite photon param passed as float: %f' % finite_photon)
    elif finite_photon in [None, False, 0]:
        finite_photon = False
    else:
        raise TypeError('`finite_photon` must be of type {True, False, float}, '
                        'got: %s' % type(finite_photon))
    
    
    # if we're getting a speedup by ignoring hydrogens, find em and slice em
    n_atoms = len(atomic_numbers)
    if ignore_hydrogens == True:
        atoms_to_keep = (atomic_numbers != 1)
        atomic_numbers = atomic_numbers[atoms_to_keep]
        n_H = n_atoms - np.sum(atoms_to_keep)
        logger.debug('Ignoring %d hydrogens (of %d atoms)' % (n_H, n_atoms))
    else:
        atoms_to_keep = np.ones(n_atoms, dtype=np.bool)
        
    
    qxyz = _qxyz_from_detector(detector)
    amplitudes = np.zeros(qxyz.shape[0], dtype=np.complex128)
    
    for i,num in enumerate(num_per_shapshot):
        
        logger.debug('Running %d molc, snapshot %d' % (num, i))
        rxyz = traj.xyz[i,atoms_to_keep,:] * 10.0 # convert nm --> Angstroms
        
        amplitudes += _cppscatter.parallel_cpp_scatter(num, rxyz, qxyz,
                                                       atom_types,
                                                       cromermann_parameters,
                                                       procs_per_node=procs_per_node,
                                                       nodes=nodes,
                                                       devices=devices,
                                                       random_state=random_state)
                                                       
    if finite_photon is not False:
        intensities = np.square(np.abs(amplitudes))
        r = _sample_finite_photon_statistics(intensities, finite_photon)
    else:
        r = amplitudes
    
    return r

    
def simulate_density(grid, grid_spacing, num_molecules, detector,
                     finite_photon=False, dont_rotate=False,
                     reshape_output=False, procs_per_node=1, nodes=[], 
                     devices=[], random_state=None):
    """
    
    Optional Parameters
    -------------------
    reshape_output : bool
        If `True`, put the output back in square grid form, such that it is the
        same shape as `grid`. Can only be used if the detector is the same
        number of pixels as the `grid` has points. Useful for comparing results
        to straight up 3D FFTs (testing). Default: False.
    
    Returns
    -------
    amplitudes : ndarray, complex128
        A flat array of the simulated amplitudes, each position corresponding
        to a scattering vector from `qxyz`.
    """
    
    if dont_rotate:
        random_state = _NonRandomState()
    
    if len(grid.shape) != 3:
        raise ValueError('`grid` must be a square 3d grid. Got a %dd grid.'
                         '' % len(grid.shape))
    
    # the below operations is equiv to flattening x/y/z individually
    gs = grid.shape
    rxyz = np.mgrid[:gs[0],:gs[1],:gs[2]].reshape(3, -1).T * grid_spacing
    
    qxyz = _qxyz_from_detector(detector)
    
    # the 9th Cromer-Mann parameter is a constant -- for a point density
    # scalar field, set this to be the density value
    atom_types = np.arange( np.product(gs) )
    cromermann_parameters = np.zeros(np.product(gs) * 9)
    cromermann_parameters[8::9] = grid.flatten()
    
    amplitudes = _cppscatter.parallel_cpp_scatter(num_molecules,
                                                  rxyz, qxyz,
                                                  atom_types,
                                                  cromermann_parameters,
                                                  procs_per_node=procs_per_node,
                                                  nodes=nodes,
                                                  devices=devices,
                                                  random_state=random_state)
    
    # put the output back in sq grid form if requested
    if reshape_output:
        amplitudes = amplitudes.T.reshape(*gs)
    
    return amplitudes
    
        
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
        
    fi = np.ones_like(q_mag) * cromermann[8]
    for i in range(4):
        fi += cromermann[i] * np.exp( - cromermann[i+4] * qo)
        
    return fi
    
    
def atomic_electrondens(atomic_Z, r_mag, radial_cutoff=None):
    """
    Evaluate the contribution of the electron density due to a particular atom
    at distance `r_mag`. This function employs an isotropic sum-of-Gaussians
    model, specifically the inverse Fourier transform of the Cromer-Mann
    atomic form factors.
    
    Parameters
    ----------
    atomic_Z : int
        The atomic number of the atom to compute the density for.
    
    r_mag : float
        The distance from the atomic center to the point at which the
        density is being evaluated

    Returns
    -------
    fi : float
        The real part of the atomic form factor.

    See Also
    --------
    atomic_formfactor : function
        A direct interface to the Cromer-Mann form factors

    atomic_to_density : function
        Employs this function to construct a model of the electron density on
        a grid.

    Notes
    -----
    The Cromer-Mann model for the atomic form factors reads

         f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]
                     i=1,4

    This function employs the numerical inverse FT of a Gaussian to compute
    the atomic contribution to the electron density. Let x = |r - r_0|,

        phi[x] = [SUM a_i * {4pi/b_i}^(3/2) * EXP{- (4pi^2 / b_i) * x^2}]
                 i=1,4
                 
    References
    ----------
    ..[1] Afonine and Urzhumtsev, Acta Cryst (2004) A60 19-32.
    """

    # OLD --->
    # xo = np.power(4.0 * r_mag, 2)
    # cromermann = cromer_mann_params[(atomic_Z,0)]
    # 
    # # retain constant term (?)
    # phi = cromermann[8] * np.ones_like(r_mag) # retain
    # #phi = np.zeros_like(r_mag)               # discard
    # 
    # for i in range(4):
    #     phi = 4.0 * cromermann[i] * np.sqrt(np.power(np.pi,3) / cromermann[i+4]) *\
    #           np.exp( - xo / cromermann[i+4])
    # END OLD <---
    
    # NEW code is based on Formulae given in [1], differs in some coefficients
    # I am currently IGNORING the constant term. It is usually pretty small...

    phi = np.zeros_like(r_mag)
    
    if radial_cutoff is not None:
        inds = (r_mag < radial_cutoff)
        if np.sum(inds) == 0:
            raise ValueError('No grid points contained in radial cutoff!s')
    else:
        inds = np.ones(r_mag.shape, dtype=np.bool)
    
    xo = 4.0 * (np.pi ** 2) * np.power(r_mag[inds], 2)
    cromermann = cromer_mann_params[(atomic_Z,0)]
    
    for i in range(4):
        phi[inds] = cromermann[i] * \
                    np.power(4.0 * np.pi / cromermann[i+4], 1.5) * \
                    np.exp( - xo / cromermann[i+4] )

    return phi

    
def sph_harm_coefficients(trajectory, q_values, weights=None,
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
    spatial correlation of scattering fluctuations. Macromolecules 10, 927-934 
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
    # from thor.refdata import sph_quad_900
    q_phi   = arctan3(sph_quad_900[:,1], sph_quad_900[:,0])
    q_theta = np.arccos(sph_quad_900[:,2])
        
    # iterate over all snapshots in the trajectory
    for i in range(trajectory.n_frames):

        for iq,q in enumerate(q_values):
            logger.info('Computing coefficients for q=%f\t(%d/%d)' % (q, iq+1, num_q_mags))
            
            # compute S, the single molecule scattering intensity
            A_q = simulate_atomic(trajectory[i], 1, q * sph_quad_900[:,:3])
            S_q = np.square(np.abs(A_q))

            # project S onto the spherical harmonics using spherical quadrature
            for il,l in enumerate(l_vals):                
                for m in range(0, l+1):
                    
                    logger.debug('Projecting onto Ylm, l=%d/m=%d' % (l, m))

                    # compute a spherical harmonic, turns out this is 
                    # unexpectedly annoying...
                    # -----------
                    # option (1) : scipy (slow & incorrect?)
                    # scipy switched the convention for theta/phi in this fxn
                    #Ylm = special.sph_harm(m, l, q_phi, q_theta)
                    
                    # -----------
                    # option (2) : roll your own
                    Ylm = sph_harm(l, m, q_theta, q_phi)
                    # -----------
                    
                    # NOTE: we're going to use the fact that negative array
                    #       indices wrap around here -- the value of m can be
                    #       negative, but those values just end up at the *end*
                    #       of the array 
                    r = np.sum( S_q * Ylm * sph_quad_900[:,3] )
                    Slm[il,  m, iq] = r
                    Slm[il, -m, iq] = ((-1) ** m) * np.conjugate(r)

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

        
        
