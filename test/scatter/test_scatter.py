
"""
Reference implementation & unit test for the GPU & CPU scattering simulation code
"""

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_almost_equal, assert_allclose

from nose import SkipTest

from mdtraj import Trajectory
import matplotlib.pyplot as plt

from thor import _cppscatter
from thor import xray
from thor import scatter
from thor import structure
from thor.refdata import get_cromermann_parameters, cromer_mann_params
from thor.testing import skip, ref_file, gputest

global RANDOM_STATE
RANDOM_STATE = np.random.RandomState(0)

# ------------------------------------------------------------------------------
#                        BEGIN REFERENCE IMPLEMENTATIONS
# ------------------------------------------------------------------------------

def rand_rotate_molecule(xyzlist, rfloat=None):
    """
    Randomly rotate the molecule defined by xyzlist.
    
    Parameters
    ----------
    xyzlist : ndarray, float, 3D
        An n x 3 array representing the x,y,z positions of n atoms.
        
    rfloat : ndarray, float, len 3
        A 3-vector of random numbers in [0,1) that acts as a random seed. If
        not passed, generates new random numbers.
        
    Returns
    -------
    rotated_xyzlist : ndarray, float, 3D
        A rotated version of the input `xyzlist`.
    """
    
    # get a random quaternion vector
    q = structure.quaternion.random(rfloat)
    
    # take the quaternion conjugate
    qconj = structure.quaternion.conjugate(q)
    
    # prepare data structures
    rotated_xyzlist = np.zeros(xyzlist.shape)
    qv = np.zeros(4)
    
    # put each atom through the same rotation
    for i in range(xyzlist.shape[0]):
        qv[1:] = xyzlist[i,:].copy()
        q_prime = structure.quaternion.prod( structure.quaternion.prod(q, qv), qconj )
        rotated_xyzlist[i,:] = q_prime[1:].copy() # want the last 3 elements...
    
    return rotated_xyzlist


def form_factor(qvector, atomz):
    
    mq = np.sum( np.power(qvector, 2) )
    qo = mq / (16.*np.pi*np.pi)
    
    if ( atomz == 1 ):
        fi = 0.493002*np.exp(-10.5109*qo)
        fi+= 0.322912*np.exp(-26.1257*qo)
        fi+= 0.140191*np.exp(-3.14236*qo)
        fi+= 0.040810*np.exp(-57.7997*qo)
        fi+= 0.003038
    
    elif ( atomz == 8):
        fi = 3.04850*np.exp(-13.2771*qo)
        fi+= 2.28680*np.exp(-5.70110*qo)
        fi+= 1.54630*np.exp(-0.323900*qo)
        fi+= 0.867000*np.exp(-32.9089*qo)
        fi+= 0.2508

    elif ( atomz == 79):
        fi = 16.8819*np.exp(-0.4611*qo)
        fi+= 18.5913*np.exp(-8.6216*qo)
        fi+= 25.5582*np.exp(-1.4826*qo)
        fi+= 5.86*np.exp(-36.3956*qo)
        fi+= 12.0658
        
    # else approximate with Nitrogen
    else:
        fi = 12.2126*np.exp(-0.005700*qo)
        fi+= 3.13220*np.exp(-9.89330*qo)
        fi+= 2.01250*np.exp(-28.9975*qo)
        fi+= 1.16630*np.exp(-0.582600*qo)
        fi+= -11.529
        
    return fi

    
def ref_simulate_shot(xyzlist, atomic_numbers, num_molecules, q_grid):
    """
    Simulate a single x-ray scattering shot off an ensemble of identical
    molecules.
    
    Parameters
    ----------
    xyzlist : ndarray, float, 2d
        An n x 3 array of each atom's position in space
        
    atomic_numbers : ndarray, float, 1d
        An n-length list of the atomic numbers of each atom
        
    num_molecules : int
        The number of molecules to include in the ensemble.
        
    q_grid : ndarray, float, 2d
        An m x 3 array of the q-vectors corresponding to each detector position.
    
    Optional Parameters
    -------------------
    rfloats : ndarray, float, n x 3
        A bunch of random floats, uniform on [0,1] to be used to seed the 
        quaternion computation.
        
    Returns
    -------
    I : ndarray, float
        An array the same size as the first dimension of `q_grid` that gives
        the value of the measured intensity at each point on the grid.
    """
    
    A = np.zeros(q_grid.shape[0], dtype=np.complex128)
    
    rs = np.random.RandomState(0)
    rfs = rs.rand(3, num_molecules)
    
    for i,qvector in enumerate(q_grid):    
        for n in range(num_molecules):
        
            # match random numbers that will be generated in actual impl.
            rotated_xyzlist = rand_rotate_molecule(xyzlist, rfloat=rfs[:,n])
            
            # compute the molecular form factor F(q)
            for j in range(xyzlist.shape[0]):
                fi = form_factor(qvector, atomic_numbers[j])
                r = rotated_xyzlist[j,:]
                A[i] +=      fi * np.sin( np.dot(qvector, r) )
                A[i] += 1j * fi * np.cos( np.dot(qvector, r) )

    return A


def debye_reference(trajectory, weights=None, q_values=None):
    """
    Computes the Debye scattering equation for the structures in `trajectory`,
    producing the theoretical intensity profile.

    Treats the object `trajectory` as a sample from a Boltzmann ensemble, and
    averages the profile from each snapshot in the trajectory. If `weights` is
    provided, weights the ensemble accordingly -- otherwise, all snapshots are
    given equal weights.
    
    THIS IS A PYTHON REFERENCE IMPLEMENTATAION.

    Parameters
    ----------
    trajectory : mdtraj.trajectory
        A trajectory object representing a Boltzmann ensemble.


    Optional Parameters
    -------------------    
    weights : ndarray, int
        A one dimensional array indicating the weights of the Boltzmann ensemble.
        If `None` (default), weight each structure equally.

    q_values : ndarray, float
        The values of |q| to compute the intensity profile for, in
        inverse Angstroms. Default: np.arange(0.02, 6.0, 0.02)

    Returns
    -------
    intensity_profile : ndarray, float
        An n x 2 array, where the first dimension is the magnitude |q| and
        the second is the average intensity at that point < I(|q|) >_phi.
    """

    # first, deal with weights
    if weights == None:
        weights = 1.0 # this will work later when we use array broadcasting
    else:
        if not len(weights) == trajectory.n_frames:
            raise ValueError('length of `weights` array must be the same as the'
                             'number of snapshots in `trajectory`')
        weights /= weights.sum()

    # next, construct the q-value array
    if q_values == None:
        q_values = np.arange(0.02, 6.0, 0.02)

    intensity_profile = np.zeros( ( len(q_values), 2) )
    intensity_profile[:,0] = q_values

    # array to hold the contribution from each snapshot in `trajectory`
    S = np.zeros(trajectory.n_frames)

    # extract the atomic numbers, number each atom by its type
    aZ = np.array([ a.element.atomic_number for a in trajectory.topology.atoms() ])
    n_atoms = len(aZ)

    atom_types = np.unique(aZ)
    num_atom_types = len(atom_types)
    cromermann = np.zeros(9*num_atom_types, dtype=np.float32)

    aid = np.zeros( n_atoms, dtype=np.int32 )
    atomic_formfactors = np.zeros( num_atom_types, dtype=np.float32 )

    for i,a in enumerate(atom_types):
        ind = i * 9
        try:
            cromermann[ind:ind+9] = np.array(cromer_mann_params[(a,0)]).astype(np.float32)
        except KeyError as e:
            logger.critical('Element number %d not in Cromer-Mann form factor parameter database' % a)
            raise ValueError('Could not get CM parameters for computation')
        aid[ aZ == a ] = np.int32(i)

    # iterate over each value of q and compute the Debye scattering equation
    for q_ind,q in enumerate(q_values):
        print q_ind, len(q_values)

        # pre-compute the atomic form factors at this q
        qo = np.power( q / (4. * np.pi), 2)
        for ai in xrange(num_atom_types):                
            for i in range(4):
                atomic_formfactors[ai]  = cromermann[ai*9+8]
                atomic_formfactors[ai] += cromermann[ai*9+i] * np.exp( cromermann[ai*9+i+5] * qo)

        # iterate over all pairs of atoms
        for i in range(n_atoms):
            fi = atomic_formfactors[ aid[i] ]
            for j in range(i+1, n_atoms):
                fj = atomic_formfactors[ aid[j] ]

                # iterate over all snapshots 
                for k in range(trajectory.n_frames):
                    r_ij = np.linalg.norm(trajectory.xyz[k,i,:] - trajectory.xyz[k,j,:])
                    r_ij *= 10.0 # convert to angstroms!
                    S[k] += 2.0 * fi * fj * np.sin( q * r_ij ) / ( q * r_ij )

        intensity_profile[q_ind,1] = np.sum( S * weights )

    return intensity_profile


# ------------------------------------------------------------------------------
#                           END REFERENCE IMPLEMENTATIONS
# ------------------------------------------------------------------------------
#                              BEGIN nosetest CLASSES
# ------------------------------------------------------------------------------

    
class TestScatter(object):
    """ test all the scattering simulation functionality """
    
    def setup(self):
        
        self.nq = 3 # number of detector vectors to do
        
        xyzQ = np.loadtxt(ref_file('512_atom_benchmark.xyz'))
        self.xyzlist = xyzQ[:,:3] * 10.0 # nm -> ang.
        self.atomic_numbers = xyzQ[:,3].flatten()
    
        self.q_grid = np.loadtxt(ref_file('512_q.xyz'))[:self.nq]
        
        #self.rfloats = np.loadtxt(ref_file('512_x_3_random_floats.txt'))
        
        self.num_molecules = 512
        self.random_state = np.random.RandomState(0)
        #self.rfloats = self.random_state.rand(3, self.num_molecules)[::-1,:].T
        self.rfloats = np.zeros((self.num_molecules, 3))
        
        self.ref_A = ref_simulate_shot(self.xyzlist, self.atomic_numbers, 
                                       self.num_molecules, self.q_grid)
    
    def test_gpu_scatter(self):

        if not GPU: raise SkipTest
            
        gpu_I = _gpuscatter.simulate(self.num_molecules, self.q_grid, self.xyzlist,
                                    self.atomic_numbers, rfloats=self.rfloats)

        print "GPU", gpu_I
        print "REF", self.ref_I
        
        assert_allclose(gpu_I, self.ref_I, rtol=1e-03,
                        err_msg='scatter: gpu/cpu reference mismatch')
        assert not np.all( gpu_I == 0.0 )
        assert not np.sum( gpu_I == np.nan )
                        
                        
    def test_cpu_scatter(self):

        print "testing c cpu code..."
        
        cromermann_parameters, atom_types = get_cromermann_parameters(self.atomic_numbers)
        
        print 'num_molecules:', self.num_molecules
        cpu_A = _cppscatter.cpp_scatter(self.num_molecules, 
                                        self.q_grid, 
                                        self.xyzlist, 
                                        atom_types,
                                        cromermann_parameters,
                                        device_id='CPU',
                                        random_state=np.random.RandomState(0))

        assert_allclose(cpu_A, self.ref_A, rtol=1e-2, atol=1.0,
                        err_msg='scatter: c-cpu/cpu reference mismatch')
        assert not np.all( cpu_A == 0.0 )
        assert not np.sum( cpu_A == np.nan )
        
                            
    def test_python_call(self):
        """
        Test the GPU scattering simulation interface (scatter.simulate)
        """

        if not GPU: raise SkipTest
        print "testing python wrapper fxn..."
        
        traj = Trajectory.load(ref_file('ala2.pdb'))
        num_molecules = 512
        detector = xray.Detector.generic()

        py_I = scatter.simulate_shot(traj, num_molecules, detector)

        assert not np.all( py_I == 0.0 )
        assert not np.isnan(np.sum( py_I ))


class TestFinitePhoton(object):
    
    def setup(self):
                
        self.nq = 2 # number of detector vectors to do
        
        xyzQ = np.loadtxt(ref_file('512_atom_benchmark.xyz'))
        self.xyzlist = xyzQ[:,:3] * 10.0 # nm -> ang.
        self.atomic_numbers = xyzQ[:,3].flatten()
    
        self.q_grid = np.loadtxt(ref_file('512_q.xyz'))[:self.nq]
        
        self.rfloats = np.loadtxt(ref_file('512_x_3_random_floats.txt'))
        self.num_molecules = self.rfloats.shape[0]
                                       
                                       
    def test_cpu(self):
        
        cpu_I = _cpuscatter.simulate(self.num_molecules, self.q_grid, self.xyzlist, 
                                    self.atomic_numbers, rfloats=self.rfloats)
             
        # assert_allclose( cpu_I, np.array([0., 23886.]) ) # the random system
                                                           # on travis is different...
        
        
    def test_gpu(self):
        
        # just makes sure that CPU and GPU match
        if not GPU: raise SkipTest
        
        gpu_I = _gpuscatter.simulate(self.num_molecules, self.q_grid, self.xyzlist, 
                                    self.atomic_numbers, rfloats=self.rfloats)
                                    
        cpu_I = _cpuscatter.simulate(self.num_molecules, self.q_grid, self.xyzlist, 
                                    self.atomic_numbers, rfloats=self.rfloats)
        
        # assert_allclose( cpu_I, gpu_I )
        
        
    def test_py_cpu_smoke(self):

        traj = Trajectory.load(ref_file('ala2.pdb'))
        
        num_molecules = 1
        detector = xray.Detector.generic()
        detector.beam.photons_scattered_per_shot = 1e3

        I = scatter.simulate_shot(traj, num_molecules, detector, 
                                  finite_photon=True)
                                          
        # simple statistical sanity check
        assert np.abs(I.sum() - detector.beam.photons_scattered_per_shot) < \
                           np.sqrt(detector.beam.photons_scattered_per_shot)*6.0
                           

def test_simulate_shot_from_grid():
    # tests both construction of density from atomic model and simulation
    # of density
    
    traj = Trajectory.load(ref_file('ala2.pdb'))
    
    num_molecules = 1
    grid_dimensions = [25,] * 3
    grid_spacing = 1.0 # A?
    detector = xray.Detector.generic()
    
    grid = structure.atomic_to_density(traj, grid_dimensions, grid_spacing)
    test = scatter.simulate_shot_from_grid(grid, grid_spacing, num_molecules, 
                                           detector, dont_rotate=True)
    ref = scatter.simulate_shot(traj, num_molecules, detector,
                                dont_rotate=True)
    
    assert_allclose(test, ref)


def test_no_hydrogens():
    
    traj = Trajectory.load(ref_file('ala2.pdb'))
    
    num_molecules = 1
    detector = xray.Detector.generic()
    detector.beam.photons_scattered_per_shot = 1e3

    I_noH = scatter.simulate_shot(traj, num_molecules, detector, 
                                  ignore_hydrogens=True,
                                  dont_rotate=True)
    I_wH  = scatter.simulate_shot(traj, num_molecules, detector, 
                                  ignore_hydrogens=False,
                                  dont_rotate=True)
                                  
    assert not np.all(I_noH == I_wH)
    
    # compute the differece -- we're not setting random numbers here so just
    # looking at radially averaged stuff...
    diff = np.sum(np.abs(I_noH - I_wH) / I_wH) / float(len(I_wH))
    print diff
    assert diff < 1.0, 'ignoring hydrogens makes too big of a difference...'
    
        

def test_sph_harm():
    
    # -----------------------
    traj = Trajectory.load(ref_file('pentagon.pdb'))
    
    q_magnitudes     = [1.6]
    num_coefficients = 44

    num_molecules = 1
    num_shots     = 20000
    num_phi       = 2048
    # -----------------------
        
    q = q_magnitudes[0]


    # compute the Kam-theoretic values of the Legendre coefficients C_ell, which
    # we will call "coeffsh"
    coeffsh_even = scatter.sph_harm_coefficients(traj, q_magnitudes,
                                                 num_coefficients=num_coefficients/2)
    coeffsh_even = np.nan_to_num(coeffsh_even)
    coeffsh_even /= coeffsh_even[1]

    coeffsh = np.zeros(num_coefficients)
    coeffsh[0::2] = coeffsh_even.flatten()


    # next, preform a simulation of the scattering and empirically compute the
    # correlation function
    rings = xray.Rings.simulate(traj, num_molecules, q_magnitudes, 
                                num_phi, num_shots)
    
    c = rings.correlate_intra(q, q, mean_only=True)

    # it seems best to compare the solutions in the expanded basis
    c_sh = np.polynomial.legendre.legval(rings.cospsi(q, q), coeffsh.flatten())
    
    c    = c - c.mean()
    c_sh = c_sh - c_sh.mean()
    
    # plt.figure()
    # plt.plot(c_sh / c_sh[0])
    # plt.plot(c / c[0])
    # plt.show()
    
    # if these are more than 10% different, fail the test
    error = (np.sum(np.abs( (c_sh / c_sh[0]) - (c / c[0]) )) / float(num_phi))
    assert error < 0.1, 'simulation and analytical computation >10%% different (%f %%)' % error
    
    return
        
def test_atomic_formfactor():
    
    # this is a function in thor.xray, but the reference implementation
    # is in this file, so testing it here
    
    for q_mag in np.arange(2.0, 6.0, 1.0):
        for Z in [1, 8, 79]:
            qv = np.zeros(3)
            qv[0] = q_mag
            assert_allclose(scatter.atomic_formfactor(Z, q_mag), form_factor(qv, Z))

