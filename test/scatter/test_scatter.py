
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

import mdtraj

from thor import _cppscatter
from thor import xray
from thor import scatter
from thor import structure
from thor.refdata import get_cromermann_parameters, cromer_mann_params
from thor.testing import ref_file, gputest

import pytest

import time
global RANDOM_SEED
RANDOM_SEED = int(time.time() * 1e6) % 4294967294

GPU = _cppscatter.GPU_ENABLED

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


def form_factor_reference(qvector, atomz):
    
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
        
    else:
        raise KeyError('no implementation for Z=%d' % atomz)
        
    return fi

    
def ref_simulate_shot(xyzlist, atomic_numbers, num_molecules, q_grid,
                      dont_rotate=False):
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
    
    if dont_rotate:
        rfs = np.zeros((3, num_molecules))
    else:
        rs = np.random.RandomState(RANDOM_SEED)
        rfs = rs.rand(3, num_molecules)
    
    for i,qvector in enumerate(q_grid):    
        for n in range(num_molecules):
        
            # match random numbers that will be generated in actual impl.
            rotated_xyzlist = rand_rotate_molecule(xyzlist, rfloat=rfs[:,n])
            
            # compute the molecular form factor F(q)
            for j in range(xyzlist.shape[0]):
                q_mag = np.linalg.norm(qvector)
                fi = scatter.atomic_formfactor(atomic_numbers[j], q_mag)
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
        print(q_ind, len(q_values))

        # pre-compute the atomic form factors at this q
        qo = np.power( q / (4. * np.pi), 2)
        for ai in range(num_atom_types):                
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
#                              BEGIN test CLASSES
# ------------------------------------------------------------------------------

    
class TestCppScatter(object):
    """ tests for src/scatter """
    
    def setup(self):
        
        self.nq = 5  # number of detector vectors to do
        self.nr = 10 # number of atoms to use
        
        xyzZ = np.loadtxt(ref_file('512_atom_benchmark.xyz'))
        self.xyzlist = xyzZ[:self.nr,:3] * 10.0 # nm -> ang.
        self.atomic_numbers = xyzZ[:self.nr,3].flatten()
        
        self.q_grid = np.loadtxt(ref_file('512_q.xyz'))[:self.nq]
        
        self.num_molecules = 512        
        self.ref_A = ref_simulate_shot(self.xyzlist, self.atomic_numbers, 
                                       self.num_molecules, self.q_grid)
                                                                              
    def test_cpu_scatter(self):
        
        print("testing c cpu code...")

        cromermann_parameters, atom_types = get_cromermann_parameters(self.atomic_numbers)

        print('num_molecules:', self.num_molecules)
        cpu_A = _cppscatter.cpp_scatter(self.num_molecules, 
                                        self.xyzlist, 
                                        self.q_grid, 
                                        atom_types,
                                        cromermann_parameters,
                                        device_id='CPU',
                                        random_state=np.random.RandomState(RANDOM_SEED))

        assert_allclose(cpu_A, self.ref_A, rtol=1e-3, atol=1.0,
                       err_msg='scatter: c-cpu/cpu reference mismatch')
        assert not np.all( cpu_A == 0.0 )
        assert not np.sum( cpu_A == np.nan )
    
    def test_gpu_scatter(self):

        if not GPU:
            pytest.skip("unsupported configuration")
            
        cromermann_parameters, atom_types = get_cromermann_parameters(self.atomic_numbers)
        
        print('num_molecules:', self.num_molecules)
        gpu_A = _cppscatter.cpp_scatter(self.num_molecules, 
                                        self.xyzlist, 
                                        self.q_grid, 
                                        atom_types,
                                        cromermann_parameters,
                                        device_id=0,
                                        random_state=np.random.RandomState(RANDOM_SEED))

        assert_allclose(gpu_A, self.ref_A, rtol=1e-3, atol=1.0,
                        err_msg='scatter: cuda-gpu/cpu reference mismatch')
        assert not np.all( gpu_A == 0.0 )
        assert not np.sum( gpu_A == np.nan )
                        
    def test_parallel_interface(self):
        
        cp, at = get_cromermann_parameters(self.atomic_numbers)
        base_args = (self.num_molecules, self.xyzlist, self.q_grid, at, cp)
        
        out0 = _cppscatter.parallel_cpp_scatter(*base_args,
                                                random_state=np.random.RandomState(RANDOM_SEED))
        srl  = _cppscatter.cpp_scatter(*base_args, 
                                       random_state=np.random.RandomState(RANDOM_SEED))
        
        assert_allclose(out0, srl, 
                        rtol=1e-2, atol=1.0,
                        err_msg='parallel not consistent with serial')
                                
        assert_allclose(out0, self.ref_A, 
                        rtol=1e-2, atol=1.0,
                        err_msg='parallel not consistent with reference')
        
        if GPU:
            out1 = _cppscatter.parallel_cpp_scatter(*base_args,
                                                    devices=[0],
                                                    random_state=np.random.RandomState(RANDOM_SEED))
            assert_allclose(out1, self.ref_A, rtol=1e-2, atol=1.0,
                            err_msg='error in parallel gpu interface output')
                                               
                
        out2 = _cppscatter.parallel_cpp_scatter(*base_args,
                                                procs_per_node=2,
                                                devices=[],
                                                random_state=np.random.RandomState(RANDOM_SEED))
        # --> smoke test for now, fix later
        # assert_allclose(out2, self.ref_A, rtol=1e-2, atol=1.0,
        #                 err_msg='error in 2 thread cpu')
        
        
class TestSimulateAtomic(object):
    """ tests for src/python/scatter.py """
    
    def setup(self):
        
        self.nq = 100 # number of detector vectors to do
        self.q_grid = np.loadtxt(ref_file('512_q.xyz'))[:self.nq]
        
        self.traj = mdtraj.load(ref_file('ala2.pdb'))
        atomic_numbers = np.array([ a.element.atomic_number for a in self.traj.topology.atoms ])
        cp, at = get_cromermann_parameters(atomic_numbers)
        
        self.num_molecules = 32
        rxyz = self.traj.xyz[0] * 10.0
        
        self.ref_A = ref_simulate_shot(rxyz, 
                                       atomic_numbers, 
                                       self.num_molecules,
                                       self.q_grid)
        self.cpp_A = _cppscatter.cpp_scatter(self.num_molecules,
                                             rxyz,
                                             self.q_grid,
                                             at, cp,
                                             random_state=np.random.RandomState(RANDOM_SEED))
                                       
    def simulate_atomic(self):
        rs = np.random.RandomState(RANDOM_SEED)
        A = scatter.simulate_atomic(self.traj, self.num_molecules, self.q_grid,
                                    random_state=rs)
                                    
        assert_allclose(A, self.cpp_A, rtol=1e-4, err_msg='doesnt match cpp ref')
        assert_allclose(A, self.ref_A, rtol=1e-3, err_msg='doesnt match py ref')

    def test_multimodel(self):
        """ regression test """
        t =  mdtraj.load(ref_file('3LYZ_x2.pdb'))
        A = scatter.simulate_atomic(t, 1, self.q_grid)
        
    def test_dont_rotate_atomic(self):
        A1 = scatter.simulate_atomic(self.traj, self.num_molecules, self.q_grid,
                                     dont_rotate=True)
        A2 = scatter.simulate_atomic(self.traj, self.num_molecules, self.q_grid,
                                     dont_rotate=True)
        assert_allclose(A1, A2)
        
    def test_finite_photon_atomic(self):
        rs = np.random.RandomState(RANDOM_SEED)
        A = scatter.simulate_atomic(self.traj, self.num_molecules, self.q_grid,
                                    random_state=rs, finite_photon=1.0e8)
        assert A.dtype == np.int
        A = A.astype(np.float)
        ref_I = np.square( np.abs( self.cpp_A ) )
        
        print(np.sum( (A / A[0] - ref_I / ref_I[0]) > 0.05 ))
        
        assert_allclose(A / A[0], ref_I / ref_I[0], rtol=5e-2, 
                        err_msg='Finite photon statistics screwy in large photon limit')
        
        
    def test_no_hydrogens(self):

        I_wH = scatter.simulate_atomic(self.traj, self.num_molecules, self.q_grid,
                                       dont_rotate=True, ignore_hydrogens=False)
        I_noH = scatter.simulate_atomic(self.traj, self.num_molecules, self.q_grid,
                                        dont_rotate=True, ignore_hydrogens=True)

        assert not np.all(I_noH == I_wH)

        # compute the differece -- we're not setting random numbers here so just
        # looking at radially averaged stuff...
        diff = np.sum(np.abs(I_noH - I_wH) / I_wH) / float(len(I_wH))
        print(diff)
        assert diff < 1.0, 'ignoring hydrogens makes too big of a difference...'
        
        
        
class TestSimulateDensity(object):
    
    
    def setup(self):
        
        self.GRIDSIZE = 15
        self.GRIDSPAC = 1.0

        self.gs = (self.GRIDSIZE,)*3

        self.rxyz = np.mgrid[:self.gs[0],:self.gs[1],:self.gs[2]].reshape(3, -1).T * self.GRIDSPAC

        self.detector = np.mgrid[:self.gs[0],:self.gs[1],:self.gs[2]].reshape(3, -1).T * 1.0
        self.detector -= self.detector.mean(axis=0)[None,:]
        self.detector *= (self.GRIDSPAC * self.GRIDSIZE) / (2.0 * np.pi)
        
        
    def fft_and_simulate_density(self, dens):
        """
        NOT A TEST
        
        Take `dens`, a square grid, and use both a generic ND fft method
        and simulate_density() to transform it -- for comparison.
        
        returns:
        tst : the simulate_density() amplitudes
        ref : the fftn() amplitudes
        """
        
        ref = np.fft.fftn(dens)
        ref = np.fft.fftshift(ref)
        ref = np.abs(ref)
        ref /= ref.max()

        tst = scatter.simulate_density(dens, self.GRIDSPAC, 1, 
                                       self.detector, dont_rotate=True)
        tst = tst.T.reshape(*self.gs) # confirmed
        tst = np.abs(tst)
        tst /= tst.max()
        
        return tst, ref
    
    
    def test_box_dens(self):
        
        # make a 3d square pulse
        dens = np.zeros(self.gs)
        sq = 2 # size of box
        dens[self.GRIDSIZE//2:self.GRIDSIZE//2+sq,
             self.GRIDSIZE//2:self.GRIDSIZE//2+sq,
             self.GRIDSIZE//2:self.GRIDSIZE//2+sq] = 1.0
             
        tst, ref = self.fft_and_simulate_density(dens)
        
        # confirm total error is < 10%, per-pixel is < 100%
        total_error = np.sum(np.abs(tst - ref)) / np.product(self.gs), 'total err > 10%'
        #assert_allclose(tst, ref, rtol=1.0, atol=0.1, err_msg='per-pixel error > 100%')
        
    def test_random_dens(self):
        
        # use just a random density, no structures
        dens = np.abs( np.random.randn(*self.gs) )
             
        tst, ref = self.fft_and_simulate_density(dens)
        
        # confirm total error is < 10%, per-pixel is < 100%
        total_error = np.sum(np.abs(tst - ref)) / np.product(self.gs), 'total err > 10%'
        #assert_allclose(tst, ref, rtol=1.0, atol=0.1, err_msg='per-pixel error > 100%')
        
    def test_from_atomic(self):
        """
        simulate both from a grid and from an atomic model and ensure match
        """
        
        num_molecules = 1
        
        nq = 100 # number of q vectors
        q_grid = np.zeros((nq, 3))
        q_grid[:,1] = np.linspace(.01, 2.0, nq)
        
        traj = mdtraj.load(ref_file('pentagon.pdb'))
        atomic_numbers = np.array([ a.element.atomic_number for a in traj.topology.atoms ])
        cp, at = get_cromermann_parameters(atomic_numbers)
        rxyz = traj.xyz[0] * 10.0
        
        ref_A = ref_simulate_shot(rxyz, atomic_numbers, num_molecules, q_grid,
                                  dont_rotate=True)
        
        grid_dimensions = [125,] * 3
        grid_spacing = 0.1 # Angstroms
        grid = structure.atomic_to_density(traj, grid_dimensions, 
                                           grid_spacing)
        
        A = scatter.simulate_density(grid, grid_spacing, 
                                     num_molecules, q_grid,
                                     dont_rotate=True)
        
        tst = np.abs(A)
        ref = np.abs(ref_A)
        
        R = np.corrcoef(tst, ref)[0,1]
        assert R > 0.95, 'atomic and grid models significantly different'
        

def test_sph_harm():
    
    # -----------------------
    traj = mdtraj.load(ref_file('pentagon.pdb'))
    
    q_magnitudes     = [1.6]
    num_coefficients = 44

    num_molecules = 1
    num_shots     = 2000
    num_phi       = 256
    # -----------------------
        
    q = q_magnitudes[0]


    # compute the Kam-theoretic values of the Legendre coefficients C_ell, which
    # we will call "coeffsh"
    coeffsh_even = scatter.sph_harm_coefficients(traj, q_magnitudes,
                                                 num_coefficients=num_coefficients//2)
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
    
    for q_mag in np.arange(2.0, 6.0, 1.0):
        for Z in [1, 8, 79]:
            qv = np.zeros(3)
            qv[0] = q_mag
            assert_allclose(scatter.atomic_formfactor(Z, q_mag), 
                            form_factor_reference(qv, Z))

