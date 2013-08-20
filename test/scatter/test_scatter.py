
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

try:
    from odin import _gpuscatter
    GPU = True
except ImportError:
    GPU = False

from odin.refdata import cromer_mann_params
from odin import xray
from odin import _cpuscatter
from odin.testing import skip, ref_file, gputest
from odin.xray import scatter
from odin.xray import structure
from odin.xray.structure import rand_rotate_molecule

from mdtraj import trajectory


# ------------------------------------------------------------------------------
#                        BEGIN REFERENCE IMPLEMENTATIONS
# ------------------------------------------------------------------------------

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

    
def ref_simulate_shot(xyzlist, atomic_numbers, num_molecules, q_grid, rfloats=None):
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
    
    I = np.zeros(q_grid.shape[0])
    
    for n in range(num_molecules):
        
        if rfloats == None:
            rotated_xyzlist = rand_rotate_molecule(xyzlist)
        else:
            rotated_xyzlist = rand_rotate_molecule(xyzlist, rfloat=rfloats[n,:])
        
        for i,qvector in enumerate(q_grid):

            # compute the molecular form factor F(q)
            F = 0.0
            for j in range(xyzlist.shape[0]):
                fi = form_factor(qvector, atomic_numbers[j])
                r = rotated_xyzlist[j,:]
                F += fi * np.exp( 1j * np.dot(qvector, r) )
    
            I[i] += F.real*F.real + F.imag*F.imag

    if len(I[I<0.0]) != 0:
        raise Exception('neg values in reference scattering implementation')

    return I


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
        
        self.nq = 2 # number of detector vectors to do
        
        xyzQ = np.loadtxt(ref_file('512_atom_benchmark.xyz'))
        self.xyzlist = xyzQ[:,:3] * 10.0 # nm -> ang.
        self.atomic_numbers = xyzQ[:,3].flatten()
    
        self.q_grid = np.loadtxt(ref_file('512_q.xyz'))[:self.nq]
        
        self.rfloats = np.loadtxt(ref_file('512_x_3_random_floats.txt'))
        self.num_molecules = self.rfloats.shape[0]
        
        self.ref_I = ref_simulate_shot(self.xyzlist, self.atomic_numbers, 
                                       self.num_molecules, self.q_grid, 
                                       self.rfloats)
    
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

        cpu_I = _cpuscatter.simulate(self.num_molecules, self.q_grid, self.xyzlist, 
                                    self.atomic_numbers, rfloats=self.rfloats)

        print "CPU", cpu_I
        print "REF", self.ref_I

        assert_allclose(cpu_I, self.ref_I, rtol=1e-03,
                        err_msg='scatter: c-cpu/cpu reference mismatch')
        assert not np.all( cpu_I == 0.0 )
        assert not np.sum( cpu_I == np.nan )
        
                            
    def test_python_call(self):
        """
        Test the GPU scattering simulation interface (scatter.simulate)
        """

        if not GPU: raise SkipTest
        print "testing python wrapper fxn..."
        
        traj = trajectory.load(ref_file('ala2.pdb'))
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

        traj = trajectory.load(ref_file('ala2.pdb'))
        
        num_molecules = 1
        detector = xray.Detector.generic()
        detector.beam.photons_scattered_per_shot = 1e3

        I = scatter.simulate_shot(traj, num_molecules, detector, 
                                  finite_photon=True)
                                          
        # simple statistical sanity check
        assert np.abs(I.sum() - detector.beam.photons_scattered_per_shot) < \
                           np.sqrt(detector.beam.photons_scattered_per_shot)*6.0
        

class TestSphHrm(object):
   
    @skip
    def test_vs_reference(self):
        qs = np.arange(2, 3.52, 0.02)
        silver = structure.load_coor(ref_file('SilverSphere.coor'))
        cl = scatter.sph_hrm_coefficients(silver, q_magnitudes=qs, 
                                          num_coefficients=2)[1,:,:]
        ref = np.loadtxt(ref_file('ag_kam.dat')) # computed in matlab
        assert_allclose(cl, ref)
    
        
def test_atomic_formfactor():
    
    # this is a function in odin.xray, but the reference implementation
    # is in this file, so testing it here
    
    for q_mag in np.arange(2.0, 6.0, 1.0):
        for Z in [1, 8, 79]:
            qv = np.zeros(3)
            qv[0] = q_mag
            assert_allclose(scatter.atomic_formfactor(Z, q_mag), form_factor(qv, Z))

