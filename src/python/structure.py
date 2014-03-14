

"""
structure.py

Functions/classes for manipulating structures.
"""

import numpy as np
from scipy.special import cbrt

from mdtraj import trajectory
from mdtraj.topology import Topology
from mdtraj.pdb.element import Element

from odin.math2 import rand_rot
from odin.refdata import periodic_table

import logging
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


class quaternion(object):
    """
    Container class for quaternion-based functions. All methods of this class
    are static, and there is no concept of an instance of this class. It is
    really just a namespace deliminator.
    """
    
    @staticmethod
    def random(rfloat=None):
        """
        Compute a quaterion representing a random rotation, uniform on the
        unit sphere.
        
        Optional Parameters
        -------------------
        rfloat : ndarray, float, len 3
            A 3-vector of random numbers in [0,1) that acts as a random seed. If
            not passed, generates new random numbers.
            
        Returns
        -------
        q : ndarray, float, len 4
            A quaternion representing a random rotation uniform on the unit 
            sphere.
        """
    
        if rfloat == None:
            rfloat = np.random.rand(3)
    
        q = np.zeros(4)
    
        s = rfloat[0]
        sig1 = np.sqrt(s)
        sig2 = np.sqrt(1.0 - s)
    
        theta1 = 2.0 * np.pi * rfloat[1]
        theta2 = 2.0 * np.pi * rfloat[2]
    
        w = np.cos(theta2) * sig2
        x = np.sin(theta1) * sig1
        y = np.cos(theta1) * sig1
        z = np.sin(theta2) * sig2
    
        q[0] = w
        q[1] = x
        q[2] = y
        q[3] = z
    
        return q

    @staticmethod
    def prod(q1, q2):
        """
        Perform the Hamiltonian product of two quaternions. Note that this product
        is non-commutative -- this function returns q1 x q2.
    
        Parameters
        ----------
        q1 : ndarray, float, len(4)
            The first quaternion.
    
        q1 : ndarray, float, len(4)
            The first quaternion.
        
        Returns
        -------
        qprod : ndarray, float, len(4)
            The Hamiltonian product q1 x q2.
        """
        
        if (len(q1) != 4) or (len(q2) != 4):
            raise TypeError('Parameters cannot be interpreted as quaternions')
    
        qprod = np.zeros(4)
    
        qprod[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        qprod[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        qprod[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        qprod[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    
        return qprod
    
    @staticmethod
    def conjugate(q):
        """
        Compute the quaternion conjugate of q.
        
        Parameters
        ----------
        q : ndarray, float, len 4
            A quaternion input.
            
        Returns
        qconj : ndarray, float, len 4
            The conjugate of `q`.
        """
        
        if len(q) != 4:
            raise TypeError('Parameter `q` cannot be interpreted as a quaternion')
    
        qconj = np.zeros(4)
        qconj[0] = q[0]
        qconj[1] = -q[1]
        qconj[2] = -q[2]
        qconj[3] = -q[3]
    
        return qconj

    @staticmethod
    def rand_rotate_vector(v):
        """
        Randomly rotate a three-dimensional vector, `v`, uniformly over the unit
        sphere.
    
        Parameters
        ----------
        v : ndarray, float, len 3
            A vector to rotatea 3-vector in x,y,z space (e.g. the atomic 
            positions of an atom)
            
        Returns
        -------
        v_prime : ndarray, float, len 3
            Another 3-vector, which is the rotated version of v.
        """
        
        if len(v) != 3:
            raise TypeError('Parameter `v` must be in R^3')
        
        # generate a quaternion vector, with the first element zero
        # the last there elements are from v
        qv = np.zeros(4)
        qv[1:] = v.copy()
    
        # get a random quaternion vector
        q = quaternion.random()
        qconj = quaternion.conjugate(q)
    
        q_prime = quaternion.prod( quaternion.prod(q, qv), qconj )
    
        v_prime = q_prime[1:].copy() # want the last 3 elements...
    
        return v_prime


def remove_COM(traj):
    """
    Remove the center of mass from all frames in a trajectory.
    
    Parameters
    ----------
    traj : mdtraj.trajectory
        A trajectory object.
        
    Returns
    -------
    centered_traj : mdtraj.trajectory
        A trajectory with the center of mass removed
    """
    
    for i in range(traj.n_frames):
        masses = [ a.element.mass for a in traj.topology.atoms ]
        traj.xyz[i,:,:] -= np.average(traj.xyz[i,:,:], axis=0, weights=masses)
        
    return traj
    

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
    q = quaternion.random(rfloat)
    
    # take the quaternion conjugate
    qconj = quaternion.conjugate(q)
    
    # prepare data structures
    rotated_xyzlist = np.zeros(xyzlist.shape)
    qv = np.zeros(4)
    
    # put each atom through the same rotation
    for i in range(xyzlist.shape[0]):
        qv[1:] = xyzlist[i,:].copy()
        q_prime = quaternion.prod( quaternion.prod(q, qv), qconj )
        rotated_xyzlist[i,:] = q_prime[1:].copy() # want the last 3 elements...
    
    return rotated_xyzlist

def rand_rotate_molecule2(xyzlist, rfloat=None):
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
    
    rotated_xyzlist = rand_rot(rfloat) * xyzlist.T
    
    rotated_xyzlist = np.array( rotated_xyzlist.T )
    
    return rotated_xyzlist
    
    
def rand_rotate_traj(traj, remove_COM=False):
    """
    Randomly rotate all the members of a trajectory.
    
    Parameters
    ----------
    xyzlist : ndarray, float, 3D
        An n x 3 array representing the x,y,z positions of n atoms.
    
    Optional Parameters
    -------------------
    remove_COM : bool
        Whether or not to translate the center of mass of the molecule to the
        origin before rotation.
        
    
    """
    
    if remove_COM:
        traj = remove_COM(traj)
        
    for i in range(traj.n_frames):
        traj.xyz = rand_rotate_molecule(traj.xyz)
    
    return traj


def multiply_conformations(traj, num_replicas, density, traj_weights=None):
    """
    Take a structure and generate a system of many conformations, such that they
    are randomly distributed & rotated in space with a given `density`.
    
    This function is useful for approximating a subset of a homogenous solution,
    gas, slurry, etc. composed of the structures indicated by `trajectory`.
    These structures can be given e.g. Boltzmann weights by using the
    `traj_weights` argument.
    
    Parameters
    ----------
    traj : mdtraj
        The structures to use to generate the system. Note this can be a single
        conformation, in which case each molecule in the system is identical,
        just translated & rotated.
    
    num_replicas : int
        The total number of molecules to include in the system. The total volume
        of the system depends on this parameter and `density`.
        
    density : float
        The number density of species, MICROMOLAR UNITS. That is, umol/L. (This
        software was written by a chemist!).
    
    Optional Parameters
    -------------------
    perc_mean : float
        Fraction of total atoms that are vacant (missing).

    traj_weights : ndarray, float
        The weights at which to include members of trajectory in the final
        system. Default is to assign equal weight to all members of trajectory.
    
    Returns
    -------
    system_structure : mdtraj
        Length-one trajectory representing the coordinates of the molecular
        system.
    """

    traj = remove_COM(traj)

    # check traj_weights
    if traj_weights != None:
        if len(traj_weights) != traj.n_frames:
            raise ValueError('Length of `traj_weights` is not the same as `traj`')
    else:
        traj_weights = np.ones(traj.n_frames)
    traj_weights /= traj_weights.sum() # normalize
        
    # generate a random ensemble, defined by a list of indices of `traj`
    num_per_shapshot = np.random.multinomial(num_replicas, traj_weights)
    
    # determine the box size
    boxvol  = (num_replicas * 1.0e27) / (density * 6.02e17) # in A^3
    boxsize = cbrt(boxvol)            # one dim of a cubic box, in Angstrom
    logger.info('Set boxsize: %f A' % boxsize)

    # find the maximal radius of each snapshot in traj
    max_radius = np.zeros(traj.n_frames)
    for i in xrange(traj.n_frames):
        max_radius[i] = np.max( np.sqrt( np.sum(np.power(traj.xyz[i,:,:]  , 2), axis=1) ) )
        
    if boxsize < np.max(max_radius)*2:
        raise ValueError('You solution is too concentrated for its constituent'
                         ' matter! There is no way it will fit. Box: '
                         '%f, Biggest Molecule: %f' % (boxsize, np.max(max_radius)))
        
    # place in space
    ind = []
    
    for x in xrange( len(num_per_shapshot) ):
        ind.extend( [x] * num_per_shapshot[x] )
    
    xyz = traj.xyz[ind,:,:]
    
    centers_of_mass = np.zeros((num_replicas, 3)) # to store these and use later
    
    # randomly orient the first molecule
    xyz[0,:,:]         = rand_rotate_molecule2(xyz[0,:,:])
    centers_of_mass[0] = np.random.uniform(low=0, high=boxsize, size=3)
    for x in xrange(3):
        xyz[0,:,x] += centers_of_mass[0,x]

    for i in xrange(1, xyz.shape[0] ):
        molecule_overlapping = True # initial cond.
        
        attempt = 0
        while molecule_overlapping:
            
            attempt += 1
        
            # suggest a random translation
            centers_of_mass[i,:] = np.random.uniform(low=0, high=boxsize, size=3)
            
            # check to see if we're overlapping another molecule already placed
            for j in xrange(i):
                molec_dist = np.linalg.norm(centers_of_mass[i,:] - centers_of_mass[j,:])
                min_allowable_dist = max_radius[ind[i]] + max_radius[ind[j]]
                
                if molec_dist > min_allowable_dist:
                    # if not, move the molecule there and do a rotation.
                    molecule_overlapping = False
                else:
                    molecule_overlapping = True
                    break
                    
            if attempt > 10000:
                raise RuntimeError('Number of attempts > 10000, density is too high.')
            
        xyz[i,:,:] = rand_rotate_molecule2(xyz[i,:,:])
        for x in xrange(3):
            xyz[i,:,x] += centers_of_mass[i,x]
        
        logger.debug('Placed molecule, took %d attempts' % attempt)
    
    # store & return the results
    out_traj = trajectory.Trajectory( xyz, traj.topology )

    return out_traj


def load_coor(filename):
    """
    Load a simple coordinate file, formatted as:
    
    x   y   z   atomic_number
    
    where `x`,`y`,`z` (float) are the positions of the atom, in angstroms, and 
    `atomic_number` (int) is the atomic number Z specifying what the atom is.
    
    Parameters
    ----------
    filename : str
        The filename to load.
        
    Returns
    -------
    structure : mdtraj.trajectory
        A meta-data minimal mdtraj instance
    """
    data = np.genfromtxt(filename)
    xyz = data[:,:3] / 10.0 # coor files are in angstoms, conv. to nm
    atomic_numbers = data[:,3]
    structure = _traj_from_xyza(xyz, atomic_numbers)
    return structure

    
def _traj_from_xyza(xyz, atomic_numbers, units='nm'):
    """
    Parameters
    ----------
    xyz : np.array, float, shape( num_atom, 3)
        array of x,y,z,a

    atomic_numbers : np.array, int, shape( num_atom, 1 )
        the atomic numbers of each of the atoms.

    Optional Parameters
    -------------------
    units : str
        if units == 'nm' then nothing happens. if units == 'ang' then
        we convert them to nm.
        
    Returns
    -------
    structure : mdtraj.trajectory
        A meta-data minimal mdtraj instance
    """
    
    if units == 'ang':
        xyz /= 10.

    top = Topology()
    chain = top.add_chain()
    residue = top.add_residue('XXX', chain)
    
    for i in range(xyz.shape[0]):
        element_symb = periodic_table[atomic_numbers[i]][1] # should give symbol
        element = Element.getBySymbol(element_symb)
        name = '%s' % element_symb
        top.add_atom(name, element, residue)
    
    structure = trajectory.Trajectory(xyz=xyz, topology=top)

    return structure

