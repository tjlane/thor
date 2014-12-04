
"""
math.py

Various mathematical functions and operations.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel("DEBUG")

import numpy as np
from random import randrange, seed

from scipy import ndimage
from scipy import stats
from scipy import optimize
from scipy import spatial
from scipy import special
from scipy.misc import factorial
from scipy.ndimage import filters, interpolation
from scipy.signal import fftconvolve
from scipy.interpolate import interpn
 
from matplotlib.path import Path
        

def smooth(x, beta=10.0, window_size=11):
    """
    Apply a Kaiser window smoothing convolution.
    
    Parameters
    ----------
    x : ndarray, float
        The array to smooth.
        
    Optional Parameters
    -------------------
    beta : float
        Parameter controlling the strength of the smoothing -- bigger beta 
        results in a smoother function.
    window_size : int
        The size of the Kaiser window to apply, i.e. the number of neighboring
        points used in the smoothing.
        
    Returns
    -------
    smoothed : ndarray, float
        A smoothed version of `x`.
    """
    
    # make sure the window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # apply the smoothing function
    s = np.r_[x[window_size-1:0:-1], x, x[-1:-window_size:-1]]
    w = np.kaiser(window_size, beta)
    y = np.convolve( w/w.sum(), s, mode='valid' )
    
    # remove the extra array length convolve adds
    b = (window_size-1) / 2
    smoothed = y[b:len(y)-b]
    
    return smoothed


def arctan3(y, x):
    """
    Compute the inverse tangent. Like arctan2, but returns a value in [0,2pi].
    """
    theta = np.arctan2(y,x)
    theta[theta < 0.0] += 2 * np.pi
    return theta


def fft_acf(data):
    '''
    Return the autocorrelation of a 1D array using the FFT
    Note: the result is normalized
    
    Parameters
    ----------
    data : ndarray, float, 1D
        Data to autocorrelate
        
    Returns
    -------
    acf : ndarray, float, 1D
        The autocorrelation function
    '''
    data = data - np.mean(data)
    result = fftconvolve(data, data[::-1])
    result = result[result.size / 2:] 
    acf = result / result[0]
    return acf


def freedman_diaconis(data):
    """
    Estimate an optimal number of histogram bins using the Freedman-Diaconis
    rule of thumb.
    
    Parameters
    ----------
    data : ndarray
        The data to histogram.
        
    Returns
    -------
    n_bins : int
        The number of bins to use.
    """
    
    data = data.flatten()
    
    q3 = stats.scoreatpercentile(data, 75.0)
    q1 = stats.scoreatpercentile(data, 25.0)
    
    h  = 2.0 * (q3 - q1) * np.power(len(data), -1.0/3.0)
    n_bins = int( ( np.max(data) - np.min(data) ) / h )
    
    return n_bins

    
def find_overlap(area_points, test_points):
    """
    Find the intersection of two sets of points. Here, `area_points` defines
    a polygon, and this function finds which of `test_points` are in that 
    polygon.
    
    Parameters
    ----------
    area_points : np.ndarray, float
        An N x M array, with N the number of points and M the dimension of the 
        space. The convex hull these points define an `area` against which
        the `test_points` are tested for inclusion.
        
    test_points : np.ndarray, float
        An N' x M array, with the same M (dimension) as `area_points`.
    
    Returns
    -------
    in_area : np.ndarray, bool
        An len N' array of booleans, True if the corresponding index of
        `test_points` is in the test area, and False otherwise.
    """
    
    if not area_points.shape[1] == test_points.shape[1]:
        raise ValueError('area_points and test_points must be two dimensional, '
                         'and their second dimension must be the same size')
    
    # http://stackoverflow.com/questions/11629064/finding-a-partial-or-complete-
    # inclusion-of-a-set-of-points-in-convex-hull-of-oth
    
    triangulation = spatial.Delaunay(area_points)
    
    # order points for matplotlib fxn
    unordered = list(triangulation.convex_hull)
    ordered = list(unordered.pop(0))
    
    while len(unordered) > 0:
        next = (i for i, seg in enumerate(unordered) if ordered[-1] in seg).next()
        ordered += [point for point in unordered.pop(next) if point != ordered[-1]]
    
    ordered_pts = area_points[ordered]
    
    path = Path(ordered_pts)
    in_area = path.contains_points(test_points)
    
    assert len(in_area) == test_points.shape[0]
    
    return in_area


def ER_rotation_matrix(axis, theta):
    """
    Compute the rotation matrix defining a rotation in 3D of angle `theta` 
    around `axis` using the Euler-Rodrigues formula.
    
    Parameters
    ----------
    axis : np.ndarray, float
        A 3-vector defining the axis of rotation.

    theta : float
        The rotation angle, in radians.

    Returns
    -------
    R : np.ndarray, float
        A 3x3 array defining a rotation matrix.

    Citation
    --------
    ..[1] https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_parameters
    """

    # note: the a,b,c,d parameters here don't correspond with their equivalents
    # elsewhere in the code

    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    R = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    return R


def rand_rot(rands = None):
    """
    Compute a uniform, random rotation matrix. 
    
    Optional Parameters
    ------------------
    rands : np.ndarray, float, shape( 3,)
        3 random numbers used to define matrix

    Returns
    -------
    RU : np.matrix, float
        A 3x3 matrix defining a uniform, random rotation.
    
    Reference
    ---------
    ..[1] http://www.google.com/url?sa=t&rct=j&q=uniform%20random%20rotation%20matrix&source=web&cd=5&ved=0CE8QFjAE&url=http%3A%2F%2Fciteseerx.ist.psu.edu%2Fviewdoc%2Fdownload%3Fdoi%3D10.1.1.53.1357%26rep%3Drep1%26type%3Dps&ei=lw2cUa2eIMKRiQKXnIDwBQ&usg=AFQjCNHViFXwwa8kv_tobzteWYM8EaKF-w&sig2=148RpesMoZvJmtse2oerjg&bvm=bv.46751780,d.cGE
    """
    
    # 3 random numbers
    np.random.seed()
    if rands == None:
        rands = np.random.random(( 3, ) )

    x1 = rands[0] * np.pi * 2
    x2 = rands[1] * np.pi * 2
    x3 = rands[2]

    # matrix for rotation about z-axis
    Rz1 = [ np.cos(x1), np.sin(x1), 0]
    Rz2 = [-np.sin(x1), np.cos(x1), 0]
    Rz = np.matrix(  [ Rz1, Rz2, [ 0, 0, 1 ] ]  )


    # matrix for rotating the pole
    v = np.array( [ np.cos(x2) * np.sqrt(x3) ,
                    np.sin(x2) * np.sqrt(x3) ,
                    np.sqrt( 1 - x3 ) ] )
    H  = np.identity( 3 ) - 2 * np.outer(v,v)
    RU = -H*Rz
    return RU


def Wigner3j(j1, j2, j3, m1, m2, m3):
    """
    Compute the Wigner 3j symbol using the Racah formula [1]. 
    
                         / j1 j2 j3 \
                         |          |  
                         \ m1 m2 m3 /
     
    Parameters
    ----------
    j : angular momentum quantum numbers
    m : magnetic quantum numbers
    
    Returns
    -------
    wigner3j : float
        The 3j symbol value.
     
    References
    ----------
    ..[1] Wigner 3j-Symbol entry of Eric Weinstein's Mathworld: 
    http://mathworld.wolfram.com/Wigner3j-Symbol.html
    
    Notes
    -----
    Adapted from Wigner3j.m by David Terr, Raytheon, 6-17-04
    """
    
    # Error checking
    if ( ( 2*j1 != np.floor(2*j1) ) | ( 2*j2 != np.floor(2*j2) ) | ( 2*j3 != np.floor(2*j3) ) | ( 2*m1 != np.floor(2*m1) ) | ( 2*m2 != np.floor(2*m2) ) | ( 2*m3 != np.floor(2*m3) ) ):
        raise ValueError('All arguments must be integers or half-integers.')
        
    # Additional check if the sum of the second row equals zero
    if ( m1+m2+m3 != 0.0 ):
        logger.debug('3j-Symbol unphysical')
        return 0.0

    if ( j1 - m1 != np.floor ( j1 - m1 ) ):
        logger.debug('2*j1 and 2*m1 must have the same parity')
        return 0.0

    if ( j2 - m2 != np.floor ( j2 - m2 ) ):
        logger.debug('2*j2 and 2*m2 must have the same parity')
        return 0.0

    if ( j3 - m3 != np.floor ( j3 - m3 ) ):
        logger.debug('2*j3 and 2*m3 must have the same parity')
        return 0.0

    if ( j3 > j1 + j2)  | ( j3 < abs(j1 - j2) ):
        logger.debug('j3 is out of bounds.')
        return 0.0

    if abs(m1) > j1:
        logger.debug('m1 is out of bounds.')
        return 0.0

    if abs(m2) > j2:
        logger.debug('m2 is out of bounds.')
        return 0.0

    if abs(m3) > j3:
        logger.debug('m3 is out of bounds.')
        return 0.0

    t1 = j2 - m1 - j3
    t2 = j1 + m2 - j3
    t3 = j1 + j2 - j3
    t4 = j1 - m1
    t5 = j2 + m2

    tmin = max( 0.0, max( t1, t2 ) )
    tmax = min( t3, min( t4, t5 ) )
    tvec = np.arange(tmin, tmax+1.0, 1.0)

    wigner = 0.0

    for t in tvec:
        wigner += (-1.0)**t / float( factorial(t) * factorial(t-t1) * factorial(t-t2) *\
                  factorial(t3-t) * factorial(t4-t) * factorial(t5-t) )

    w3j = wigner * (-1)**(j1-j2-m3) * np.sqrt( factorial(j1+j2-j3) * \
          factorial(j1-j2+j3) * factorial(-j1+j2+j3) / factorial(j1+j2+j3+1) * \
          factorial(j1+m1) * factorial(j1-m1) * factorial(j2+m2) * \
          factorial(j2-m2) * factorial(j3+m3) * factorial(j3-m3) )
    
    return w3j


def assoc_legendre(l, m, x):
    """
    Compute and return the associated Legendre polynomial of degree l and order
    m, evaluated at each point x. This is commonly written
    
        P_l^m (x)
        l = 0, 1, 2, ...
        m = -1, -l+1, ..., l
        x in (-1, 1)
        
    Parameters
    ----------
    l : int
        The polynomial degree
        
    m : int
        The polynomial order
        
    x : np.ndarray, float
        The points to evaluate the function at
        
    Returns
    -------
    P : np.ndarray, float
        The polynomial evaluted at the appropriate values.
    """
    
    if np.abs(m) > l:
        raise ValueError('The associated Legendre polynomial is only defined'
                         'for |m| < l')
    
    if m > 0:
        prefix = np.power(-1, m) * np.product(np.arange(l-m+1, l+m+1))
        m = -1 * m
    else:
        prefix = 1
        
    t1 = 1.0 / special.gamma(1.0-m)
    
    if m == 0:
        t2 = np.ones_like(x)
    elif m < 0:
        t2 = 1.0 / (np.power((1.0+x) / (1.0-x), np.abs(m)/2.0) + 1e-16)
    else:
        t2 = np.power((1.0+x) / (1.0-x), m/2.0)
    t2[ np.abs(t2) < (t2.max() * 1e-50) ] = 0.0 # avoid underflow
    
    t3 = special.hyp2f1(-l, l+1.0, 1.0-m, (1.0-x)/2.0)
    
    return prefix * t1 * t2 * t3


def sph_harm(l, m, theta, phi):
    """
    Compute the spherical harmonic Y_lm(theta, phi).
    """
    
    theta = np.array(theta)
    phi   = np.array(phi)
    
    if np.any( np.isnan(theta) + np.isinf(theta) ):
        raise ValueError('NaN or inf in theta -- must be floats in [0, pi]')
    if np.any( np.isnan(phi) + np.isinf(phi) ):
        raise ValueError('NaN or inf in phi -- must be floats in [0, 2pi]')
    
    # avoid P_lm(1.0) = inf
    cos_theta = np.cos(theta)
    cos_theta[(cos_theta >= 1.0)] = 1.0 - 1e-8
    
    N = np.sqrt( (2. * l + 1) * special.gamma(l-m+1) /
                 ( 4. * np.pi * special.gamma(l+m+1) ) )
    
    Plm = assoc_legendre(l, m, cos_theta) 
    
    Ylm = N * np.exp( 1j * m * phi ) * Plm
    
    if (np.any(np.isnan(Ylm)) or (np.any(np.isinf(Ylm)))):
        raise RuntimeError('Could not compute Ylm, l=%d/m=%d' % (l, m))
    
    return Ylm


def kabsch(P, Q):
    """ 
    Employ the Kabsch algorithm to compute the rotation matrix U that when
    applied to solid object P, minimizes the RMSD to Q. I.E. choose rotation U
    such that RMSD(PU, Q) is as small as possible.
    
    Parameters
    ----------
    P : np.ndarray, float
        N x D array representing an object in space, where N is the number of 
        points in the object and D is the dimension.
    
    Q : np.ndarray, float
        A second N x D array representing a second object in the same space,
        with an equal number of points.
    
    
    Returns
    -------
    U : np.ndarray, float
        A D x D rotation matrix, that minimizes RMSD(PU, Q)
    
    References
    ----------
    ..[1] https://github.com/charnley/rmsd
    ..[2] http://en.wikipedia.org/wiki/Kabsch_algorithm
    ..[3] Kabsch, Wolfgang, (1976) "A solution for the best rotation to relate
          two sets of vectors", Acta Crystallographica 32:922
    
    Example
    -------
    >>> U = kabsch(P, Q)
    >>> rotated = np.dot(P, U)
    >>> print (rotated - Q)
    """
    
    if not P.shape == Q.shape:
        raise ValueError('`P` and `Q` must be the same shape')
    
    A = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(A)
    
    # ensure right-handed coordinate system (see wikipedia)
    if ((np.linalg.det(V) * np.linalg.det(W)) < 0.0):
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]
        
    U = np.dot(V, W)
    
    return U
    
    
def LDA_direction(positives, negatives):
    """
    Compute the maximally discriminating direction between two sets
    of points in a common vector space.

    Parameters
    ----------
    positives/negatives : np.ndarray
        The two sets. Will find the linear direction that best
        describes their differences.

    Returns
    -------
    w : np.ndarray
        A vector representing the discriminatory direction.

    Notes
    -----
    The boundary for a binary classifier, as usually employed
    in LDA, will be the hyperplane perpendicular to w.
    """
    
    if not positives.shape[1] == negatives.shape[1]:
        raise ValueError('`positives` and `negatives` should have the same dimensionality!')
    
    S1 = np.cov(positives.T)
    S0 = np.cov(negatives.T)
    
    mu1 = positives.mean(axis=0)
    mu0 = negatives.mean(axis=0)
    
    assert S0.shape[0] == mu0.shape[0]
    assert S1.shape[0] == mu1.shape[0]
    
    w = np.dot(np.linalg.inv(S0 + S1), (mu1 - mu0)) 
    
    return w


def interp_grid_to_spherical(grid, radii, num_phi, num_theta, 
                             grid_origin=(0,0,0), return_spherical_coords=False):
    """
    Compute interpolated values in 3D spherical coordinates from a 3D square 
    grid. The interpolated values lie equally spaced along the azumithal and
    polar directions for a series of concentric spheres.
    
    Interpolation used is linear.
    
    Parameters
    ----------
    grid : np.ndarray, float
        The 3D square grid of values definiting a scalar field
    radii : np.ndarray, float
        The radial values of the interpolant grid
    num_theta : int
        The number of points along the polar angle to interpolate
    num_phi : int
        The number of points along the azmuthal angle to interpolate
    grid_origin : 3-tuple, floats
        The origin of the grid, which forms the center of the interpolant 
        spheres
        
    Optional Parameters
    -------------------
    return_spherical_coords : bool
        If true, the spherical coordiantes used are also returned as an N x 3
        array.
        
    Returns
    -------
    interpolated : np.ndarray
        A 3D array of the interpolated values. The dimensions are 
        (radial, polar [theta], azmuthal [phi]).
    """
    
    # find the cartesian x,y,z values for each interpolant
    xi = np.zeros( (len(radii) * num_theta * num_phi, 3), dtype=grid.dtype )
    
    thetas = np.arange(0.0, 2.0*np.pi, 2.0*np.pi / num_theta)
    phis = np.arange(0.0, np.pi, np.pi / num_phi)
    assert len(thetas) == num_theta, 'thetas len mistmatch %d %d' % (len(thetas), num_theta)
    assert len(phis) == num_phi, 'phi len mistmatch %d %d' % (len(phis), num_phi)
    
    # the repeat rate will be important for the reshape, below
    r = np.repeat(radii, num_theta * num_phi)            # radius, slowest
    t = np.repeat( np.tile(thetas, num_phi), len(radii)) # theta
    p = np.tile(phis, len(radii) * num_theta)            # phi, fastest
    
    xi[:,0] = r * np.sin(t) * np.cos(p) # x
    xi[:,1] = r * np.sin(t) * np.sin(p) # y
    xi[:,2] = r * np.cos(t)             # z
    
    xi += np.array(grid_origin)[None,:]
    
    # compute an interpolator for the rectangular grid
    gi = [ np.arange(l) for l in grid.shape ]
    interpolated = interpn(gi, grid, xi, bounds_error=False)
    
    res = interpolated.reshape(len(radii), num_theta, num_phi)
    
    if return_spherical_coords:
        rtp = np.array([r, t, p])
        return res, rtp
    else:
        return res
        
        
class IncrementalVariance(object):
    """
    http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    
    def __init__(self, size=1):
        self._n = 0
        self._mean = np.zeros(size)
        self._M2 = np.zeros(size)

    def add(self, x):
        self._n = self._n + 1
        delta = x - self._mean
        self._mean = self._mean + delta/self._n
        self._M2 = self._M2 + delta*(x - self._mean)
        return

    def remove(self, x):
        self._n = self._n - 1
        delta = x - self._mean
        self._mean = self._mean - delta/self._n
        self._M2 = self._M2 - delta*(x - self._mean)
        return

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._M2/float(self._n - 1)
        
    @property
    def std(self):
        return np.sqrt(self.variance)
        
    @property
    def num_samples(self):
        return self._n
        
            