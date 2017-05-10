
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
from scipy.ndimage import filters, interpolation, morphology
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


def find_local_maxima(arr):
    """
    Find local maxima in a multidimensional array `arr`.
    
    Parameters
    ----------
    arr : np.ndarray
        The array to find maxima in
    
    Returns
    -------
    indices : tuple of np.ndarray
        The indices of local maxima in `arr`
    """
    
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    
    # neighborhood is simply a 3x3x3 array of True
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_max = ( filters.maximum_filter(arr, footprint=neighborhood) == arr )
    
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    background = ( arr == 0 )
    eroded_background = morphology.binary_erosion(background,
                                                  structure=neighborhood,
                                                  border_value=1)
        
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_max = local_max ^ eroded_background # ^ = XOR
    
    return np.where(detected_max)


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


def sample_vMF(mu, kappa, num_samples):
    """
    Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu in R^N with concentration kappa.
    
    Parameters
    ----------
    mu: float, np.ndarray
        The vMF mean parameter(s). Pass an array of length N to sample the vMF
        distribution in R^N.
    
    kappa: float
        The vMF concentration/shape parameter
    
    num_samples : int
        The number of samples to draw
    
    Returns
    -------
    samples : np.ndarray
        A num_samples x len(mu)

    Reference
    ---------
    ..[1] https://raw.githubusercontent.com/clara-labs/spherecluster/develop/spherecluster/util.pyhttps://raw.githubusercontent.com/clara-labs/spherecluster/develop/spherecluster/util.py
    ..[2] http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
    ..[3] https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    ..[4] http://www.stat.pitt.edu/sungkyu/software/randvonMisesFisher3.pdf
    """
    
    dim = len(mu)
    result = np.zeros((num_samples, dim))
    
    for nn in range(num_samples):
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight(kappa, dim)

        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal_to(mu)

        # compute new point
        result[nn, :] = v * np.sqrt(1. - w**2) + w * mu

    return result


def _sample_weight(kappa, dim):
    """
    Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    
    dim = dim - 1 # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa**2 + dim**2) + 2 * kappa)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x**2)

    while True:
        z = np.random.beta(dim / 2., dim / 2.)
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
            return w


def _sample_orthonormal_to(mu):
    """
    Sample point on sphere orthogonal to mu.
    """
    v = np.random.randn(mu.shape[0])
    proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
    orthto = v - proj_mu_v
    return orthto / np.linalg.norm(orthto)


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
        
            