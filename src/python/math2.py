
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

from scipy import ndimage, stats, optimize, spatial
from scipy.misc import factorial
from scipy.ndimage import filters, interpolation
from scipy.signal import fftconvolve
 
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
