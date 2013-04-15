
"""
math.py

Various mathematical functions and operations.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

import multiprocessing as mp
from utils import parmap

import numpy as np
from random import randrange, seed

from scipy import ndimage, stats, optimize, spatial
from scipy.ndimage import filters
from scipy.signal import fftconvolve
 
from matplotlib import nxutils
from matplotlib import pyplot as plt

from odin.interp import Bcinterp

    
        
def find_center(image2d, mask=None, initial_guess=None, pix_res=0.1):
    """
    Locates the center of a circular image.
    """
    
    logger.info('Finding the center of the strongest ring...')
    
    x_size = image2d.shape[0]
    y_size = image2d.shape[1]
    
    if mask != None:
        image2d *= mask.astype(np.bool)
    
    if initial_guess == None:
        initial_guess = np.array(image2d.shape) / 2.0
        
    # generate a radial grid
    num_phi = 360
    phi = np.linspace(0.0, 2.0 * np.pi * (float(num_phi-1)/num_phi), num=num_phi)
    r   = np.arange(pix_res, np.min([x_size/3., y_size/3.]), pix_res)
    num_r = len(r)
    
    rx = np.repeat(r, num_phi) * np.cos(np.tile(phi, num_r))
    ry = np.repeat(r, num_phi) * np.sin(np.tile(phi, num_r))
        
    # interpolate the image
    interp = Bcinterp(image2d.T.flatten(), 1.0, 1.0, x_size, y_size, 0.0, 0.0)
    
    def objective(center):
        """
        Returns the peak height in radial space.
        """
        
        ri = interp.evaluate(rx + center[0], ry + center[1])
        a = np.mean( ri.reshape(num_r, num_phi), axis=1 )
        m = np.max(a)

        return -1.0 * m
    
    logger.debug('optimizing center position')
    center = optimize.fmin_powell(objective, initial_guess, xtol=pix_res, disp=0)
    logger.debug('Optimal center: %s (Delta: %s)' % (center, initial_guess-center))
    
    return center
        

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
    

def rand_pairs(numItems,numPairs):
	seed()
	i = 0
	pairs = []
	while i < numPairs:
		ind1 = randrange(numItems)
		ind2 = ind1
		while ind2 == ind1:
			ind2 = randrange(numItems)
		pair = [ind1,ind2]
		if pairs.count(pair) == 0:
			pairs.append(pair)
			i += 1
	return pairs


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
    in_area = nxutils.points_inside_poly(test_points, ordered_pts)
    
    assert len(in_area) == test_points.shape[0]
    
    return in_area

        
