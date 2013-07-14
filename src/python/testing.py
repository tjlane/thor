
import os
import os.path
import functools
from pkg_resources import resource_filename

from nose import SkipTest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

# decorator to skip tests
def skip(rason):
    def wrap(test):
        @functools.wraps(test)
        def inner(*args, **kwargs):
            raise SkipTest
            print "After f(*args)"
        return inner
    return wrap
    
# decorator to mark tests as expected failure
def expected_failure(test):
    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except BaseException:
            raise SkipTest
        else:
            raise AssertionError('Failure expected')
    return inner
    
# decorator to skip GPU tests
def gputest(rason):
    def wrap(test):
        @functools.wraps(test)
        def inner(*args, **kwargs):
            if 'GPU' not in globals().keys():
                try:
                    import gpuscatter
                    GPU = True
                except ImportError as e:
                    GPU = False
            if not GPU: 
                raise SkipTest
            print "After f(*args)"
        return inner
    return wrap
    
    
def ref_file(filename):
    """
    Returns the egg-ed path to the reference data `filename`.
    """
    fn = resource_filename('odin', os.path.join('../reference', filename))
    
    # if that doesn't work, try to find the refdata in the git repo
    # this is mostly for Travis CI
    if not os.path.exists(fn):
        
        # check if we're in the base odin dir of the repo
        if os.path.exists('reference'):
            fn = os.path.join(os.path.abspath(os.curdir), 'reference', filename)
            
        # else if we're in  the odin/test directory in the repo
        elif os.path.exists('../reference'):
            fn = os.path.join(os.path.abspath(os.curdir), '../reference', filename)
            
        else:
            logger.info('working dir: %s' % os.path.abspath(os.curdir))
            raise RuntimeError('Could not locate reference file: %s' % filename)

    return fn
    
    
def assert_dict_equal(t1, t2, decimal=6):
    """
    Assert two dicts are equal.
    This method should actually
    work for any dict of numpy arrays/objects
    """

    # make sure the keys are the same
    eq_(t1.keys(), t2.keys())

    for key, val in t1.iteritems():        
        # compare numpy arrays using numpy.testing
        if isinstance(val, np.ndarray):
            if val.dtype.kind ==  'f':
                # compare floats for almost equality
                assert_array_almost_equal(val, t2[key], decimal)
            else:
                # compare everything else (ints, bools) for absolute equality
                assert_array_equal(val, t2[key])
        else:
            eq_(val, t2[key])


def assert_spase_matrix_equal(m1, m2, decimal=6):
    """Assert two scipy.sparse matrices are equal."""

    # delay the import to speed up stuff if this method is unused
    from scipy.sparse import isspmatrix
    from numpy.linalg import norm

    # both are sparse matricies
    assert isspmatrix(m1)
    assert isspmatrix(m1)

    # make sure they have the same format
    eq_(m1.format, m2.format)

    # even though its called assert_array_almost_equal, it will
    # work for scalars
    assert_array_almost_equal((m1 - m2).sum(), 0, decimal=decimal)
    
    
def brute_force_masked_correlation(x, mask, normed=True):
    """
    Performes a masked correlation in a brute-force manner. Used in 
    test/test_xray.py to test methods in xray.py.
    
    Parameters
    ----------
    x : np.ndarray
        The array to (auto) correlate. Should be one-D
        
    mask : np.ndarray, np.bool
        The mask to use. True is keep, false discard.
        
    normed : bool
        Returns the std/trace normalized correlation function, if True. Else
        no normalization.
        
    Returns
    -------
    ref_corr : np.ndarray
        The correlation function, where ref_corr[i] is the correlation between
        x and itself circlularly permuted by i indices.
    """
    
    mask = mask.astype(np.bool)
    x = x * mask.astype(np.float)
    x_bar = np.ma.masked_array( x, mask=np.logical_not(mask) ).mean()
    
    n_x = len(x)
    ref_corr = np.zeros(n_x)
    
    for delta in range(n_x):
        n_delta = 0.0
        
        for i in range(n_x):    
            j = (i + delta) % n_x

            if np.logical_and(mask[i], mask[j]):
                ref_corr[delta] += (x[i] - x_bar) * (x[j] - x_bar)
                n_delta += 1.0

        if n_delta > 0.0:
            ref_corr[delta] /= n_delta
        else:
            ref_corr[delta] = 1e-300 # zero that can be divided

    if normed:
        ref_corr /= ref_corr[0]
            
    return ref_corr
