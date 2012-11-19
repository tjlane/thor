
import os
import os.path
import functools
from pkg_resources import resource_filename

from nose import SkipTest

from numpy.testing import assert_array_almost_equal, assert_array_equal


# decorator to skip tests
def skip(rason):
    def wrap(test):
        @functools.wraps(test)
        def inner(*args, **kwargs):
            raise SkipTest
            print "After f(*args)"
        return inner
    return wrap
    
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
    fn = resource_filename('odin', os.path.join('reference', filename))
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