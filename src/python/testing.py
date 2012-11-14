
import os
import functools

from nose import SkipTest

# decorator to skip tests
def skip(rason):
    def wrap(test):
        @functools.wraps(test)
        def inner(*args, **kwargs):
            raise SkipTest
            print "After f(*args)"
        return inner
    return wrap