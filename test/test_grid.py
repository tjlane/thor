
import numpy as np
from thor import math2
from thor import grid as tg

# 90-deg around y-axis, from x/y into z/y plane
axis = np.array([0.0, 1.0, 0.0])
theta = np.pi / 2.0
R = math2.ER_rotation_matrix(axis, theta)


def test_rotated_slice():
    extent = (9,9)
    rs = tg._rotated_slice(extent, 1.0, R)
    assert rs.shape == (9,9,3)
    x_total = np.sum(np.abs(rs[:,:,0]))
    #assert x_total < 1e-8, x_total # NOT VALID NEED BETTER

def test_grid_loadings():
    extent = (2,2)
    R2 = math2.ER_rotation_matrix(np.random.randn(3), theta)
    rs = tg._rotated_slice(extent, 1.0, R2)
    l = tg._grid_loadings(rs)
    ls = l.sum(2)
    assert np.sum(np.abs(ls - np.ones(extent))) < 1e-8, ls

def test_slice_grid():
    v = np.random.randn()
    grid = v * np.ones([3,4,5]) 
    R2 = math2.ER_rotation_matrix(np.random.randn(3), theta)
    s = tg.slice_grid(grid, R2)
    assert np.all( np.abs( s - v) < 1e-8 )

def test_add_slice_to_grid():
    v = np.random.randn()
    gs = v * np.ones([3,4])
    grid = np.zeros([3,4,2])
    R2 = math2.ER_rotation_matrix(np.random.randn(3), theta)
    g2 = tg.add_slice_to_grid(grid, gs, R2)
    # BAD TESTS TODO
    assert (g2.sum() / v) > 0.0
    assert (g2.sum() / v) < 20.0



if __name__ == '__main__':
    test_rotated_slice()
    test_grid_loadings()
    test_slice_grid()
    test_add_slice_to_grid()

