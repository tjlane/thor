
import numpy as np

loading_indices = [ np.array([0, 0, 0]), 
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                    np.array([1, 0, 1]),
                    np.array([0, 1, 1]),
                    np.array([1, 1, 0]),
                    np.array([1, 1, 1])]


def _rotated_slice(extent, res, rot_matrix):
    """
    Returns
    -------
    slice_xyz : np.ndarray
        An n x m x 3 grid of the x/y/z coordinates of a rotated x/y plane.
    """

    # slice_xyz is N x 3 array of xyz coords
    slice_xyz = res * np.mgrid[0:extent[0], 0:extent[1]]
    slice_xyz = np.moveaxis(slice_xyz, 0, 2)
    slice_xyz = np.concatenate([slice_xyz, np.zeros([extent[0], extent[1], 1])], axis=2)

    # rotate slice
    com = slice_xyz.mean(0).mean(0)
    slice_xyz = np.dot(slice_xyz - com[None,None,:], rot_matrix) + com[None,None,:]

    return slice_xyz


def _grid_loadings(slice_xyz):
    """
    Returns the n x m x 8 loading array
    """

    r_xyz = slice_xyz % 1.0

    loadings = np.zeros([slice_xyz.shape[0], slice_xyz.shape[1], 8])

    # ugly, but explicit!
    loadings[:,:,0] = (1.0 - r_xyz[:,:,0]) * (1.0 - r_xyz[:,:,1]) * (1.0 - r_xyz[:,:,2]) # 000
    loadings[:,:,1] = (      r_xyz[:,:,0]) * (1.0 - r_xyz[:,:,1]) * (1.0 - r_xyz[:,:,2]) # 100
    loadings[:,:,2] = (1.0 - r_xyz[:,:,0]) * (      r_xyz[:,:,1]) * (1.0 - r_xyz[:,:,2]) # 010
    loadings[:,:,3] = (1.0 - r_xyz[:,:,0]) * (1.0 - r_xyz[:,:,1]) * (      r_xyz[:,:,2]) # 001
    loadings[:,:,4] = (      r_xyz[:,:,0]) * (1.0 - r_xyz[:,:,1]) * (      r_xyz[:,:,2]) # 101
    loadings[:,:,5] = (1.0 - r_xyz[:,:,0]) * (      r_xyz[:,:,1]) * (      r_xyz[:,:,2]) # 011
    loadings[:,:,6] = (      r_xyz[:,:,0]) * (      r_xyz[:,:,1]) * (1.0 - r_xyz[:,:,2]) # 110
    loadings[:,:,7] = (      r_xyz[:,:,0]) * (      r_xyz[:,:,1]) * (      r_xyz[:,:,2]) # 111

    return loadings


def _grid_indices(slice_xyz):
    """
    an N x 3 array of integers: idx
    such that: grid[idx] = lower left corner
    """
    return (slice_xyz // 1).reshape(-1, 3).astype(np.int)


def slice_grid(grid, rot_matrix, slice_res=1.0, slice_extent=None):
    """
    Compute the planar slice of a cubic `grid`.

    The slice is a linearly interpolated planar section through a 3d volume.

    Parameters
    ----------
    grid : np.ndarray
        A 3-d cubic grid to slice.

    rot_matrix : np.ndarray
        A 3 x 3 rotation matrix that dictates how the grid should be rotated
        with respect to the grid before slicing. The grid starts in the x/y
        plane, with normal vector pointing to +z.

    slice_res : float
        The resolution of the slice, in unit of the grid spacing (ie
        slice_res = 1.0 means the slice is a 2d cubic array with each side
        length 1.0 times the 3d grid spacing).

    slice_extent : 3-tuple of ints
        The size of the slice in each dimension. Default: the size of the
        x/y plane of the passed `grid`.

    Returns
    -------
    grid_slice : np.ndarray
        The rotated slice of the grid.
    """

    if len(grid.shape) != 3:
        raise ValueError('grid is not 3d: %s' % str(grid.shape))

    if rot_matrix.shape != (3,3):
        raise ValueError('rotation matrix is not 3x3!')

    if slice_extent is None:
        slice_extent = grid.shape[:-1]

    slice_xyz = _rotated_slice(slice_extent, slice_res, rot_matrix)
    idx000   = _grid_indices(slice_xyz)
    loadings = _grid_loadings(slice_xyz)

    grid_slice = np.zeros(slice_extent)
    for i,idx in enumerate(loading_indices):
        ii = np.clip( (idx000 + idx).reshape(-1,3), 0, np.array(grid.shape)-1 )

        #  n x m               nm                          --> n x m                n x m
        grid_slice += grid[ii[:,0], ii[:,1], ii[:,2]].reshape(*slice_extent) * loadings[:,:,i]

    return grid_slice


def add_slice_to_grid(grid, grid_slice, rot_matrix, slice_res=1.0):
    """
    Add the 2d array `grid_slice` to the 3d `grid` using linear interpolation.

    Parameters
    ----------
    grid : np.ndarray
        A 3-d cubic grid to slice.

    grid_slice : np.ndarray
        A 2-d slice to add to `grid`

    rot_matrix : np.ndarray
        A 3 x 3 rotation matrix that dictates how the grid should be rotated
        with respect to the grid before slicing. The grid starts in the x/y
        plane, with normal vector pointing to +z.

    slice_res : float
        The resolution of the slice, in unit of the grid spacing (ie
        slice_res = 1.0 means the slice is a 2d cubic array with each side
        length 1.0 times the 3d grid spacing).

    Returns
    -------
    grid : np.ndarray
        An updated grid with `slice` added to it.
    """

    if len(grid.shape) != 3:
        raise ValueError('grid is not 3d')

    if rot_matrix.shape != (3,3):
        raise ValueError('rotation matrix is not 3x3!')

    slice_extent = grid_slice.shape

    slice_xyz = _rotated_slice(slice_extent, slice_res, rot_matrix)
    idx000   = _grid_indices(slice_xyz)
    loadings = _grid_loadings(slice_xyz)

    for i,idx in enumerate(loading_indices):
        ii = np.clip( (idx000 + idx).reshape(-1,3), 0, np.array(grid.shape)-1 )

        #              nm                             n x m
        grid[ii[:,0], ii[:,1], ii[:,2]] += (grid_slice * loadings[:,:,i]).flatten()

    return grid



