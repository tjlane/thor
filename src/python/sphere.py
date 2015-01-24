#!/usr/bin/env python

"""
Code pertaining to spherical harmonics and other spherical represenataions
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

import numpy as np
from scipy import sparse
from scipy import special
from scipy.interpolate import interpn

from thor.math2 import arctan3


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
    

def radial_sph_harm(n, l, m, u, theta, phi, Rtype='legendre', **kwargs):
    """
    Return the radially augmented spherical harmonic Y_nlm evaluated at
    (r, theta, phi).
    
    These functions are the regurlar spherical harmonics with a Legendre
    radial component:
    
        Y_nlm = R_n * Y_lm
        R_n(u) = u^n
        
    Where P_n are the Legendre polynomials
        
    """
    
    if not (theta.shape == phi.shape) and (theta.shape == u.shape):
        raise ValueError('`u`, `theta`, and `phi` must all have the same shape!')
    
    Ylm = sph_harm(l, m, theta, phi)
    
    if Rtype == 'monomial':
        Rn = np.power(u, n)
        
    elif Rtype == 'legendre':
        
        if not 'delta' in kwargs.keys():
            raise TypeError('with Rtype=legendre, you must provide a `delta` '
                            'kwarg parameter.')
        if not 'radius' in kwargs.keys():
            raise TypeError('with Rtype=legendre, you must provide a `radius` '
                            'kwarg parameter.')
        
        d  = kwargs['delta'] 
        r0 = kwargs['radius']
        f  = special.legendre(n)
        
        Rn = np.sqrt( (2.0*n+1.0) / (2.0*d) ) / (d * u + r0) * f(u)
        
    else:
        raise NotImplementedError('No known implementation for radial basis '
                                  'function type: `%s`' % Rtype)
    
    return Rn * Ylm


def interp_grid_to_spherical(grid, radii, num_phi, num_theta, 
                             grid_origin=(0,0,0), return_spherical_coords=False,
                             theta_offset=0.0):
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

    thetas = np.arange(0.0, 2.0*np.pi, 2.0*np.pi / num_theta) + theta_offset
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


class SphHarmGrid(object):
    """
    Implements the algorithm of Misner for evaluating a three dimensional
    spherical harmonic expansion of a scalar filed sampled on a regular
    cubic grid. The specific implementation here is the one due to Rupright.
    """
    
    def __init__(self, grid_dimensions, origin, radii, L_max, Rtype='flat'):
        """
        
        Parameters
        ----------
        grid_dimensions : tuple
            A 3-tuple describing the number of grid points along the x/y/z
            dimensions, respectively. Assumes a regular square grid.
        
        origin : tuple
            A 3-tuple (or list or array) defining the center of the grid from
            which to expand in harmonics. This value is given in units of grid
            spacing.
            
        radii : ndarray, float
            The radii at which to evaluate the spherical harmonic expansion,
            in units of grid spacing.
            
        L_max : int
            The spherical harmonic order (ell) cutoff at which to truncate the
            expansion.
        """
        
        
        self._delta = 5.0 / 4.0
        self._radii = radii
        self._L_max = L_max
        self._grid_dimensions = grid_dimensions
        
        # different types of radial basis functions, with orders chosen to
        # achieve high accuracy while maintaining efficiency
        
        self.Rtype = Rtype
        
        if Rtype == 'flat':
            self._n_values = [0]
        elif Rtype == 'legendre':
            self._n_values = [0,2,4]
        elif Rtype == 'monomial':
            self._n_values = [0,1,2,3]
        else:
            raise ValueError('No known radial type (Rtype): %s' % Rtype)
            
        
        self._num_grid_points = np.product(self._grid_dimensions)
        
        if not len(origin) == 3:
            raise ValueError('`origin` must be a 3-tuple (x,y,z)')
        self._origin = origin
        
        
        # ------- compute & cache some geometric properties of the grid
        # figure out how far each grid point is from the center
        gd = self._grid_dimensions # just shorthand
        
        self._nxyz   = np.mgrid[:gd[0],:gd[1],:gd[2]]                # index of each pt
        self._c_nxyz = self._nxyz - self._origin[:,None,None,None]   # center subtracted
        
        self._grid_distances = np.sqrt(np.sum( np.square(self._c_nxyz), axis=0))
        self._u_nxyz = self._c_nxyz / (self._grid_distances + 1e-16) # unit vectors
        
        self._theta = np.arctan2(self._u_nxyz[1,:,:,:], self._u_nxyz[0,:,:,:])
        self._phi   = np.arccos(self._u_nxyz[2,:,:,:])
                
        
        # ------- compute the grid projections
        # this will be keyed by (r, l, m), and return a sparse vector of length
        # `self._num_grid_points`, which describes how to project a flattened
        # 3d grid to obtain a spherical harmonic coefficient
        self._projections = {}
        self._compute_projections()
        
        return
       
       
    @property 
    def grid_dimensions(self):
        return self._grid_dimensions
    
        
    @property
    def radii(self):
        return self._radii
    
        
    @property
    def num_radii(self):
        return len(self.radii)
    
        
    @property
    def L_max(self):
        return self._L_max
        
    
    @property
    def n_values(self):
        return self._n_values
    
        
    def __call__(self, grid):
        """
        Evaluate the spherical harmonic expansion of the scalar field `grid`.
        
        Parameters
        ----------
        grid : np.ndarray
            The 3d scalar field. Must be the same shape as 
            `SphHarmGrid.grid_dimensions`.
            
        Returns
        -------
        coefficients : np.ndarray
            A three dimensional array of coefficients, indexed by (radii, l, m).
            Note that the 'm' values wrap around, so that for m < 0 the 
            index is 2l+1-m.
        """
        
        if not np.all(grid.shape == self.grid_dimensions):
            raise ValueError('Grid is incorrect shape. Expected %s, got %s'
                             '.' % (str(self.grid_dimensions), str(grid.shape)) )
        
        coefficients = np.zeros( (len(self.radii), self._L_max, 
                                  2*self._L_max + 1),
                                dtype=np.complex128 )
        
        for ir, r in enumerate(self.radii):
            for l in range(self._L_max):
                for m in range(-l, l+1):
                    coefficients[ir, l, m] = self._evaluate_projection(grid, r, l, m)
        
        return coefficients
        
        
    def _compute_shell(self, radius):
        """
        Determine which points are in the radial shell (set "S")
        """
        
        u = (self._grid_distances - radius) / self._delta
        S = ( np.abs(u) < 1.0 )
        u[ np.logical_not(S) ] = 0.0
        
        assert np.all(u >= -1.0)
        assert np.all(u <=  1.0)
        
        return S, u
        
    
    def _projection_index(self, n, l, m):
        """
        Retrieve the column index of (nlm) in the projections/P matrix
        """
        
        if not n in self.n_values:
            raise ValueError('n=%d not in n_values (%s)' % (n, str(self.n_values)))
        else:
            n_ind = self.n_values.index(n)
        
        # this index should map (n,l,m) to 0,1,2,... in the expected order
        i = (n_ind * self.L_max**2) + l * (l+1) + m
        
        if (i < 0) or (i > len(self.n_values) * self.L_max ** 2):
            raise RuntimeError('(nlm) index (%d,%d,%d) out of bounds' % (n, l, m))
        
        return i
    
        
    def _compute_projections(self):
        """
        Compute the spherical harmonic projection vectors for the specific grid.
        """
        
        for ir, r in enumerate(self._radii):
            
            logger.debug('Computing projections for radius: %f' % r)
            S, u = self._compute_shell(r)
            
            # determine weights, tricube rule
            w = np.power(1 - np.power(np.abs(u)[S], 3), 3)
            W = np.diag( w.astype(np.complex128) )
            
            # compute Y, the N x M matrix where each column is the function
            # Y_nlm evaluated at all the points S in the shell
            
            N = np.sum(S)
            M = len(self.n_values) * self._L_max ** 2
            Y = np.zeros((N, M), dtype=np.complex128)
            logger.debug('Computing %d x %d (points x functions) design matrix '
                         'Y' % (N, M))
            
            for n in self.n_values:
                for l in range(self._L_max):
                    for m in range(-l, l+1):
                        
                        i = self._projection_index(n, l, m)
                        
                        logger.debug('computing projection for %d-th function, '
                                    '(n l m) = (%d %d %d)' % (i, n, l, m))
                                    
                        if self.Rtype == 'flat':
                            if n != 0:
                                raise ValueError('n must be 0 for Rtype `flat` (got n=%d)' % n)
                            Y[:,i] = sph_harm(l, m, self._theta[S], self._phi[S])

                        elif self.Rtype == 'monomial':
                            Y[:,i] = radial_sph_harm(n, l, m,
                                                     u[S],
                                                     self._theta[S],
                                                     self._phi[S],
                                                     Rtype='monomial')

                        elif self.Rtype == 'legendre':
                            Y[:,i] = radial_sph_harm(n, l, m, 
                                                     u[S],
                                                     self._theta[S],
                                                     self._phi[S],
                                                     Rtype='legendre',
                                                     delta=self._delta,
                                                     radius=r)
                        else:
                            raise ValueError('Unknown Rtype: `%s`' % Rtype)
                                              
                    
            # perform weighted LSQ
            G = np.dot(np.conjugate(Y).T, np.dot(W, Y))
            B = np.dot(np.conjugate(Y).T, W)
            
            assert G.shape == (M, M)
            assert B.shape == (M, N)
            
            P = np.linalg.solve(G, B)
            assert P.shape == (M, N)
        
            # store the projection for later use
            self._projections[r] = P
        
        return
        
        
    def _evaluate_projection(self, grid, radius, l, m):
        
        if (radius not in self._projections) or (l > self.L_max) or (np.abs(m) > l):
            raise KeyError('The spherical harmonic shell you requested, %s, is '
                           'not included in the parameter space of this '
                           'object.' % str( (radius, l, m) ))
                           
        S, _ = self._compute_shell(radius)
        p = np.zeros(self.grid_dimensions, dtype=np.complex128)
        
        if self.Rtype == 'monomial':
            p[S] += self._projections[radius][self._projection_index(0,l,m)]
        
        else:
            for n in self.n_values:
                # CONSIDER if m < 0: np.power(-1, m) * np.conjugate(P)
                p[S] += self._projections[radius][self._projection_index(n,l,m)]
        

        expn_coefficient = np.sum(p * grid)
        
        return expn_coefficient
    
        
    def expand_sph_harm_order(self, A_ell, ell):
        """
        Evaluate a single order term in the expansion of a scalar field in 
        spherical harmonics. Specifically, if field phi(r) has been expanded
        
            phi(r) = sum sum { A_{l,m}(|r|)  * Y_lm(Omega) }
                      l   m
                      
        then this function takes A_{ell,m} for one value of ell and returns
        
            phi_ell(r) = sum { A_{l,m}(|r|)  * Y_lm(Omega) }
                          m
                          
        evaluated at points on a regular square grid.
        
        Parameters
        ----------
        A_ell : np.ndarray, float
            A shape ( 2`ell`+1, len(`radii`) ) array of spherical harmonic
            coefficients for a single ell band.
        
        ell : int
            The value of l for which to compute the term.
            
        Returns
        -------
        grid_term : np.ndarray, complex128
            One term, representing an-ell band in the spherical harmonic
            expansion of an object.
            
        See Also
        --------
        expand_sph_harm : method
            Evaluate the expansion for all terms {ell} simultaneously.
        """
        
        l = ell # rename
        if not A_ell.shape == (2*l+1, len(self.radii)):
            raise ValueError('`A_ell` has an invalid shape. Must be a (2l+1) x'
                             ' len(radii) shape array. Got '
                             '(%s)' % str(A_ell.shape))
                             
        
        grid_term = np.zeros(self._grid_dimensions, dtype=np.complex128)
        
        for ir,r in enumerate(self._radii):
            
            S, u = self._compute_shell(r)
            
            w_grid = np.zeros_like(grid)
            w_grid[S] = np.power(1 - np.power(np.abs(u)[S], 3), 3)

            for m in range(-l, l+1):
                Y_lm = sph_harm(l, m, self._theta, self._phi)
                grid_term += coefficients[ir,l,m] * w_grid * Y_lm
            
        return grid_term
    
        
    def expand_sph_harm(self, coefficients):
        """
        Evaluate a field phi(r) on a regular grid by expanding a series of
        spherial harmonic coefficients:
        
            phi(r) = sum sum { A_{l,m}(r) * Y_lm(Omega) }
                      l   m
                      
        
        Parameters
        ----------
        coefficients : np.ndarray
            A three dimensional array of coefficients, indexed by (radii, l, m).
            Note that the 'm' values wrap around, so that for m < 0 the 
            index is 2l+1-m.
            
        Returns
        -------
        grid : np.ndarray, complex128
            The expansion evaluated on the regular grid.
        
        See Also
        --------
        expand_sph_harm_order : method
            Evaluate the expansion for only a single order {ell}.
        """
        
        if not coefficients.shape[0] == len(self.radii):
            raise ValueError('`coefficients` first dimension must be the same '
                             'length as self.radii')
            
        l_max = coefficients.shape[1]
        
        grid = np.zeros(self._grid_dimensions, dtype=np.complex128)
        for ir,r in enumerate(self._radii):
            
            S, u = self._compute_shell(r)
            
            w_grid = np.zeros_like(grid)
            w_grid[S] = np.power(1 - np.power(np.abs(u)[S], 3), 3)

            for l in range(self.L_max):
                for m in range(-l, l+1):
                    Y_lm = sph_harm(l, m, self._theta, self._phi)
                    grid += coefficients[ir,l,m] * w_grid * Y_lm
        
        return grid
    
        

    
