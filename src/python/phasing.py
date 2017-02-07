#!/usr/bin/env python

"""
Implement the 
"""

import numpy as np

from thor import math2
from sphhrmgrid import SphHarmGrid


class ErrorReducer(object):
    
    
    def __init__(self, support, legendre_coefficients, q_values,
                 enforce_positivity=True, enforce_reality=True, constraints=[]):
                 
        # initialize necessary values
        self._support = support
        self._legendre_coefficients = legendre_coefficients
        self._q_values = q_values
        
        self._density = self._random_density()
        self._delta = np.inf
        
        self._constraints = constraints
        
        self._factor_coefficients(legendre_coefficients)
        
        # initialize a spherical harmonic projection class
        self._sph_harm_projector = SphHarmGrid(self._density.shape, 
                                               np.array(self._density.shape)/2.0, # ft grid origin
                                               self.q_values,
                                               self.order_cutoff)
        
        return
        
        
    @property
    def density(self):
        return self._density
    
        
    @property
    def error(self):
        raise NotImplementedError()
        return
        
        
    @property
    def q_values(self):
        return self._q_values
    
        
    @property
    def order_cutoff(self):
        return self._legendre_coefficients.shape[0]
        
        
    @property
    def num_q(self):
        return len(self.q_values)
    
        
    @property
    def delta(self):
        """
        The change since last iteration
        """
        return self._delta
    
        
    def _random_density(self):
        rd = self._support * np.random.rand(self._support.shape)
        rd = self._enforce_constraints(rd)
        return rd
    
    
    @staticmethod
    def _S(ell):
        r"""
        Return a matrix S_ell composed of {-1, 0, 1} such that:
        
            A_ell = S_ell * B_ell
            
        Where:
        
            -- A_ell is a matrix of spherical harmonic coefficients for
                     m = -ell, ..., ell
            -- B_ell is the matrix A_ell truncated at m >= 0
            
        The easiest way to think about S is as a matrix who's elements are given
        by
        
            S_m',m = (-1)^m * delta_|m'|, m
            
        where m' is an index for A and m is an index for B.
            
        Note that because in this class we're employing the python convention
        of "wrapping" negative indices to deal with m < 0, "S" is a
        (2l+1) x (l+1) matrix with two diagonals:
        
                        2l + 1
                  -----------------
                  |\        
                  | \       \ 
         S^T: l+1 |  \       \         [sorry, right now this is a
                  |   \       \            transposed view !]
                  |    \       \
              
                  ^--------^-------^
                      D1       D2
              m:    (0->l)  (-l->-1)
              size:  l+1        l
              
        Here we construct both "diagonal" components separately then stitch them
        together.
        """
        
        d1 = np.ones(ell+1)
        d1[1::2] = -1
        D1 = np.diag(d1)
        assert D1.shape == (ell+1, ell+1)
        
        d2 = np.ones(ell)
        d2[::-2] = -1
        D2 = np.hstack(( np.zeros((ell,1)), np.diag(d2) )) # ell = 0
        assert D2.shape == (ell, ell+1)
        
        S_ell = np.vstack(( D1, D2 ))
        assert S_ell.shape == (2*ell+1, ell+1)
        
        return S_ell
    
    
    def _factor_coefficients(self, legendre_coefficients, verify=True):
        """
        Factor the experimental coefficients into one solution A_ell for
        all observed ell. Store the solution in dictionary:
        
            self._A_ell_expt[l] --> (2l+1) x n_q array
            
            
        Parameters
        ----------
        legendre_coefficients : np.ndarray
            The (ell, q, q) array of legendre coefficients, presumably from
            an experiment.
            
        verify : bool
            Perform some extra computation to check the result.
        """
        
        self._A_ell_expt = {}
        for l in range(legendre_coefficients):
            logger.debug('Factoring experimental coefficients for ell=%d' % l)
            
            S = self._S(l)
            Q = np.dot(S.T, S)
            
            C = legendre_coefficients[l,:,:]
            
            # check this is OK norm
            def objective(b):
                B = b.reshape(B_shape)
                return np.linalg.norm( np.dot( np.conj(B.T), np.dot(Q, B)) - C )
            first_deriv = lambda b: np.dot(np.conj(b.reshape(B_shape).T), Q + Q.T)
            secnd_deriv = lambda b: Q + Q.T
            
            B_shape = (ell+1, self.num_q)
            b0 = (np.random.randn(*B_shape) + 1j * np.random.randn(*B_shape)).flatten()
            
            if verify:
                err = optimize.check_grad(objective, first_deriv, b0.flatten())
                norm_err = (err / np.prod(B_shape))
                if norm_err > 1e-6:
                    raise RuntimeError('Analytical derivative does not pass '
                                       'numerical double-check. RMS error per '
                                       'value: %f' % norm_err)
            
            # not sure if CG is most efficient
            b_opt = optimize.fmin_cg(objective, b0, fprime=first_deriv)
            A_ell = np.dot(S, b_opt.reshape(B_shape))
            
            if verify:
                err = np.linalg.norm( np.dot(np.conj(A_ell.T), A_ell) - C )
                if err > 1e-6:
                    raise RuntimeError('Factorization inaccurate. ||A*A - C||'
                                       ' = %f > 1e-6' % err)
            
            self._A_ell_expt[l] = A_ell
        
        return
    
        
    def realspace_projection(self, density):
        """
        Apply the real-space constraints and return an updated density
        """
        
        if enforce_reality:
            self._density = np.real(self._density)
            
        if enforce_positivity:
            self._density[self._density < 1e-8] = 1e-8
            
        if len(self._constraints) > 0:
            for c in self._constraints:
                self._density = c(self._density)
        
        return updated_density
    
        
    def fourier_projection(self, density):
        
        ft_density = np.fft(density)
        
        # -m wraps around array, but that should be OK....
        # --> self._A_ell_expt[l] is shape (2l+1) x n_q  = (m x q)
        #     ft_coefficients[:,l,:] is shape n_q x (2l+1) = (q x m) (tranposed later)
        ft_coefficients = self._sph_harm_projector(ft_density)
        
        
        # zero out the array, use it to store the next iter
        ft_density[:,:,:] = 0.0 + 1j * 0.0
        
        for l in range(self.order_cutoff):
            
            A_ell_model = ft_coefficients[:,l,:].T # (2l+1) x n_q  = (m x q)
        
            # find U that rotates the experimental vectors into as close
            # agreement as possible as the model vectors
            U = math2.kabsch(self._A_ell_expt[l], A_ell_model)
    
            # update: k --> k+1
            A_ell_prime = np.dot(self._A_ell_expt[l], U)
            ft_density_prime += self._sph_harm_projector.expand_sph_harm_order(A_ell_prime, l)
            
        updated_density = np.ifft(ft_density)
            
        return updated_density
    
        
    def iterate(self, iterations=1):
        """
        Run the algorith for `iterations` steps. Currently implemented is error
        reduction.
        
        Parameters
        ----------
        iterations : int
            The number of iterations to perform
            
        Notes
        -----
        A good way to implement a new algorithm for phasing is to subclass this
        class and overwrite this method.
        """
        
        density = self._density.copy()
        
        for i in range(iterations):
            density = self.realspace_projection(density)
            density = self.fourier_projection(density)
        
        self._delta = np.sum(np.abs( init_density - self._density ))
        
        return
        
        
    def converge(self, tolerance=1e-6):
        """
        Iterate until the algorithm converges, e.g. absolute changes in the
        electron density map are less than `tolerance`.
        
        Parameters
        ----------
        tolerance : float
            The tolerance for convergance.
        """
        
        while self.delta < tolerance:
            self.iterate()
        
        return
    

# ------------------------------------------------------------------
# create new algorithms by subclassing ErrorReducer
# the iterate() method is likely all that needs to be overwritten...
# ------------------------------------------------------------------

# class HIO(ErrorReducer):
#     
#     def iterate(self, iterations=1):
#         """
#         Run the algorith for `iterations` steps. Currently implemented is error
#         reduction.
#         
#         Parameters
#         ----------
#         iterations : int
#             The number of iterations to perform
#             
#         Notes
#         -----
#         A good way to implement a new algorithm for phasing is to subclass this
#         class and overwrite this method.
#         """
#         
#         density = self._density.copy()
#         
#         for i in range(iterations):
#             density = self.realspace_projection(density)
#             density = self.fourier_projection(density)
#         
#         self._delta = np.sum(np.abs( init_density - self._density ))
#         
#         return


def main():
    
    # to test: -- simulate C_ell from a known model, do the decomposition, and
    #             then reconstruct it and make sure sum_m A_ell,m are good
    #          -- 

    
    return

if __name__ == '__main__':
    main()
    
