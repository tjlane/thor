
"""
Tests: src/python/xray.py
"""

import os
import sys
import warnings
import tables

import numpy as np

from scipy import stats

from thor import utils
from thor import math2
from thor import utils
from thor import xray
from thor import parse
from thor import structure
from thor import scatter
from thor.testing import (ref_file, expected_failure, 
                          brute_force_masked_correlation)
from thor.refdata import cromer_mann_params, get_cromermann_parameters

import pytest

import mdtraj
from mdtraj import Trajectory, io

from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_equal)


class TestBeam(object):

    def setup(self):
        self.n_photons = 100.0

    def test_unit_convs(self):
        beam = xray.Beam(self.n_photons, energy=1.0)
        assert_allclose(beam.wavelength, 12.398, rtol=1e-3)
        assert_allclose(beam.frequency, 2.4190e17, rtol=1e-3)
        assert_allclose(beam.wavenumber, (2.0 * np.pi)/12.398, rtol=1e-3)


class TestBasisGrid(object):

    def setup(self):
        self.p = np.array([0.0, 0.0, 1.0])
        self.s = np.array([1.0, 0.0, 0.0])
        self.f = np.array([0.0, 2.0, 0.0])
        self.shape = (10, 10)
        self.grid_list = [(self.p, self.s, self.f, self.shape)]
        self.bg = xray.BasisGrid(self.grid_list)

    def test_add_grid(self):
        nbg = xray.BasisGrid()
        nbg.add_grid(*self.grid_list[0])
        assert_array_almost_equal(nbg.to_explicit(), self.bg.to_explicit())

    def test_add_using_center(self):
        center = np.array([4.5, 9, 1.0])
        nbg = xray.BasisGrid()
        nbg.add_grid_using_center(center, self.s, self.f, self.shape)
        assert_array_almost_equal(nbg.to_explicit(), self.bg.to_explicit())

    def test_num_pixels(self):
        assert self.bg.num_pixels == np.product(self.shape)

    def test_grid_corners(self):
        c = self.bg.get_grid_corners(0)
        assert_array_almost_equal(c[0,:], self.p)
        assert_array_almost_equal(c[1,:], np.array([1.0*10, 0.0, 1.0])) # slow
        assert_array_almost_equal(c[2,:], np.array([0.0, 2.0*10, 1.0])) # fast
        assert_array_almost_equal(c[3,:], np.array([1.0*10, 2.0*10, 1.0]))

    def test_get_grid(self):
        assert self.bg.get_grid(0) == self.grid_list[0]

    def test_to_explicit(self):
        ref = np.zeros((100,3))
        mg = np.mgrid[0:9:10j,0:18:10j]
        ref[:,0] = mg[0].flatten()
        ref[:,1] = mg[1].flatten()
        ref[:,2] = 1.0
        assert_array_almost_equal(self.bg.to_explicit(), ref)

    def test_grid_as_explicit(self):
        ref = np.zeros((10,10,3))
        mg = np.mgrid[0:9:10j,0:18:10j]
        ref[:,:,0] = mg[0]
        ref[:,:,1] = mg[1]
        ref[:,:,2] = 1.0
        assert_array_almost_equal(self.bg.grid_as_explicit(0), ref)
        
    def test_array_typechecking(self):
        bad_p = np.array([0.0, 0.0])
        grid_list = [(bad_p, self.s, self.f, self.shape)]
        try:
            bg = xray.BasisGrid(grid_list)
        except:
            pass
        else:
            raise Exception('should have failed : bad typecheck')


class TestDetector(object):

    def setup(self):
        self.spacing   = 0.05
        self.lim       = 10.0
        self.energy    = 0.7293
        self.n_photons = 100.0
        self.l         = 50.0
        self.d = xray.Detector.generic(spacing = self.spacing,
                                       lim = self.lim,
                                       energy = self.energy,
                                       photons_scattered_per_shot = self.n_photons,
                                       l = self.l)

    def test_implicit_to_explicit(self):
        xyz_imp = self.d.real
        self.d.implicit_to_explicit()
        assert_array_almost_equal(xyz_imp, self.d.real)

    def test_evaluate_qmag(self):
        # doubles as a test for _evaluate_theta
        x = np.zeros((5, 3))
        x[:,0] = np.random.randn(5)
        x[:,2] = self.l

        S = x.copy()
        S = S / np.sqrt( np.sum( np.power(S, 2), axis=1 ) )[:,None]
        S -= self.d.beam_vector

        b = xray.Beam(1, energy=self.energy)
        qref = b.k * np.sqrt( np.sum( np.power(S, 2), axis=1 ) )

        qmag = self.d.evaluate_qmag(x)
        assert_allclose(qref, qmag)

    def test_recpolar_n_reciprocal(self):
        q1 = np.sqrt( np.sum( np.power(self.d.reciprocal,2), axis=1) )
        q2 = self.d.recpolar[:,0]
        assert_array_almost_equal(q1, q2)

    def test_polar_space(self):

        # this is the "generic" detector in real space
        x = np.arange(-self.lim, self.lim+self.spacing, self.spacing)
        xx, yy = np.meshgrid(x, x)

        # one slice along the horizontal direction in real space
        r     = self.d.polar[:,0]
        theta = self.d.polar[:,1]
        phi   = self.d.polar[:,2]

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        assert_array_almost_equal(yy.flatten(), x)
        assert_array_almost_equal(xx.flatten(), y)

    def test_reciprocal_space(self):
        qx = self.d.reciprocal[:,0]
        qy = self.d.reciprocal[:,1]

        Shat    = self.d._unit_vector(self.d.real)
        Sx_unit = Shat[:,0]
        Sy_unit = Shat[:,1]

        assert_array_almost_equal(qx/self.d.k, Sx_unit)
        assert_array_almost_equal(qy/self.d.k, Sy_unit)

    def test_recpolar_space(self):

        # build a reference conversion, using a different geometrical calc
        ref1 = np.zeros(self.d.xyz.shape)
        hd = np.sqrt( np.power(self.d.xyz[:,0], 2) + np.power(self.d.xyz[:,1], 2) )

        # |q| = k*sqrt{ 2 - 2 cos(theta) }
        ref1[:,0] = self.d.k * np.sqrt( 2.0 - 2.0 * np.cos(self.d.polar[:,1]) )

        # q_theta = theta / 2 (one-theta convention)
        ref1[:,1] = self.d.polar[:,1] / 2.0 # not working atm

        # q_phi is the same as polar
        ref1[:,2] = self.d.polar[:,2].copy()

        assert_array_almost_equal(ref1[:,0], self.d.recpolar[:,0], err_msg='|q|')
        assert_array_almost_equal(ref1[:,1], self.d.recpolar[:,1], err_msg='theta')
        assert_array_almost_equal(ref1[:,2], self.d.recpolar[:,2], err_msg='phi')

    def test_compute_intersect(self):

        # build a simple grid and turn it into a detector
        bg = xray.BasisGrid()
        p = np.array([0.0, 0.0, 1.0])
        s = np.array([1.0, 0.0, 0.0])
        f = np.array([0.0, 1.0, 0.0])
        shape = (10, 10)
        bg.add_grid(p, s, f, shape)
        d = xray.Detector(bg, 2.0*np.pi/1.4)

        # compute a set of q-vectors corresponding to a slightly offset grid
        xyz_grid = bg.to_explicit()
        xyz_off = xyz_grid.copy()
        xyz_off[:,0] += 0.5
        xyz_off[:,1] += 0.5
        q_vectors = d._real_to_reciprocal(xyz_off)

        # b/c s/f vectors are unit vectors, where they intersect s/f is simply
        # their coordinates. The last row and column will miss, however
        intersect_ref = np.logical_and( (xyz_off[:,0] <= 9.0),
                                        (xyz_off[:,1] <= 9.0) )

        pix_ref = xyz_off[intersect_ref,:2]

        # compute the intersection from code
        pix, intersect = d._compute_intersections(q_vectors, 0) # 0 --> grid_index
        print(pix, intersect)

        assert_array_almost_equal(intersect_ref, intersect)
        assert_array_almost_equal(pix_ref, pix)

    def test_serialization(self):
        s = self.d._to_serial()
        d2 = xray.Detector._from_serial(s)
        assert_array_almost_equal(d2.xyz, self.d.xyz)

    def test_io(self):
        if os.path.exists('r.dtc'): os.system('rm r.dtc')
        self.d.save('r.dtc')
        d = xray.Detector.load('r.dtc')
        if os.path.exists('r.dtc'): os.system('rm r.dtc')
        assert_array_almost_equal(d.xyz, self.d.xyz)

    def test_q_max(self):
        ref_q_max = np.max(self.d.recpolar[:,0])
        assert_almost_equal(self.d.q_max, ref_q_max, decimal=2)


class TestShotset(object):

    def setup(self):
        self.q_values = np.array([1.0, 2.0])
        self.num_phi  = 360
        self.l = 50.0
        self.d = xray.Detector.generic(spacing=0.4, l=self.l)
        self.num_shots = 2
        self.i = np.abs( np.random.randn(self.num_shots, self.d.num_pixels) )
        self.t = Trajectory.load(ref_file('ala2.pdb'))
        self.shot = xray.Shotset(self.i, self.d)

    def test_mask_argument(self):
        # simple smoke test to make sure we can interpolate with a mask
        q_values = np.array([1.0, 2.0])
        num_phi = 360
        mask = np.random.binomial(1, 0.1, size=(self.d.num_pixels,))
        s = xray.Shotset(self.i, self.d, mask=mask)
        s.interpolate_to_polar(q_values, self.num_phi)
        
    def test_filters(self):
        # to do : work out a better test

        def test_filter(i):
            return np.zeros_like(i)

        self.shot._add_intensity_filter(test_filter)
        ip = self.shot.intensity_profile()[:,1]
        assert_allclose( np.zeros_like(ip), ip )
        
    def test_num_shots(self):
        assert self.shot.num_shots == self.num_shots
        
    def test_num_px(self):
        assert self.shot.num_pixels == self.d.num_pixels
        
    def test_intensities(self):
        assert np.all( self.shot.intensities == self.i )
        
    def test_intensities_iter_smoke(self):
        n = 0
        for i in self.shot.intensities_iter:
            n += 1
        print("n/num_shots", n, self.num_shots)
        assert n == self.num_shots
            
    def test_len(self):
        assert len(self.shot) == self.num_shots

    @pytest.mark.skip(reason="not in yet")
    def test_add(self):
        ss = self.shot + self.shot
        assert len(ss) == 2 * len(self.shot)
        
    def test_average_intensity(self):
        assert_array_almost_equal(self.i.mean(0), self.shot.average_intensity)

    # missing test: num_phi_to_values
    
    # missing test: num_phi_to_spacing

    # missing test: test_assemble (ok for now)
    
    def test_augment_mask(self):
        # smoke test only (really)
        n_masked = np.sum(self.shot.mask)
        self.shot.augment_mask(first_moment_cutoff=1.0, second_moment_cutoff=1.0)
        assert np.sum(self.shot.mask) > n_masked # make sure at least one got masked

    def test_polar_grid(self):
        pg = self.shot.polar_grid([1.0], 360)
        pg_ref = np.zeros((360, 2))
        pg_ref[:,0] = 1.0
        pg_ref[:,1] = np.arange(0, 2.0*np.pi, 2.0*np.pi/float(360))
        assert_array_almost_equal(pg, pg_ref)

    def test_polar_grid_as_cart(self):
        pg = self.shot.polar_grid(self.q_values, self.num_phi)
        pgc = self.shot.polar_grid_as_cart(self.q_values, self.num_phi)
        mag = np.sqrt(np.sum(np.power(pgc,2), axis=1))
        assert_array_almost_equal( mag, pg[:,0] )
        maxq = self.q_values.max()
        assert np.all( mag <= (maxq + 1e-6) )

    def test_interpolate_to_polar(self):
        # doubles as a test for _implicit_interpolation
        q_values = np.array([2.0, 2.67, 3.7]) # should be a peak at |q|=2.67
        t = structure.load_coor(ref_file('gold1k.coor'))
        s = xray.Shotset.simulate(t, self.d, 3, 1)
        pi, pm = s.interpolate_to_polar(q_values, self.num_phi)
        ip = np.sum(pi[0,:,:], axis=1)
        assert ip[1] > ip[0]
        assert ip[1] > ip[2]

    def test_explicit_interpolation(self):
        # doubles as a test for _explicit_interpolation
        q_values = np.array([2.0, 2.67, 3.7]) # should be a peak at |q|=2.67
        t = structure.load_coor(ref_file('gold1k.coor'))
        self.d.implicit_to_explicit()
        s = xray.Shotset.simulate(t, self.d, 3, 1)
        pi, pm = s.interpolate_to_polar(q_values, self.num_phi)
        ip = np.sum(pi[0,:,:], axis=1)
        assert ip[1] > ip[0]
        assert ip[1] > ip[2]

    @pytest.mark.skip()
    def test_interpolation_consistency(self):
        q_values = np.array([2.0, 4.0])
        de = xray.Detector.generic(spacing=0.4, force_explicit=True)
        s1 = xray.Shotset(self.i, self.d)
        s2 = xray.Shotset(self.i, de)
        p1, m1 = s1.interpolate_to_polar(q_values, self.num_phi)
        p2, m2 = s2.interpolate_to_polar(q_values, self.num_phi)
        p1 /= p1.max()
        p2 /= p2.max()
        
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(p1[0,0,1:].flatten())
        # plt.plot(p2[0,0,1:].flatten())
        # plt.show()
        
        assert_allclose(p1[0,0,1:].flatten(), p2[0,0,1:].flatten(), err_msg='interp intensities dont match',
                        rtol=1.0, atol=0.5)
        
    def test_multi_panel_interp(self):
        # regression test ensuring detectors w/multiple basisgrid panels
        # are handled correctly

        t = structure.load_coor(ref_file('gold1k.coor'))
        q_values = np.array([2.66])
        multi_d = xray.Detector.load(ref_file('lcls_test.dtc'))
        num_phi = 1080
        num_molecules = 1

        # generate a set of random numbers that we can use to make sure the
        # two simulations have the same molecular orientation (and therefore)
        # output
        rfloats = np.random.rand(num_molecules, 3)

        # --- first, scatter onto a perfect ring
        q_grid = xray._q_grid_as_xyz(q_values, num_phi, multi_d.k)

        ring_a = scatter.simulate_atomic(t, num_molecules, q_grid, dont_rotate=True)
        ring_i = np.abs(ring_a)
        perf = xray.Rings(q_values, ring_i[None,None,:], multi_d.k)

        # --- next, to the full detector
        q_grid2 = multi_d.reciprocal
        real_a = scatter.simulate_atomic(t, num_molecules, q_grid2, dont_rotate=True)
        real_i = np.abs(real_a)

        # interpolate
        ss = xray.Shotset(real_i, multi_d)
        real = ss.to_rings(q_values, num_phi)

        # count the number of points that differ significantly between the two
        diff = ( np.abs((perf.polar_intensities[0,0,:] - real.polar_intensities[0,0,:]) \
                 / (real.polar_intensities[0,0,:] + 1e-300) ) > 1e-2)
        print(np.sum(diff))
        assert np.sum(diff) < 300

    def test_i_profile(self):
        # doubles as a test for intensity_maxima()
        t = structure.load_coor(ref_file('gold1k.coor'))
        s = xray.Shotset.simulate(t, self.d, 5, 1)
        p = s.intensity_profile()
        m = s.intensity_maxima()
        assert np.any(np.abs(p[m,0] - 2.67) < 1e-1) # |q| = 2.67 is in {maxima}

    def test_rotated_beam(self):
        # shift a detector up (in x) a bit and test to make sure there's no diff
        t = structure.load_coor(ref_file('gold1k.coor'))
        s = xray.Shotset.simulate(t, self.d, 5, 1)
            
        sh = 50.0 # the shift mag
        xyz = self.d.xyz.copy()
        shift = np.zeros_like(xyz)
        shift[:,0] += sh
        beam_vector = np.array([ sh/self.l, 0.0, 1.0 ])
        
        # note that the detector du is further from the interaction site
        du = xray.Detector(xyz + shift, self.d.k, beam_vector=beam_vector)
        su = xray.Shotset.simulate(t, du, 5, 1)
        
        p1 = s.intensity_profile(q_spacing=0.05)
        p2 = su.intensity_profile(q_spacing=0.05)

        p1 /= p1.max()
        p2 /= p2.max()
        p1 = p2[:10,:]
        p2 = p2[:p1.shape[0],:]
        
        assert_allclose(p1, p2, rtol=0.1)

    def test_sim(self):
        # todo
        shot = xray.Shotset.simulate(self.t, self.d, 512, 1)

    # def test_simulate_cpu_only(self):
    #     # todo
    #     d = xray.Detector.generic(spacing=0.6)
    #     x = xray.Shotset.simulate(self.t, d, 1, 1)
    # 
    # def test_simulate_gpu_only(self):
    #     # todo
    #     d = xray.Detector.generic(spacing=0.6)
    #     x = xray.Shotset.simulate(self.t, d, 512, 1)
    # 
    # def test_simulate_gpu_and_cpu(self):
    #     # todo
    #     d = xray.Detector.generic(spacing=0.6)
    #     x = xray.Shotset.simulate(self.t, d, 513, 1)

    def test_to_rings(self):
        
        t = structure.load_coor(ref_file('gold1k.coor'))
        shot = xray.Shotset.simulate(t, self.d, 1, 2)

        shot_ip = shot.intensity_profile(0.1)
        q_values = shot_ip[:,0]        
        rings = shot.to_rings(q_values)
        assert rings.num_shots == shot.num_shots
        
        rings_ip = rings.intensity_profile()
        
        # normalize to the 6th entry, and discard values before that
        # which are usually just large + uninformative
        rings_ip[:,1] /= rings_ip[5,1]
        shot_ip[:,1] /= shot_ip[5,1]
        
        # for some reason assert_allclose not working, but this is
        x = np.sum( np.abs(rings_ip[5:,1] - shot_ip[5:,1]) )
        x /= float(len(rings_ip[5:,1]))
        print(x)
        assert x < 0.2 # intensity mismatch
        assert_allclose(rings_ip[:,0], shot_ip[:,0], err_msg='test impl error')
        
    def test_polar_mask_conversion(self):

        # make a real mask that is a circle, and then it should be easy to check
        # that the polar mask is correct
        
        q_cutoff_index = 5
        q_values = np.arange(3.5, 4.5, 0.1)
        q_cutoff = q_values[q_cutoff_index]
        
        rm = np.ones(self.d.num_pixels, dtype=np.bool)
        rm[self.d.recpolar[:,0] < q_cutoff] = np.bool(False)
        
        shot = xray.Shotset(self.i, self.d, mask=rm)
        r = shot.to_rings(q_values)
        
        ref_pm = np.ones((len(q_values), r.num_phi), dtype=np.bool)
        ref_pm[:q_cutoff_index+1,:] = np.bool(False)
        
        print("num px masked", ref_pm.sum(), r.polar_mask.sum())
        assert ref_pm.sum() == r.polar_mask.sum() # same num px masked
        assert_array_equal(ref_pm, r.polar_mask)

    def test_to_rings_on_disk(self):
        # this test uses the Rings `rings_filename` flag
        
        t = structure.load_coor(ref_file('gold1k.coor'))
        shot = xray.Shotset.simulate(t, self.d, 1, 1)
        q_values = [1.0, 2.0]
        rings_ref = shot.to_rings(q_values)
        
        if os.path.exists('tmp.ring'): os.remove('tmp.ring')
        shot.to_rings(q_values, rings_filename='tmp.ring')
        rings = xray.Rings.load('tmp.ring')
        
        assert_array_almost_equal(rings_ref.polar_intensities,
                                  rings.polar_intensities)
                                  
        if os.path.exists('tmp.ring'): os.remove('tmp.ring')

    def test_io(self):
        if os.path.exists('test.shot'): os.remove('test.shot')
        self.shot.save('test.shot')
        s = xray.Shotset.load('test.shot')
        assert_allclose( s.intensity_profile(),
                         self.shot.intensity_profile(), rtol=1e-3 )
        if os.path.exists('test.shot'): os.remove('test.shot')
                                  
    def test_io_memforce(self):
        if os.path.exists('test.shot'): os.remove('test.shot')
        self.shot.save('test.shot')
        s = xray.Shotset.load('test.shot', force_into_memory=True)
        assert_allclose(s.intensity_profile(),
                        self.shot.intensity_profile(), rtol=1e-2 )
        if os.path.exists('test.shot'): os.remove('test.shot')
                                  
    def test_io_subset(self):
        if os.path.exists('test.shot'): os.remove('test.shot')
        self.shot.save('test.shot')
        s = xray.Shotset.load('test.shot', to_load=[0])
        print(s.num_shots)
        assert s.num_shots == 1
        if os.path.exists('test.shot'): os.remove('test.shot')
        
    # missing test : load from cxi
    
    def test_fromfiles(self):
        ss = xray.Shotset.fromfiles( [ref_file('test_cbf.cbf'),
                                      ref_file('test_cbf2.cbf')] )
        assert ss.num_shots == 2
        n = 0
        for i in ss.intensities:
            n += 1
        assert ss.num_shots == n
        assert np.sum(ss.mask) > 1000
        
    def test_write_cxidb(self):
        # smoke test for now
        self.shot.save_as_cxi('testx.cxi')
        if os.path.exists('testx.cxi'): os.remove('testx.cxi')
        
    def test_append(self):
        if os.path.exists('test.shot'): os.remove('test.shot')

        self.shot.save('test.shot')
        ss = xray.Shotset.load('test.shot', force_into_memory=True)

        self.shot.append(ss, allow_disk_modification=True)
        assert self.shot.num_shots == 2 * self.num_shots

        if os.path.exists('test.shot'): os.remove('test.shot')
    
        
class TestRings(object):

    def setup(self):
        self.q_values  = np.array([1.0, 2.0])
        self.num_phi   = 360
        self.traj      = Trajectory.load(ref_file('ala2.pdb'))
        self.num_shots = 4
        self.rings     = xray.Rings.simulate(self.traj, 1, self.q_values,
                                             self.num_phi, self.num_shots) # 1 molec

    def test_simulation(self):
        rings = xray.Rings.simulate(self.traj, 1, self.q_values,
                                    self.num_phi, 1) # 1 molec, 1 shots
        
    def test_polar_intensities_type(self):
        assert self.rings._polar_intensities_type == 'array'
        
    def test_polar_intensities(self):
        pi = self.rings.polar_intensities
        assert pi.shape == (self.num_shots, len(self.q_values), self.num_phi)
        
    def test_polar_intensities_iter(self):
        n = 0
        for x in self.rings.polar_intensities_iter:
            n += 1
        print("n / num_shots", n, self.num_shots)
        assert n == self.num_shots
        
    def test_polar_intensities_consistency(self):
        pi = self.rings.polar_intensities.copy()
        n = 0
        for x in self.rings.polar_intensities_iter:
            assert np.all(pi[n,:,:] == x[:,:])
            n += 1
        
    def test_filters(self):
        # to do : work out a better test
        
        def test_filter(i):
            return np.zeros_like(i)
        
        self.rings._add_intensity_filter(test_filter)
        assert_allclose( np.zeros(self.rings.num_q), 
                         self.rings.intensity_profile()[:,1] )
        
    def test_num_shots(self):
        assert self.rings.num_shots == self.num_shots
        
    def test_phi_values(self):
        assert len(self.rings.phi_values) == self.num_phi
    
    def test_q_values(self):
        assert np.all(self.rings.q_values == self.q_values)
    
    def test_num_phi(self):
        assert self.rings.num_phi == self.num_phi
        
    def test_num_datapoints(self):
        assert self.rings.num_datapoints == self.num_phi * len(self.q_values)

    def test_cospsi(self):
        cospsi = self.rings.cospsi(1.0, 1.0)
        assert_allclose(cospsi[0], 1.0, rtol=0.01, err_msg='0 fail')
        theta_max = np.pi/2. + np.arcsin( 1.0 / (2.*self.rings.k) )
        assert_allclose(cospsi.min(), np.cos(2.0 * theta_max), rtol=0.01, err_msg='pi fail')

    def test_q_index(self):
        assert self.rings.q_index(self.q_values[0]) == 0
        assert self.rings.q_index(self.q_values[1]) == 1
        
    def test_correct_polarization(self):
        
        yaxis_polarization = 0.99 # chosen for use w/real data below
        
        # two part test -- first take a real (polarized) ring and make sure
        # it gets corrected
        
        # TEST NOT IN (todo) -- the experimental data are quite noisy
        
        # second part of the test is to check it against this stable ref implmt
        # note this isn't really a correction, since we're doing it to
        # simulation data w/o any polarization
        
        # --> dermen's code below (reference implementation)
        
        out_of_plane = yaxis_polarization
        
        qs = self.rings.q_values
        wave = 2. * np.pi / self.rings.k
        ref_i = self.rings.polar_intensities.copy()
        phis = self.rings.phi_values
        
        for i in range( len ( qs ) ):
            q = qs[i]
            theta = np.arcsin( q*wave / 4./ np.pi)
            SinTheta = np.sin( 2 * theta )
            correctn = out_of_plane * ( 1. - SinTheta**2 * np.cos( phis )**2 )
            correctn += (1.-out_of_plane) * ( 1. - SinTheta**2 * np.sin( phis )**2 )
            ref_i[:,i,:] /= correctn
            
        # <-- end reference implementation
        
        self.rings.correct_polarization(yaxis_polarization)
        ip = self.rings.polar_intensities
        assert_allclose(ip / ip[None,None,0], ref_i / ref_i[None,None,0])

    def test_intensity_profile(self):
        q_values = [2.4, 2.67, 3.0] # should be a peak at |q|=2.67
        t = structure.load_coor(ref_file('gold1k.coor'))
        rings = xray.Rings.simulate(t, 20, q_values, self.num_phi, 1) # 3 molec, 1 shots
        ip = rings.intensity_profile()
        assert ip[1,1] > ip[0,1]
        assert ip[1,1] > ip[2,1]

    def test_brute_correlation_wo_mask(self):
        # this might be better somewhere else, but is here for now:
        # this tests the brute force implementation in testing.py used below
        
        x = np.random.randn(100) + 0.1 * np.arange(100)
        mask = np.ones(100, dtype=np.bool)

        c = brute_force_masked_correlation(x, mask, normed=False)
        c_norm = brute_force_masked_correlation(x, mask, normed=True)
        
        ref = np.zeros(100)
        for delta in range(100):
            ref[delta] = np.sum( (x - x.mean()) * \
                                 (np.roll(x, delta) - x.mean()) ) \
                                  / float(len(x))

        assert_allclose(ref / ref[0], c_norm, err_msg='normalized fail')
        assert_allclose(ref, c, err_msg='unnormalized fail')

    def test_corr_rows_no_mask(self):

        q1 = 1.0 # chosen arb.
        q_ind = self.rings.q_index(q1)
        x = self.rings.polar_intensities[0,q_ind,:].flatten()

        ring = self.rings._correlate_rows(x, x)

        ref = np.zeros(len(x))
        for delta in range(len(x)):
            ref[delta] = np.sum( (x - x.mean()) * \
                                 (np.roll(x, delta) - x.mean()) ) \
                         / float(len(x))
        
        assert_allclose(ref / ref[0], ring / ring[0], 
                        err_msg='failed w/forced normalization')
        assert_allclose(ref, ring, err_msg='error in normalization')
        
    def test_corr_rows_w_mask(self):

        q1 = 1.0 # chosen arb.
        q_ind = self.rings.q_index(q1)
        
        x = self.rings.polar_intensities[0,q_ind,:].flatten().copy()
        x_mask = np.random.binomial(1, 0.9, size=len(x)).astype(np.bool)
        no_mask = np.ones_like(x_mask)
        
        corr = self.rings._correlate_rows(x, x, x_mask, x_mask)
        true_corr = self.rings._correlate_rows(x, x)
        ref_corr = brute_force_masked_correlation(x, x_mask, normed=False)

        normed_ref = ref_corr / ref_corr[0]
        normed_corr = corr / corr[0]
        
        # big tol, but w/a lot of masking there is a ton of noise
        assert_allclose(normed_ref, normed_corr, atol=1e-2,
                        rtol=1e-2, err_msg='correlation incorrect')

        # make sure the normalization is OK
        assert np.sum( np.abs(corr - ref_corr) ) / (corr[1]*float(len(corr))) < 1e-2

        # make sure masked and unmaksed are somewhat similar
        assert_allclose(true_corr / true_corr[0], normed_corr, atol=0.1, 
                        err_msg='masked correlation very different from unmasked version')
        
    def test_mask_nomask_consistency(self):
        
        q1 = 1.0 # chosen arb.
        q_ind = self.rings.q_index(q1)
        
        x = self.rings.polar_intensities[0,q_ind,:].flatten().copy()
        blank_mask = np.ones(x.shape[0], dtype=np.bool)
        
        corr_nomask = self.rings._correlate_rows(x, x)
        corr_mask   = self.rings._correlate_rows(x, x, blank_mask)
        corr_mask2  = self.rings._correlate_rows(x, x, blank_mask, blank_mask)

        assert_allclose(corr_mask, corr_mask2)
        assert_allclose(corr_mask, corr_nomask)
        
    def test_fft_vs_cpp(self):
        
        q1 = 1.0 # chosen arb.
        q_ind = self.rings.q_index(q1)
        
        x = self.rings.polar_intensities[0,q_ind,:].flatten().copy()
        mask = np.random.binomial(1, 0.9, size=len(x)).astype(np.bool)
        
        fft = self.rings._correlate_rows(x, x, mask, use_fft=True)
        cpp = self.rings._correlate_rows(x, x, mask, use_fft=True)

        assert_allclose(fft, cpp)
        
    def test_correlate_intra(self, rtol=0.1, atol=0.1):

        # test autocorrelator
        intra = self.rings.correlate_intra(1.0, 1.0, normed=True)
        assert intra.shape == (self.rings.num_phi,)
        
        q_ind = self.rings.q_index(1.0)
        ref_corr = np.zeros(self.num_phi)
        for i in range(self.num_shots):
            x = self.rings.polar_intensities[i,q_ind,:].flatten().copy()
            ref_corr += brute_force_masked_correlation(x, np.ones(len(x), dtype=np.bool), normed=True)
        ref_corr /= float(self.num_shots)
        
        assert_allclose(intra / intra[0], ref_corr / ref_corr[0], 
                        rtol=rtol, atol=atol,
                        err_msg='doesnt match reference implementation')
        #assert_allclose(intra, ref_corr, rtol=rtol, atol=atol,
        #                err_msg='doesnt match reference implementation normalization')
        
        # test norming
        assert np.abs(intra[0] - 1.0) < rtol, 'first normalized entry not 1, is: %f' % intra[0]
        intra_unnorm = self.rings.correlate_intra(1.0, 1.0, normed=False)
        assert not np.abs(intra_unnorm[0] - 1.0) < rtol
        assert_allclose(intra, intra_unnorm / intra_unnorm[0],
                        rtol=rtol, atol=atol, err_msg='normalization broken')
        
        # test cross correlator
        intra = self.rings.correlate_intra(1.0, 2.0, normed=True)
        assert not np.abs(intra_unnorm[0] - 1.0) < 1e-8 # x-corr normed when it shouldnt be
        
        # test the limit on 
        intra = self.rings.correlate_intra(1.0, 1.0, num_shots=1, mean_only=False)
        assert intra.shape == (1, self.rings.num_phi)

    def test_correlate_inter(self, rtol=0.01, atol=0.01):
        
        q = 1.0
        q_ind = self.rings.q_index(q)
        
        inter = self.rings.correlate_inter(q, q, mean_only=False, normed=False).mean(0)
        #inter = self.rings.correlate_inter(q, q, mean_only=True, normed=False)

        # reference
        ref = np.zeros(self.rings.num_phi)
        n = 0.0
        for i,j in utils.all_pairs(self.rings.num_shots):
            print('pair:', i,j)
            x = self.rings.polar_intensities[i,q_ind,:].flatten()
            y = self.rings.polar_intensities[j,q_ind,:].flatten()
            ref += self.rings._correlate_rows(x, y)
            n += 1.0
        ref /= float(n)
        
        print('tols:', rtol, atol)
        assert_allclose(ref / ref[0], inter / inter[0], rtol=rtol, atol=atol, 
                        err_msg='doesnt match reference implementation')
        assert_allclose(ref, inter, rtol=rtol, atol=atol, 
                        err_msg='normalization doesnt match reference implementation')
        
        # also smoke test random pairs
        rings2 = xray.Rings.simulate(self.traj, 1, self.q_values, self.num_phi, 3) # 1 molec, 3 shots
        inter = rings2.correlate_inter(q, q, mean_only=True, num_pairs=1)
        
    def test_correlate_inter_mean_only(self, rtol=0.001, atol=0.001):
        q = 1.0
        inter1 = self.rings.correlate_inter(q, q, mean_only=True,  normed=False)
        inter2 = self.rings.correlate_inter(q, q, mean_only=False, normed=False)
        inter2_mean = inter2.mean(axis=0)
        
        assert_allclose(inter1 / inter1[0], inter2_mean / inter2_mean[0],
                        rtol=rtol, atol=atol, 
                        err_msg='mean_only and rand pairs dont match')

        assert_allclose(inter1, inter2_mean, rtol=rtol, atol=atol, 
                        err_msg='mean_only and rand pairs non-std normalization doesnt match')
                        
        inter1 = self.rings.correlate_inter(q, q, mean_only=True,  normed=True)
        inter2 = self.rings.correlate_inter(q, q, mean_only=False, normed=True)
        inter2_mean = inter2.mean(axis=0)
        
        assert_allclose(inter1 / inter1[0], inter2_mean / inter2_mean[0], rtol=rtol, atol=atol, 
                        err_msg='mean_only and rand pairs std normalization doesnt match')
        assert np.abs( 1.0 - inter1[0] / inter2_mean[0] ) < 0.05, 'mean_only and rand pairs std normalization doesnt match'
                        
                        
    def test_correlate_difference(self, rtol=1e-4, atol=0.0):
        # compute diff corr for 2 shots
        q = 1.0
        
        r = xray.Rings.simulate(self.traj, 1, self.q_values, self.num_phi, 2)
        assert r.num_shots == 2, 'this test will only work w/a 2 shot Rings'
        
        iq = r.q_index(q)
        pi = r.polar_intensities.copy()
        d = r.correlate_difference(q, q, normed=True)
        nd = d / d[0]
        d2 = r.correlate_difference(q, q, normed=False)
        
        # regression test for in-place modification
        pi2 = r.polar_intensities.copy()
        assert np.all(pi == pi2), 'inappropriate in-place modification of intensities'
        
        # compute the reference
        minus = pi[0,iq,:] - pi[1,iq,:]
        normed_ref = brute_force_masked_correlation(minus, np.ones(len(minus), dtype=np.bool), normed=True)
        ref = brute_force_masked_correlation(minus, np.ones(len(minus), dtype=np.bool), normed=False)
        
        
        # make sure the results are close
        assert_allclose(nd, normed_ref, rtol=rtol, atol=atol, 
                        err_msg='shapes of norm difference correlator dont match')
        assert_allclose(d, normed_ref, rtol=rtol, atol=atol, 
                        err_msg='normalization constant doesnt match')
        assert_allclose(d2, ref, rtol=rtol, atol=atol, 
                        err_msg='non-normalized results mismatch')
                        
        # regression test for nans when intensity_normed=True
        d2 = r.correlate_difference(q, q, normed=True, intensity_normed=True)
        assert not np.any( np.isnan(d2) ), 'nans found when intensity_normed=True'
        
    @pytest.mark.skip()
    def test_convert_to_kam(self):
        intra = self.rings.correlate_intra(1.0, 1.0, mean_only=True)
        kam_corr = self.rings._convert_to_kam(1.0, 1.0, intra)
        assert kam_corr.shape[1] == 2
        assert kam_corr[:,0].min() >= -1.0
        assert kam_corr[:,0].max() <=  1.0
        
        half = kam_corr.shape[0] / 2
        
        assert_array_almost_equal(kam_corr[:half,0], -kam_corr[half:,0][::-1])
        # assert_array_almost_equal(kam_corr[1:half,1],  kam_corr[half:-1,1][::-1])

    def test_coefficients_smoke(self):
        order = 6
        cl1 = self.rings.legendre(self.q_values[0], self.q_values[0], order)
        assert cl1.shape == (order,)

        cl2 = self.rings.legendre_matrix(order)
        assert cl2.shape == (order, self.rings.num_q, self.rings.num_q)

    def test_legendre(self):
            
        order = 300
        q1 = 1.0
        with warnings.catch_warnings(record=True) as w:
            cl = self.rings.legendre(q1, q1, order) # keep only q1, q1 correlation
        assert len(cl) == order

        # make sure it matches up with the raw correlation
        ring = self.rings.correlate_intra(q1, q1, mean_only=True)
    
        # reconstruct the correlation function
        pred = np.polynomial.legendre.legval(self.rings.cospsi(q1, q1), cl)
        assert_allclose(pred, ring, rtol=0.1, atol=0.1)
        
    def test_correlation_significance(self):
        
        # accept null hypothesis
        fake_intra = np.random.randn(1000, 360)
        fake_inter = np.random.randn(1000, 360)
        p = self.rings.correlation_significance(1.0, 1.0, intra=fake_intra, inter=fake_inter, trim=0)
        print('accept p:', p)
        assert p > 0.01 # null hypothesis should be accepted
        
        # reject null hypothesis
        fake_intra = np.random.randn(1000, 360)
        fake_inter = np.random.randn(1000, 360) + 0.10
        p = self.rings.correlation_significance(1.0, 1.0, intra=fake_intra, inter=fake_inter, trim=0)
        print('reject p:', p)
        assert p < 0.01 # null hypothesis should be rejected
        
        # ensure univariate version gives same result as scipy
        # first when we should accept null hypothesis
        fake_intra = np.random.randn(1000, 1)
        fake_inter = np.random.randn(1000, 1)
        p = self.rings.correlation_significance(1.0, 1.0, intra=fake_intra, inter=fake_inter, trim=0)
        _, p_ref = stats.ttest_ind(fake_intra[:,0], fake_inter[:,0])
        assert_allclose(p_ref, p)
        
        # smoke test version where it computes correlators
        #p = self.rings.correlation_significance(1.0, 1.0)
        
    def test_simulate_density(self):
        # generate a rings object both from an atomic and density model and
        # ensure the correlations match
        
        num_shots = 100
        num_phi   = 1024
        
        nq = 100 # number of q vectors
        q_values = [1.0, 2.0]
        
        # atomic model
        traj = mdtraj.load(ref_file('pentagon.pdb'))        
        r1 = xray.Rings.simulate(traj, 1, q_values, num_phi, num_shots)
                                  
        # density model
        grid_dimensions = [151,] * 3
        grid_spacing = 1.0 # Angstroms
        grid = structure.atomic_to_density(traj, grid_dimensions, 
                                           grid_spacing)
                                           
        r2 = xray.Rings.simulate_density(grid, grid_spacing, num_shots, 
                                         q_values, num_phi)        
        
        # compute correlations & ensure match
        c1 = r1.correlate_intra(1.0, 1.0)
        c2 = r2.correlate_intra(1.0, 1.0)
        R = np.corrcoef(c1, c2)[0,1]
        assert R > 0.95
        
        c1 = r1.correlate_intra(2.0, 2.0)
        c2 = r2.correlate_intra(2.0, 2.0)
        R = np.corrcoef(c1, c2)[0,1]
        assert R > 0.95

    def test_io(self):
        if os.path.exists('test.ring'): os.remove('test.ring')
        self.rings.save('test.ring')
        r = xray.Rings.load('test.ring')
        assert_allclose(self.rings.polar_intensities, r.polar_intensities)
        if os.path.exists('test.ring'): os.remove('test.ring')
        
    def test_io_forcemem(self):
        if os.path.exists('test.ring'): os.remove('test.ring')
        self.rings.save('test.ring')
        r = xray.Rings.load('test.ring', force_into_memory=True)
        assert_allclose(self.rings.polar_intensities, r.polar_intensities)
        assert type(r.polar_intensities) == np.ndarray
        if os.path.exists('test.ring'): os.remove('test.ring')
    
    def test_append(self):
        if os.path.exists('test.ring'): os.remove('test.ring')
        
        self.rings.save('test.ring')
        r = xray.Rings.load('test.ring', force_into_memory=True)

        self.rings.append(r, allow_disk_modification=True)
        assert self.rings.num_shots == 2 * self.num_shots
        
        if os.path.exists('test.ring'): os.remove('test.ring')
        

class TestMisc(object):

    def test_q_values(self):

        q_values = np.array([1.0, 2.0, 3.0])
        num_phi = 360
        k = 2.0 * np.pi / 1.4

        qxyz = xray._q_grid_as_xyz(q_values, num_phi, k)

        # assert that the above vectors are the correct length
        assert np.all( np.abs( np.sqrt( np.sum( np.power(qxyz,2), axis=1 ) ) - \
                               np.repeat(q_values, num_phi)) < 1e-6 )

    # this test is not working quite right, it fails a lot
    # maybe we can make it deterministic in the future
    @pytest.mark.skip()
    def test_iprofile_consistency(self):

        t = structure.load_coor(ref_file('gold1k.coor'))
        d = xray.Detector.generic()
        s = xray.Shotset.simulate(t, d, 5, 1)

        q_values = np.arange(1.0, 4.0, 0.02)
        num_phi = 360

        # compute from polar interp
        pi, pm = s._implicit_interpolation(q_values, num_phi)
        pi = pi.reshape(len(q_values), num_phi)
        ip1 = np.zeros((len(q_values), 2))
        ip1[:,0] = q_values
        ip1[:,1] = pi.sum(1)

        # compute from detector
        ip2 = s.intensity_profile(0.02)

        # compute from rings
        r = xray.Rings.simulate(t, 10, q_values, 360, 1)
        ip3 = r.intensity_profile()

        # make sure maxima are all similar
        ind1 = utils.maxima( math2.smooth(ip1[:,1], beta=15.0, window_size=21) )
        ind2 = utils.maxima( math2.smooth(ip2[:,1], beta=15.0, window_size=21) )
        ind3 = utils.maxima( math2.smooth(ip3[:,1], beta=15.0, window_size=21) )
        
        m1 = ip1[ind1,0]
        m2 = ip2[ind2,0]
        m3 = ip3[ind3,0]
        
        # discard the tails of the sim -- they have weak/noisy peaks
        # there should be strong peaks at |q| ~ 2.66, 3.06
        m1 = m1[(m1 > 2.0) * (m1 < 3.2)]
        m2 = m2[(m2 > 2.0) * (m2 < 3.2)]
        m3 = m3[(m3 > 2.0) * (m3 < 3.2)]

        # I'll let them be two q-brackets off
        assert_allclose(m1, m2, atol=0.045)
        assert_allclose(m1, m3, atol=0.045)
        assert_allclose(m2, m3, atol=0.045)
