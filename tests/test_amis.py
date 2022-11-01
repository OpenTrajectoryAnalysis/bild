import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass
from copy import deepcopy

import numpy as np
np.seterr(all='raise')

from scipy import stats
from scipy.special import logsumexp

import unittest
from unittest.mock import patch

import noctiluca as nl
from context import bild
amis = bild.amis

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'TestDirichlet',
    'TestCFC',
    'TestFixedkSampler',
]

# Extend unittest.TestCase's capabilities to deal with numpy arrays
class myTestCase(unittest.TestCase):
    def assert_array_equal(self, array1, array2):
        try:
            np.testing.assert_array_equal(array1, array2)
            res = True
        except AssertionError as err: # pragma: no cover
            res = False
            print(err)
        self.assertTrue(res)

    def assert_array_almost_equal(self, array1, array2, decimal=10):
        try:
            np.testing.assert_array_almost_equal(array1, array2, decimal=decimal)
            res = True
        except AssertionError as err: # pragma: no cover
            res = False
            print(err)
        self.assertTrue(res)

class TestDirichlet(myTestCase):
    def test_logpdf(self):
        lp = amis.Dirichlet().logpdf(np.array([0.5, 4]),
                                     np.array([[0., 1]]))
        self.assertEqual(lp, np.inf)

    def test_methodofmoments(self):
        ss = np.array([[0.0, 1.0],
                       [0.5, 0.5],
                       [1.0, 0.0]])
        a = amis.Dirichlet().estimate(ss, np.zeros(len(ss))/len(ss))
        self.assert_array_equal(a, [0.25, 0.25])
        a = amis.Dirichlet().estimate(ss, np.array([1, 1, -np.inf]))
        self.assert_array_equal(a, [0.5, 1.5])

class TestCFC(myTestCase):
    def test_pathological(self):
        # should be enough for uniform_marginals() and logp_uniform()
        # Impossible to leave state 1
        cfc = amis.CFC([[0, 1, 1],
                        [0, 0, 0],
                        [1, 1, 0]])

        log_marg = cfc.uniform_marginals(4)
        self.assert_array_equal(log_marg[1, :-1], -np.inf)
        self.assertNotEqual(log_marg[1, -1], -np.inf)

        logp = cfc.logp_uniform(4)
        self.assert_array_equal(logp[1, :-1], -np.inf)
        self.assertNotEqual(logp[1, -1], -np.inf)

        # Impossible to enter state 1
        cfc = amis.CFC([[0, 0, 1],
                        [1, 0, 1],
                        [1, 0, 0]])

        log_marg = cfc.uniform_marginals(4)
        self.assert_array_equal(log_marg[1, 1:], -np.inf)
        self.assertNotEqual(log_marg[1, 0], -np.inf)

        logp = cfc.logp_uniform(4)
        self.assert_array_equal(logp[1, 1:], -np.inf)
        self.assertNotEqual(logp[1, 0], -np.inf)

        logf = -np.log(2)*np.ones(3)
        logf[1] = -np.inf
        logp = cfc.solve_marginals_single(logf, np.array([-np.inf, 0., -np.inf]))
        self.assert_array_equal(logp, logf)

    def test_full_sample(self):
        cfc = amis.CFC([[0, 1, 1],
                        [1, 0, 0],
                        [1, 1, 0]])
        self.assert_array_equal(cfc.full_sample(0), [[0], [1], [2]])
        self.assert_array_equal(cfc.full_sample(1), [[0, 1], [0, 2], [1, 0], [2, 0], [2, 1]])

        cfc = amis.CFC([[0, 1, 1],
                        [1, 0, 0],
                        [0, 1, 0]])
        self.assert_array_equal(cfc.full_sample(1), [[0, 1], [0, 2], [1, 0], [2, 1]])

        with self.assertRaises(ValueError):
            _ = cfc.full_sample(100)

        cfc = amis.CFC([[0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]])
        self.assert_array_equal(cfc.full_sample(1), [[0, 2], [1, 0], [2, 1]])
        self.assertEqual(len(cfc.full_sample(5)), 3)

    def test_sample(self):
        cfc = amis.CFC([[0, 1, 1],
                        [1, 0, 0],
                        [1, 1, 0]])

        full_sample = cfc.full_sample(1) # (5, 2)
        self.assert_array_equal(full_sample, [[0, 1], [0, 2], [1, 0], [2, 0], [2, 1]])

        for k in range(5):
            full_sample = cfc.full_sample(k)
            sample = cfc.sample(cfc.logp_uniform(k), N=10*len(full_sample)) # (N, 2)
            eq = np.sum(sample[:, None, :] == full_sample[None, :, :], axis=-1) == k+1 # (N, 5)

            # every sampled trace corresponds to exactly one in full sample
            self.assert_array_equal(np.sum(eq, axis=1), 1)

            # every trace in full sample is sampled at least once
            self.assertTrue(np.all(np.sum(eq, axis=0) > 0))

    def test_logpmf(self):
        cfc = amis.CFC([[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]])
        sample = cfc.full_sample(4)
        logL = cfc.logpmf(np.ones((3, 5)), sample)
        self.assert_array_equal(logL, logL[0])

        cfc = amis.CFC([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
        sample = cfc.full_sample(9)
        logL = cfc.logpmf(np.zeros((3, 10)), sample)
        self.assert_array_equal(logL, -np.log(3))

    def test_estimate(self):
        # also covers logp_from_marginals and solve_marginals_single
        cfc = amis.CFC([[0, 1, 1],
                        [1, 0, 0],
                        [1, 1, 0]])

        logp = np.log(1-np.random.rand(3, 3))
        logp -= logsumexp(logp, axis=0)
        sample = cfc.sample(logp, N=500)

        est = cfc.estimate(sample, log_weights=np.zeros(len(sample)))
        with np.errstate(under='ignore'):
            self.assertTrue(np.all(np.abs(np.exp(est)-np.exp(logp)) < 0.2))

        with self.assertRaises(RuntimeError):
            cfc.MOM_maxiter = 0
            _ = cfc.estimate(sample, log_weights=np.zeros(len(sample)))

    def test_N_total(self):
        cfc = amis.CFC([[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]])
        for k in range(10):
            self.assertEqual(cfc.N_total(k), 3*2**k)

        cfc = amis.CFC([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])

        self.assertEqual(cfc.N_total(0), 3)
        self.assertEqual(cfc.N_total(1), 4)
        self.assertEqual(cfc.N_total(2), 6)

        cfc = amis.CFC([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
        for k in range(10):
            self.assertEqual(cfc.N_total(k), 3)

class TestFixedkSampler(myTestCase):
    def setUp(self):
        self.traj = nl.Trajectory([0.1, 1, 2, 3, 4, 5])
        self.model = bild.models.FactorizedModel([stats.maxwell(scale=0.1),
                                                  stats.maxwell(scale=1.0)])

    def test_st2profile(self):
        sampler = amis.FixedkSampler(self.traj, self.model, k=2)
        profile = sampler.st2profile([0.25, 0.5, 0.25], [0, 1, 0])
        self.assert_array_equal(profile[:], [0, 0, 1, 1, 0, 0])

    def test_logL(self):
        sampler = amis.FixedkSampler(self.traj, self.model, k=1)

        ss = np.array([[0.1, 0.9],
                       [0.5, 0.5],
                       [0.9, 0.1]])
        thetas = np.array([[1, 0],
                           [1, 0],
                           [1, 0]])

        logLs = sampler.logL(ss, thetas)
        self.assertTrue(np.all(np.isfinite(logLs)))

    def test_sampling(self):
        sampler0 = amis.FixedkSampler(self.traj, self.model, k=0)
        self.assertFalse(sampler0.step())
        self.assert_array_equal(sampler0.MAP_profile()[:], [1, 1, 1, 1, 1, 1])

        sampler1 = amis.FixedkSampler(self.traj, self.model, k=1)
        self.assertFalse(sampler1.step())
        self.assert_array_equal(sampler1.MAP_profile()[:], [0, 1, 1, 1, 1, 1])
        
        self.assertGreater(sampler1.tstat(sampler0), 10)

        sampler2 = amis.FixedkSampler(self.traj, self.model, k=2,
                                      N=10, max_fev=25)
        self.assertTrue(sampler2.step())
        self.assertTrue(sampler2.step())
        self.assertFalse(sampler2.step())

        samplerK = amis.FixedkSampler(self.traj, self.model, k=10)
        self.assertFalse(samplerK.step())

        # Also check posterior, since we need a sample for that
        with np.errstate(under='ignore'):
            logpost = sampler1.log_marginal_posterior()
            self.assert_array_almost_equal(logsumexp(logpost, axis=0), np.zeros(logpost.shape[1]))
            logpost = sampler2.log_marginal_posterior()
            self.assert_array_almost_equal(logsumexp(logpost, axis=0), np.zeros(logpost.shape[1]))

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
