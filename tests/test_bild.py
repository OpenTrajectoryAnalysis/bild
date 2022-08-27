import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass
from copy import deepcopy

import numpy as np
np.seterr(all='raise')
import scipy.stats

import unittest
from unittest.mock import patch

import noctiluca as nl
from context import bild

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'TestUtilLoopingprofile',
    'TestUtilStateProbabilities',
    'TestModels',
    'TestAMIS',
    'TestCore',
    'TestPostproc',
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

class TestUtilLoopingprofile(myTestCase):
    def setUp(self):
        self.profile = bild.Loopingprofile([0, 0, 0, 1, 1, 0, 3, 3])

    def test_init(self):
        lp = bild.Loopingprofile()
        self.assert_array_equal(lp.state, np.array([]))
        lp = bild.Loopingprofile([1, 2, 3])
        self.assert_array_equal(lp.state, np.array([1, 2, 3]))

    def test_copy(self):
        new_profile = self.profile.copy()
        self.assert_array_equal(new_profile.state, self.profile.state)
        new_profile[2] = 5
        self.assertEqual(self.profile[2], 0)

    def test_implicit_functions(self):
        # len
        self.assertEqual(len(self.profile), 8)

        # getitem
        self.assertEqual(self.profile[3], 1)
        self.assert_array_equal(self.profile[2:4], np.array([0, 1]))

        # setitem
        self.profile[2] = 3
        self.assertEqual(self.profile[2], 3)
        with self.assertRaises(AssertionError):
            self.profile[5] = 3.74

        # eq
        self.assertEqual(self.profile, bild.Loopingprofile([0, 0, 3, 1, 1, 0, 3, 3]))
        self.assertNotEqual(self.profile, bild.Loopingprofile([1, 0, 3]))

    def test_count_switches(self):
        self.assertEqual(self.profile.count_switches(), 3)
        self.profile[5] = 1
        self.assertEqual(self.profile.count_switches(), 2)
        self.profile[4] = 2
        self.assertEqual(self.profile.count_switches(), 4)

    def test_intervals(self):
        ivs = self.profile.intervals()
        ivs_true = [(None, 3, 0), (3, 5, 1), (5, 6, 0), (6, None, 3)]

        self.assertEqual(len(ivs), len(ivs_true))
        for iv, iv_true in zip(ivs, ivs_true):
            self.assertTupleEqual(iv, iv_true)

        ivs = bild.Loopingprofile([1, 1, 1, 1]).intervals()
        self.assertEqual(len(ivs), 1)
        self.assertTupleEqual(ivs[0], (None, None, 1))

    def test_plottable(self):
        t, y = self.profile.plottable()
        self.assert_array_equal(t, np.array([-1, 2, 2, 4, 4, 5, 5, 7]))
        self.assert_array_equal(y, np.array([0, 0, 1, 1, 0, 0, 3, 3]))

class TestUtilStateProbabilities(myTestCase):
    def test_(self):
        profiles = [bild.Loopingprofile([0, 1, 0, 1, 0]),
                    bild.Loopingprofile([1, 1, 1, 1, 1]),
                   ]
        self.assert_array_equal(bild.util.state_probabilities(profiles),
                                [[0.5, 0, 0.5, 0, 0.5],
                                 [0.5, 1, 0.5, 1, 0.5]])
        self.assert_array_equal(bild.util.state_probabilities(profiles, nStates=3),
                                [[0.5, 0, 0.5, 0, 0.5],
                                 [0.5, 1, 0.5, 1, 0.5],
                                 [  0, 0,   0, 0,   0],
                                ])

class TestModels(myTestCase):
    def setUp(self):
        self.traj = nl.Trajectory(np.array([1, 2, np.nan, 4]), localization_error=[0.5])
        self.profile = bild.Loopingprofile([1, 1, 0, 0])

    def test_base(self):
        model = bild.models.MultiStateRouse(20, 1, 5, d=1)

        # Check base implementation
        profile = bild.models.MultiStateModel.initial_loopingprofile(model, self.traj)
        self.assertEqual(len(profile), 4)

    def test_Rouse(self):
        model = bild.models.MultiStateRouse(20, 1, 5, d=1)
        logL = model.logL(self.profile, self.traj)
        self.assertTrue(logL > -100 and logL < 0)

        traj = deepcopy(self.traj)
        traj.localization_error = None
        with self.assertRaises(ValueError):
            _ = model.logL(self.profile, traj)

        model = bild.models.MultiStateRouse(20, 1, 5, d=1,
                                            localization_error=0.5)
        logL2 = model.logL(self.profile, self.traj)
        self.assertEqual(logL, logL2)

        profile = model.initial_loopingprofile(self.traj)
        self.assert_array_equal(profile.state, np.array([1, 0, 0, 0]))

        traj = model.trajectory_from_loopingprofile(bild.Loopingprofile([0, 0, 0, 1, 1, 1]),
                                                    localization_error=0.1)
        self.assertEqual(len(traj), 6)

        traj = model.trajectory_from_loopingprofile(bild.Loopingprofile(np.ones(20)),
                                                    localization_error=0.1,
                                                    missing_frames = 0.9,
                                                    )
        self.assertLess(traj.count_valid_frames(), 18)

        traj = model.trajectory_from_loopingprofile(bild.Loopingprofile(np.ones(20)),
                                                    missing_frames = 12,
                                                    )
        self.assertEqual(traj.count_valid_frames(), 8)

    def test_Factorized(self):
        model = bild.models.FactorizedModel([
            scipy.stats.maxwell(scale=1),
            scipy.stats.maxwell(scale=4),
            ], d=1)

        self.assertEqual(model.nStates, 2)

        logL = model.logL(self.profile, self.traj)
        profile = model.initial_loopingprofile(self.traj)
        self.assertTrue(logL > -100 and logL < 0)
        self.assert_array_equal(profile.state, np.array([0, 0, 1, 1]))

        model.clear_memo()
        logL = model.logL(self.profile, self.traj)
        profile = model.initial_loopingprofile(self.traj)
        self.assertTrue(logL > -100 and logL < 0)
        self.assert_array_equal(profile.state, np.array([0, 0, 1, 1]))

        traj = model.trajectory_from_loopingprofile(bild.Loopingprofile([0, 0, 0, 1, 1, 1]))
        self.assertEqual(len(traj), 6)

class TestAMIS(myTestCase):
    def setUp(self):
        self.traj = nl.Trajectory([0.1, 1, 2, 3, 4, 5])
        self.model = bild.models.FactorizedModel([scipy.stats.maxwell(scale=0.1),
                                                  scipy.stats.maxwell(scale=1.0)])

    def test_st2profile(self):
        profile = bild.amis.st2profile([0.25, 0.5, 0.25], 0, self.traj)
        self.assert_array_equal(profile[:], [0, 0, 1, 1, 0, 0])

    def test_dirichlet_methodofmoments(self):
        ss = np.array([[0.0, 1.0],
                       [0.5, 0.5],
                       [1.0, 0.0]])
        a = bild.amis.dirichlet_methodofmoments(ss, np.ones(len(ss))/len(ss))
        self.assert_array_equal(a, [0.25, 0.25])
        a = bild.amis.dirichlet_methodofmoments(ss, np.array([0.5, 0.5, 0]))
        self.assert_array_equal(a, [0.5, 1.5])

    def test_logL(self):
        s = np.array([0.5, 0.5])
        theta = 1
        logL = bild.amis.logL(((s, theta), (self.traj, self.model)))

        ss = np.array([[0.1, 0.9],
                       [0.5, 0.5],
                       [0.9, 0.1]])
        thetas = np.array([1, 1, 0])

        logLs = bild.amis.calculate_logLs(ss, thetas, self.traj, self.model)
        self.assertTrue(np.all(np.isfinite(logLs)))
        self.assertEqual(logL, logLs[1])

    def test_proposal(self):
        a = np.array([0.5, 1, 4])
        m = 0.7

        ss, thetas = bild.amis.sample_proposal(a, m, 100)
        dens = bild.amis.proposal(a, m, ss, thetas)
        self.assertTrue(np.all(np.isfinite(dens)))

        a_fit, m_fit = bild.amis.fit_proposal(ss, thetas,
                                              np.ones(len(thetas))/len(thetas))
        self.assertTrue(np.all(np.abs(np.log(a_fit/a)) < 1))
        self.assertLess(np.abs(m_fit - m), 0.2)

    def test_sampler(self):
        sampler0 = bild.amis.FixedkSampler(self.traj, self.model, k=0)
        self.assertFalse(sampler0.step())
        self.assert_array_equal(sampler0.MAP_profile()[:], [1, 1, 1, 1, 1, 1])

        sampler1 = bild.amis.FixedkSampler(self.traj, self.model, k=1)
        self.assertFalse(sampler1.step())
        self.assert_array_equal(sampler1.MAP_profile()[:], [0, 1, 1, 1, 1, 1])

        sampler2 = bild.amis.FixedkSampler(self.traj, self.model, k=2,
                                          N=100, max_fev=150)
        self.assertTrue(sampler2.step())
        self.assertFalse(sampler2.step())
        # The trajectory is so blatantly state 1 (except for the first point)
        # that the inference will reliably make vanishingly small 0-state
        # intervals.
        self.assert_array_equal(sampler2.MAP_profile()[:], [1, 1, 1, 1, 1, 1])
        self.assertGreater(sampler0.t_stat(sampler2), 3)

        self.model.distributions = self.model.distributions[::-1]
        self.model.clear_memo()
        sampler2i = bild.amis.FixedkSampler(self.traj, self.model, k=2)
        self.assertTrue(sampler2i.step())
        self.assertTrue(sampler2i.step())
        self.assert_array_equal(sampler2i.MAP_profile()[:], [0, 0, 0, 0, 0, 0])

class TestCore(myTestCase):
    def setUp(self):
        self.traj = nl.Trajectory([0.1, 0.05, 6, 3, 4, 0.01, 5, 7])
        self.model = bild.models.FactorizedModel([scipy.stats.maxwell(scale=0.1),
                                                  scipy.stats.maxwell(scale=1)])

    def test_sample(self):
        for _ in range(5):
            res = bild.sample(self.traj, self.model,
                              init_runs=5,
                              sampler_kw={'max_fev' : 1000}, # runtime
                             )

            self.assertGreater(len(res.k), 4)
            self.assertGreaterEqual(np.argmax(res.evidence), 3) # See profile comparison below
            self.assertTrue(np.all(res.evidence_se > 0))
            self.assert_array_equal(res.best_profile()[:], res.best_profile(dE=2)[:])

    def test_insignificance_resolution_after_main_loop(self):
        model = bild.models.FactorizedModel([scipy.stats.maxwell(scale=0.1),
                                             scipy.stats.maxwell(scale=1)],
                                            d=1,
                                           )
        # the point of equal likelihood for two Maxwell's with scales a and b
        # is x = ab * sqrt(6*log(b/a) / (b^2 - a^2))
        x = 0.1*np.sqrt(6*np.log(0.1) / -0.99)
        traj = nl.Trajectory(x*np.ones(5))
        res = bild.sample(traj, model,
                          init_runs=5,
                          sampler_kw={'max_fev' : 1000}, # runtime
                          k_max = 10, # will always be hit here
                         )

        self.assertTrue(np.allclose(res.evidence, [np.mean(res.evidence)],
                                    atol = 0.1))

class TestPostproc(myTestCase):
    def setUp(self):
        self.traj = nl.Trajectory([0.1, 0.05, 6, 3, 4, 0.01, 5, 7])
        self.model = bild.models.FactorizedModel([scipy.stats.maxwell(scale=0.1),
                                                  scipy.stats.maxwell(scale=1)])

    def test_optimize_boundary(self):
        bad_profile = bild.Loopingprofile([0, 1, 1, 1, 0, 0, 0, 1])
        better_profile = bild.postproc.optimize_boundary(bad_profile, self.traj, self.model)
        self.assert_array_equal(better_profile[:], [0, 0, 1, 1, 1, 0, 1, 1])

        with self.assertRaises(RuntimeError):
            bild.postproc.optimize_boundary(bad_profile, self.traj, self.model, max_iteration=2) # 3 are necessary at least

        bad_profile = bild.Loopingprofile([0, 1, 0, 1, 0, 0, 0, 1])
        with self.assertRaises(bild.postproc.BoundaryEliminationError):
            bild.postproc.optimize_boundary(bad_profile, self.traj, self.model)

        bad_profile = bild.Loopingprofile([1, 1, 1, 1, 1, 1, 1, 1])
        _ = bild.postproc.optimize_boundary(bad_profile, self.traj, self.model, max_iteration=1) # nothing should run, there are no boundaries here

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
