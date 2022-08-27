"""
Inference models and the interface they should conform to

The `MultiStateModel` defines the interface used for inference by the rest of
the BILD module. We provide specific implementations, the `MultiStateRouse`
model, as well as the `FactorizedModel`, which essentially constitutes an HMM.
"""
import abc

from tqdm.auto import tqdm

import numpy as np
import scipy.stats

import rouse
from noctiluca import Trajectory
from .util import Loopingprofile

LOG_SQRT_2_PI = 0.5*np.log(2*np.pi)

class MultiStateModel(metaclass=abc.ABCMeta):
    """
    Abstract base class for inference models

    The most important capability of any model is the likelihood function
    `logL` for a combination of `Loopingprofile` and `Trajectory`. Furthermore,
    a model should be able to tell how many states it has and what number of
    spatial dimensions it expects through the `nStates` and `d` properties
    respectively.
    
    Finally, there are some convenience functions that are recommended, but not
    required to implement:

     + provide an initial guess for a good profile by `initial_loopingprofile`
     + sample from the likelihood by `trajectory_from_loopingprofile`
    """
    @property
    def nStates(self):
        """
        How many internal states does this model have?
        """
        raise NotImplementedError # pragma: no cover

    @property
    def d(self):
        """
        Spatial dimension
        """
        raise NotImplementedError # pragma: no cover

    def initial_loopingprofile(self, traj):
        """
        Give a quick guess for a good `Loopingprofile` for a `Trajectory`.

        The default implementation gives a random `Loopingprofile`.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Loopingprofile
        """
        return Loopingprofile(np.random.choice(self.nStates, size=len(traj)))

    @abc.abstractmethod
    def logL(self, loopingprofile, traj):
        """
        Calculate log-likelihood for (`Loopingprofile`, `Trajectory`) pair.

        Parameters
        ----------
        loopingprofile : Loopingprofile
        traj : Trajectory

        Returns
        -------
        float
            log-likelihood associated with the inputs
        """
        raise NotImplementedError # pragma: no cover

    def trajectory_from_loopingprofile(self, profile,
                                       localization_error=None,
                                       missing_frames=None,
                                       preproc=None):
        """
        Generate a `Trajectory` for the given `Loopingprofile`

        Parameters
        ----------
        profile : Loopingprofile
        localization_error : float or (d,) np.ndarray, dtype=float
            how much Gaussian noise to add to the trajectory. Specify either as
            a single float, in which case a Gaussian random variable of that
            standard deviation will be added to each dimension, or as an array
            with one value of the standard deviation for each dimension
            individually.
        missing_frames : None, float in [0, 1), int, or np.ndarray
            frames to remove from the generated trajectory. Can be
             + ``None`` or ``0`` : remove no frames
             + ``float in (0, 1)`` : remove frames at random, with this
               probability
             + ``int`` : remove this many frames at random
             + ``np.ndarray, dtype=int`` : indices of the frames to remove
        preproc : string
            specific to this base implementation. Set to
            ``'localization_error'`` or ``'missing_frames'`` to resolve the
            corresponding parameter according to the rules outlined above and
            return it.

        Returns
        -------
        Trajectory

        Notes
        -----
        Even though this method provides default preprocessing for
        ``localization_error`` and ``missing_frames``, implementations are not
        forced to use them. So the meaning of these arguments might be
        different from what's outlined here.
        """
        if preproc == 'localization_error':
            if np.isscalar(localization_error):
                localization_error = self.d*[localization_error]
            localization_error = np.asarray(localization_error)
            if localization_error.shape != (self.d,):
                raise ValueError("Did not understand localization_error") # pragma: no cover
            return localization_error

        elif preproc == 'missing_frames':
            if missing_frames is None or missing_frames == 0:
                missing_frames = np.array([], dtype=int)
            if np.isscalar(missing_frames):
                if 0 < missing_frames and missing_frames < 1:
                    missing_frames = np.nonzero(np.random.rand(len(profile)) < missing_frames)[0]
                else:
                    missing_frames = np.random.choice(len(profile), size=missing_frames, replace=False)
                    missing_frames = missing_frames.astype(int)

            return missing_frames

        else: # pragma: no cover
            raise NotImplementedError


class MultiStateRouse(MultiStateModel):
    """
    A multi-state Rouse model

    This inference model uses a given number of `rouse.Model` instances to
    choose from for each propagation interval. In the default use case this
    switches between a looped and unlooped model, but it could be way more
    general than that, e.g. incorporating different looped states, loop
    positions, numbers of loops, etc.

    Parameters
    ----------
    N : int
        number of monomers
    D, k : float
        Rouse parameters: 1d diffusion constant of free monomers and backbone
        spring constant
    d : int, optional
        spatial dimension
    looppositions : list of tuples
        specification of the extra bonds to add for the different states.  Each
        entry corresponds to one possible state of the model and should be a
        tuple ``(left_mon, right_mon, rel_strength=1)`` or a list of such;
        specification of ``rel_strength`` is optional. This will introduce an
        additional bond between the monomer indexed ``left_mon`` and the one
        indexed ``right_mon``, with strength ``rel_strength*k``. Note that you
        can remove the i-th backbone bond by giving ``(i, i+1, -1)``. Remember
        that (if relevant) you also have to include the unlooped state, which
        you can do by giving ``None`` instead of a tuple; alternatively, give a
        vacuous extra bond like ``(0, 0)``.
    measurement : "end2end" or (N,) np.ndarray
        which distance to measure. The default setting "end2end" is equivalent
        to specifying a vector ``np.array([-1, 0, ..., 0, 1])``, i.e. measuring
        the distance from the first to the last monomer.
    localization_error : float, (d,) np.array, or None, optional
        localization error assumed by the model, e.g. in calculating the
        likelihood or generating trajectories. If ``None``, try to use
        ``traj.localization_error`` where possible. If scalar, will be
        broadcast to all spatial dimensions.

    Attributes
    ----------
    models : list of `rouse.Model`
        the Rouse models for the individual states
    measurement : (N,) np.ndarray
        the measurement vector
    localization_error : array or None
        if ``None``, use ``traj.localization_error`` where possible

    Notes
    -----
    This class also provides a function to convert to a `FactorizedModel`,
    using the exact steady state distributions of the Rouse models. This is
    used for example to quickly guess an initial looping profile.

    See also
    --------
    MultiStateModel, rouse.Model
    """
    def __init__(self, N, D, k, d=3,
                 looppositions=(None, (0, -1)), # no mutable default parameters!
                                                # (thus tuple instead of list)
                 measurement="end2end",
                 localization_error=None,
                 ):
        self._d = d

        if str(measurement) == "end2end":
            measurement = np.zeros(N)
            measurement[0]  = -1
            measurement[-1] =  1

        assert len(measurement) == N
        self.measurement = measurement

        if localization_error is not None and np.isscalar(localization_error):
            localization_error = localization_error*np.ones(d)
        self.localization_error = localization_error

        self.models = []
        for loop in looppositions:
            if loop is not None and np.isscalar(loop[0]):
                loop = [loop]
            mod = rouse.Model(N, D, k, d, add_bonds=loop)
            self.models.append(mod)

    @property
    def nStates(self):
        return len(self.models)

    @property
    def d(self):
        return self._d

    def _get_noise(self, traj):
        # for internal use: get the localization error that should apply to a
        # given trajectory
        if self.localization_error is not None:
            return np.asarray(self.localization_error)
        elif traj.localization_error is not None:
            return np.asarray(traj.localization_error)
        else:
            raise ValueError("No localization error specified (use MultiStateModel.localization_error or Trajectory.localization_error)")

    def initial_loopingprofile(self, traj):
        """
        Give an initial guess for a looping profile

        Parameters
        ----------
        traj : Trajectory
            the trajectory under investigation

        Returns
        -------
        Loopingprofile
        """
        return self.toFactorized().initial_loopingprofile(traj)

    def logL(self, profile, traj):
        """
        Rouse likelihood, evaluated by Kalman filter

        Parameters
        ----------
        profile : Loopingprofile
        traj : Trajectory

        Returns
        -------
        float
        """
        localization_error = self._get_noise(traj)

        # Evolution of covariance matrix for each dimension depends only on the
        # localization error, i.e. is independent of the actual data. This
        # means we can optimize here by not executing the actual propagation
        # for dimensions with equal localization error
        # Idea: always use C[Cind[d]] instead of C[d], so C actually has only
        #       the distinct covariance matrices
        unique_errors, Cind = np.unique(localization_error, return_inverse=True)
        s2 = unique_errors*unique_errors

        for model in self.models:
            model.check_dynamics()

        model = self.models[profile[0]]
        M, C_single = model.steady_state()
        C = len(unique_errors) * [C_single]

        valid_times = np.nonzero(~np.any(np.isnan(traj[:]), axis=1))[0]
        L_log = np.empty((len(valid_times), self.d), dtype=float)

        def Kalman_update(t, M, C, L_log, i_write):
            # Innovation
            w = self.measurement
            m = w @ M
            xmm = traj[t] - m

            # Updates of covariances
            Cw = [c @ w for c in C]
            S = [w @ Cw[i] + s2[i]                   for i in range(len(C))]
            K = [Cw[i] / S[i]                        for i in range(len(C))]
            C = [C[i] - K[i][:, None]*Cw[i][None, :] for i in range(len(C))]

            # Mean and likelihoods
            M              = M + np.stack( [K[Cind[d]]*xmm[d]                             for d in range(self.d)], axis=-1)
            L_log[i_write] = -0.5*np.array([xmm[d]*xmm[d]/S[Cind[d]] + np.log(S[Cind[d]]) for d in range(self.d)]) - LOG_SQRT_2_PI

            return M, C

        # First update
        i_write = 0
        if 0 in valid_times:
            M, C = Kalman_update(0, M, C, L_log, i_write)
            i_write += 1

        # Propagate, then update
        for t, state in enumerate(profile[1:], start=1):
            model = self.models[state]

            # Propagate
            M = model.propagate_M(M, check_dynamics=False)
            C = [model.propagate_C(myC, check_dynamics=False) for myC in C]

            # Update
            if t in valid_times:
                M, C = Kalman_update(t, M, C, L_log, i_write)
                i_write += 1

        if i_write != len(L_log):
            raise RuntimeError("Internal inconsistency (i.e. bug)") # pragma: no cover

        return np.sum(L_log)

    def trajectory_from_loopingprofile(self, profile,
                                       localization_error=None,
                                       missing_frames=None,
                                       ):
        """
        Generative model

        Parameters
        ----------
        profile : Loopingprofile
            the profile from whose associated ensemble to sample
        localization_error : float or (d,) np.ndarray, dtype=float
            see `MultiStateModel.trajectory_from_loopingprofile`
        missing_frames : None, float in [0, 1), int, or np.ndarray
            see `MultiStateModel.trajectory_from_loopingprofile`

        Returns
        -------
        Trajectory
        """
        # Pre-processing
        # localization_error
        if localization_error is None:
            if self.localization_error is None:
                raise ValueError("Need to specify either localization_error or model.localization_error") # pragma: no cover
            else:
                localization_error = self.localization_error
        localization_error = super().trajectory_from_loopingprofile(profile, preproc='localization_error', localization_error=localization_error)

        # missing_frames
        missing_frames = super().trajectory_from_loopingprofile(profile, preproc='missing_frames', missing_frames=missing_frames)

        # Assemble trajectory
        data = np.empty((len(profile), self.d), dtype=float)
        data[:] = np.nan

        model = self.models[profile[0]]
        conf = model.conf_ss()
        data[0, :] = self.measurement @ conf

        for i in range(1, len(profile)):
            model = self.models[profile[i]]
            conf = model.evolve(conf)
            data[i, :] = self.measurement @ conf

        # Kick out frames that should be missing
        data[missing_frames, :] = np.nan

        # Add localization error
        data += localization_error[None, :] * np.random.normal(size=data.shape)

        # Return as Trajectory
        return Trajectory(data,
                          localization_error=localization_error,
                          loopingprofile=profile,
                         )

    def toFactorized(self):
        """
        Give the corresponding `FactorizedModel`

        This is the model that simply calculates likelihoods from the steady
        state probabilities of each of the individual states.

        Returns
        -------
        FactorizedModel
        """
        distributions = []
        noise2_per_d = np.sum(self.localization_error**2)/self.d if self.localization_error else 0
        for mod in self.models:
            _, C = mod.steady_state()
            s2 = self.measurement @ C @ self.measurement + noise2_per_d
            distributions.append(scipy.stats.maxwell(scale=np.sqrt(s2)))

        return FactorizedModel(distributions, d=self.d)

class FactorizedModel(MultiStateModel):
    """
    A simplified model, assuming time scale separation

    This model assumes that each point is sampled from one of a given list of
    distributions, where there is no correlation between the choice of
    distribution for each point.

    Parameters
    ----------
    distributions : list of distribution objects
        these will usually be ``scipy.stats.rv_continuous`` objects (e.g.
        Maxwell), but can be pretty arbitrary. The only function they have to
        provide is ``logpdf()``, which should take a scalar or vector of
        distance values and return a corresponding number of outputs. If you
        plan on using `trajectory_from_loopingprofile`, the distributions
        should also have an ``rvs()`` method for sampling.

    Attributes
    ----------
    distributions : list of distribution objects

    Notes
    -----
    This being a heuristical model, we assume that the localization error is
    already incorporated in the `!distributions`, as would be the case if they
    came from experimental data. Therefore, this class ignores the
    ``localization_error`` attribute of `Trajectory`.

    The ``d`` attribute mandated by the `MultiStateRouse` interface is used
    only for generation of trajectories.

    Instances of this class memoize trajectories they have seen before. To
    reset the memoization, you can either reinstantiate, or clear the cache
    manually:
    
    >>> model = FactorizedModel(model.distributions)
    ... model.clear_memo()

    If using ``scipy.stats.maxwell``, make sure to use it correctly, i.e. you
    have to specify ``scale=...``. Writing ``scipy.stats.maxwell(5)`` instead
    of ``scipy.stats.maxwell(scale=5)`` shifts the distribution instead of
    scaling it and leads to ``-inf`` values in the likelihood.

    Examples
    --------
    Experimentally measured distributions can be used straightforwardly using
    ``scipy.stats.gaussian_kde``: assuming we have measured ensembles of
    distances ``dists_i`` for reference states ``i``, we can use

    >>> model = FactorizedModel([scipy.stats.gaussian_kde(dists_0),
    ...                          scipy.stats.gaussian_kde(dists_1),
    ...                          scipy.stats.gaussian_kde(dists_2)])

    """
    def __init__(self, distributions, d=3):
        self.distributions = distributions
        self._d = d
        self._known_trajs = dict()

    @property
    def nStates(self):
        return len(self.distributions)

    @property
    def d(self):
        return self._d

    def _memo(self, traj):
        # memoize `traj`
        if traj not in self._known_trajs:
            with np.errstate(divide='ignore'): # nans in the trajectory raise 'divide by zero in log'
                logL_table = np.array([dist.logpdf(traj.abs()[:][:, 0]) 
                                       for dist in self.distributions
                                       ])
            self._known_trajs[traj] = {'logL_table' : logL_table}

    def clear_memo(self):
        """
        Clear the memoization cache
        """
        self._known_trajs = dict()

    def initial_loopingprofile(self, traj):
        """
        Gives the MLE profile for the given trajectory

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Loopingprofile
        """
        self._memo(traj)

        valid_times = np.nonzero(~np.any(np.isnan(traj[:]), axis=1))[0]
        best_states = np.argmax(self._known_trajs[traj]['logL_table'][:, valid_times], axis=0)

        states = np.zeros(len(traj), dtype=int)
        states[:(valid_times[0]+1)] = best_states[0]
        last_time = valid_times[0]

        for cur_time, cur_state in zip(valid_times[1:], best_states[1:]):
            states[(last_time+1):(cur_time+1)] = cur_state
            last_time = cur_time

        if last_time < len(traj):
            states[(last_time+1):] = best_states[-1]

        return Loopingprofile(states)

    def logL(self, profile, traj):
        self._memo(traj)
        return np.nansum([self._known_trajs[traj]['logL_table'][profile[i], i] for i in range(len(profile))])

    def trajectory_from_loopingprofile(self, profile,
                                       localization_error=0.,
                                       missing_frames=None):
        """
        Generative model

        Parameters
        ----------
        profile : Loopingprofile
            the profile from whose associated ensemble to sample
        localization_error : float or (d,) np.ndarray, dtype=float
            see `MultiStateModel.trajectory_from_loopingprofile`; note that
            since the localization error should already be accounted for in the
            `distributions` of the model, it is *not* added to the trajectory
            here. Instead, it is just written to ``traj.localization_error``.
        missing_frames : None, float in [0, 1), int, or np.ndarray
            see `MultiStateModel.trajectory_from_loopingprofile`

        Returns
        -------
        Trajectory

        Notes
        -----
        The `FactorizedModel` only contains distributions for the scalar
        distance between the points, amounting to the assumption that the full
        distribution of distance vectors is isotropic. Thus, in generating the
        trajectory we sample a magnitude from the given distributions and a
        direction from the unit sphere.
        """
        # Pre-proc
        localization_error = super().trajectory_from_loopingprofile(profile, preproc='localization_error', localization_error=localization_error)
        missing_frames = super().trajectory_from_loopingprofile(profile, preproc='missing_frames', missing_frames=missing_frames)

        # Note that the distributions in the model give us only the length, not
        #   the orientation. So we also have to sample unit vectors
        # Furthermore, localization_error should not be added, since
        #   self.distributions already contain it. It will be written to the
        #   meta entry though!
        magnitudes = np.array([self.distributions[state].rvs() for state in profile[:]])
        data = np.random.normal(size=(len(magnitudes), self.d))
        data *= np.expand_dims(magnitudes / np.linalg.norm(data, axis=1), 1)
        data[missing_frames, :] = np.nan

        return Trajectory(data,
                          localization_error=localization_error,
                          loopingprofile=profile,
                         )
