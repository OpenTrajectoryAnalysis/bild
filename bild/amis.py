"""
XXXXXXXXXXX unfinished implementation with free choice of possible transitions XXXXXXXXXXXXX

This was an attempt to implement blocked state transitions (e.g. prohibit 1-->2
in a three state model). This turned out to be tricky, since there is no good
way to identify the uniform proposal distribution (it's no longer the one with
all weights equal to 1, nor the one with all uniform marginals). Beyond that
it's also not quite clear that there will be a real use case for this anyways.

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Implementation of AMIS for posterior sampling at fixed k

This module provides the implementation of the AMIS (`Cornuet et al. 2012
<https://doi.org/10.1111/j.1467-9469.2011.00756.x>`_) sampling scheme for BILD,
at a fixed number of switches ``k`` in the profile. The `FixedkSampler` class
provides the full interface, while the other functions in the module define the
sampling scheme.

We parametrize the binary profiles as ``(s, theta)`` where
 + s is a vector of interval lengths, as fraction of the whole trajectory. It
   thus has dimension ``k+1`` and satisfies ``sum(s) = 1``, as well as ``s_i >
   0 forall i``.
 + theta is the looping state of the first interval. Since the profiles are
   binary, this determines the state of all subsequent intervals. Clearly ``θ
   in {0, 1}``.

The conversion from the parametrization ``(s, θ)`` to discretely sampled
profiles is according to the following scheme:

.. code-block:: text

    profile:            θ     p     p     p     p     p
    trajectory:       -----x-----x-----x-----x-----x-----x
    position:             0.0   0.2   0.4   0.6   0.8   1.0

    example (s):           ==0.25==|=====0.5=====|==0.25==
    switch positions:              |(0.25)       |(0.75)
    conversion:            |<<<<<|<<<<<|<<<<<|<<<<<|<<<<<
    result (θ=0):       0     0  '  1     1  '  0     0

Note that the conversion step amounts to a simple ``floor()`` operation on the
switch positions. We chose this conversion over the maybe more "natural" one of
rounding to the nearest integer because it guarantees

 + that a uniformly distributed continuous switch position will still be
   uniformly distributed over its possible discrete positions.
 + conservation of switches: a switch at .99 would disappear from the profile
   with the rounding scheme. Importantly, note that in either case an interval
   shorter than the framerate will likely not be represented in the profile,
   and thus switches are not always conserved. However, if such profiles are
   the main contribution to the evidence for this number of switches, we should
   (and do!) just choose the appropriate lower number of switches instead.
"""
from abc import ABCMeta, abstractmethod
import itertools

import math
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from numpy         import logaddexp
from numpy.linalg  import matrix_power

from noctiluca import parallel
from .util import Loopingprofile

def st2profile(s, theta, traj):
    """
    Convert AMIS parameters (s, θ) to Loopingprofile for given trajectory

    Parameters
    ----------
    s, theta : see module doc.
    traj : Trajectory
        needed to determine discretization. In fact, only ``len(traj)`` is
        evaluated.

    Returns
    -------
    Loopingprofile
    """
    states = theta[0]*np.ones(len(traj))
    if len(s) > 1:
        switchpos = np.cumsum(s)[:-1]
        
        switches = np.floor(switchpos*(len(traj)-1)).astype(int) + 1 # floor(0.0) + 1 = 1 != ceil(0.0)
        for i in range(1, len(switches)):
            states[switches[i-1]:switches[i]] = theta[i]
            
        states[switches[-1]:] = theta[-1]
    
    return Loopingprofile(states)

### Likelihood, i.e. target distribution ###

def logL(s, theta, traj, model):
    """
    Evaluate the model likelihood for a given profile

    Parameters
    ----------
    s, theta : see module doc
    traj : Trajectory
    model : MultiStateModel

    Returns
    -------
    float

    See also
    --------
    calculate_logLs
    """
    profile = st2profile(s, theta, traj)
    return model.logL(profile, traj)

def _logL_for_parallelization(params):
    # Helper function, unpacking arguments
    ((s, theta), (traj, model)) = params
    return logL(s, theta, traj, model)

def calculate_logLs(ss, thetas, traj, model):
    """
    Evaluate the likelihood on an ensemble of profiles

    Parameters
    ----------
    ss : (N, k+1) np.ndarray, dtype=float
    thetas : (N,) np.ndarray, dtype=int
    traj : Trajectory
    model : MultiStateModel

    Returns
    -------
    (N,) np.ndarray, dtype=float
        the likelihoods associated with each profile

    Notes
    -----
    This function is parallel-aware (ordered) (c.f. `noctiluca.util.parallel`).
    But overhead might be large.
    """
    todo = itertools.product(zip(ss, thetas), [(traj, model)])
    imap = parallel._map(_logL_for_parallelization, todo)
    return np.array(list(imap))

### Proposal distribution and its components ###

class ProposalDistribution(metaclass=ABCMeta):
    """
    ABC defining the interface we require from proposal distributions
    """
    @abstractmethod
    def sample(self, params, N=1):
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, params, points):
        raise NotImplementedError

    @abstractmethod
    def estimate(self, points, log_weights):
        raise NotImplementedError
            
class Dirichlet(ProposalDistribution):
    """
    The Dirichlet is implemented in ``scipy.stats.dirichlet``; we wrap it here
    for consistency with the ``ProposalDistribution`` interface, and add the
    method of moments estimator
    """
    def sample(self, a, N=1):
        """
        Sample from the Dirichlet distribution

        Parameters
        ----------
        a : (k+1,) np.ndarray
            concentration parameters
        N : int, optional
            how many samples to draw

        Returns
        -------
        (N, k+1) np.ndarray
        """
        return stats.dirichlet(a).rvs(N)

    def log_likelihood(self, a, ss):
        """
        Evaluate Dirichlet distribution

        Parameters
        ----------
        a : (k+1,) np.ndarray
            concentration parameters
        ss : (N, k+1) np.ndarray
            the samples for which to evaluate

        Returns
        -------
        (N,) np.ndarray
        """
        try:
            return stats.dirichlet(a).logpdf(ss.T)
        except ValueError:
            # if some a < 1 and associated s == 0
            logLs = []
            for s in ss:
                try:
                    logLs.append(stats.dirichlet(a).logpdf(s))
                except ValueError:
                    logLs.append(np.inf)
            return np.array(logLs)
    
    def estimate(self, ss, log_weights):
        """
        Method of moments estimator for the Dirichlet distribution

        Parameters
        ----------
        ss : (N, k+1) np.ndarray, dtype=float
            the switch positions ``s`` for a sample of size ``n``
        log_weights : (N,) np.ndarray, dtype=float
            the associated weights for all samples (unnormalized)

        Returns
        -------
        (k+1,) np.ndarray
            the concentration parameters for the fitted Dirichlet distribution

        Notes
        -----
        We use the method of moments, because it is straight-forward to apply to a
        weighted sample. Given an ensemble of switch positions ``s``, we have

         + the mean positions ``m = <s>``
         + the variances ``v = <(s-m)²>``
         + from this we can find the total concentration as ``A = mean(m*(1-m)/v) -
           1``
         + thus yielding the final estimate of ``α = A*m``.
        """
        with np.errstate(under='ignore'):
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.sum(weights)

            m = weights @ ss
            v = weights @ (ss - m[np.newaxis, :])**2

        if np.any(v == 0):
            # this is almost a pathological case, but it is possible; best solution
            # is to return very concentrated (but finite!) distribution and let the
            # concentration brake do its work.
            s = 1e10 # pragma: no cover
        else:
            s = np.mean(m*(1-m)/v) - 1
        return s*m

class CFC(ProposalDistribution):
    """
    Conflict Free Categorical (CFC)

    The CFC is our proposal distribution for state traces θ. These are vectors
    of length ``k+1``, containing integers between 0 and ``n-1``—``k`` being
    the number of switches and ``n`` the number of states we consider. The
    fundamental twist is that neighboring entries cannot take the same value,
    which makes it difficult to define a reasonable proposal distribution over
    this space of state traces. 

    Generalizing the above constraint, ``CFC`` accepts the ``transitions``
    argument, which specifies which state transitions are allowed or forbidden.

    The CFC is parametrized by weights ``p`` for each entry being in any of the
    available states (``p.shape = (n, k+1)``). The constraint is then enforced
    by a causal sampling scheme:

     + sample ``θ[0] ~ Categorical(p[:, 0])``
     + select the weights associated with all allowed states (given ``θ[0]``)
       from ``p[:, 1]`` and renormalize
     + sample ``θ[1] ~ Categorical(p[:, 1])`` (with the adapted ``p[:, 1]``)
     + continue iteratively

    This scheme makes sampling and likelihood evaluation relatively
    straight-forward. For estimation, we have to convert the actually observed
    marginals in a sample back to the weights ``p``. Letting ``f_n`` and
    ``g_n`` denote the marginals at step ``i`` and ``i-1`` respectively, we
    have that ``p_n := p[:, i]`` is given by the solution of
    ```
        p_n = f_n / sum_{m in C_n}( g_m / ( sum_{k in C_m} 1-p_m ) ) ,
    ```
    where the sums run over index sets ``C_n``, indicating which transitions
    are allowed from state ``n``. Solving this equation by fixed point
    iteration, we find approximate parameters ``p`` for a given sample of state
    traces.

    Knowing how to sample, evaluate, and estimate the CFC is all we need in
    order to implement the ``ProposalDistribution`` interface.

    For numerical stability, we work with ``log(p)`` instead of ``p``.
    
    Parameters
    ----------
    k : int
        number of switches
    transitions : (n, n) np.ndarray, dtype=bool
        ``transitions[i, j]`` indicates whether the transition from state ``i``
        to state ``j`` is allowed or not.
    """
    def __init__(self, transitions):
        self.transitions = transitions

    @property
    def n(self):
        return self.transitions.shape[0]

    def sample(self, logp, N=1):
        """
        Sample from the Conflict Free Categorical

        Parameters
        ----------
        logp : (n, k+1) np.ndarray
            the weight parameters defining the distribution
        N : int, optional
            how many traces to sample

        Returns
        -------
        thetas : (N, k+1) np.ndarray
            the sampled state traces
        """
        k = logp.shape[1]-1
        assert k >= 0

        # For sampling we actually need p, not log(p)
        with np.errstate(under='ignore'):
            p = np.exp(logp)

        thetas = np.empty((N, k+1), dtype=int)
        thetas[:, 0] = np.random.choice(self.n, size=N, p=p[:, 0])
        for i in range(1, k+1):
            p_cur = p[None, :, i] * self.transitions[thetas[:, i-1]] # (N, n)
            P = np.cumsum(p_cur, axis=1)
            P /= P[:, [-1]]

            # "vectorize np.random.choice" by hand
            thetas[:, i] = np.argmax(P > np.random.rand(N, 1), axis=1) # argmax gives first occurence

        return thetas

    def log_likelihood(self, logp, thetas):
        """
        Evaluate the Conflict Free Categorical

        Parameters
        ----------
        logp : (n, k+1) np.ndarray
            the weights defining the CFC.
        thetas : (N, k+1) np.ndarray
            the ``N`` state traces for which to evaluate the distribution

        Returns
        -------
        (N,) np.ndarray
            the logarithm of the CFC evaluated at the given state traces
        """
        #                                    (1, n, k+1)     x    (N, 1, k+1)    -->    (N, 1, k+1)
        logp_theta = np.take_along_axis(logp[None, :,  :], thetas[:, None, :  ], axis=1)[:, 0, :] # (N, k+1)
        with np.errstate(under='ignore'):
            #                             (1, k, n)              (N, k, n)
            log_norm = logsumexp(logp.T[None, 1:, :] , b=self.transitions[thetas[:, :-1]], axis=-1) # (N, k)

        return np.sum(logp_theta, axis=1) - np.sum(log_norm, axis=1)

    def estimate(self, thetas, log_weights):
        """
        Method of "Moments" (Marginals) estimation for the CFC

        Parameters
        ----------
        thetas : (N, k+1) np.ndarray, dtype=int
            the sample of state traces
        log_weights : (N,) np.ndarray, dtype=float
            the associated weights for all samples (unnormalized)

        Returns
        -------
        logp : (n, k+1) np.ndarray
            the estimated weight parameters; normalized to satisfy
            ``logsumexp(logp, axis=0) == 0``.
        """
        indicators = thetas[None, :, :] == np.arange(self.n)[:, None, None] # (n, N, k+1)
        with np.errstate(under='ignore'):
            log_marginals = logsumexp(log_weights[None, :, None], b=indicators, axis=1) # (n, k+1)
            log_marginals -= logsumexp(log_marginals, axis=0, keepdims=True)

        return self.logp_from_marginals(log_marginals)

    def logp_from_marginals(self, log_marginals, **kwargs):
        """
        Calculate weight parameters from marginals

        Parameters
        ----------
        log_marginals : (n, k+1) np.ndarray, dtype=float
            the marginals for each of the ``k+1`` slots; should be normalized:
            ``logsumexp(log_marginals, axis=0) == 0``
        kwargs : keywords
            forwarded to `solve_marginals_single`

        Returns
        -------
        logp : (n, k+1) np.ndarray, dtype=float
            the weight parameters corresponding to the given marginals

        See also
        --------
        solve_marginals_single
        """
        k = log_marginals.shape[1]-1
        assert k >= 0

        logp = np.empty(log_marginals.shape, dtype=float)
        logp[:, 0] = log_marginals[:, 0]
        for i in range(1, k+1):
            logp[:, i] = self.solve_marginals_single(log_marginals[:, i], log_marginals[:, i-1], **kwargs)

        with np.errstate(under='ignore'):
            return logp - logsumexp(logp, axis=0)

    def solve_marginals_single(self, logf, logg, max_iter=1000, precision=1e-2):
        """
        Convert marginals to weight parameters

        This is a helper function that runs the iteration needed in `estimate`.

        Parameters
        ----------
        logf, logg : (n,) np.ndarray
            marginals for current and previous step, respectively; should be
            normalized (``logsumexp(logf) == 0``).
        precision : float
            the iteration stops when the absolute difference between successive
            ``logp`` iterates is less than this value.
        max_iter : int, optional
            upper limit on iterations

        Raises
        ------
        RuntimeError
            if running out of iterations (c.f. `!max_iter`)

        Returns
        -------
        logp : (n,) np.ndarray
            the weight parameters for the step associated with the marginal
            `!f`.
        """
        # Check whether marginals are Kronecker δ's
        if np.any(logf == 0):
            return logf.copy()
        if np.any(logg == 0):
            assert np.all(logf[logg == 0] == -np.inf)
            return logf.copy()
        
        logp_old = logf
        for _ in range(max_iter):
            with np.errstate(under='ignore'):
                log_norm = logsumexp(logp_old[None, :], b=self.transitions, axis=1) # sum over j --> ?
                logg_norm = logg - log_norm
                logp = logf - logsumexp(logg_norm[:, None], b=self.transitions, axis=0) # sum over ? --> i

            if np.max(np.abs(logp-logp_old)) < precision:
                return logp
            else:
                logp_old = logp
        else:
            raise RuntimeError("Iteration did not converge")

    def uniform_marginals(self, k):
        """
        Compute the state marginals for the uniform distribution

        This allows estimating the corresponding weight parameters through
        `solve_marginals()`.

        Parameters
        ----------
        k : int
            the total number of switches

        Returns
        -------
        log_marginals : (n, k+1) np.ndarray, dtype=float
            the marginals under the uniform distribution (normalized such that
            ``logsumexp(log_marginals, axis=0) == 0``)

        Notes
        -----
        The uniform CFC is a bit tricky, because its marginals are not uniform.
        Furthermore, if any non-trivial transitions are blocked (i.e. anything
        beyond prohibiting self-transitions), then the weight parameters for
        the uniform distribution are also not all equal. So getting the
        parameters for the uniform CFC with general transition matrix is
        non-trivial.

        This function estimates the marginals under the uniform CFC by counting
        trajectories passing through a given state slot, which can be done by
        taking powers of the transition matrix. For long trajectories, this
        might result in an overflow of the ``np.integer`` types, which is why
        this function uses python's built-in ``int``, which has arbitrary
        precision. The return value is normalized and cast to regular floats.
        """
        # Implementation Notes
        # --------------------
        # + casting an integer numpy array to dtype=object makes numpy use
        #   python's built-in int—which has arbitrary precision—instead of the
        #   64-bit np.integer
        # + math.log (as opposed to np.log) can deal with large integers
        #   correctly, if applied to single variables (not numpy arrays)
        T = self.transitions.astype(int).astype(object)
        p = np.empty((self.n, k+1), dtype=object)
        for i in range(k+1):
            p[:, i] = (  ( np.ones(self.n, dtype=int) @ matrix_power(T, i)    )
                       * ( matrix_power(T, k-i)  @ np.ones(self.n, dtype=int) ) )

        mathlog = np.vectorize(math.log)
        return (mathlog(p) - mathlog(np.sum(p, axis=0))).astype(float)

    def logp_uniform(self, k):
        """
        Weight parameters for uniform distribution with `!k` switches

        Parameters
        ----------
        k : int

        Returns
        -------
        logp : (n, k+1) np.ndarray, dtype=float

        Notes
        -----
        This is just a convenience function, returning
        ``logp_from_marginals(uniform_marginals(k))``.
        """
        return self.logp_from_marginals(self.uniform_marginals(k))

    def N_total(self, k, log=False):
        """
        Give the total number of state traces with k switches

        Parameters
        ----------
        k : int
        log : bool
            if ``True``, return ``math.log(N_total)``. This is safer than
            ``np.log(N_total)``, since ``N_total`` might be large.

        Returns
        -------
        int
        """
        N = np.sum(matrix_power(self.transitions.astype(object), k))
        if log:
            return math.log(N)
        else:
            return N

    def full_sample(self, k, Nmax=1000):
        """
        Give the full sample of trajectories with `!k` switches

        Parameters
        ----------
        k : int
        Nmax : int
            assemble the sample only if it contains less than `!Nmax` traces.

        Returns
        -------
        (N, k+1) np.ndarray, dtype=int
        
        Raises
        ------
        ValueError
            if ``self.N_total(k) > Nmax``
        """
        N = self.N_total(k)
        if N > Nmax:
            raise ValueError(f"full sample would be {N} > Nmax = {Nmax} traces")

        T = self.transitions.astype(int).astype(object)
        to_list = [np.nonzero(T[i])[0].tolist() for i in range(len(T))]     # possible states from i
        ns = [matrix_power(T, i).sum(axis=1) for i in reversed(range(k+1))] # #ways to continue from state i@t

        vals = np.arange(len(T)).tolist()
        thetas = np.empty((N, k+1), dtype=int)
        thetas[:, 0] = sum((ns[0][val]*[val] for val in vals), [])
        for i in range(1, k+1):
            vals = sum((to_list[i] for i in vals), [])
            thetas[:, i] = sum((ns[i][val]*[val] for val in vals), [])

        return thetas

### Sampling ###

class ExhaustionImpractical(ValueError):
    pass
            
class FixedkSampler:
    """
    Running the AMIS scheme for a fixed number of switches k

    The most important method is `step`, which performs one AMIS iteration,
    i.e. it samples `!N` profiles from the current proposal and updates the
    weights and proposal parameters accordingly.

    Parameters
    ----------
    traj : Trajectory
        the trajectory we are sampling for
    model : MultiStateModel
        the model to use (defines the likelihood)
    k : int
        the number of switches for this run
    N : int, optional
        the sample size for each AMIS step
    concentration_brake : float, optional
        limits changes in the total concentration of the Dirichlet proposal by
        constraining ``|log(new_concentration/old_concentration)| <=
        N*concentration_brake``, where ``concentration = sum(α)``.
    polarization_brake : float, optional
        limits changes in the polarization (Bernoulli part of the proposal) by
        constraining ``|m_new - m_old| <= N*polarization_brake``.
    max_fev : int, optional
        limit on likelihood evaluations. If this limit is reached, the sampler
        will go to an "exhausted" state, where it does not allow further
        evaluations. Will be rounded down to integer multiple of `!N`.

    Attributes
    ----------
    traj, model : Trajectory, MultiStateModel
        setup for the sampling. See Parameters
    k, N : int
        see Parameters
    brakes : (concentration, polarization)
        see Parameters
    max_fev : int
        see Parameters
    exhausted : bool
        the state of the sampler. See `step`.
    samples : list
        list of samples. Each sample is a dict; the entries ``['ss', 'thetas',
        'logLs']`` are guaranteed, others might exist.
    dirichlet : Dirichlet
        the proposal distribution over switch intervals
    cfc : CFC
        the proposal distribution over state traces
    parameters : [((k+1,) np.ndarray, (n, k+1) np.ndarray)]
        proposal parameters for the samples in `!samples`, as tuple ``(a,
        logp)``.
    evidences : [(logE, dlogE, KL)]
        estimated evidence at each step. Similar to `!samples` and
        `!parameters`, this is a list with an entry for each `step`. The
        entries are tuples of log-evidence, standard error on the log-evidence,
        and Kullback-Leibler divergence ``D_KL( posterior || proposal )`` of
        posterior on proposal. The latter can come in handy in interpreting
        convergence.
    max_logL : float
        maximum value of the likelihood. Used internally to prevent overflows.
    logprior : float
        value of the uniform prior over profiles. Used internally.
    """
    # Potential further improvements:
    #  + make each proposal a mixture of Dirichlet's to catch multimodal behavior
    #  + use MAP instead of MOM estimation, which would replace the brake
    #    parameters with proper priors; this turns out to be technically quite
    #    involved and numerically unstable, so stick with MOM + brakes for now.
    def __init__(self, traj, model, k,
                 N=100,
                 concentration_brake=1e-2,
                 polarization_brake=1e-3,
                 max_fev = 20000,
                ):
        self.k = k
        self.N = N
        self.brakes = (concentration_brake, polarization_brake)
        
        self.max_fev = max_fev - (max_fev % self.N)
        self.exhausted = False
        
        self.traj = traj
        self.model = model

        if self.k >= len(self.traj):
            # This is by construction unidentifiable
            # Should we warn the user? Probably more annoying than necessary;
            # this will a) probably not be relevant in production and b) in any
            # case just serve as early termination in core.sample()
            self.evidences = [(-np.inf, 1e-10, np.inf)]
            self.exhausted = True
            return
        
        self.dirichlet = Dirichlet()
        self.cfc = CFC(model.transitions)
        self.parameters = [(np.ones(self.k+1), self.cfc.logp_uniform(self.k))]

        # Value of the uniform prior over profiles
        # Profiles are defined by θ, which has CFC.N_total() possible values and s
        # in the unit simplex, which has volume 1/k!. Consequently, the uniform
        # prior should be k!/( CFC.N_total() ).
        # Note that for k = 0, ``sum([]) == 0 == log(0!)`` still works
        self.logprior = np.sum(np.log(np.arange(self.k)+1)) - self.cfc.N_total(self.k, log=True)

        self.samples = [] # each sample is a dict with keys ['ss', 'thetas', 'logLs', 'logδs', 'log_weights']
        self.evidences = [] # each entry: (logev, dlogev, KL)
        
        # Sample exhaustively, if possible
        try:
            self.fix_exhaustive()
        except ExhaustionImpractical:
            pass

    @property
    def n(self):
        """
        Alias for ``self.model.nStates``
        """
        return self.model.nStates

    def log_proposal(self, parameters, ss, thetas):
        return ( self.dirichlet.log_likelihood(parameters[0], ss)
                     + self.cfc.log_likelihood(parameters[1], thetas) )

    def fix_exhaustive(self, Nmax=1000):
        """
        Evaluate by exhaustive sampling of the parameter space

        Parameters
        ----------
        Nmax : int, optional
            threshold up to which exhaustive sampling is considered worthwhile

        Raises
        ------
        ExhaustionImpractical
            if parameter space is too big to warrant exhaustive sampling; i.e.
            there would be more than `!Nmax` profiles to evaluate.

        Notes
        -----
        Since in this case the evidence is exact, its standard error should be
        zero. To avoid numerical issues, we set ``dlogev = 1e-10``.
        """
        Nsamples = self.cfc.N_total(self.k)
        for i in range(self.k):
            Nsamples *= len(self.traj) - i - 1
            if Nsamples > Nmax:
                raise ExhaustionImpractical(f"Parameter space too large for exhaustive sampling (number of profiles = {Nsamples} > Nmax = {Nmax})")

        # Assemble full sample
        # 1. switches
        switch_iter = itertools.combinations(np.arange(len(self.traj)-1)+0.5, self.k)
        normed_switches = np.array(list(switch_iter)) / (len(self.traj)-1)
        normed_switches = np.append(np.insert(normed_switches, 0, 0, axis=1),
                                    np.ones((len(normed_switches), 1)), axis=1)
        ss = np.diff(normed_switches, axis=1)

        # 2. states
        thetas = self.cfc.full_sample(self.k, Nmax=Nmax) # (note: different Nmax; but fine)
        
        # 3. multiply
        ss = np.tile(ss, (len(thetas), 1))
        thetas = np.tile(thetas[:, None, :], (1, len(ss), 1)).reshape(-1, thetas.shape[-1])

        # For exhaustive sampling, the proposal is uniform over the profiles,
        # i.e. equal to the prior, which thus drops out.
        # The expressions here are thus slightly different from the ones used
        # in `step()` below.
        sample = {'ss' : ss, 'thetas' : thetas}
        sample['logLs'] = calculate_logLs(sample['ss'], sample['thetas'],
                                          self.traj, self.model,
                                          )
        self.samples.append(sample)
        
        # Evidence & KL
        # do logsumexp manually, which allows calculating KL
        max_logL = np.max(sample['logLs'])
        with np.errstate(under='ignore'):
            weights_o = np.exp(sample['logLs'] - max_logL)
        ev_o = np.mean(weights_o)

        logev = np.log(ev_o) + max_logL
        dlogev = 1e-10
        KL = np.mean(sample['logLs'] * weights_o) / ev_o - logev
        
        self.evidences.append((logev, dlogev, KL))

        # Prevent the actual sampling from running
        self.exhausted = True

    def step(self):
        """
        Run a single step of the AMIS sampling scheme

        Returns
        -------
        bool
            whether the sample was performed. ``False`` if sampler is
            exhausted, ``True`` otherwise.
        """
        if self.exhausted:
            return False
        
        # Update δ's on old samples
        # Keep track of evaluation of current proposal, which we use for KL below
        for sample in self.samples:
            sample['cur_log_proposal'] = self.log_proposal(self.parameters[-1], sample['ss'], sample['thetas'])
            with np.errstate(under='ignore'):
                sample['logδs'] = logaddexp(sample['logδs'], sample['cur_log_proposal'])

        # Put together new sample
        sample = {
            'ss'     : self.dirichlet.sample(self.parameters[-1][0], self.N),
            'thetas' :       self.cfc.sample(self.parameters[-1][1], self.N),
        }
        sample['logLs'] = calculate_logLs(sample['ss'], sample['thetas'], self.traj, self.model)
        sample['cur_log_proposal'] = self.log_proposal(self.parameters[-1], sample['ss'], sample['thetas'])
        with np.errstate(under='ignore'):
            sample['logδs'] = logsumexp([self.log_proposal(params, sample['ss'], sample['thetas'])
                                         for params in self.parameters[:-1]
                                         ] + [sample['cur_log_proposal']], axis=0)
        self.samples.append(sample)

        # Calculate weights for all samples
        logNsteps = np.log(len(self.parameters)) # normalization for δs (should be means)
        for sample in self.samples:
            sample['log_weights'] = sample['logLs'] - sample['logδs'] + logNsteps

        # Assemble full ensemble
        full_ensemble = {key : np.concatenate([sample[key] for sample in self.samples], axis=0)
                         for key in self.samples[-1]
                         }

        # Update proposal
        old_a, old_logp = self.parameters[-1]
        new_a    = self.dirichlet.estimate(full_ensemble['ss'],     full_ensemble['log_weights'])
        new_logp =       self.cfc.estimate(full_ensemble['thetas'], full_ensemble['log_weights'])

        # Keep concentration from exploding
        log_concentration_ratio = np.log( np.sum(new_a) / np.sum(old_a) )
        if np.abs(log_concentration_ratio) > self.N*self.brakes[0]:
            new_a *= np.exp(np.sign(log_concentration_ratio)*self.N*self.brakes[0] - log_concentration_ratio)

        # Keep polarizations from exploding (each i individually)
        # the interpolation for this works naturally only in linear space, so
        # let's work there
        with np.errstate(under='ignore'):
            old_p = np.exp(old_logp)
            new_p = np.exp(new_logp)

        for i in range(new_p.shape[1]):
            delta = new_p[:, i] - old_p[:, i]
            max_abs_delta = np.max(np.abs(delta))
            if max_abs_delta > self.N*self.brakes[1]:
                new_logp[:, i] = np.log( old_p[:, i] + self.N*self.brakes[1] * delta/max_abs_delta )

        self.parameters.append((new_a, new_logp))

        # Evidence & KL
        # do logsumexp manually, which allows calculating sem & KL
        max_log_weight = np.max(full_ensemble['log_weights'])
        with np.errstate(under='ignore'):
            weights_o = np.exp(full_ensemble['log_weights'] - max_log_weight)
        ev_o = np.mean(weights_o)

        logev = np.log(ev_o) + max_log_weight + self.logprior
        dlogev = stats.sem(weights_o) / ev_o # offset and prior cancel
        with np.errstate(under='ignore', invalid='ignore'):
            # old, reevaluated samples with new proposal = 0 will have weight =
            # 0, cur_log_proposal = -np.inf and thus raise "invalid value in
            # multiply" and return np.nan. This is fine, we can just ignore
            # them; but note that ev_o is normalized to the full length of
            # weights_o, regardless of whether some are zero. So instead of
            # np.nanmean, use np.nansum / len for the normalized summation.
            KL = (
                    np.nansum(weights_o * ( full_ensemble['logLs']
                                           -full_ensemble['cur_log_proposal'] )
                              ) / len(weights_o) / ev_o
                   -logev
                   +self.logprior
                 )
        
        self.evidences.append((logev, dlogev, KL))
        
        # Check whether we can still sample more in the future
        if len(self.samples)*self.N >= self.max_fev:
            self.exhausted = True
        return True
        
    def t_stat(self, other):
        """
        Calculate separation (by evidence) from another sampler

        Inspired by the t-statistic from frequentist inference, we use the
        separation score ``(logev - other_logev) / sqrt( dlogev² +
        other_dlogev² )``.

        Returns
        -------
        float
        """
        logev0, dlogev0 = self.evidences[-1][:2]
        logev1, dlogev1 = other.evidences[-1][:2]

        effect = logev0 - logev1
        return effect / np.sqrt( dlogev0**2 + dlogev1**2 )
    
    def MAP_profile(self):
        """
        Give the current MAP estimate

        Returns
        -------
        Loopingprofile
        """
        best_logL = -np.inf
        for sample in self.samples:
            i = np.argmax(sample['logLs'])
            if sample['logLs'][i] > best_logL:
                best_logL = sample['logLs'][i]
                s = sample['ss'][i]
                t = sample['thetas'][i]
                
        return st2profile(s, t, self.traj)
