"""
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
import itertools

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from numpy         import logaddexp

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

class Dirichlet:
    """
    Namespace for stuff associated with Dirichlet distribution

    The Dirichlet is implemented in ``scipy.stats.dirichlet``; we just wrap it
    here for consistency with `CFC` and `Proposal`.
    """
    def sample(a, N=1):
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

    def log_likelihood(a, ss):
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
    
    def estimate(ss, log_weights):
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

class CFC:
    """
    Namespace for the Conflict Free Categorical (CFC)

    The CFC is our proposal distribution for state traces θ. These are vectors
    of length ``k+1``, containing integers between 0 and ``n-1``—``k`` being
    the number of switches and ``n`` the number of states we consider. The
    fundamental twist is that neighboring entries cannot take the same value,
    which makes it difficult to define a reasonable proposal distribution over
    this space of state traces.

    The CFC is parametrized by weights ``p`` for each entry being in any of the
    available states (``p.shape = (n, k+1)``). The constraint is then enforced
    by a causal sampling scheme:

     + sample ``θ[0] ~ Categorical(p[:, 0])``
     + remove the weight corresponding to ``θ[0]`` from ``p[:, 1]`` and
       renormalize
     + sample ``θ[1] ~ Categorical(p[:, 1])`` (with the adapted ``p[:, 1]``)
     + continue iteratively

    This scheme makes sampling and likelihood evaluation relatively
    straight-forward. For estimation, we have to convert the actually observed
    marginals in a sample back to the weights ``p``. Letting ``f_n`` and
    ``g_n`` denote the marginals at step ``i`` and ``i-1`` respectively, we
    have that ``p_n := p[:, i]`` is given by the solution of
    ```
        p_n = f_n / sum_{m!=n}( g_m / (1-p_m) ) .
    ```
    Solving this equation by fixed point iteration, we find approximate
    parameters ``p`` for a given sample of state traces.

    Knowing how to sample, evaluate, and estimate the CFC is all we need in
    order to use it as proposal in AMIS.

    For numerical stability, we work with ``log(p)`` instead of ``p``.
    """
    def sample(logp, N=1):
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
        n = logp.shape[0]
        k = logp.shape[1]-1
        assert k >= 0

        # For sampling we actually need p, not log(p)
        with np.errstate(under='ignore'):
            p = np.exp(logp)

        thetas = np.empty((N, k+1), dtype=int)
        thetas[:, 0] = np.random.choice(n, size=N, p=p[:, 0])
        for i in range(1, k+1):
            p_cur = np.tile(p[:, i], (N, 1)) # (N, n)                  # current weight parameters
            np.put_along_axis(p_cur, thetas[:, [i-1]], 0, axis=1)      # remove the one corresponding to last entry

            P = np.cumsum(p_cur, axis=1)                               # cumulative weights
            P /= P[:, [-1]]                                            # normalize

            # "vectorize np.random.choice" by hand                     # sample
            thetas[:, i] = np.argmax(P > np.random.rand(N, 1), axis=1) # argmax gives first occurence

        return thetas

    def log_likelihood(logp, thetas):
        """
        Evaluate the Conflict Free Categorical

        Parameters
        ----------
        logp : (n, k+1) np.ndarray
            the weights defining the CFC. Should satisfy ``logsumexp(logp,
            axis=0) == 0`` if you want the distribution to be properly
            normalized; this is not checked internally.
        thetas : (N, k+1) np.ndarray
            the ``N`` state traces for which to evaluate the distribution

        Returns
        -------
        (N,) np.ndarray
            the logarithm of the CFC evaluated at the given state traces
        """
        #                                  (1, n, k+1)     x    (N, 1, k+1)    -->    (N, 1, k+1)
        logp_direct = np.take_along_axis(logp[None, :,  :], thetas[:, None, :  ], axis=1)[:, 0, :] # (N, k+1)
        logp_shift  = np.take_along_axis(logp[None, :, 1:], thetas[:, None, :-1], axis=1)[:, 0, :] # (N, k)

        log_numerator = np.sum(logp_direct, axis=1)
        with np.errstate(under='ignore', divide='ignore'):
            log_denominator = np.sum(np.log(1-np.exp(logp_shift)), axis=1)
        log_denominator[log_denominator == -np.inf] = np.inf
        # re: divide = 'ignore' & switching -∞ --> +∞
        # encountering log(0) in the above expression means that p has at least
        # one slot that is a Kronecker δ, i.e. 1 somewhere, 0 elsewhere. When
        # trying to evaluate this distribution on a trace that does not satisfy
        # that constraint, we get 0/0 from the analytical expression; but of
        # course the distribution should just evaluate to 0 here. So we switch
        # the denominator to get 0/∞ instead of 0/0, which evaluates correctly.

        return log_numerator - log_denominator

    def estimate(thetas, log_weights, n=None):
        """
        Method of "Moments" (Marginals) estimation for the CFC

        Parameters
        ----------
        thetas : (N, k+1) np.ndarray, dtype=int
            the sample of state traces
        log_weights : (N,) np.ndarray, dtype=float
            the associated weights for all samples (unnormalized)
        n : int, optional
            the total number of states. If omitted, use ``np.max(thetas)``;
            however, this is not very safe, so usually one should give ``n``
            explicitly.

        Returns
        -------
        logp : (n, k+1) np.ndarray
            the estimated weight parameters; normalized to satisfy
            ``logsumexp(logp, axis=0) == 0``.
        """
        k = thetas.shape[1] - 1
        assert k >= 0

        if n is None:
            n = np.max(thetas)

        indicators = np.tile(thetas, (n, 1, 1)) == np.arange(n)[:, None, None]
        with np.errstate(under='ignore'):
            log_marginals = logsumexp(log_weights[None, :, None], b=indicators, axis=1) # (n, k+1)
            log_marginals -= logsumexp(log_marginals, axis=0, keepdims=True)

        logp = np.empty((n, k+1), dtype=float)
        logp[:, 0] = log_marginals[:, 0]
        for i in range(1, k+1):
            logp[:, i] = CFC.log_solve_marginals(log_marginals[:, i], log_marginals[:, i-1])
            
        with np.errstate(under='ignore'):
            return logp - logsumexp(logp, axis=0)

    def log_solve_marginals(logf, logg, max_iter=1000):
        """
        Convert marginals to weight parameters

        This is a helper function that runs the iteration needed in
        `CFC.estimate`.

        Parameters
        ----------
        logf, logg : (n,) np.ndarray
            marginals for current and previous step, respectively; should be
            normalized (``logsumexp(logf) == 0``).
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
                gkpk = np.tile(logg - np.log(1-np.exp(logp_old)), (len(logg), 1))
                np.put_along_axis(gkpk, np.arange(len(logg))[:, None], -np.inf, axis=1)
                logp = logf - logsumexp(gkpk, axis=1)

            if np.max(np.abs(logp-logp_old)) < 1e-2:
                return logp
            else:
                logp_old = logp
        else:
            raise RuntimeError("Iteration did not converge")
            
class Proposal:
    """
    Namespace for the full proposal distribution (Dirichlet x CFC)
    """
    def sample(a, logp, N=1):
        """
        Sample from the proposal distribution

        Parameters
        ----------
        a : (k+1,) np.ndarray, dtype=float
            the concentration parameters for the Dirichlet
        logp : (n, k+1) np.ndarray, dtype=float
            the weight parameters for the CFC
        N : int, optional
            size of the sample

        Returns
        -------
        ss : (N, k+1) np.ndarray, dtype=float
            switch positions
        thetas : (N, k+1) np.ndarray, dtype=int
            state traces
        """
        return Dirichlet.sample(a, N), CFC.sample(logp, N)

    def log_likelihood(a, logp, ss, thetas):
        """
        Evaluate the proposal distribution at a given point

        Parameters
        ----------
        a : (k+1,) np.ndarray, dtype=float
            the concentration parameters for the Dirichlet
        logp : (n, k+1) np.ndarray, dtype=float
            the weight parameters for the CFC
        ss : (N, k+1) np.ndarray, dtype=float
            sample of switch positions
        thetas : (N,) np.ndarray, dtype=int
            associated sample of state traces

        Returns
        -------
        (N,) np.ndarray, dtype=float
        """
        return Dirichlet.log_likelihood(a, ss) + CFC.log_likelihood(logp, thetas)

    def estimate(ss, thetas, log_weights, n=None):
        """
        Fit the proposal distribution to a weighted sample

        Parameters
        ----------
        ss : (N, k+1) np.ndarray, dtype=float
            sample of switch positions
        thetas : (N,) np.ndarray, dtype=int
            associated sample of state traces
        log_weights : (N,) np.ndarray, dtype=float
            the weights associated with the sample ``(ss, thetas)``
        n : int, optional
            the number of states we consider. This should be specified, since
            the state traces in `!thetas` might not cover all possible states.

        Returns
        -------
        a : (k+1,) np.ndarray, dtype=float
            the concentration parameters for the Dirichlet
        logp : (n, k+1) np.ndarray, dtype=float
            the weight parameters for the CFC
        """
        return Dirichlet.estimate(ss, log_weights), CFC.estimate(thetas, log_weights, n)

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
    parameters : [((k+1,) np.ndarray, float)]
        proposal parameters for the samples in `!samples`, as tuple ``(a, m)``.
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
        
        self.parameters = [(np.ones(self.k+1), -np.log(self.n)*np.ones((self.n, self.k+1)))]
        self.samples = [] # each sample is a dict with keys ['ss', 'thetas', 'logLs', 'logδs', 'log_weights']
        self.evidences = [] # each entry: (logev, dlogev, KL)

        # Value of the uniform prior over profiles
        # Profiles are defined by θ, which has n*(n-1)^k possible values and s
        # in the unit simplex, which has volume 1/k!. Consequently, the uniform
        # prior should be k!/( n*(n-1)^k ).
        # Note that for k = 0, ``sum([]) == 0 == log(0!)`` still works
        self.logprior = np.sum(np.log(np.arange(self.k)+1)) - np.log(self.n) - self.k*np.log(self.n-1)
        
        # Sample exhaustively, if possible
        try:
            self.fix_exhaustive()
        except ExhaustionImpractical:
            pass

    @property
    def n(self):
        """
        The number of states we consider

        This is just an alias for ``self.model.nStates``
        """
        return self.model.nStates

    def fix_exhaustive(self):
        """
        Evaluate by exhaustive sampling of the parameter space

        Raises
        ------
        ExhaustionImpractical
            if parameter space is too big to warrant exhaustive sampling.
            Currently this means that there are more than 10^3 possible
            profiles

        Notes
        -----
        Since in this case the evidence is exact, its standard error should be
        zero. To avoid numerical issues, we set ``dlogev = 1e-10``.
        """
        Nsamples = self.n * (self.n-1)**self.k
        for i in range(self.k):
            Nsamples *= len(self.traj) - i - 1
            if Nsamples > 1000:
                raise ExhaustionImpractical(f"Parameter space too large for exhaustive sampling (number of profiles = {Nsamples} > 1000)")

        # Assemble full sample
        # 1. switches
        def switchpos2ss(switches):
            normed = switches / (len(self.traj)-1)
            return np.diff([0] + normed.tolist() + [1])

        switch_iter = itertools.combinations(np.arange(len(self.traj)-1)+0.5, self.k)
        ss_iter = map(switchpos2ss, map(np.asarray, switch_iter))

        # 2. states
        def red2theta(red):
            for i in range(1, len(red)):
                red[i] += int(red[i] >= red[i-1])
            return red

        red_iter = itertools.product(*([np.arange(self.n)] + self.k*[np.arange(self.n-1)]))
        thetas_iter = map(red2theta, map(np.asarray, red_iter))
        
        # 3. multiply
        ss_thetas = list(itertools.product(ss_iter, thetas_iter))
        ss     = np.array([s for s, _ in ss_thetas])
        thetas = np.array([t for _, t in ss_thetas])

        # For exhaustive sampling, the proposal is uniform over the profiles,
        # i.e. equal to the prior, which thus drops out.
        # The expressions here are thus slightly different from the ones used
        # in `step()` below.
        sample = {
                'ss'     : np.array([s for s, _ in ss_thetas]),
                'thetas' : np.array([t for _, t in ss_thetas]), 
                }
            
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
            sample['cur_log_proposal'] = Proposal.log_likelihood(*self.parameters[-1], sample['ss'], sample['thetas'])
            with np.errstate(under='ignore'):
                sample['logδs'] = logaddexp(sample['logδs'], sample['cur_log_proposal'])

        # Put together new sample
        sample = dict()
        sample['ss'], sample['thetas'] = Proposal.sample(*self.parameters[-1], self.N)
        sample['logLs'] = calculate_logLs(sample['ss'], sample['thetas'], self.traj, self.model)

        sample['cur_log_proposal'] = Proposal.log_likelihood(*self.parameters[-1], sample['ss'], sample['thetas'])
        with np.errstate(under='ignore'):
            sample['logδs'] = logsumexp([Proposal.log_likelihood(a, logp, sample['ss'], sample['thetas'])
                                         for a, logp in self.parameters[:-1]
                                         ] + [sample['cur_log_proposal']], axis=0)
        sample['log_weights'] = None # will be written below
        self.samples.append(sample)

        # Calculate weights for all samples
        logNsteps = np.log(len(self.parameters)) # normalization for δs (should be means)
        for sample in self.samples:
            sample['log_weights'] = sample['logLs'] - sample['logδs'] + logNsteps

        # Update proposal
        full_ensemble = {key : np.concatenate([sample[key] for sample in self.samples], axis=0)
                         for key in self.samples[-1]
                         }

        old_a, old_logp = self.parameters[-1]
        new_a, new_logp = Proposal.estimate(full_ensemble['ss'],
                                            full_ensemble['thetas'],
                                            full_ensemble['log_weights'],
                                            n = self.n,
                                            )


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
