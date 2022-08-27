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
import numpy as np
from scipy import stats

from noctiluca.util import parallel
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
    states = theta*np.ones(len(traj))
    if len(s) > 1:
        switchpos = np.cumsum(s)[:-1]
        
        switches = np.floor(switchpos*(len(traj)-1)).astype(int) + 1 # floor(0.0) + 1 = 1 != ceil(0.0)
        for i in range(1, len(switches)):
            states[switches[i-1]:switches[i]] = theta if i % 2 == 0 else 1-theta
            
        states[switches[-1]:] = theta if len(switches) % 2 == 0 else 1-theta
    
    return Loopingprofile(states)

def dirichlet_methodofmoments(ss, normalized_weights):
    """
    Method of moments estimator for the Dirichlet distribution

    Parameters
    ----------
    ss : (n, k+1) np.ndarray, dtype=float
        the switch positions ``s`` for a sample of size ``n``
    normalized_weights : (n,) np.ndarray, dtype=float
        the associated weights for all samples. Should satisfy
        ``np.sum(normalized_weights) = 1``.

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

    See also
    --------
    sample_proposal
    """
    m = normalized_weights @ ss
    v = normalized_weights @ (ss - m[np.newaxis, :])**2
    if np.any(v == 0):
        # this is almost a pathological case, but it is possible; best solution
        # is to return very concentrated (but finite!) distribution and let the
        # concentration brake do its work.
        s = 1e10 # pragma: no cover
    else:
        s = np.mean(m*(1-m)/v) - 1
    return s*m

### Likelihood, i.e. target distribution ###

def logL(params):
    """
    Evaluate the model likelihood for a given profile

    The argument to this function is an aggregate of all the necessary
    information, such that we can evaluate it in parallel (see
    `calculate_logLs`). Its structure is ``params = ((s, theta), (traj,
    model))``.

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
    ((s, theta), (traj, model)) = params
    profile = st2profile(s, theta, traj)
    return model.logL(profile, traj)

def calculate_logLs(ss, thetas, traj, model):
    """
    Evaluate the likelihood on an ensemple of profiles

    Parameters
    ----------
    ss : (n, k+1) np.ndarray, dtype=float
    thetas : (n,) np.ndarray, dtype=int
    traj : Trajectory
    model : MultiStateModel

    Returns
    -------
    (n,) np.ndarray, dtype=float
        the likelihoods associated with each profile

    Notes
    -----
    This function is parallel-aware (ordered). See `noctiluca.util.parallel`
    """
    todo = zip(ss, thetas)
    todo = zip(todo, len(ss)*[(traj, model)])
    imap = parallel._map(logL, todo)
    return np.array(list(imap))

### Proposal distribution ###

def proposal(a, m, ss, thetas):
    """
    Evaluate the proposal distribution at a given point

    Parameters
    ----------
    a, m : (k+1,) np.ndarray, dtype=float; float in [0, 1]
        the parameters for the proposal distribution (Dirichlet(α) x
        Bernoulli(m))
    ss : (n, k+1) np.ndarray, dtype=float
    thetas : (n,) np.ndarray, dtype=int

    Returns
    -------
    float

    See also
    --------
    sample_proposal
    """
#     try: # this was supposed to catch instances where dirichlet.pdf raises a
           # ValueError due to the sample points not lying strictly within the
           # simplex, i.e 0 < s < 1. But this should not happen in production
           # anyways, so it seems appropriate to raise this error.
    with np.errstate(under='ignore'): # pdf(.) == exp(_logpdf(.)) and exp may underflow
        return (
            stats.dirichlet(a).pdf(ss.T)
            * ( m*thetas + (1-m)*(1-thetas) )
        )
#     except ValueError:
#         # dirichlet.pdf got an argument that has a zero somewhere, but its
#         # alpha is < 1.
#         if len(s.shape) == 1:
#             s = s[None, :]
#             theta = np.asarray(theta)
# 
#         ind = np.any(s[:, a < 1] == 0, axis=1)
#         if np.sum(ind) == 0:
#             raise RuntimeError("Could not identify 0s in sample")
# 
#         out = np.empty(len(s), dtype=float)
#         if np.sum(~ind) > 0:
#             out[~ind] = proposal(a, m, s[~ind], theta[~ind])
#         out[ind] = np.inf
# 
#         return out

# def log_proposal(a, m, s, theta0):
#     return (
#         stats.dirichlet(a).logpdf(s.T)
#         + np.log( m*theta0 + (1-m)*(1-theta0) )
#     )

def sample_proposal(a, m, N):
    """
    Sample from the proposal distribution

    Parameters
    ----------
    a, m : (k+1,) np.ndarray, dtype=float; float in [0, 1]
        the parameters for the proposal distribution (Dirichlet(α) x
        Bernoulli(m))
    N : int
        size of the sample

    Returns
    -------
    ss : (N, k+1) np.ndarray, dtype=float
        the switch positions for each sample
    thetas : (N,) np.ndarray, dtype=int
        the initial state for each sample

    See also
    --------
    fit_proposal
    """
    ss = stats.dirichlet(a).rvs(N)
    thetas = (np.random.rand(N) < m).astype(int)
    return ss, thetas

def fit_proposal(ss, thetas, weights):
    """
    Fit the proposal distribution to a weighted sample

    Parameters
    ----------
    ss : (n, k+1) np.ndarray, dtype=float
    thetas : (n,) np.ndarray, dtype=int
    weights : (n,) np.ndarray, dtype=float
        the weights associated with the sample ``(ss, thetas)``

    Returns
    -------
    a, m : (k+1,) np.ndarray, dtype=float; float in [0, 1]
        the parameters for the proposal distribution (Dirichlet(α) x
        Bernoulli(m))

    See also
    --------
    sample_proposal, dirichlet_methodofmoments
    """
    weights = weights / np.sum(weights)
    a = dirichlet_methodofmoments(ss, weights)
    m = thetas @ weights
    return a, m

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
        list of samples. Each samples is a list ``[ss, thetas, logLs, deltas,
        weights]`` where each entry is an ``(N, ...) np.ndarray``. ``ss`` and
        ``thetas`` define the `!N` sample points, ``logLs`` are the associated
        likelihoods, ``deltas`` are auxiliary variables for AMIS, and
        ``weights`` are the associated weights for each point (updated at each
        `step`).
    parameters : [((k+1,) np.ndarray, float)]
        proposal parameters for the samples in `!samples`, as tuple ``(a, m)``.
    evidences : [(logE, dlogE, KL)]
        estimated evidence at each step. Similar to `!samples` and
        `!parameters`, this is a list with an entry for each `step`. The
        entries are tuples of log-evidence, standard error on the log-evidence,
        and Kullback-Leibler divergence ``D_KL( proposal || posterior )`` of
        proposal on posterior. The latter can come in handy in interpreting
        convergence.
    max_logL : float
        maximum value of the likelihood. Used internally to prevent overflows.
    logprior : float
        value of the uniform prior over profiles. Used internally.
    """
    # Potential further improvements:
    #  + make each proposal a mixture of Dirichlet's to catch multimodal behavior
    #  + improve proposal fitting / braking (better than MOM, maybe more something gradient like?)
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
        
        self.parameters = [(np.ones(k+1), 0.5)]
        self.samples = [] # each sample is a list: [s, theta, logL, δ, w], where each entry is an (N, ...) array
        self.evidences = [] # each entry: (logev, dlogev, KL)
        self.max_logL = -np.inf

        # Value of the uniform prior over profiles
        # Noting that the profiles are defined by θ in {0, 1} and s in the unit
        # k-simplex, whose volume is 1/k!, the prior should be k!/2.
        # Note that for k = 0, ``sum([]) == 0 == log(0!)`` still works
        self.logprior = np.sum(np.log(np.arange(self.k)+1)) - np.log(2)
        
        # Sample exhaustively, if possible
        try:
            self.fix_exhaustive()
        except ExhaustionImpractical:
            pass

    def fix_exhaustive(self):
        """
        Evaluate by exhaustive sampling of the parameter space

        Raises
        ------
        ValueError
            if parameter space is too big to warrant exhaustive sampling.
            Currently this means ``self.k >= 2``.

        Notes
        -----
        Since in this case the evidence is exact, its standard error should be
        zero. To avoid numerical issues, we set ``dlogev = 1e-10``.
        """
        # For exhaustive sampling, the proposal is uniform over the profiles,
        # i.e. equal to the prior.
        if self.k == 0:
            sample = [np.ones((2, 1)), np.array([0, 1]),
                      None, np.exp(self.logprior)*np.ones(2), None]
        elif self.k == 1:
            switches = (np.arange(len(self.traj)-1)+0.5)/(len(self.traj)-1)
            ss = np.concatenate(2*[np.array([switches, 1-switches]).T], axis=0)
            thetas = np.concatenate([np.zeros(len(switches)), np.ones(len(switches))], axis=0).astype(int)
            sample = [ss, thetas,
                      None, np.exp(self.logprior)*np.ones(len(ss)), None]
        else:
            raise ExhaustionImpractical(f"Parameter space too large for exhaustive sampling (k = {self.k} > 1)")
            
        sample[2] = calculate_logLs(sample[0], sample[1], self.traj, self.model)
        self.max_logL = np.max(sample[2])
        with np.errstate(under='ignore'):
            Ls = np.exp(sample[2] - self.max_logL)
        sample[4] = Ls / sample[3]
        
        self.samples.append(sample)
        
        ev_offac = np.mean(sample[4]) # offset by a factor exp(max_logL)
        dlogev = 1e-10
        logev_offac = np.log(ev_offac) + self.logprior
        with np.errstate(divide='ignore'): # we might get log(0), but in that case KL is just +inf, that's fine
            KL = logev_offac - np.mean(np.log(sample[4]))
        self.evidences.append((logev_offac + self.max_logL, dlogev, KL))

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
        for sample in self.samples:
            sample[3] += proposal(*self.parameters[-1], sample[0], sample[1])

        # Put together new sample
        sample = 5*[None]
        sample[0], sample[1] = sample_proposal(*self.parameters[-1], self.N)
        sample[2] = calculate_logLs(sample[0], sample[1], self.traj, self.model)

        self.max_logL = max(self.max_logL, np.max(sample[2])) # Keep track of max(logL) such that we can properly convert to weights later

        sample[3] = np.zeros(self.N)
        for a, m in self.parameters:
            sample[3] += proposal(a, m, sample[0], sample[1])

        self.samples.append(sample)

        # Calculate weights for all samples
        for sample in self.samples:
            with np.errstate(under='ignore'):
                Ls = np.exp(sample[2] - self.max_logL)
            sample[4] = Ls / (sample[3] / len(self.parameters))

        # Update proposal
        # full_ensemble is [s, theta, w_offac]
        full_ensemble = [np.concatenate([sample[i] for sample in self.samples], axis=0) for i in [0, 1, 4]]

        old_a, old_m = self.parameters[-1]
        new_a, new_m = fit_proposal(*full_ensemble)

        # Keep concentration from exploding
        concentration_ratio = np.sum(new_a) / np.sum(old_a)
        if np.abs(np.log(concentration_ratio))/self.N > self.brakes[0]:
            logfac = self.N*self.brakes[0]
            if concentration_ratio < 1:
                logfac *= -1 # pragma: no cover # can't think of a test case

            new_a = old_a * np.exp(logfac)

        # Keep polarization from exploding
        if np.abs(new_m - old_m)/self.N > self.brakes[1]:
            if new_m > old_m:
                new_m = old_m + self.N*self.brakes[1]
            else:
                new_m = old_m - self.N*self.brakes[1]

        self.parameters.append((new_a, new_m))

        # Evidence & KL
        ev_offac = np.mean(full_ensemble[2]) # offset by a factor exp(max_logL)
        dlogev = stats.sem(full_ensemble[2]) / ev_offac
        logev_offac = np.log(ev_offac) + self.logprior
        
        with np.errstate(divide='ignore'): # we might get log(0), but in that case KL is just +inf, that's fine
            KL = logev_offac - np.mean(np.log(full_ensemble[2]))
        
        self.evidences.append((logev_offac + self.max_logL, dlogev, KL))
        
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
            i = np.argmax(sample[2])
            if sample[2][i] > best_logL:
                best_logL = sample[2][i]
                s = sample[0][i]
                t = sample[1][i]
                
        return st2profile(s, t, self.traj)
