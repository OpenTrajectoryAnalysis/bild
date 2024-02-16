"""
Core implementation of the `bild` module
"""
from tqdm.auto import tqdm

import numpy as np
from scipy.special import logsumexp

from noctiluca import make_Trajectory

from .amis import FixedkSampler
            
"""
exec 'norm jj^d}O' | let @a="\n'" | exec 'g/^\(def\|class\)/exec ''norm w"Ayw'' | let @a=@a."'',\n''"' | exec 'norm i__all__ = ["ap}kcc]kV?__all__?+>k'
"""
__all__ = [
    'sample',
    'SamplingResults',
]

def sample(traj, model,
           dE = 0,
           init_runs = 20,
           certainty_in_k = 0.99,
           k_lookahead = 2,
           k_max = 20,
           sampler_kw = {},
           choice_kw = {},
           show_progress=False,
          ):
    """
    Entry point for BILD

    This function executes the whole BILD scheme, i.e. it takes a trajectory
    and returns a best-fit looping profile (plus some of the internally stored
    variables that might be of interest for downstream analysis).

    Parameters
    ----------
    traj : noctiluca.Trajectory, numpy.ndarray, pandas.DataFrame
        the trajectory to sample for.
        
        This input is handled by the `userinput.make_Trajectoy()
        <https://noctiluca.readthedocs.io/en/latest/noctiluca.util.html#noctiluca.util.userinput.make_Trajectory>`_
        function of the `noctiluca` package and thus accepts a range of
        formats. For a trajectory with ``N`` loci, ``T`` frames, and ``d``
        spatial dimensions, numpy arrays can be of shape ``(N, T, d)``, ``(T,
        d)``, ``(T,)``, while pandas dataframes should have columns ``(x1, y1,
        z1, x2, y2, z2, ..., frame (optional))``. For precise specs see
        `noctiluca.util.userinput
        <https://noctiluca.readthedocs.io/en/latest/noctiluca.util.html#module-noctiluca.util.userinput>`_.
    model : models.MultiStateModel
        the model defining the likelihood function
    dE : float, optional
        the evidence margin ΔE to apply in finding the actual point estimate.
        Note that this can also be set post-hoc when evaluating the results;
        c.f. `SamplingResults.best_profile()`.

    Other Parameters
    ----------------
    init_runs : int
        minimum number of AMIS runs for a new value of ``k``.
    significant_separation_sem_fold : float
        `FixedkSampler.tstat`
    k_lookahead : int
        how far we look ahead for global maxima. As long as there is a current
        candidate for global optimum at ``k >= k_max - k_lookahead`` (where
        ``k_max`` is the largest ``k`` sampled so far), sample larger ``k``
        instead of refining the ones we have. The default value of 2 comes from
        the fact that for binary profiles the evidence follows an odd-even
        pattern, so it is necessary to look ahead by 2 additional switches.
    k_max : int
        the maximum number of switches to sample to
    sampler_kw : dict
        keyword arguments for the `amis.FixedkSampler`.
    show_progress : bool
        whether to show a progress bar

    Returns
    -------
    SamplingResults

    Notes
    -----
    This function takes a value for the evidence margin ΔE; in practice it is
    often more useful to set ΔE = 0 when running the sampling, and then
    studying the results under changing ΔE afterwards, using
    ``SamplingResults.best_profile(dE=...)``.

    Post-processing functionality is provided in the `postproc` module, but
    usually the profile found by sampling alone is already pretty good.

    See also
    --------
    SamplingResults, amis.FixedkSampler, postproc, models
    """
    bar = tqdm(disable = not show_progress)
    traj = make_Trajectory(traj)

    # Initialize
    samplers = []

    ## Logging a few useful things (mostly for diagnostics)
    log = {
        'k'    : [], # which k was sampled
        'pk'   : [], # choice distribution
        'KLD'  : [], # estimated information gain for more sampling
        'I_la' : [], # importance over the lookahead region
    }

    # Steps / conditions of the iterative scheme
    def add_sample(k):
        # If sampler is exhausted, this will just do nothing
        if samplers[k].step()
            bar.update()
            for key in log:
                log[key].append(None)
            log['k'][-1] = k

    def determine_next_step():
        k_new = len(samplers) # k for an eventual new sampler

        # k_new == k_lookahead is the edge case where exactly all the samplers
        # are in the lookahead region; in this case, the information gain from
        # the lookahead region is infinite.
        # 
        # So we can run a new sampler right away, IF k_new <= k_max, i.e. we
        # are allowed to set up a new sampler. In all other cases, we have to
        # run the ChoiceSampler
        # 
        # Logic:
        # (k_new <  k_lookahead+1, k_new <= k_max) : run new sampler right away, log I_la = np.inf
        # (k_new >= k_lookahead+1, k_new <= k_max) : run ChoiceSampler (default case)
        # (k_new <  k_lookahead+1, k_new >  k_max) : run ChoiceSampler, but only KLD (I_la = np.inf)
        # (k_new >= k_lookahead+1, k_new >  k_max) : run ChoiceSampler; I_la < np.inf will be logged, but ignored.

        if k_new < k_lookahead+1 and k_new <= k_max:
            k_next = k_new
            pk = None
            KLD = None
            I_la = np.inf

        else:
            logE  = np.array([sampler.evidences[-1][0]                              for sampler in samplers])
            dlogE = np.array([sampler.evidences[-1][1]                              for sampler in samplers])
            N     = np.array([np.inf if sampler.exhausted else len(sampler.samples) for sampler in samplers])

            cs = ChoiceSampler(logE, dlogE**2, N, dE, **choice_kw)
            pk = cs.n0 / cs.samplesize

            # Information gain from more samples
            KLD = cs.KLD_moreSamples()
            k_KLD = np.argmax(KLD)

            # Information gain from new k
            if k_new >= k_lookahead+1:
                I_la = cs.KLD_omitK(np.arange(k_new-k_lookahead, k_new))
            else:
                I_la = np.inf

            k_next = k_KLD
            if I_la > KLD[k_KLD] and k_new <= k_max:
                k_next = k_new

        log['pk'  ][-1] = pk
        log['KLD' ][-1] = KLD
        log['I_la'][-1] = I_la
        return k_next

    def add_sampler(k):
        assert k == len(samplers)
        samplers.append(FixedkSampler(traj, model, k=k, **sampler_kw))

        for _ in range(init_runs):
            add_sample(k)

    def all_exhausted(samplers):
        return all(sampler.exhausted for sampler in samplers)
    
    # Main loop
    k_next = 0
    run_condition = True
    while run_condition:
        if k_next < len(samplers):
            add_sample(k)
        elif k_next == len(samplers):
            add_sampler(k_next)
        else: # pragma: no cover
            raise RuntimeError("Trying to sample outside of existing range; this is a bug")

        k_next = determine_next_step()

        # Stopping condition
        # Given by certainty, unless we determined that we need higher k, which
        # takes precedence.
        if k_next == len(samplers):
            run_condition = True
        else:
            run_condition  = np.max(log['pk'][-1]) < certainty_in_k
            run_condution &= not all_exhausted(samplers)
        
    bar.close()
    
    return SamplingResults(traj, model, samplers, log)

class SamplingResults():
    """
    Standard format for `sample` output

    Attributes
    ----------
    traj : Trajectory
        the trajectory these results pertain to
    model : MultiStateModel
        the model used in obtaining them 
    dE : float
        evidence margin applied during the sampling
    samplers : list of FixedkSampler
        the samplers run for this inference run
    k_updates : np.ndarray, optional
        an array containing the order in which the samplers were updated
    k : np.ndarray
        list of evaluated ``k`` values
    evidence : np.ndarray
        the corresponding evidences for each ``k``
    evidence_se : np.ndarray
        the standard error on the evidence
    """
    def __init__(self, traj, model, dE, samplers, log=None):
        self.traj = traj
        self.model = model
        self.dE = dE
        self.samplers = samplers

        def logentry_as_array(entry):
            shape = np.asarray(entry[np.nonzero([x is not None for x in entry])[0][0]]).shape
            nans = np.empty(shape=shape)
            nans[:] = np.nan
            return np.array([nans if x is None else x for x in entry])

        self.log = {}
        if log is not None:
            for key in log:
                self.log[key] = logentry_as_array(log[key])

    @property
    def k(self):
        return np.array([sampler.k for sampler in self.samplers])

    @property
    def evidence(self):
        return np.array([sampler.evidences[-1][0] for sampler in self.samplers])

    @property
    def evidence_se(self):
        return np.array([sampler.evidences[-1][1] for sampler in self.samplers])

    def best_k(self, dE=None):
        """
        Find the best k at given ΔE

        Parameters
        ----------
        dE : float >= 0 or None
            the evidence margin to apply; defaults to ``self.dE``

        Returns
        -------
        int

        See also
        --------
        best_profile
        """
        if dE is None:
            dE = self.dE
        ks_plausible = self.k[self.evidence >= np.max(self.evidence) - dE]
        return np.min(ks_plausible)

    def best_profile(self, dE=None):
        """
        Find the best profile at given ΔE

        Parameters
        ----------
        dE : float >= 0 or None
            the evidence margin to apply; defaults to ``self.dE``

        Returns
        -------
        Loopingprofile

        See also
        --------
        best_k
        """
        return self.samplers[self.best_k(dE)].MAP_profile()

    def log_marginal_posterior(self, dE=None):
        """
        Calculate posterior marginals

        Parameters
        ----------
        dE : float >= 0 or None
            the evidence margin to apply. If ``None`` (default): instead of
            picking the best ``k``, average over ``k``, weighted by evidence.

        Returns
        -------
        (n, T) np.ndarray, dtype=float
        """
        if dE is None:
            with np.errstate(under='ignore'):
                logpost = logsumexp([sampler.log_marginal_posterior() + logev
                                     for sampler, logev in zip(self.samplers, self.evidence)],
                                    axis=0,
                                    )
                return logpost - logsumexp(logpost, axis=0)
        else:
            return self.samplers[self.best_k(dE)].log_marginal_posterior()
