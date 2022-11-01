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
           significant_separation_sem_fold = 5,
           k_lookahead = 2,
           k_max = 20,
           sampler_kw = {},
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

    # Some conditions that come in handy
    def all_exhausted(samplers):
        return all(sampler.exhausted for sampler in samplers)
    def within_lookahead(k, candidates, k_lookahead=k_lookahead):
        return k - np.max([sampler.k for sampler in candidates]) <= k_lookahead
    def is_significant(sampler, against, t_thres=significant_separation_sem_fold):
        return np.abs(sampler.tstat(against)) > t_thres
    
    # Initialize
    samplers = []
    insignificants = []
    k_updates = [] # Keeps track of evaluation order (diagnostic)
    
    # Body of the insignificance resolution
    def step_highest_sem(samplers, k_updates=k_updates):
        candidates = [sampler for sampler in samplers if not sampler.exhausted]
        i_worst = np.argmax([candidate.evidences[-1][1] for candidate in candidates])
        k_updates.append(candidates[i_worst].k)
        candidates[i_worst].step() # no need to pay attention to return values, this is done by the loop condition
    def get_insignificants(samplers):
        # Note on the selection of insignificants:
        # we pick them such that the whole set is mutually insignificant, as
        # opposed to say picking everything with an insignificant comparison
        # against the best sample. The reason is that if for any k we can find
        # a significantly better one, we should use that and forget about the
        # worse one. Also, comparing only to the best sample is insane when
        # that happens to have large sem and is exhausted already (which is the
        # practical situation that alerted me to this issue).
        evidences = np.array([sampler.evidences[-1][0] for sampler in samplers])
        ks = np.argsort(evidences)[::-1]
        insignificants = []
        # Notes:
        # + since any([]) == False the first iteration works fine
        # + ks are ordered descending, such that on that first iteration we're
        #   checking the currently best estimate
        for k in ks: 
            if not any(is_significant(samplers[k], other) for other in insignificants):
                insignificants.append(samplers[k])
                
        return insignificants
    
    # Run
    for k in range(k_max+1):
        samplers.append(FixedkSampler(traj, model, k=k, **sampler_kw))
        assert len(samplers) == k+1 # Paranoia (samplers[k] vs. sampler.k)
        
        # Initial sampling
        for _ in range(init_runs):
            if not samplers[k].step():
                break
            k_updates.append(k)
            bar.update()
            
        # Update significances
        insignificants = get_insignificants(samplers)
        
        # Insignificance resolution
        # Note that we only resolve insignificances if it is "important", i.e.
        # if we have exhausted the lookahead already. This ensures that we
        # don't get trapped resolving irrelevant insignificances, i.e. those
        # away from the optimum
        while (not within_lookahead(k+1, insignificants)
               and not all_exhausted(insignificants)
               and len(insignificants) > 1
              ):
            step_highest_sem(insignificants)
            bar.update()
            insignificants = get_insignificants(samplers)
            
        # If the next iteration would exceed the lookahead, stop
        if not within_lookahead(k+1, insignificants):
            break

    # There might be insignificancies left, in which case we will continue sampling until
    #  a) they are resolved, or
    #  b) we exhaust all the samplers
    while (len(insignificants) > 1
           and not all_exhausted(insignificants)
          ):
        step_highest_sem(insignificants)
        bar.update()
        insignificants = get_insignificants(samplers)
        
    bar.close()
    
    return SamplingResults(traj, model, samplers, k_updates)

class SamplingResults():
    """
    Standard format for `sample` output

    Attributes
    ----------
    traj : Trajectory
        the trajectory these results pertain to
    model : MultiStateModel
        the model used in obtaining them 
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
    def __init__(self, traj, model, samplers, k_updates=None):
        self.traj = traj
        self.model = model
        self.samplers = samplers
        self.k_updates = np.asarray(k_updates)

    @property
    def k(self):
        return np.array([sampler.k for sampler in self.samplers])

    @property
    def evidence(self):
        return np.array([sampler.evidences[-1][0] for sampler in self.samplers])

    @property
    def evidence_se(self):
        return np.array([sampler.evidences[-1][1] for sampler in self.samplers])

    def best_k(self, dE=0):
        """
        Find the best k at given ΔE

        Parameters
        ----------
        dE : float >= 0
            the evidence margin to apply

        Returns
        -------
        int

        See also
        --------
        best_profile
        """
        ks_plausible = self.k[self.evidence >= np.max(self.evidence) - dE]
        return np.min(ks_plausible)

    def best_profile(self, dE=0):
        """
        Find the best profile at given ΔE

        Parameters
        ----------
        dE : float >= 0
            the evidence margin to apply

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
