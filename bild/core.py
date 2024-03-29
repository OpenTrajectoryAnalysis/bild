"""
Core implementation of the `bild` module
"""
from tqdm.auto import tqdm

import numpy as np
from scipy.special import logsumexp

from noctiluca import make_Trajectory

from .amis import FixedkSampler
from .choicesampler import ChoiceSampler
            
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
        c.f. `SamplingResults.best_profile()`, which however might lead to less
        accurate results.

    Other Parameters
    ----------------
    init_runs : int
        minimum number of AMIS runs for a new value of ``k``.
    certainty_in_k : float
        the level of certainty (in the number of switches k) that we need to
        reach before we stop sampling.
    k_lookahead : int
        how far we look ahead for global maxima. If any of the positions at ``k
        >= k_max - k_lookahead`` have a noticeable influence on the final
        choice of ``k``, we explore higher k before sampling the existing ones
        more. The specific condition for "noticable" is that the information
        gain from all the existing samples in this region is more than the
        expected information gain from one additional sample anywhere. The
        default value of 2 comes from the fact that for binary profiles the
        evidence follows an odd-even pattern, so it is necessary to look ahead
        by 2 additional switches.
    k_max : int
        the maximum number of switches to sample to
    sampler_kw : dict
        keyword arguments for the `amis.FixedkSampler`.
    choice_kw : dict
        keyword arguments for `choicesampler.ChoiceSampler`.
    show_progress : bool
        whether to show a progress bar

    Returns
    -------
    SamplingResults

    Notes
    -----
    There are two possibilities for applying the evidence margin ΔE: while
    sampling, or post hoc through ``SamplingResults.best_profile(dE=...)``. The
    benefit of specifying the evidence margin during the sampling is that we
    can take it into account for the sample selection, i.e. the positions
    relevant to this setting of ΔE will be sampled sufficiently. When setting
    ΔE post hoc, one always runs the risk of pushing the actual result of the
    inference into a regime where the sampling machinery decided that we do not
    need to know a lot of detail. This effect should of course small for small
    ΔE.

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

    memory = { # we need some persistent variables for the sampling loop
        'fresh sample' : False,
    }

    # Steps / conditions of the iterative scheme
    def add_sample(k):
        # If sampler is exhausted, this will just do nothing
        if samplers[k].step():
            bar.update()
            for key in log:
                log[key].append(None)
            log['k'][-1] = k
            memory['fresh sample'] = True

    def determine_next_step():
        k_new = len(samplers) # k for an eventual new sampler

        if not memory['fresh sample']:
            if len(log['k']) == 0:
                return k_new
            else: # pragma: no cover
                return log['k'][-1]

        # p(k) should always be evaluated, because this is the stop criterion
        logE  = np.array([sampler.evidences[-1][0]                              for sampler in samplers])
        dlogE = np.array([sampler.evidences[-1][1]                              for sampler in samplers])
        N     = np.array([np.inf if sampler.exhausted else len(sampler.samples) for sampler in samplers])

        cs = ChoiceSampler(logE, dlogE**2, N, dE, **choice_kw)
        pk = cs.n0 / cs.samplesize

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
            KLD = None
            I_la = np.inf
        else:
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
        memory['fresh sample'] = False
        return k_next

    def add_sampler(k):
        assert k == len(samplers)
        samplers.append(FixedkSampler(traj, model, k=k, **sampler_kw))

        for _ in range(init_runs):
            add_sample(k)

    # Main loop
    k_next = 0
    run_condition = True
    try:

        while run_condition:
            if k_next < len(samplers):
                add_sample(k_next)
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

                # Check that the proposed new sample actually gives information
                # This fails if all the *relevant* samplers are exhausted.
                if log['KLD'][-1] is not None:
                    run_condition &= log['KLD'][-1][k_next] > 0
        
        bar.close()

    # Allow clean abortion of execution, when done by hand
    except KeyboardInterrupt: # pragma: no cover
        pass 
    finally:
    
        return SamplingResults(traj, model, dE, samplers, log)

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
    log : dict
        records from the sampling. Mostly useful for inspection / debugging.
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

        def list_to_2D_array_nanpatching(list_2d):
            def len_nonesafe(obj):
                if obj is None:
                    return 1
                else:
                    return len(obj)

            dim0 = len(list_2d)
            max_dim1 = max(map(len_nonesafe, list_2d))

            arr = np.empty(shape=(dim0, max_dim1))*np.nan
            for i, item in enumerate(list_2d):
                if item is not None:
                    arr[i, :len(item)] = item

            return arr

        self.log = {}
        keys_1d = {'k', 'I_la'}
        if log is not None:
            for key in log.keys() & keys_1d:
                self.log[key] = np.array(log[key])
            for key in log.keys() - keys_1d:
                self.log[key] = list_to_2D_array_nanpatching(log[key])

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
        dE : float >= 0, None, or 'average'
            the evidence margin to apply. If set ``'average'`` (default): instead of
            picking the best ``k``, average over ``k``, weighted by evidence.
            Defaults to ``None``, which means "use the internal value
            ``self.dE``.

        Returns
        -------
        (n, T) np.ndarray, dtype=float
        """
        if dE == 'average':
            with np.errstate(under='ignore'):
                logpost = logsumexp([sampler.log_marginal_posterior() + logev
                                     for sampler, logev in zip(self.samplers, self.evidence)
                                     if sampler.evidences[-1][0] > -np.inf], # sanity in case k > len(traj)
                                    axis=0,
                                    )
                return logpost - logsumexp(logpost, axis=0)
        elif dE is None:
            dE = self.dE

        return self.samplers[self.best_k(dE)].log_marginal_posterior()
