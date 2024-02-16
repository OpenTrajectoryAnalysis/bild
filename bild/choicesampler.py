import numpy as np

class ChoiceSampler:
    """
    Sample selection for iterative AMIS

    When running BILD, we are faced with the sample selection issue: which k is
    most important to sample next, given our current estimate of the evidence
    curve? This turns out to be a non-trivial question, if we take into account
    the evidence margin ΔE. This class implements an information-based sample
    selection scheme, described in the Notes.

    Parameters
    ----------
    muhat : (k,) np.ndarray, dtype=float
        the current point estimates for the evidence
    shat : (k,) np.ndarray, dtype=float
        the variance of the estimated means `muhat`, i.e. the squared standard
        error of the mean
    N : (k,) np.ndarray, dtype=int (or float, to allow for np.inf)
        the number of samples used to estimate evidence at each k. "One sample"
        here is one of the steps we are asking about where to take next; i.e.
        since for each AMIS iteration we actually sample (usually) 100
        profiles, "one sample" here corresponds to 100 profiles.
    dE : float
        the evidence margin ΔE to apply in the sampling scheme
    samplesize : int
        the sample size to use for KLD estimation. Technical hyperparameter.

    Attributes
    ----------
    muhat : (k,) np.ndarray, dtype=float
        see Parameters
    shat : (k,) np.ndarray, dtype=float
        see Parameters
    N : (k,) np.ndarray, dtype=int (or float, for np.inf)
        see Parameters
    dE : float
        see Parameters
    samplesize : int
        see Parameters
    kmax : int
        number of k positions under consideration. Shorthand for
        ``len(self.muhat)``
    EDmu2 : (k,) np.ndarray, dtype=float
        expected squared update in evidence upon one more sample at the
        corresponding k. Calculated analytically.
    Dmu : (k,) np.ndarray, dtype=float
        root mean squared update in evidence at k; i.e. ``sqrt(EDmu2)``
    _scaled_rvs : (samplesize, k) np.ndarray, dtype=float
        the standard normal sample underlying the subsequent estimations
    bestk : (samplesize,) np.ndarray, dtype=int
        for each sample, the k selected under ΔE scheme.
    best_is_k : (samplesize, k) np.ndarray, dtype=bool
        truth table corresponding to ``self.bestk``
    n0 : (k,) np.ndarray, dtype=int
        histogram of `bestk`.

    Notes
    -----
    The basic idea of this sample selection scheme is to ask "which additional
    sample do we expect to yield the most useful information?". This is done by
    investigating the evolution of the "choice distribution" p(k): the
    categorical distribution that describes our belief about which k is best
    under ΔE, given the evidence curve with its uncertainty. The way it is
    calculated is simply by sampling from the error bars of the evidence curve
    `samplesize` times. Accordingly, `samplesize` should be chosen such that
    ``samplesize / kmax >> 1``. We can then predict by roughly how much the
    evidence is likely to change with an additional sample, for each k; this
    allows us to estimate an expected information gain, as the Kullback-Leibler
    divergence between the new (predicted) distribution and the current one.

    When working through this scheme analytically, note the following subtle
    point: the expected change in the choice distribution is zero; but the
    expected KLD is non-zero and meaningful, since it is quadratic in this
    change. Intuitively: regardless of whether the probability for a given k
    goes up or down, we learn something.

    See also
    --------
    amis, core.sample
    """
    def __init__(self, muhat, shat, N, dE,
                 samplesize = 10000,
                ):
        self.dE         = dE
        self.muhat      = muhat
        self.shat       = shat
        self.N          = N
        self.samplesize = samplesize
        
        self.kmax = len(muhat)

        self.EDmu2 = self.shat / (self.N+1)
        self.Dmu = np.sqrt(self.EDmu2)
        
        self.init_sample()
    
    def init_sample(self):
        """
        Initialize the internal random sample

        This is called in the constructor; it is its own function mainly for
        code cleanliness; it allows individual instances to be reinitialized.
        """
        self._scaled_rvs = np.sqrt(self.shat[None, ...]) * np.random.normal(loc=0, scale=1, size=(self.samplesize, self.kmax))
        
        self.bestk = self.evaluate() # point estimate
        self.best_is_k = self.bestk[:, None] == np.arange(self.kmax)[None, :] # (samp, k)
        self.n0 = np.sum(self.best_is_k, axis=0)
    
    def evaluate(self, k_change=None, n_step=0, omit_k=None):
        """
        Generate sample from choice distribution, possibly with some updates

        Parameters
        ----------
        k_change : int, index array, or None
            which k to change
        n_step : float
            how far to move, in units of the natural step size `Dmu` for this
            k (so usually: +1, -1, +-0.5, etc.).
        omit_k : int, index array, or None
            pretend the evidence curve has not been evaluated at these k, i.e.
            just ignore them. This is used to estimate the importance of any
            given k in the "lookahead" scheme.

        Returns
        -------
        k : (samplesize,) np.ndarray, dtype=int
            sample from the choice distribution

        Notes
        -----
        All samples rely on the same underlying standard normal sample. This
        makes them highly correlated, which reduces the variance in estimating
        differences.
        """
        myM = self.muhat.copy()
        
        if k_change is not None:
            myM[k_change] += n_step*self.Dmu[k_change]
            
        if omit_k is not None:
            myM[omit_k] = np.nan
        
        x = self._scaled_rvs + myM                     # (samp, k)
        m = np.nanmax(x, axis=1, keepdims=True)        # (samp, 1)
        k = np.nanargmax(m - self.dE - x <= 0, axis=1) # (samp,)
        
        return k
    
    def Dn(self):
        """
        Calculate the expected change in choice distribution

        Returns
        -------
        (k, k) np.ndarray, dtype=float
            ``[k1, k2]`` gives the expected change in the histogram counts
            for k=k2 upon adding a sample at k=k1.
        """
        new_ks = np.array([[self.evaluate(k, step) for k in range(self.kmax)] # (2, k_change, samp)
                           for step in (-0.5, 0.5)])
        new_n = np.sum(new_ks[..., None] == np.arange(self.kmax), axis=-2)    # (2, k_change, k)
        return new_n[1] - new_n[0] # (k_change, k)
    
    def KLD_moreSamples(self):
        """
        Calculate expected KLD upon additional sampling

        Returns
        -------
        (k,) np.ndarray, dtype=float
            estimated KLD for one additional sample at each k
        """
        Dn = self.Dn() # (k_change, k)
        return 0.5/self.samplesize * np.sum(Dn**2 / (self.n0 + 1)[None, :], axis=-1)
    
    def KLD_omitK(self, omit_k=None):
        """
        "Importance" of any given k (or combinations)

        Parameters
        ----------
        omit_k : int, index array, or None
            which k to omit

        Returns
        -------
        float
            the information gain from including these positions in the
            selection scheme. This can be used to gauge the information gain of
            an additional sample.
        """
        # Note: we calculate D_KL( full || omitted ), i.e. we pretend that we start from the
        # choice distribution with some k omitted and ask about the information gain of
        # adding those in as well.
        # Thus, the notation in the code here is such that "old" refers to omitting the given
        # k, while "new" is the full distribution.
        old_k = self.evaluate(omit_k=omit_k) # (samp,), contains nan's
        old_n = np.sum(old_k[:, None] == np.arange(self.kmax)[None, :], axis=0) # (k,)
        old_n = old_n / np.sum(old_n) * self.samplesize # renormalize missing samples
        
        Dn = self.n0 - old_n
        Dn[omit_k] = 0 # we're not interested in the changes here
                       # they would contribute infinite KLD,
                       # because we calculate KLD( new || old ) and old_n[omit_k] == 0
        
        return 0.5/self.samplesize * np.sum(Dn**2 / (old_n + 1))
