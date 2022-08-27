"""
Statistical tools
"""
import numpy as np
from scipy import stats, optimize

def KM_survival(data, censored, conf=0.95, Tmax=np.inf, S1at=0):
    """
    Kaplan-Meier survival estimator on censored data

    This is the standard survival estimator for right-censored data, i.e. data
    points that are marked as "censored" enter the estimate as ``t_true > t``.

    Parameters
    ----------
    data : (N,) array-like
        individual survival times
    censored : (N,) array-like, boolean
        indicate for each data point whether it is right-censored or not.
    conf : float in (0, 1), optional
        the confidence bounds on the survival curve to calculate
    Tmax : float
        can be used to compute survival only up to some time ``Tmax``.
    S1at : float, optional
        give a natural lower limit for survival times where S = 1.

    Returns
    -------
    (T, 4) array
        the columns of this array are: t, S(t), l(t), u(t), where l and u are
        lower and upper confidence levels respectively.
    """
    data = np.asarray(data)
    censored = np.asarray(censored).astype(bool)

    t = np.unique(data[~censored]) # unique also sorts
    t = t[t <= Tmax]
    S = np.zeros(len(t)+1)
    S[0] = 1
    V = np.zeros(len(t)+1)
    Vsum = 0
    for n, curt in enumerate(t, start=1):
        d_n = np.count_nonzero(data[~censored] == curt)
        N_n = np.count_nonzero(data >= curt)

        S[n] = S[n-1]*(1-d_n/N_n)
        if N_n > d_n:
            Vsum += d_n/(N_n*(N_n-d_n))
            V[n] = np.log(S[n])**(-2)*Vsum
        else:
            Vsum += np.inf
            V[n] = 0

    z = stats.norm().ppf((1-conf)/2)
    lower = S**(np.exp( z*np.sqrt(V)))
    upper = S**(np.exp(-z*np.sqrt(V)))

    if S1at is not None:
        t = np.insert(t, 0, S1at)
    else:
        S = S[1:]
        lower = lower[1:]
        upper = upper[1:]

    return np.stack([t, S, lower, upper], axis=-1)

def MLE_censored_exponential(data, censored, conf=0.95):
    """
    MLE estimate for exponential distribution, given right-censored data

    Parameters
    ----------
    data : array-like, dtype=float
        the (scalar) values
    censored : array-like, dtype=bool, same shape as data
        whether the corresponding value in `!data` is censored (``True``) or
        completely observed (``False``).
    conf : float in [0, 1], optional
        the confidence level to use in calculating bounds

    Returns
    -------
    m, low, high : float
        point estimate for the mean of the exponential distribution, and
        confidence bounds.
    """
    data = np.asarray(data).flatten()
    censored = np.asarray(censored, dtype=bool).flatten()

    n = np.count_nonzero(~censored)
    alpha = 1-conf

    # Point estimate
    m = np.sum(data) / n

    # Confidence interval
    c = stats.chi2(1).isf(alpha) / (2*n)
    def fitfun(beta): return np.exp(beta) - 1 - beta - c

    res = optimize.root_scalar(fitfun, bracket=(-c-1, 0))
    if not res.flag == 'converged': # pragma: no cover
        raise RuntimeError("Root finding did not converge for upper confidence interval")
    beta_m = res.root

    res = optimize.root_scalar(fitfun, bracket=(0, 2*np.sqrt(c)))
    if not res.flag == 'converged': # pragma: no cover
        raise RuntimeError("Root finding did not converge for lower confidence interval")
    beta_p = res.root

    return m, m*np.exp(-beta_p), m*np.exp(-beta_m)
