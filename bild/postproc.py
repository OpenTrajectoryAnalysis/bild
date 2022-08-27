"""
Some post-processing of inferred profiles

This module provides an iterative optimization scheme for inferred looping
profiles. Briefly, at each step we evaluate how much the likelihood increases
if we were to move each boundary of the profile and then do the move that gives
the largest increase.
"""
from copy import deepcopy

import numpy as np

def logLR_boundaries(profile, traj, model):
    """
    Give log-likelihood ratios for moving each boundary either direction

    Parameters
    ----------
    profile : Loopingprofile
        the profile whose boundaries we want to optimize
    traj : Trajectory
        the trajectory to which the profile applies
    model : MultiStateModel
        the model providing the likelihood function

    Returns
    -------
    (k, 2) np.ndarray, dtype=float
        the log(likelihood ratio) for moving any of the ``k`` boundaries to the
        left (``[i, 0]``) / right (``[i, 1]``).

    See also
    --------
    optimize_boundary
    """
    def likelihood(profile, traj=traj, model=model):
        return model.logL(profile, traj)
    
    boundaries = np.nonzero(np.diff(profile.state))[0] # boundary between b and b+1
    if len(boundaries) == 0:
        return np.array([])
    
    profile_new = profile.copy()
    Ls = np.empty((len(boundaries), 2))
    Ls[:] = np.nan
    for i, b in enumerate(boundaries):
        old_states = deepcopy(profile_new[b:(b+2)])
        
        # move left
        profile_new[b] = profile_new[b+1]
        Ls[i, 0] = likelihood(profile_new)
        profile_new[b] = old_states[0]
        
        # move right
        profile_new[b+1] = profile_new[b]
        Ls[i, 1] = likelihood(profile_new)
        profile_new[b+1] = old_states[1]
        
    return Ls - likelihood(profile)

class BoundaryEliminationError(Exception):
    pass

def optimize_boundary(profile, traj, model,
                      max_iteration = 10000,
                     ):
    """
    Locally optimize the boundaries of a profile

    Parameters
    ----------
    profile : Loopingprofile
        the profile whose boundaries we want to optimize
    traj : Trajectory
        the trajectory to which the profile applies
    model : MultiStateModel
        the model providing the likelihood function
    max_iteration : int, optional
        limit on the number of iterations through the (find boundaries) -->
        (make best move) cycle

    Raises
    ------
    BoundaryEliminationError
        if the local optimization tries to shrink an interval to zero, such
        that the total number of boundaries would change. This usually suggests
        that the original sampling might not have been extensive enough.

    Returns
    -------
    Loopingprofile
        the optimized profile
    """
    profile_new = profile.copy()
    for _ in range(max_iteration):
        logLR = logLR_boundaries(profile_new, traj, model)
        if len(logLR) == 0:
            break
            
        i, j = np.unravel_index(np.argmax(logLR), logLR.shape)
        
        if logLR[i, j] > 0:
            boundaries = np.nonzero(np.diff(profile_new.state))[0]
            if (   (j == 0 and boundaries[i] == 0)
                or (j == 0 and profile_new[boundaries[i]-1] == profile_new[boundaries[i]+1])
                or (j == 1 and boundaries[i] == len(traj)-2)
                or (j == 1 and profile_new[boundaries[i]+2] == profile_new[boundaries[i]])
               ):
                raise BoundaryEliminationError(f"Trying to abolish boundary at {boundaries[i]}")

            profile_new[boundaries[i]+j] = profile_new[boundaries[i]+(1-j)]
        else:
            break
    else:
        raise RuntimeError(f"Exceeded max_iteration = {max_iteration}")
        
    return profile_new
