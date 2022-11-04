import numpy as np

LOG_2PI = np.log(2*np.pi)

def Kalman_update(w, x, M, C, s2, Cind):
    """
    Kalman update step for the likelihood computation

    Parameters
    ----------
    w : (N,) np.ndarray
    x : (d,) np.ndarray
        current trajectory point
    M : (N, d) np.ndarray
        prior value of mean
    C : (d*, N, N) np.ndarray
        prior value of covariance
    s2 : (d*,) np.ndarray
        unique localization errors
    Cind : (d,) np.ndarray, dtype=int
        conversion ``d* --> d``; i.e. for each dimension, contains the
        index into ``d*`` arrays to use

    Returns
    -------
    M : (N, d) np.ndarray
        posterior mean
    C : (d*, N, N) np.ndarray
        posterior covariance
    logL : (d,) np.ndarray
        likelihood of the current observation `!x`
    """
    # Implementation Notes
    #  + minimal benchmarking hints at in-place addition/subtraction
    #    actually being slower than just creating new arrays for M, C at
    #    each step

    # Innovation
    m = w @ M            # (d,)
    xmm = x - m          # (d,)

    # Updates of covariances
    Cw = C @ w                           # (d*, N)
    S = Cw @ w + s2                      # (d*,)      # due to squeezing: (C @ w) @ w :=: (w @ C) @ w
    K = Cw / S[:, None]                  # (d*, N)
    C = C - K[:, :, None]*Cw[:, None, :] # (d*, N, N)

    # Mean and likelihoods
    M = M + K[Cind].T * xmm # (N, d)
    logL = -0.5 * ( xmm*xmm / S[Cind] + np.log(S)[Cind] + LOG_2PI ) # (d,)

    return M, C, logL

def MSRouse_logL(model, profile, traj):
    """
    Rouse likelihood, evaluated by Kalman filter

    Parameters
    ----------
    model : models.MultiStateRouse
    profile : Loopingprofile
    traj : noctiluca.Trajectory

    Returns
    -------
    float
    """
    localization_error = model._get_noise(traj)

    # Evolution of covariance matrix for each dimension depends only on the
    # localization error, i.e. is independent of the actual data. This
    # means we can optimize here by not executing the actual propagation
    # for dimensions with equal localization error
    # Idea: always use C[Cind[d]] instead of C[d], so C actually has only
    #       the distinct covariance matrices
    unique_errors, Cind = np.unique(localization_error, return_inverse=True)
    s2 = unique_errors*unique_errors

    w = model.measurement

    for mod in model.models:
        mod.check_dynamics()

    M, C_single = model.models[profile[0]].steady_state()
    C = np.tile(C_single, (len(unique_errors), 1, 1))

    valid_times = np.nonzero(~np.any(np.isnan(traj[:]), axis=1))[0]
    L_log = np.empty((len(valid_times), model.d), dtype=float)

    def get_vt(i):
        try:
            return valid_times[i]
        except IndexError:
            return np.nan

    # First update
    i_write = 0
    next_vt = get_vt(i_write)
    if 0 == next_vt:
        M, C, L_log[i_write] = Kalman_update(w, traj[0], M, C, s2, Cind)
        i_write += 1
        next_vt = get_vt(i_write)

    # Propagate, then update
    for t, state in enumerate(profile[1:], start=1):
        mod = model.models[state]

        # Propagate
        M = mod.propagate_M(M, check_dynamics=False)
        C = mod.propagate_C(C, check_dynamics=False)

        # Update
        if t == next_vt:
            M, C, L_log[i_write] = Kalman_update(w, traj[t], M, C, s2, Cind)
            i_write += 1
            next_vt = get_vt(i_write)

    if i_write != len(L_log):
        raise RuntimeError("Internal inconsistency (i.e. bug)") # pragma: no cover

    return np.sum(L_log)
