import numpy as np
cimport numpy as np
np.import_array()

cimport cython
from cython cimport view
from libc.math cimport log
from scipy.linalg.cython_blas cimport ( daxpy, dcopy, ddot, dgemv,
                                        dger, dscal, dsymv )

ctypedef double FLOAT_t
ctypedef unsigned long SIZE_t

cdef FLOAT_t LOG_2PI = np.log(2*np.pi)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void Kalman_update(FLOAT_t[::1]       w,
                        FLOAT_t[::1]       x,
                        FLOAT_t[:, ::1]    M,
                        FLOAT_t[:, :, ::1] C,
                        FLOAT_t[::1]       s2,
                        SIZE_t[::1]        Cind,
                        FLOAT_t[::1]       logL,
                        FLOAT_t[::1]       xmm,
                        FLOAT_t[:, ::1]    K,
                        FLOAT_t[::1]       Sinv,
                        FLOAT_t[:, ::1]    Cw,
                        ):
    """
    Kalman update step

    Parameters
    ----------
    w :    (N,)       float
    x :    (d,)       float
    M :    (N, d)     float
    C :    (d*, N, N) float
    s2 :   (d*,)      float
    Cind : (d,)       uint
    logL : (d,)       float : where to write the output values
    xmm :  (d,)       float : memoryview
    K :    (d*, N)    float : memoryview
    Sinv : (d*,)      float : memoryview
    Cw :   (d*, N)    float : memoryview to use for intermediate results
                              (assigning new every time is expensive)
    """
    cdef SIZE_t d
    cdef int inc=1, N=M.shape[0], D=M.shape[1], Dstar=C.shape[0]
    cdef double one=1., _one=-1., zero=0.

    for d in range(Dstar):
        # Cw = C @ w
        dsymv("u", &N, &one,
              &C[d, 0, 0], &N,
              &w[0], &inc,
              &zero, &Cw[d, 0], &inc,
              )

        # S = Cw @ w + s2
        # we use Sinv = 1/S, since we have to multiply by 1/S next
        Sinv[d] = 1/( s2[d] + ddot(&N, &Cw[d, 0], &inc, &w[0], &inc) )

        # K = Cw / S[:, None]
        dscal(&N, &zero, &K[d, 0], &inc) # initialize K[d] to zeros
        daxpy(&N, &Sinv[d], &Cw[d, 0], &inc, &K[d, 0], &inc)

        # C -= K[:, :, None]*Cw[:, None, :]
        # C is symmetric, thus C-order = F-order
        dger(&N, &N, &_one,
             &K[d, 0], &inc,
             &Cw[d, 0], &inc,
             &C[d, 0, 0], &N,
             )

    for d in range(D):
        # xmm = x - w @ M
        xmm[d] = x[d] - ddot(&N, &w[0], &inc, &M[0, d], &D)

        # M += K[Cind].T * xmm[None, :]
        daxpy(&N, &xmm[d],
              &K[Cind[d], 0], &inc,
              &M[0, d], &D,
              )

        # Gaussian likelihood
        logL[d] = -0.5*( xmm[d]*xmm[d] * Sinv[Cind[d]] - log(Sinv[Cind[d]]) + LOG_2PI )

    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def logL(object model,   # bild.models.MultiStateModel
         object profile, # bild.util.Loopingprofile
         object traj,    # noctiluca.Trajectory
         ):
    ### Declarations ###
    cdef FLOAT_t            logL_total  # final output value
    cdef FLOAT_t[:, :, ::1] C           # prior covariances for Kalman
    cdef FLOAT_t[:, :, ::1] Bs          # \
    cdef FLOAT_t[:, :, ::1] Gs          #  > internals from the Rouse models
    cdef FLOAT_t[:, :, ::1] Sigs        # /
    cdef FLOAT_t[:, ::1]    C_post      # posterior covariance (in 1D)
    cdef FLOAT_t[:, ::1]    M           # prior mean for Kalman
    cdef FLOAT_t[:, ::1]    M_post      # posterior mean
    cdef FLOAT_t[:, ::1]    logL        # all individual likelihood values
    cdef FLOAT_t[:, ::1]    trajdat     # trajectory data in C
    cdef FLOAT_t[::1]       BCn0        # temporary for matmul
    cdef FLOAT_t[::1]       s2          # unique localization errors, squared
    cdef FLOAT_t[::1]       w           # measurement vector
    cdef SIZE_t             i_write     # index for Kalman output
    cdef SIZE_t             t, state, n0, n1, n2, n3, d # loop indices
    cdef SIZE_t[::1]        Cind        # conversion d --> d*
    cdef SIZE_t[::1]        states      # Loopingprofile in C
    cdef SIZE_t[::1]        valid_times # times for which we have data in traj

    # Dummy memory allocations for Kalman updates
    cdef FLOAT_t[:, ::1]    Kalman_Cw
    cdef FLOAT_t[:, ::1]    Kalman_K
    cdef FLOAT_t[::1]       Kalman_Sinv
    cdef FLOAT_t[::1]       Kalman_xmm

    ### Setup ###
    # This has a lot of python --> C conversion and interplay; but that's fine,
    # since it's not the runtime critical part
    # all python variables get py_ prefix

    # localization errors
    py_localization_error = model._get_noise(traj)
    py_unique_errors, py_Cind = np.unique(py_localization_error, return_inverse=True)
    s2 = py_unique_errors**2
    Cind = py_Cind.astype(np.uint)

    # individual Rouse models and propagation dynamics
    w = model.measurement

    for py_state, py_m in enumerate(model.models):
        py_m.check_dynamics()

    Bs   = np.array([m._dynamics['B']   for m in model.models])
    Gs   = np.array([m._dynamics['G']   for m in model.models])
    Sigs = np.array([m._dynamics['Sig'] for m in model.models])

    # initial condition (steady state)
    py_M, py_C_single = model.models[profile[0]].steady_state()

    M = py_M
    C = np.tile(py_C_single, (len(py_unique_errors), 1, 1))

    assert tuple(M.shape) == tuple(Gs[0].shape)
    assert tuple(C[0].shape) == tuple(Sigs[0].shape)

    Kalman_Cw = np.empty((C.shape[0], C.shape[1]), dtype=float)
    Kalman_K = Kalman_Cw.copy()
    Kalman_Sinv = s2.copy()
    Kalman_xmm = traj[0].copy()

    # Get trajectory and profile data to C
    trajdat = traj[:]
    states = profile[:].astype(np.uint)

    # Find the times for which we have data
    valid_times = np.nonzero(~np.any(np.isnan(traj[:]), axis=1))[0].astype(np.uint)
    
    # Initialize output
    logL_total = 0.
    logL = np.empty((valid_times.shape[0], model.d), dtype=float)
    i_write = 0

    # Get likelihood from first data point in steady state
    if valid_times[i_write] == 0:
        Kalman_update(w, np.zeros(traj.d, dtype=float), M, C, s2, Cind,
                      logL[i_write], Kalman_xmm, Kalman_K, Kalman_Sinv, Kalman_Cw,
                      )

        i_write += 1

    # sizes and factors for BLAS
    cdef int inc=1, N=M.shape[0], D=M.shape[1], Dstar=C.shape[0]
    cdef double one=1., zero=0.

    # Initialize temp variables for matrix multiplications
    M_post = np.empty((N, D), dtype=float)
    C_post = np.empty((N, N), dtype=float)
    BCn0   = np.empty((N,),   dtype=float)

    ### Kalman filtering loop ###
    # Runtime critical, should be pure C
    for t in range(1, states.shape[0]):
        state = states[t]

        for d in range(D):
            # M <-- B @ M + G
            # B is symmetric, thus C-order = F-order
            dcopy(&N, &Gs[state, 0, d], &D, &M_post[0, d], &D)
            dsymv("u", &N, &one,
                  &Bs[state, 0, 0], &N,
                  &M[0, d], &D,
                  &one, &M_post[0, d], &D,
                  )

        M[...] = M_post

        # C <-- B @ C @ B + Sig
        # calculate as: Sig + (B @ C) @ B
        for d in range(Dstar):
            for n0 in range(C.shape[1]):
                # C_post = Sig
                dcopy(&N, &Sigs[state, n0, 0], &inc, &C_post[n0, 0], &inc)

                # BCn0[:] = B[n0, :] @ C[d, :, :]
                # C is symmetric, thus C-order = F-order
                dsymv("u", &N, &one,
                      &C[d, 0, 0], &N,
                      &Bs[state, n0, 0], &inc,
                      &zero, &BCn0[0], &inc,
                      )

                # C_post[n0, :] += BCn0[:] @ Bs[state, :, :]
                # B is symmetric, thus C-order = F-order
                dsymv("u", &N, &one,
                      &Bs[state, 0, 0], &N,
                      &BCn0[0], &inc,
                      &one, &C_post[n0, 0], &inc,
                      )

            C[d, ...] = C_post
        
        # Kalman update
        if i_write < valid_times.shape[0] and t == valid_times[i_write]:
            Kalman_update(w, trajdat[t], M, C, s2, Cind,
                          logL[i_write], Kalman_xmm, Kalman_K, Kalman_Sinv, Kalman_Cw,
                          )
            i_write += 1

    # Sum everything and return
    logL_total = 0.
    for d in range(logL.shape[0]):
        for t in range(logL.shape[1]):
            logL_total += logL[d, t]

    return logL_total
