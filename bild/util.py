"""
Utilities for BILD
"""
import numpy as np

class Loopingprofile:
    """
    This class represents a single looping profile

    This is a thin wrapper around a `!state` array, whose entries indicate
    which state of the model should be used for propagation from frame to
    frame. Specifically, the following scheme applies to the relation between a
    `Loopingprofile` ``profile`` and the associated `Trajectory` ``traj``:

    .. code-block:: text
        
        profile[0]              profile[1]                                  profile[-1]
        -------------> traj[0] ------------> traj[1] --- ... ---> traj[-2] -------------> traj[-1]

    Note that ``profile[i]`` can be interpreted as "the state of the model used
    to propagate to ``traj[i]``". Furthermore, ``profile[0]`` indicates the
    model state used to calculate the steady state ensemble each trajectory
    starts from.

    Operators defined for a `Loopingprofile` ``profile``:
     + ``len(profile)`` gives the length in frames
     + ``profile[i]`` for element-access (get/set)
     + ``profile0 == profile1`` if the associated states are equal

    Attributes
    ----------
    states : np.ndarray, dtype=int
        the internal state array. If this is not given upon initialization,
        initialized to an empty array.
    """
    def __init__(self, states=None):
        if states is None:
            self.state = np.array([], dtype=int)
        else:
            self.state = np.asarray(states, dtype=int)

    def copy(self):
        """
        Copy to a new object

        Because we can copy just ``self.state``, this is faster than doing
        ``deepcopy(profile)``.

        Returns
        -------
        Loopingprofile
        """
        new = Loopingprofile()
        new.state = self.state.copy()
        return new

    def __len__(self):
        return len(self.state)

    def __getitem__(self, key):
        return self.state[key]

    def __setitem__(self, key, val):
        # we check type instead of casting, since trying to write float values
        # to a Loopingtrace is more probable to indicate an error somewhere
        val = np.asarray(val)
        assert np.issubdtype(val.dtype, np.integer)
        self.state[key] = val

    def __eq__(self, other):
        try:
            if len(self) != len(other):
                raise RuntimeError # numpy will raise this itself at some point in the future
            return np.all(self.state == other.state)
        except:
            return False

    def count_switches(self):
        """
        Give number of switches in the profile

        Returns
        -------
        int
        """
        # the != construct is significantly faster than np.diff()
        return np.count_nonzero(self.state[1:] != self.state[:-1])

    def intervals(self):
        """
        Find intervals of constant state in the profile

        Returns
        -------
        list of tuples
            each entry signifies an interval, in the format ``(start, end,
            state)``, where ``start`` or ``end`` are ``None`` for the
            first/last interval in the profile, respectively.
        """
        boundaries = np.nonzero(np.diff(self.state))[0] + 1 # need indices on the right of the switch
        boundaries = [None] + boundaries.tolist()

        ivs = []
        for bl, br in zip(boundaries[:-1], boundaries[1:]):
            ivs.append((bl, br, self.state[br-1]))
        ivs.append((boundaries[-1], None, self.state[-1]))

        return ivs

    def plottable(self):
        """
        For plotting profiles

        Returns
        -------
        t, y : np.ndarray
            coordinates for plotting the profile as proper step function. See
            below for usage

        Example
        -------
        Given a profile ``profile``, you might visualize it as

        >>> from matplotlib import pyplot as plt
        ... plt.plot(*profile.plottable(), color='tab:green')
        ... plt.show()

        Note that another way to achieve the same thing is to just use
        `!matplotlib`'s ``stairs`` function:

        >>> plt.stairs(profile, edges=np.arange(-1, len(profile)))
        """
        ivs = self.intervals()
        ivs[0] = (0, ivs[0][1], ivs[0][2])
        ivs[-1] = (ivs[-1][0], len(self), ivs[-1][2])
        ivs = np.asarray(ivs)

        t = ivs[:, :2].flatten() - 1
        y = np.stack([ivs[:, 2], ivs[:, 2]], axis=-1).flatten()

        return t, y

def state_probabilities(profiles, nStates=None):
    """
    Calculate marginal probabilities for an ensemble of profiles

    Parameters
    ----------
    profiles : list of Loopingprofile
        the profiles to use
    nStates : int, optional
        the number of states in the model associated with the profiles. Since a
        profile might be (e.g.) flat zero, it is not in general possible to
        determine this from the profiles. Does not affect the calculated
        probabilities, but ("just") the shape of the output. If ``None``, will
        be identified from the profiles (i.e. ``nStates = (largest state index
        found) + 1``)

    Returns
    -------
    (nStates, T) np.ndarray
        the marginal probabilities for each state at each time point
    """
    allstates = np.array([profile[:] for profile in profiles])
    if nStates is None:
        nStates = np.max(allstates)+1

    counts = np.array([np.count_nonzero(allstates == i, axis=0) for i in range(nStates)])
    return counts / allstates.shape[0]
