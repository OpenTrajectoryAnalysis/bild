"""
Bayesian Inference of Loop Dynamics (BILD)

This module provides the implementation of BILD, proposed by `Gabriele,
Brandão, Grosse-Holz, et al. <https://doi.org/10.1126/science.abn6583>`_. Since
the ideas behind this scheme are explained in the paper (and its supplementary
information), in the code we only provide technical documentation, assuming
knowledge of the reference text.
"""
from . import util
from .util import Loopingprofile
from . import models
from . import amis
from . import postproc
from .core import *
