from warnings import warn

try:
    from .bin.MSRouse_logL import MSRouse_logL
except ImportError: # pragma: no cover
    warn("Did not find compiled code for MSRouse_logL, falling back to python")
    from .src.MSRouse_logL_py import MSRouse_logL
