import os
from setuptools import dist, Extension, setup

# dist.Distribution().fetch_build_eggs(['numpy>=1.20'])
import numpy

CYTHONIZE = os.getenv('CYTHONIZE') == "1"
PYTHON_ONLY = os.getenv('PYTHON_ONLY') == "1"

if PYTHON_ONLY:
    extensions = []
else:
    ext = '.pyx' if CYTHONIZE else '.c'
    extensions = [
        Extension(
            "bild.bin.MSRouse_logL", ["bild/src/MSRouse_logL"+ext],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ),
    ]

if CYTHONIZE:
    # Get dependencies of the cython build
    # These are only needed when cythonizing .pyx --> .c, which should always
    # done locally. These are thus separate from the install requirements in
    # pyproject.toml
    # fetch_build_eggs() also installs stuff if necessary, so this does not
    # require pre-installed packages
    with open('cython_requirements.txt') as f:
        reqs = [l for l in [l.strip() for l in f] if not l.startswith('#')]

    dist.Distribution().fetch_build_eggs(reqs)
    from Cython.Build import cythonize

    extensions = cythonize(extensions,
                    compiler_directives = {'language_level' : '3'},
                )

    # Remove hash strings from comments in the .c to prevent the file from
    # changing on every build (which freaks out git)
    import re
    for ext in extensions:
        for cfile in ext.sources:
            with open(cfile, 'r') as f:
                lines = f.readlines()
            with open(cfile, 'w') as f:
                for line in lines:
                    m = re.search('(pip-build-env-)[^/]*/', line)
                    if m:
                        line = ( line[:m.start()]
                                  + m[1]
                                  + '<hash_removed>'
                                  + line[m.end()-1:]
                                )
                    f.write(line)

setup(
    ext_modules  = extensions,
    include_dirs = [numpy.get_include()],
    zip_safe     = False,
)
