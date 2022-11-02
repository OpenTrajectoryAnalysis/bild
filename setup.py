import os
from setuptools import dist, Extension, setup

dist.Distribution().fetch_build_eggs(['numpy>=1.20'])
import numpy

CYTHONIZE = os.getenv('CYTHONIZE') == "1"
ext = '.pyx' if CYTHONIZE else '.c'
extensions = [
    Extension(
        "bild.cython.MSRouse_logL", ["bild/cython/MSRouse_logL"+ext],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ),
]

if CYTHONIZE:
    # hardcode cython code dependencies for now
    dist.Distribution().fetch_build_eggs(['Cython>=0.25', 'scipy>=1.6'])
    from Cython.Build import cythonize

    extensions = cythonize(extensions,
                    compiler_directives = {'language_level' : '3'},
                )

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    ext_modules  = extensions,
    include_dirs = [numpy.get_include()],
    zip_safe     = False,

    name         = 'bild',
    version      = '0.0.3',
    author       = 'Simon Grosse-Holz',
    author_email = 'sgh256@mit.edu',
    description  = 'Bayesian Inference of Looping Dynamics',
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    license      = 'LICENSE',
    python_requires = '>=3.7',
    classifiers  = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        "noctiluca >=0",
        "rouse >=0",
        "numpy >=1.20",
        "scipy >=1.6",
        "tqdm >=4",
    ],
    url = "https://github.com/OpenTrajectoryAnalysis/bild",
    project_urls = {
        "Documentation" : "https://bild.readthedocs.org/en/latest",
        "Bug Tracker"   : "https://github.com/OpenTrajectoryAnalysis/bild/issues",
    },
)
