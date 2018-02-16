#!/usr/bin/env python

# .:==============:.
# .:    PyRIS     :.
# .: -------------:.
# .: Setup Script :.
# .:==============:.

# Python - RIvers from Satellites

from distutils.core import setup

description = 'PyRIS :: Python - RIvers by Satellites'
long_description = '\n'.join((
    description,
    '''

    See Publication:
    
    --------------------------------------------------------------------------------------------------
    Monegaglia et al.
    "Automated extraction of meandering river morphodynamics from multitemporal remotely sensed data"
    (under consideration for publication in "Environmental Modeling & Software")
    --------------------------------------------------------------------------------------------------

    Requires: NumPy, SciPy, MatPlotLib, GDAL
    
    '''
))

setup(
    name = 'pyris',
    version = '1.0',
    author = 'Federico Monegaglia',
    author_email = 'f.monegaglia@gmail.com',
    maintainer = 'Federico Monegaglia',
    maintainer_email = 'f.monegaglia@gmail.com',
    description = description,
    long_description = long_description,
    url = 'https://github.com/fmonegaglia/pyris',
    install_requires = [ 'numpy', 'matplotlib', 'scikit-image', 'gdal' ],
    packages = [ 'pyris', 'pyris.config', 'pyris.misc', 'pyris.raster', 'pyris.vector' ],
    py_modules = [ 'misc', 'raster', 'vector' ],
    scripts = ['pyris/bin/pyris'],
    
)
    
