# ===================================================================================
#
# Package: PyRIS
# Author: Federico Monegaglia
# Date: 2016
# Description: A Toolbox for extracting river planform features from satellite images
#
# ===================================================================================

'''
PyRIS :: Python - RIvers from Satellite
====================================
'''

# Imports
from skimage.util import img_as_ubyte
from skimage.io import imread
from scipy import ndimage
import warnings

# Suppress Warnings
warnings.filterwarnings("ignore")

__author__ = 'Federico Monegaglia'
__email__ = 'f.monegaglia@gmaial.com'
__version__ = '0.1'
__year__ = '2016'

__all__ = [
    # modules
    'raster', 'vector', 'misc',
    # pkg data
    '__author__', '__version__', '__email__', '__year__',
    # from standard packages
    'img_as_ubyte', 'imread', 'ndimage',
    # mlpy compatibility
    'HAS_MLPY', 'MLPYException', 'MLPYmsg',
    # misc
    'GeoReference', 'find_nearest', 'NaNs', 'BW',
    # raster
    'CleanIslands', 'RemoveSmallObjects', 'Skeletonize',
    'Pruner', 'Pruning',
    'Thresholding', 'SegmentationIndex',
    # vector
    'AxisReader', 'ReadAxisLine',
    'InterpPCS', 'CurvaturePCS', 'WidthPCS',
    'MigRateBend', 'LoadLandsatData',
    ]

# Check if correct version of MLPY is installed
try:
    import mlpy.wavelet as wave # mlpy>=3.5 compiled from source
    HAS_MLPY = True
except ImportError:
    HAS_MLPY = False

class MLPYException( ImportError ):
    pass
MLPYmsg = '''Module MPLY not found or the version is old.'''

# Import Everything from SubModules
from raster import *
from vector import *
from misc import *
from config import *
