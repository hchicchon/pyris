from __future__ import division
from ..misc import isRGB, BW
import numpy as np
from skimage.filter import threshold_otsu, rank
from skimage.util import img_as_ubyte
import warnings


def Thresholding( rgb, band=None ):
    '''Thresholding(rgb) - Apply Otsu's Thresholding Method'''
    # Check rgb image
    if not isRGB( rgb ):
        raise TypeError, \
          'Input must be an RGB image'
    # Assign band
    if band is None: idx = 0 # Defaut band is R
    elif isinstance(band, str):
        if band.lower()[0] == 'r': idx = 0
        elif band.lower()[0] == 'g': idx = 1
        elif band.lower()[0] == 'b': idx = 2
    # Apply Threshold to selected Band
    img = rgb[:,:,idx] # Band Index
    thresh = threshold_otsu( img ) # Compute Otsu's Threshold
    bw = img < thresh # Apply Threshold
    return BW( bw )


def SegmentationIndex( *args, **kwargs ):
    '''Apply Index'''
    R = kwargs.pop( 'R', np.nan ).astype( float )
    G = kwargs.pop( 'G', np.nan ).astype( float )
    B = kwargs.pop( 'B', np.nan ).astype( float )
    NIR = kwargs.pop( 'NIR', np.nan ).astype( float )
    MIR = kwargs.pop( 'MIR', np.nan ).astype( float )
    Bawei = kwargs.pop( 'Bawei', np.nan ).astype( float )
    index = kwargs.pop( 'index', None )
    rad = kwargs.pop( 'radius', 20 )
    method = kwargs.pop( 'method', 'local' )

    if index == 'NDVI':
        IDX =  (NIR - R) / (NIR + R)
    elif index == 'MNDWI':
        IDX =  (G - MIR) / (G + MIR)
    elif index == 'AWEI':
        warnings.warn('AWEI index has not been tested and may contain bugs. Please check your results')
        IDX =  4 * ( G - MIR ) - ( 0.25*NIR + 2.75*Bawei ) # TODO: verify
    else:
        err = 'Index %s not recognized' % IDX
        raise ValueError, err
    
    # Apply Local Otsu's Method
    globthresh = threshold_otsu( IDX[np.isfinite(IDX)] )
    if method == 'local':
        print "   Local Otsu's Method - This may require some time..."
        selem = mm.disk( rad )
        thresh = rank.otsu( img_as_ubyte(IDX), selem )
    else:
        thresh = globthresh
    print '   ...done'
    if index == 'NDVI': MASK = img_as_ubyte(IDX) <= thresh
    else: MASK = img_as_ubyte(IDX) >= thresh

    return IDX, BW( MASK ), globthresh
