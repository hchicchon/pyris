from __future__ import division
import os, sys
import numpy as np
from skimage.filter import threshold_otsu, rank
from skimage import morphology as mm
from skimage.util import img_as_ubyte


def Thresholding( rgb, band=None ):
    '''Thresholding(rgb) - Apply Otsu's Thresholding Method'''
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
    return bw


def SegmentationIndex( *args, **kwargs ):
    '''Apply Index'''

    R = kwargs['R'].astype( float )
    G = kwargs['G'].astype( float )
    B = kwargs['B'].astype( float )
    NIR = kwargs.pop( 'NIR', np.full(R.shape,np.nan) ).astype( float )
    MIR = kwargs.pop( 'MIR', np.full(R.shape,np.nan) ).astype( float )
    SWIR = kwargs.pop( 'SWIR', np.full(R.shape,np.nan) ).astype( float )
    index = kwargs.pop( 'index', None )
    rad = kwargs.pop( 'radius', 20 )
    method = kwargs.pop( 'method', 'local' )

    if index == 'NDVI':
        IDX =  (NIR - R) / (NIR + R)
    elif index == 'MNDWI':
        IDX =  (G - MIR) / (G + MIR)
    elif index == 'BAR':
        IDX = SWIR
    elif index == 'MIX':
        IDX = (NIR - R) / (NIR + R)
        IDXX = (G - MIR) / (G + MIR)
        IDXXX = SWIR
    elif index == 'AWEI':
        raise NotImplementedError
        IDX =  4 * ( G - MIR ) - ( 0.25*NIR + 2.75*Bawei ) # TODO: verify
    else:
        err = 'Index %s not recognized' % IDX
        raise ValueError, err
    # Apply Local Otsu's Method
    selem = mm.disk( rad )
    globthresh = threshold_otsu( IDX[np.isfinite(IDX)] )

    if index=='MIX': globthreshX = threshold_otsu( IDXX[np.isfinite(IDXX)] )

    if method == 'local':
        print "applying local Otsu method - this may require some time... ", 
        thresh = rank.otsu( img_as_ubyte(IDX), selem ).astype(float)
        if index=='MIX': threshX = rank.otsu( img_as_ubyte(IDXX), selem ).astype(float)
        print 'done'
    else:
        thresh = globthresh
        if index=='MIX':
            threshX = globthreshX
            threshXX = 90

    #from matplotlib import pyplot as plt
    #plt.figure()
    #plt.imshow(SWIR, cmap='spectral', interpolation='none')
    #plt.colorbar()
    #plt.contour( IDX>thresh, 1 )
    #plt.title('%f' % thresh)
    #plt.axis('tight')
    #plt.show()

    if index == 'NDVI': MASK = IDX <= thresh
    elif index == 'MIX':
        MASK = np.logical_or( ( IDX<=thresh ) * ( mm.binary_dilation(IDXX>=threshX,mm.disk(0.3*rad)) ), IDXXX>threshXX)
    else: MASK = IDX >= thresh

    return IDX, MASK.astype( int ), globthresh


