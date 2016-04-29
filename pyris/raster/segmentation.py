from __future__ import division
import os, sys
from ..misc import isRGB, BW, LoadLandsatData
from .morphology import RemoveSmallObjects, CleanIslands
import numpy as np
from scipy import ndimage
from skimage.filter import threshold_otsu, rank
from skimage import morphology as mm
from skimage.util import img_as_ubyte
from matplotlib import pyplot as plt
import warnings
import pickle

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
    R = kwargs['R'].astype( float )
    G = kwargs['G'].astype( float )
    B = kwargs['B'].astype( float )
    NIR = kwargs.pop( 'NIR', np.full(R.shape,np.nan) ).astype( float )
    MIR = kwargs.pop( 'MIR', np.full(R.shape,np.nan) ).astype( float )
    Bawei = kwargs.pop( 'Bawei', np.full(R.shape,np.nan) ).astype( float )
    index = kwargs.pop( 'index', None )
    rad = kwargs.pop( 'radius', 20 )
    method = kwargs.pop( 'method', 'local' )

    if index == 'NDVI':
        IDX =  (NIR - R) / (NIR + R)
    elif index == 'MNDWI':
        IDX =  (G - MIR) / (G + MIR)
    elif index == 'AWEI':
        raise NotImplementedError
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
        print '   ...done'
    else:
        thresh = globthresh
    if index == 'NDVI': MASK = img_as_ubyte(IDX) <= thresh
    else: MASK = img_as_ubyte(IDX) >= thresh

    return IDX, MASK, globthresh


def run_all_segmentation( landsat_dirs, geodir, maskdir, config, auto_label=False ):
    '''
    Iterate over all the Landsat Directories and perform image segmentation
    landsat_dirs:     directory containing all the landsat directories
    geodir:           directory where GeoTransf instances are stored
    maskdir:          directory where channel masks are stored
    config:           PyRIS' RawConfigParser instance
    '''
    to_skip = []
    # Iterate over Landsat Directories
    for landsat in landsat_dirs:
        # input
        landsatname = os.path.basename( landsat )
        year = landsatname[9:13]
        jday = landsatname[13:16]
        name = '_'.join( ( year, jday ) )
        # output
        maskfile = os.path.join( maskdir, '.'.join( (name, 'npy') ) )
        geofile = os.path.join( geodir, '.'.join( (name, 'p') ) )

        # skip the files which have already been processes
        if all( map( os.path.isfile, [ maskfile, geofile ] ) ):
            print 'data found for file %s - skipping '  % ( landsatname )
            to_skip.append( name )
            continue

        print
        print 'Processing file %s' % ( landsatname )

        bands, GeoTransf = LoadLandsatData( landsat )

        # Apply White Mask
        white = np.zeros( bands[0].shape, dtype=int )
        for s in eval( config.get( 'Segmentation', 'white_masks' ) ):
            white[ s[0], s[1] ] = 1

        # Apply Black Mask
        black = np.ones( bands[0].shape, dtype=int )
        for s in eval( config.get( 'Segmentation', 'black_masks' ) ):
            black[ s[0], s[1] ] = 0

        # Mark Landsat NoData
        nanmask = np.where( bands[0]==0, 1, 0 )
        nanmask = mm.binary_dilation( nanmask, mm.disk( 5 ) )
        [ R, G, B, NIR, MIR ] =  [ np.where(white*black==1, band, np.nan) for band in bands ]

        # Set Dimensions
        pixel_width = config.getfloat('Data', 'channel_width') / GeoTransf['PixelSize'] # Channel width in Pixels
        radius = 10 * pixel_width # Circle Radius for Local Thresholding
        # Compute Mask
        print 'computing mask...'
        _, mask, _ = SegmentationIndex( R=R, G=G, B=B, NIR=NIR, MIR=MIR, index=config.get('Segmentation', 'method'), 
                                        rad=radius, method=config.get('Segmentation', 'thresholding') )
        mask = np.where( nanmask==1, 0, mask*white*black )
        # Image Cleaning
        print 'cleaning mask...'
        mask = RemoveSmallObjects( mask, 100*pixel_width**2 ) # One Hundred Widths of Channel at Least is Required
        radius = max( np.floor( 0.5 * ( pixel_width ) ) - 3, 0 )
        mask = mm.binary_opening( mask, mm.disk( radius ) ) # Break Small Connectins
        mask = RemoveSmallObjects( mask, 100*pixel_width**2 ) # Remove New Small Objects
        print 'saving  mask and GeoTransf data...'
        np.save( maskfile, mask )
        with open( geofile, 'w' ) as gf: pickle.dump( GeoTransf, gf )

    # Label Masks
    print 'labelling feature in channel mask...'
    for maskfile in sorted( os.listdir(maskdir) ):
        if os.path.splitext( maskfile )[0] in to_skip: continue
        mask = np.load( os.path.join( maskdir, maskfile ) ).astype( int )
        mask_lab, num_features = ndimage.measurements.label( mask )
        print 'found %s features in river mask %s...' % ( num_features, maskfile )

        if not auto_label:
            plt.figure()
            plt.imshow( mask_lab, cmap=plt.cm.spectral, interpolation='none' )
            plt.title( 'Indentify the label(s) corresponding to the river planform.' )
            for ifeat in xrange(1,num_features+1):
                c0 = np.column_stack( np.where( mask_lab==ifeat ) )[0]
                plt.text( c0[1], c0[0], '%s' % ifeat, fontsize=30, bbox=dict( facecolor='white' ) )
            plt.show()
            labs = raw_input( 'Please enter the label(s) do you want to use? (if more than one, separate them with a space) ' ).split(' ')
            mask *= 0
            for lab in labs: mask += np.where( mask_lab==int(lab), int(lab), 0 )
        else:
            # The larger element in the image will be used.
            warnings.warn('automated labels may lead to errors! please check your results!')
            feats = np.zeros( num_features+1 )
            for ifeat in xrange( 1, num_features+1 ):
                feats[ifeat] = np.count_nonzero( np.where( mask_lab==ifeat, 1, 0 ) )
                fmax = feats.argmax()
                mask = ( mask_lab==fmax ).astype( np.uint8 )
        np.save( os.path.join( maskdir, maskfile), mask )
    return None
