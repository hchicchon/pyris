# -*- coding: utf-8 -*-
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
import os, sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from skimage import morphology as mm
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


# =====================
# Main Script Functions
# =====================

def segment_all( landsat_dirs, geodir, config, maskdir, auto_label=None ):
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
        mask = CleanIslands( mask, 2*pixel_width**2 ) # Clean Holes Inside the Planform
        mask = RemoveSmallObjects( mask, 100*pixel_width**2 ) # Remove New Small Objects
        mask = mask.astype( int )

        # Label Masks
        mask_lab, num_features = ndimage.measurements.label( mask ) # FIXME: apply labelling to the skeleton and add --fast flag to pyris (remove minimum values from skeleton to make pruning faster)!
        print 'labelling feature in channel mask...'
        print 'found %s features in river mask %s...' % ( num_features, os.path.basename(maskfile) )

        if auto_label is None:
            plt.figure()
            plt.imshow( mask_lab, cmap=plt.cm.spectral, interpolation='none' )
            plt.title( 'Indentify the label(s) corresponding to the river planform.' )
            for ifeat in xrange(1,num_features+1):
                c0 = np.column_stack( np.where( mask_lab==ifeat ) )[0]
                plt.text( c0[1], c0[0], '%s' % ifeat, fontsize=30, bbox=dict( facecolor='white' ) )
            plt.show()
            labs = raw_input( 'Please enter the label(s) do you want to use? (if more than one, separate them with a space): ' ).split(' ')
            for lab in labs: mask += np.where( mask_lab==int(lab), int(lab), 0 )
        else:
            # The larger element in the image will be used.
            warnings.warn('automated labels may lead to errors! please check your results!')
            if auto_label == 'auto': # FIXME: use MNDWI threshold to define which labels to use
                if eval( config.get( 'Segmentation', 'white_masks' ) ) == []:
                    auto_label = 'max'
                else:
                    auto_label = 'all'
            if auto_label == 'all':
                mask = mask_lab
            elif auto_label == 'max':
                labs, counts = np.unique( mask_lab[mask_lab>0], return_counts=True )
                mask = mask_lab==labs[ counts.argmax() ]
            else:
                e = "labelling method '%s' not known. choose either 'auto', 'max', 'all' or None"
                raise ValueError, e

        print 'saving  mask and GeoTransf data...'
        np.save( maskfile, mask )
        with open( geofile, 'w' ) as gf: pickle.dump( GeoTransf, gf )
    return None


def vectorize_all( geodir, maskdir, config, axisdir ):

    maskfiles = sorted( [ os.path.join(maskdir, f) for f in os.listdir(maskdir) ] )
    geofiles = sorted( [ os.path.join(geodir, f) for f in os.listdir(geodir) ] )

    for ifile, (maskfile, geofile) in enumerate( zip( maskfiles, geofiles ) ):
        # input
        name = os.path.splitext( os.path.basename( maskfile ) )[0]
        # output
        axisfile = os.path.join( axisdir, '.'.join(( name, 'npy' )) )

        # skip the files which have already been processes
        if os.path.isfile( axisfile ):
            print 'data found for file %s - skipping '  % ( axisfile )
            continue
        print
        print 'Processing file %s' % ( maskfile )

        # Load mask and GeoFile
        GeoTransf = pickle.load( open( geofile ) )
        mask = np.load( maskfile )

        # Skeletonization
        print 'skeletonizing...'
        skel, dist = Skeletonize( np.where(mask>0,1,0).astype( int ) ) # Compute Axis and Distance
        labelled_skel = skel.astype(int) * mask.astype(int)

        # Pruning
        print 'pruning n=%d labelled elements...' % mask.max()
        pruned = np.zeros( mask.shape, dtype=int )
        for lab in xrange( 1, mask.max()+1 ):
            print 'pruning label %d...' % lab
            pruned += Pruning( labelled_skel==lab, int(config.get('Pruning', 'prune_iter')), smooth=False ) # Remove Spurs

        # Centerline Extraction
        print 'extracting centerline of n=%d labelled elements...' % pruned.max()
        axis = Line2D()
        for lab in xrange( pruned.max(), 0, -1 ): # FIXME!!!! Dipende da flow_from!!!! -> forse meglio ruotare l'immagine a seconda del flow from già prina della fase di labelling oppure trovare la label giusta dalla posizione
            print 'extracting label %d...' % lab
            axis.join( ReadAxisLine( dist*(pruned==lab), flow_from=config.get('Data', 'flow_from'),
                                     method=config.get('Axis', 'reconstruction_method') ) )

        # Interpolation
        print 'parametric cublic spline interpolation of the centerline...'
        step = max( 1, int( 2*axis.B.mean() ) ) # Discard points if too close
        Npoints = min( max( axis.L / (0.5*axis.B.mean()), 3000 ), 5000 ) # Spacing = width/4 but between (3000,5000)
        PCSs = axis.x[::step].size # Degree of smoothness = n. of data points
        
        # Pixelled PCS
        xp_PCS, yp_PCS, d1xp_PCS, d1yp_PCS, d2xp_PCS, d2yp_PCS = InterpPCS( # Build Spline
            axis.x[::step], axis.y[::step], s=PCSs, N=Npoints
            )
        sp_PCS, thetap_PCS, Csp_PCS = CurvaturePCS( xp_PCS, yp_PCS, d1xp_PCS, d1yp_PCS, d2xp_PCS, d2yp_PCS,
                                                    method=2, return_diff=False )
        Bp_PCS = WidthPCS( axis.s/axis.L*sp_PCS[-1], axis.B, sp_PCS )
        for ifilter in xrange(10): Bp_PCS[1:-1] = 0.25 * ( Bp_PCS[:-2] + 2*Bp_PCS[1:-1] + Bp_PCS[2:] ) # Filter channel width
       
        # GeoReferenced PCS
        s_PCS, theta_PCS, Cs_PCS, B_PCS = \
            sp_PCS*GeoTransf['PixelSize'], thetap_PCS, Cs_PCS/GeoTransf['PixelSize'], Bp_PCS*GeoTransf['PixelSize']
        GR = GeoReference( pruned, GeoTransf )
        x_PCS, y_PCS = GR.RefCurve( xp_PCS, yp_PCS )

        # Save Axis Properties
        print 'saving main channel data...'
        np.save( ofile, ( xp_PCS, yp_PCS, x_PCS, y_PCS, s_PCS, theta_PCS, Cs_PCS, B_PCS ) )
