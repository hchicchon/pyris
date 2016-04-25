#!/usr/bin/env python
# ===============================================================
#
# Script Name: SARA (SAtellite River Analysis)
# Author: Federico Monegaglia
# Date: November, 2015
# Description: Read Landsat Bands and Extracts River Features
#
# Branch: PRUT
#
# UniTN, QMUL 
#
# ===============================================================

from __future__ import division
import os, sys, shutil
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from skimage import morphology as mm
import pickle
import xlrd
from pyris import *
import warnings

# ==============================================================
# Settable
# ==============================================================
# Code enclosed here must be set by the user
# --------------------------------------------------------------
# Name of the Input Directory ( path: ./inputs/RIVER/<landsat_directories> )

try: RIVER=sys.argv[1]
except IndexError: RIVER = 'ucayali'
# Run Interactively only if you want to manually select object labels (recommended)
RUN_INTERACTIVE = True
# Index used to isolate the river channel from the surrounding planform
# ( can be either LGR (log(Green/Red)), LGB (log(Green/Blue)),
#   NDVI (normalized difference veg, index), MNDWI (water index) )
SEGMENTATION_METHOD = 'NDVI'
# Thresholding method for Otsu's Threshold ('global' or 'local')
THRESH_METHOD = 'global'
# In order to store river properties streamwise, define where the water is coming from
# ( b:bottom, t:top, l:left, r:right)
FLOW_FROM = 'b'
# In order to remove tributaries and useless branches from the image mask,
# enter a scale for the average channel width (in meters)
RIVER_WIDTH = 300
# Skeletonization leaves some spurs on the planform. Pruning algorithm helps to get rid
# of them, but it is very expensive. If a length based approach is used to reconstruct the planform shape
# (RECONSTRUCTION_METHOD='length'), pruning may be skipped (PRUNE=False) or PRUNE_ITER can be kept very small.
# If RECONSTRUCTION_METHOD='width', a high limit of iterations is required (i.e., PRUNE_ITER=250).
# Be aware that, on complex planforms, PRUNE_ITER>500 could result in many hours of iteration.
PRUNE = True # Use pruning after skeletonization (require if width is used as axis bifurcation selector)
PRUNE_ITER = 50 # Maximum Number of Iterations in Spurs Removal Algorithm
# During channel axis vectorization, external branches, bifurcations andconflunces can be found.
# In order to choose one of them automatically, select width- or length-based approach by setting
# RECONSTRUCTION_METHOD either to 'width', 'length' or 'std'.
# Standard 'std' method uses width if the width of the branches is similar, length otherwise
RECONSTRUCTION_METHOD = 'std'
# --------------------------------------------------------------

# Directories
# -----------
idir = os.path.join( 'inputs', RIVER ) # Input Directory containing Landsat Data Directories
odir = os.path.join( 'outputs', RIVER ) # Main Output Directory
sdirs = { # Output Directories for each of the Segmentation Methods
    'NDVI' : os.path.join( odir, 'NDVI' ),
    'MNDWI' : os.path.join( odir, 'MNDWI' ),
    'AWEI' : os.path.join( odir, 'AWEI' )
    }

# Files & Folders
# ---------------
SM = SEGMENTATION_METHOD
sdir = sdirs[SM] # Where all files obtained by the same segmentation method are stored
geodir = os.path.join( sdir, 'geotransf' )
maskdir = os.path.join( sdir, 'mask' ) # Where Mask Stored
skeldir = os.path.join( sdir, 'skeleton' ) # Where Skeleton and Distance are Stored
prundir = os.path.join( sdir, 'pruned' ) # Where Pruned Skeleton and Distance are Stored
axisdir = os.path.join( sdir, 'axis' ) # Where Axis Attributes are Stored
labdir = os.path.join( sdir, 'label' ) # labeled masks

# Create Missing Directories
dirs_to_make = [ geodir, maskdir, skeldir, prundir, axisdir, labdir ]
map( os.makedirs, ( d for d in dirs_to_make if not os.path.isdir( d ) ) )

# Settings
# --------
CMAP = 'Spectral'
#plt.set_cmap( CMAP ) # Set Default Colormap
mpl.rc( 'font', size=20 )
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{siunitx}',
    r'\siteup{detect-all}',
    r'\usepackage{helvet}',
    r'\usepackage{sansmath}',
    r'\sansmath'
]
plt.close( 'all' )


# ==========================================
# Skeletonization of the Centerline Planform
# ==========================================

title = 'ANALYZING RIVER %s' % RIVER
print '=' * len(title)
print title
print '=' * len(title) 

print
print 'CREATING MASK'
print

# Input Landsat Data Directories

landsat_dirs = sorted( [ os.path.join( idir, f ) for f in os.listdir( idir ) if os.path.isdir( os.path.join( idir, f ) ) ] )

for landsat in landsat_dirs:

    # File Names
    # ----------
    landsatname = os.path.split(landsat)[-1]
    year = int( landsatname[9:13] )
    jday = int( landsatname[13:16] )
    name = '_'.join( ( landsatname[9:13], landsatname[13:16] ) )
    ofile = os.path.join( skeldir, '.'.join((name, 'npy')) )
    maskfile = os.path.join( maskdir, '.'.join((name, 'npy')) )
    geofile = os.path.join( geodir, '.'.join((name, 'p')) )

    if os.path.isfile( maskfile ):
        # Has the axis been extracted already?
        print 'DATA FOUND FOR FILE %s - FOR DATE %s - SKIPPING '  % ( landsatname, name )
        continue

    print 'PROCESSING FILE %s - FOR DATE %s' % (landsatname, name)
    
    # Load Landsat Data
    # -----------------    
    [ R, G, B, NIR, MIR, Bawei ], GeoTransf = LoadLandsatData( landsat )

    # Image Segmentation
    # ------------------
    print ' * Applying %s-based segmentation' % SM

    # Compute Water or Vegetation Index
    # ---------------------------------
    print '   Performing Segmentation'
    pixel_width = RIVER_WIDTH/GeoTransf['PixelSize']
    radius = 10 * pixel_width
    IDX, mask, globthresh = SegmentationIndex( R=R, G=G, B=B, NIR=NIR, MIR=MIR, index=SM, rad=radius, method=THRESH_METHOD )
    ## ShowRasterData( IDX, label='', title='Choose a threshold' ).show()

    # Create Clean Channel Mask
    # -------------------------
    print '   Processing Mask'
    mask = RemoveSmallObjects( mask.bw, 100*pixel_width**2 ) # One Hundred Widths of Channel at Least is Required
    print '   Small Objs Removed'
    radius = np.max( np.floor( 0.5 * ( pixel_width ) ) - 3, 0 )
    selem = mm.disk( radius )
    mask = BW( mm.binary_opening( mask.bw, selem ) )
    print '   Binary Opening Done'
    mask = RemoveSmallObjects( mask.bw, 100*pixel_width**2 ) # One Hundred Widths of Channel at Least is Required
    print '   More Objs Removed'    

    # Manage Landsat NaNs
    # -------------------
    print '   Removing NoData Values'
    nanmask = mm.binary_dilation( np.where( R==0,1,0 ), mm.disk(30) ) # NaNs Mask
    print '   NoData Dilated'
    mask = BW( np.where( nanmask==1, np.nan, mask.bw ) ) # Remove NaNs from Indexed Mask
    print '   NaNs added where NoData'

    # We need to add X-dimension to GeoTransg in order to make the correct georeference
    #GeoTransf['Lx'] = mask.bw.shape[0] # I put this in LoadLandsatData!!! TODO : Verify
    #GeoTransf['Ly'] = mask.bw.shape[1] # I put this in LoadLandsatData!!! TODO : Verify

    # Dump Data
    # ---------
    np.save( maskfile, mask.bw )
    with open( geofile, 'w' ) as gf: pickle.dump( GeoTransf, gf )
    print '   Data Dumped'


print
print 'LABELLING MASK'
print

# Input Landsat Data Directories
mask_files = sorted([os.path.join(maskdir, f) for f in os.listdir(maskdir)])
for mask_file in mask_files:

    # Landsat Files Name and Data
    # ---------------------------
    name = os.path.splitext(os.path.split( mask_file )[-1])[0]
    labfile = os.path.join( labdir, '.'.join((name, 'npy')) )
    geofile = os.path.join( geodir, '.'.join((name, 'p')) )

    if os.path.isfile( labfile ):
        # Has the axis been extracted?
        print 'DATA FOUND FOR FILE %s - SKIPPING '  % ( labfile )
        continue

    print 'PROCESSING FILE %s' % ( labfile )

    mask = BW( np.load(mask_file) ) # Load Mask

    # Identify and Label Image Features
    # ---------------------------------
    mask_lab, num_features = ndimage.measurements.label( mask.bw )
    masksm = RemoveSmallObjects(np.where(mask_lab > 0, 1, 0), 10000 ).bw
    mask_lab, num_features = ndimage.measurements.label( masksm )
    print '   Found %s features in river mask' % num_features

    # Label Selection
    # ---------------
    # This part shall be run with the RUN_INTERACTIVE check set on True
    if RUN_INTERACTIVE:
        plt.figure()
        plt.imshow( mask_lab, cmap=plt.cm.spectral, interpolation='nearest' )
        plt.colorbar()
        plt.title( 'Indentify the label corresponding to the river planform' )
        for ifeat in xrange(1,num_features+1):
            c0 = np.column_stack( np.where( mask_lab==ifeat ) )[0]
            plt.text( c0[1], c0[0], '%s' % ifeat, fontsize=30, bbox=dict( facecolor='white' ) )
        plt.show()
        fmax = int( raw_input( 'Which label do you want to use? ' ) )
    else:
        # The biggest element in the image will be used. THIS CAN LEAD TO FAILURE!!!
        warnings.warn( 'Label selection should be made by hand, otherwiseit may lead to important errors.\n' +
            'Please consider running this part of the script with RUN_INTERACTIVE set to True and with a graphical backend.')
        feats = np.zeros( num_features+1 )
        for ifeat in xrange( 1, num_features+1 ):
            feats[ifeat] = np.count_nonzero( np.where( mask_lab==ifeat, 1, 0 ) )
            fmax = feats.argmax()

    # Mask Selected Element and Dump
    # ------------------------------
    mask = BW( ( mask_lab==fmax ).astype( np.uint8 ) )
    print '   Mask Completed'
    np.save( labfile, mask.bw )
    print '   Data Dumped'


print
print 'MASK SKELETONIZATION'
print 

lab_files = sorted([os.path.join(labdir, f) for f in os.listdir(labdir)])
for ilab, lab_file in enumerate(lab_files):

    # Landsat Files Name and Data
    # ---------------------------
    name = os.path.splitext(os.path.split(lab_file)[-1])[0]
    ofile = os.path.join( skeldir, '.'.join((name, 'npy')) )
    geofile = os.path.join( geodir, '.'.join((name,'p')) )
    with open( geofile ) as gf: GeoTransf = pickle.load( gf )

    if os.path.isfile(ofile):
        print 'SKELETON FOUND FOR %s - SKIPPING' % name
        continue
    print 'PROCESSING %s' % name

    mask = BW( np.load( lab_file ) )

    # Mask Processing
    # ---------------
    pixel_width = RIVER_WIDTH/GeoTransf['PixelSize']
    mask = CleanIslands( mask.bw, 2*pixel_width**2 ) # Clean Holes Inside the Planform
    print '   Internal Spots cleaned...'
    mask = RemoveSmallObjects( mask.bw, 2*pixel_width**2 ) # Remove External Noise
    print '   Small objects removed...'
    skel, dist, skeldist = Skeletonize( mask.bw ) # Compute Axis and Distance
    print '   Skeletonization done...'
    np.save( ofile, skeldist )
    print '   Skeleton dumped'

    
print
print 'SKELETON PRUNING'
print

skel_files = sorted([os.path.join(skeldir, f) for f in os.listdir(skeldir)])
for skel_file in skel_files:

    if not PRUNE:
        print 'Spurs will not be removed (set PRUNE to True in order to apply spurs removal to skeleton.)'
        break
    
    # Landsat Files Name and Data
    # ---------------------------
    skelname = os.path.splitext( os.path.split( skel_file )[-1] )[0]
    name = skelname
    geofile = os.path.join( geodir, '.'.join((name, 'p')) )
    ofile = os.path.join( axisdir, '.'.join(( name, 'npy' )) )
    pruname = os.path.join( prundir, '.'.join(( name, 'npy' )) )

    # Skip if Required and Data has already been Computed
    if os.path.isfile( pruname ):
        print 'DATA FOUND FOR FILE %s - FOR DATE %s - SKIPPING..' % ( skelname, name )
        continue
    print 'PRUNING SKELETON %s - FOR DATE %s' % ( skelname, name )

    with open( geofile, 'r' ) as gf: GeoTransf = pickle.load( gf ) # Load GeoReferencing

    skeldist = np.load( skel_file ) # Load Skeletonized Image #
    skel = np.where( skeldist.astype(np.uint8) > 0, 1, 0 ).astype( np.uint8 )

    # Remove eventual small objects arising from river flowing too close to image borders
    mlab, nfeats = ndimage.measurements.label( skel, structure=np.ones((3,3)) )
    if nfeats > 1:
        print '   Removing smallest features from skeletonized image'
    feats = np.zeros( nfeats+1 )
    for ifeat in xrange( 1, nfeats+1 ):
        feats[ ifeat ] = np.count_nonzero( np.where( mlab==ifeat, 1, 0 ) )
        fmax = feats.argmax()
    skel = ( mlab==fmax ).astype( np.uint8 )
    skeldist *= skel

    # Prune
    # -----
    print ' * Removing Spurs'    
    pdist = Pruning( skeldist, PRUNE_ITER, smooth=False ) # Remove Spurs
    print '   Spurs removed...'
    np.save( pruname, pdist )
    print '   Pruned map dumped'


# =================================
# Extract Data from Pruned Skeleton
# =================================

print
print 'EXTRACTING CENTERLINE PROPERTIES'
print

# Centerline Extraction
# =====================
if PRUNE: prune_files = sorted([os.path.join(prundir, f) for f in os.listdir(prundir)])
else: prune_files = skel_files

for prune_file in prune_files:

    # Landsat Files Name and Data
    # ---------------------------
    pruname = os.path.splitext(os.path.split( prune_file )[-1])[0]
    name = pruname
    geofile = os.path.join( geodir, '.'.join((name, 'p')) )
    ofile = os.path.join( axisdir, '.'.join(( name, 'npy' )) )

    if os.path.isfile( ofile ):
        print 'DATA FOUND FOR FILE %s - SKIPPING...' % ofile
        continue

    print 'PROCESSING SKELETON %s - FOR DATE %s' % ( prune_file, name )

    with open( geofile, 'r' ) as gf: GeoTransf = pickle.load( gf ) # Load GeoReferencing
    pdist = np.load( prune_file )[::-1,:] # Load Pruned Skeletonized Image
    # Somewhere we reversed the image for some reason =/ # FIXME !


    # Axis Properties Extraction
    print ' * Computing Centerline Properties'
    # Here we get the axis both in pixels and in the georeferenced system
    axis = ReadAxisLine( pdist, GeoTransf, flow_from=FLOW_FROM, method=RECONSTRUCTION_METHOD )

    # Axis PCS Interpolation
    Npoints = min( max( axis.d['L'] / (0.5*axis.d['B'].mean()), 3000 ), 5000 )
    m = axis.x.size
    PCSs = m # We use the highest degree of smoothness
    x_PCS, y_PCS, d1x_PCS, d1y_PCS, d2x_PCS, d2y_PCS = InterpPCS(
        axis.x[::20], axis.y[::20], # Use some step to reduce the noise
        s=PCSs, N=Npoints )
    s_PCS, theta_PCS, Cs_PCS = CurvaturePCS( x_PCS, y_PCS,
        d1x_PCS, d1y_PCS, d2x_PCS, d2y_PCS,
        method=2, return_diff=False )
    B_PCS = WidthPCS( axis.d['s']/axis.d['s'][-1]*s_PCS[-1], axis.d['B'], s_PCS )
    print '   Filtering channel half-width'
    for ifilter in xrange(10): B_PCS[1:-1] = 0.25 * ( B_PCS[:-2] + 2*B_PCS[1:-1] + B_PCS[2:] )
    print '   PCS Interpolated...'    

    if RUN_INTERACTIVE:
        plt.figure()
        plt.plot( x_PCS/1000, y_PCS/1000 ) # We assume the reference system is in meters
        plt.xlabel( r'x[km]' )
        plt.ylabel( r'y[km]' )
        plt.title( r'River Centerline' )
        plt.show()

    # Save Axis Properties
    np.save( ofile, (x_PCS, y_PCS, s_PCS, theta_PCS, Cs_PCS, B_PCS) )
    print '    Axis Properties dumped on %s' % ofile


# Migration Rates
# ===============
print
print 'SEPARATING MEANDER BENDS AND COMPUTING MIGRATION RATES'
print


# Results are appended to the axis file
axis_files = sorted([os.path.join(axisdir, f) for f in os.listdir(axisdir)])
geo_files = sorted([os.path.join(geodir, f) for f in os.listdir(geodir)])

# Eventually make a plot of the centerlines
if RUN_INTERACTIVE:
    # Set Plot Colors
    number_of_lines= len( axis_files )
    cm_subsection = np.linspace(0, 1, number_of_lines) 
    colors = [ plt.cm.jet(x) for x in cm_subsection ]
    lws = np.linspace(0.2, 2, number_of_lines)
    # Plot
    plt.figure()
    plt.hold('on')
    for i, (axis, geo) in enumerate(zip(axis_files, geo_files)):
        data = np.load(axis)
        ilab = int( os.path.splitext(os.path.split(axis)[-1])[0].split('_')[0] )
        lab = r'$%d$' % ilab
        x = data[1] / 1000
        y = data[0] / 1000
        plt.plot( y, x, c=colors[i], lw=lws[i], label=lab )
    plt.xlabel(r'$x[km]$')
    plt.ylabel(r'$y[km]$')
    plt.title(r'UTM Coordinates')
    plt.legend( loc='center left', bbox_to_anchor=(1,0.5), ncol=2 )
    plt.axis('equal')
    plt.show()
    plt.close()


# List Planforms and Times
# ------------------------
data = []
times = []
for i_file, axis_file in enumerate( axis_files ):    
    # Landsat Files Name and Data
    # ---------------------------
    axisname = os.path.splitext( os.path.split( axis_file )[-1] )[0]
    name = axisname
    times.append( name )
    data.append( np.load( axis_file )[:6] )


# Compute Migration Rates
# -----------------------
D = MigRateBend( data, T=times )( filter_reduction=0.33 )

# Append Data to the Axis File and Dump It
# ----------------------------------------
for i, pruname in enumerate( axis_files ):
    pname = os.path.splitext(os.path.split(pruname)[-1])[0]
    ofile = os.path.join( axisdir, '.'.join(( pname, 'npy' )) )
    stuff_to_append = [
                D['Csf1'][i], D['Csf3'][i], D['B1'][i], D['B12'][i], D['BUD'][i],
                D['dx'][i], D['dy'][i], D['dz'][i],
                D['rot'][i], D['Dx'][i], D['Dy'][i], D['dA'][i], D['dL'][i], D['dS'][i] 
            ]
    data_to_dump = np.vstack( ( data[i], stuff_to_append ) )
    np.save( ofile, data_to_dump )

# Eventually Plot Bends Separation
if RUN_INTERACTIVE:
    BENDS = np.unique( D['B1'][0][D['B1'][0]>0] )
    nbends = BENDS.size
    lws = np.linspace( 0.5, 5, len(data) )
    colors = [ plt.cm.Set1(x) for x in np.linspace(0, 1, nbends) ]
    plt.figure()
    for iB, B in enumerate( BENDS ):
        for i, d in enumerate( data ):
            xi, yi, si = d[0], d[1], d[2]
            BI = D['B1'][i]
            B12 = D['B12'][i]
            bend = BI==B
            X = xi[ bend ]
            Y = yi[ bend ]
            S = si[ bend ]
            if i==0:
                plt.plot( X, Y, lw=lws[i], color=colors[iB], label='%s' % B )
            else:
                plt.plot( X, Y, lw=lws[i], color=colors[iB] )
            B = ( B12[ bend ][ 0 ] )
    plt.legend( ncol=4 )
    plt.axis('equal')
    plt.show()

if RUN_INTERACTIVE:
    B = int( np.unique( D['B1'][0][D['B1'][0]>0] ).max() / 2 ) # Plot a BEND in the middle
    lws = np.linspace( 0.5, 5, len(data) )
    plt.figure()
    for i, d in enumerate( data ):
        xi, yi, si = d[0], d[1], d[2]
        BI = D['B1'][i]
        B12 = D['B12'][i]
        dx = D['dx'][i]
        dy = D['dy'][i]
        dz = D['dz'][i]
        bend = BI==B
        X = xi[ bend ]
        Y = yi[ bend ]
        S = si[ bend ]
        DX = dx[ bend ]
        DY = dy[ bend ]
        DZ = dz[ bend ]
        A = np.arctan2( DY,DX )
        if i==0:
            plt.plot( X, Y, lw=lws[i], color=colors[i], label='%s' % B )
        else:
            plt.plot( X, Y, lw=lws[i], color=colors[i] )
        for j in xrange(X.size):
            plt.arrow( X[j], Y[j], DZ[j]*np.cos(A[j]), DZ[j]*np.sin(A[j]), fc='k', ec='k' )
        B = ( B12[ bend ][ 0 ] )
    plt.legend( ncol=4 )
    plt.axis('equal')
    plt.show()
