#!/usr/bin/env python
# ===============================================================
#
# Script Name: SINBA (SINgle Bend Analysis)
# Author: Federico Monegaglia
# Date: November, 2015
# Description: Read Landsat Bands and Extracts River Features
# 
# ===============================================================

from __future__ import division
import os, sys, shutil
import warnings
import pickle
import matplotlib as mpl
from matplotlib import gridspec
from pyris.Raster.Raster import *
from pyris.Vector.Vector import *

# Suppress Warnings
warnings.filterwarnings("ignore")

# ==============================================================
# Settable
# ==============================================================
# Code enclosed here must be set by the user
# --------------------------------------------------------------
# Name of the Input Directory ( path: ./inputs/RIVER/<landsat_directories> )
RIVER = 'upstream_beni'
SEGMENTATION_METHOD = 'NDVI'
# In order to store river properties streamwise, define where the water is coming from
# ( b:bottom, t:top, l:left, r:right)
# Reconstruction of the Two-Dimensional Structure in Intrinsic Coordinates (s,n)
# of a Given Raster Band (R, G, B, NIR, MIR).
# Put None to skip
BAND2D = 'R'
# --------------------------------------------------------------

# Directories
# -----------
idir = os.path.join( 'inputs', RIVER ) # Input Directory containing Landsat Data Directories
odir = os.path.join( 'outputs', RIVER ) # Main Output Directory
sdirs = { # Output Directories for each of the Segmentation Methods
    'LGR' : os.path.join( odir, 'LGR' ),
    'LGB' : os.path.join( odir, 'LGB' ),
    'NDVI' : os.path.join( odir, 'NDVI' ),
    'MNDWI' : os.path.join( odir, 'MNDWI' )
    }
snames = { # Full Names of the Segmentation Methods
    'LGR' : r'log(G / R)',
    'LGB' : r'log(G / B)',
    'NDVI' : r'NDVI',
    'MNDWI' : r'MNDVI',
    }
sthresh = { # Default Thresholds for Segmentation Methods
    'LGR' : 0, # To Be Defined
    'LGB' : 0, # To Be Defined
    'NDVI' : 0.15,
    'MNDWI' : 0 # To Be Defined
    }    


# Files & Folders
# ---------------
SM = SEGMENTATION_METHOD
sdir = sdirs[SM] # Where all files obtained by the same segmentation method are stored
axisdir = os.path.join( sdir, 'axis' ) # Where Axis Attributes are Stored
axis_files = sorted([os.path.join(axisdir, f) for f in os.listdir(axisdir)])

# Settings
# --------
CMAP = 'Spectral'
#plt.set_cmap( CMAP ) # Set Default Colormap
mpl.rc( 'font', size=12 )
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{siunitx}',
    r'\siteup{detect-all}',
    r'\usepackage{helvet}',
    r'\usepackage{sansmath}',
    r'\sansmath'
]
plt.close( 'all' )


BEND = 5
# Set some Lists
sinuosity = []

plt.figure()
# Now Finally Plot Some Bend Evolution Stuff
for i_file, (axis_file) in enumerate( axis_files ):
        
    axisname = os.path.splitext( os.path.split( axis_file )[-1] )[0]
    [year, jday] = [int(n) for n in axisname.split('_')]
    name = '%d_%d' % ( year, jday )

    axis = np.load( axis_file )
    
    [x, y, s, theta, Cs, B, Csmooth, b1, b2] = axis[:9]

    BI = np.abs(b1[1:] - b1[:-1]).astype(np.uint8)
    #for i, bi in enumerate( np.unique( b1[np.isfinite(b1)] ) ):
                
    xi, yi, si, ni = x[b1==BEND], y[b1==BEND], s[b1== BEND], b2[b1==BEND]

    Lm = si[-1] - si[0] # Intrinsic Bend Length
    Lc = np.sqrt((xi[0]-xi[-1])**2 + (yi[0]-yi[-1])**2) # Cartesian Bend Length

    # Same Bend at the Next Time - Dict?
    plt.plot(xi,yi,label=name) #+'_%s_%s' % (int(np.unique(ni)), int(np.unique(BEND))))#sinuosity)
    plt.plot(x[1:][BI>0],y[1:][BI>0], 'o')

    print ' - %s' % name
    if (ni == -1).all():
        plt.plot(x,y,'k-')
        break
    BEND = np.unique(ni)
    
plt.legend()
plt.close()
#plt.show()

#sys.exit()


# TODO :: MAKE FUNCTION !!!!
# TODO :: Make Normal/Dimless/Georeferenced converter Class
# TODO :: Class converter (s,n) |--> (Xc,Yc) [Zc]

# Two-Dimensional Structure of Raster Band or Band Ratio
# ======================================================
geodir = os.path.join( sdir, 'geotransf' )
landsat_dirs = sorted([os.path.join(idir, f) for f in os.listdir(idir)])

geo_files = sorted([os.path.join(geodir, f) for f in os.listdir(geodir)])

for i_file, (axis_file, geo_file) in enumerate( zip( axis_files, geo_files ) ):

    # Look for the corresponding Landsat Folder
    year, day = [ int(v) for v in os.path.splitext(os.path.split(axis_file)[-1])[0].split('_') ]
    name = '%s%s' % (year, day)
    for landsat_dir in landsat_dirs:
        lname = os.path.splitext(os.path.split(landsat_dir)[-1])[0][9:16]
        if name == lname: break

    Npts = 100 # Number of Transverse Points
    with open(geo_file) as gf: GeoTransf = pickle.load( gf ) # Load Georeference Transform
    axis = np.load( axis_file ) # Load Properties Table

    # ReLoad Landsat Data
    R, G, B, NIR, MIR, GeoTransf = LoadLandsatData( landsat_dir )
    bands = {
        'R' : R,
        'G' : G,
        'B' : B,
        'NIR' : NIR,
        'MIR' : MIR
    }

    # Load LandsatLook Image and Georeference It
    imgname = './images/LE70010702002311EDC00.tif' # './images/upstream_beni_2005_RGB.tiff' #
    img = gdal.Open( imgname ) # Load it Once for Spatial Reference
    Zi = np.rollaxis(img.ReadAsArray().T,0,2) #imread( imgname ) # Reload it properly to be shown

    GT = {
        'PixelSize' : abs( img.GetGeoTransform()[1] ),
        'X'  : img.GetGeoTransform()[0],
        'Y'  : img.GetGeoTransform()[3],
        'Lx' : Zi.shape[0],
        'Ly' : Zi.shape[1]
    }
    GF1 = GeoReference( Zi[:,:,0], GT )
    Xi, Yi = GF1.RefImage()

    GF0 = GeoReference( R, GeoTransf )
    X0, Y0 = GF0.RefImage()

    R = R.astype( float )
    G = G.astype( float )
    B = B.astype( float )
    NIR = NIR.astype( float )
    MIR = MIR.astype( float )
    NDVI = (NIR-R) / (NIR+R)
    MNDWI = (MIR-G) / (MIR+G)

    #WMASK = np.where()

    BAND = np.log(B)  # np.log((B*G)/(R**2))  #MNDWI  #np.log( G.astype(float) / B.astype(float) )
    BAND = np.where(np.isfinite(BAND), BAND, 0)
    #BAND = np.where(BAND>0.1, BAND, 0)
    bname = r'$\log(\lambda_G/\lambda_R)$'

    # Bande che funzionano (lin)
    # B, R, G, NDVI
    # NIR (solo sopra s.l.), MIR (solo dove c'e' o non c'e' acqua)

    # Bande che funzionano (log)
    # B, G, R, 1/NIR
    # 1/MIR (water or not)

    # SI units river planform
    x = axis[0]
    y = axis[1]
    s = axis[2]
    theta = axis[3]
    Cs = axis[4]
    b = axis[5]
    Csmooth = axis[6]

    for ifilter in xrange(100): Cs[1:-1] = 0.25 * (Cs[:-2] + 2*Cs[1:-1] + Cs[2:]) # Filter Curvature

    # Pixel units river planform    
    X = ( x -  ( GeoTransf['X'] ) ) / GeoTransf['PixelSize']
    Y = ( y -  ( (1-BAND.shape[0])*GeoTransf['PixelSize'] + GeoTransf['Y'] ) ) / GeoTransf['PixelSize']
    S = s / GeoTransf['PixelSize']
    B = b / GeoTransf['PixelSize']

    # Transverse Axis
    N = np.linspace( -1.05, 1.05, Npts, endpoint=True )
    
    # Knots
    Xc, Yc = np.zeros((S.size, N.size)), np.zeros((S.size, N.size))
    dS = np.gradient( S )
    angle = np.arctan2( np.gradient( Y ),  np.gradient( X ) )

    # Create Cartesian Coorinates Array for Intrinsic Coordinate Grid
    for i in xrange( S.size ):
        # TODO : VectorizeMe!
        n = N * B[i] # Pixel Units Transverse Coordinate
        Xc[i,:] = X[i] + n*np.cos( angle[i]-np.pi/2 )
        Yc[i,:] = Y[i] + n*np.sin( angle[i]-np.pi/2 )

    XC = Xc*GeoTransf['PixelSize'] + GeoTransf['X']
    YC = Yc*GeoTransf['PixelSize'] + GeoTransf['Y'] + (1-BAND.shape[0])*GeoTransf['PixelSize']
    Sc, Nc = np.meshgrid( s/b.mean(), N )
    Zc = ndimage.interpolation.map_coordinates( BAND[::-1,:], [Yc, Xc] )


    # Plot Comparison Between Band and RGB Image
    f = plt.figure()
    
    gs = gridspec.GridSpec( 12, 2 )
    ax1 = plt.subplot( gs[:1, :] ) # Colorbar
    ax2 = plt.subplot( gs[2:, 0] ) # Topography
    ax3 = plt.subplot( gs[2:, 1], sharex=ax2, sharey=ax2 ) # RGB
    
    ax2.imshow( G, cmap='binary', extent=[X0.min(), X0.max(), Y0.min(), Y0.max()] )
    pcm = ax2.pcolormesh( XC, YC, Zc, cmap='Spectral_r', alpha=0.75 )
    ax2.contour( XC, YC, Zc )
    plt.plot( XC[:,0], YC[:,0], 'k', lw=2 )
    plt.plot( XC[:,-1], YC[:,-1], 'k', lw=2 )

    # Draw Cross Sections
    for i in xrange(1, s.size-1, 10):
        ax2.plot( XC[i,:], YC[i,:], 'k' )
        ax2.text( XC[i,-1], YC[i,-1], 's=%s' % int(s[i]/b.mean()) )
        ax3.plot( XC[i,:], YC[i,:], 'k' )
        ax3.text( XC[i,-1], YC[i,-1], 's=%s' % int(s[i]/b.mean()) )

    ax3.imshow( Zi, extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()] )
    cb = plt.colorbar( pcm, cax=ax1, orientation='horizontal' )
    cb.set_label( bname )
    plt.axis('equal')


    plt.figure()
    gs = gridspec.GridSpec( 60, 60 )
    ax1 = plt.subplot( gs[7:11, :] ) # Colorbar
    ax2 = plt.subplot( gs[15:25,:] ) # Surface
    ax3 = plt.subplot( gs[30:40, :], sharex=ax2 ) # Width
    ax4 = plt.subplot( gs[48:58, :], sharex=ax2 ) # Curvature

    # Surface
    pcm = ax2.pcolormesh( s/b.mean(), N, Zc.T, cmap='Spectral_r' )
    ax2.contour( Sc, Nc, Zc.T )
    ax2.set_ylabel( r'$n^*/B^*$' )

    # Colorbar
    cb = plt.colorbar( pcm, cax=ax1, orientation='horizontal' )
    
    # Width
    ax3.plot( s/b.mean(), b/b.mean() )
    ax3.set_ylabel( r'$B^*/B_0^*$' )

    # Curvature
    ax4.plot( s/b.mean(), Cs )
    ax4.set_ylabel( r'$\mathcal{C^*}$' )
    ax4.set_xlabel( r'$s^*/B_0^*$' )

    plt.axis('tight')
    plt.show( ) #block=False )
    # Plot Transverse Distribution of BAND
    #splot = float (raw_input( 'Enter a value for dimensionless longitudinal distance s: ' ) )
    splot = 77
    iplot = abs ( s/b.mean() - splot ).argmin()

    plt.figure()
    gs = gridspec.GridSpec( 12, 12 )
    ax1 = plt.subplot( gs[4:9,:] )
    
    Ni = N*b[iplot]; Ni -= Ni[0]
    ax1.plot( Ni, Zc[iplot,:] )
    ax1.set_xlabel( r'$N^*[m]$' )
    ax1.set_ylabel( bname )
    plt.show()
    sys.exit()

    # Wavelet del centro del canale
    print np.log(Zc)[:,int(Npts/2)].shape
    scales = wave.autoscales( N=s.size, dt=s[1]-s[0], dj=0.25, wf='morlet', p=2 )
    cwt = wave.cwt( x=np.log(Zc)[:,int(Npts/2)], dt=s[1]-s[0], scales=scales, wf='morlet', p=2 )
    power = cwt**2

    plt.figure()
    plt.contourf( s, scales, power, cmap=plt.cm.Spectral, levels=np.linspace(power.min(), power.max(), 100) )
    plt.show()
