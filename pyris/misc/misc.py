from __future__ import division
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.io import imread
import gdal
import warnings

mpl.rc( 'font', size=20 )
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{siunitx}',
    r'\siteup{detect-all}',
    r'\usepackage{helvet}',
    r'\usepackage{sansmath}',
    r'\usepackage{amsmath}',
    r'\sansmath'
]
mpl.rcParams['axes.formatter.limits'] = [-3,3]
plt.close( 'all' )

# Classes
# =======
class Line2D( object ): # This must be put down better!
    
    def __init__( self, x=np.array([]), y=np.array([]), B=np.array([]) ):
        dx = ediff1d0( x )
        dy = ediff1d0( y )
        ds = np.sqrt( dx**2 + dy**2 )
        s = np.cumsum( ds )
        try: L = s[-1]
        except IndexError: L = 0
        self.x, self.y, self.B, self.s, self.L = x, y, B, s, L
        return None

    def join( self, line2d ):
        if len(self.x) == 0:
            ds = 0
        else:
            ds = np.sqrt( (self.x[-1]-line2d.x[-1])**2 + (self.y[-1]-line2d.y[-1])**2 )
        self.x = np.concatenate( (self.x, line2d.x) )
        self.y = np.concatenate( (self.y, line2d.y) )
        self.B = np.concatenate( (self.B, line2d.B) )
        self.s = np.concatenate( (self.s, line2d.s+ds) )
        self.L = self.s[-1]
        return None

def ediff1d0( x ):
    if len( x ) == 0:
        return np.ediff1d( x )
    return np.ediff1d( x, to_begin=0 )


class GeoReference( object ):
    '''Provide Georeferenced Coordinates for an Image Object'''

    def __init__( self, GeoTransf ):
        self.GeoTransf = GeoTransf
        self.extent = [ GeoTransf['X'], # xmin
                        GeoTransf['X'] + GeoTransf['PixelSize']*GeoTransf['Lx'], # xmax
                        GeoTransf['Y'] - GeoTransf['PixelSize']*GeoTransf['Ly'], # ymin
                        GeoTransf['Y'] # ymax
                    ]
        return None

    def RefCurve( self, X, Y, inverse=False ):
        X, Y = np.asarray(X), np.asarray(Y)
        if inverse:
            Cx = ( X - self.extent[0] ) / self.GeoTransf['PixelSize']
            Cy = -( Y - self.extent[3] ) / self.GeoTransf['PixelSize']
            return Cx, Cy
        else:
            self.Cx, self.Cy = X, Y
            self.CX = self.extent[0] + self.Cx*self.GeoTransf['PixelSize']
            self.CY = self.extent[3] - self.Cy*self.GeoTransf['PixelSize']
            return self.CX, self.CY


class interactive_mask( object ):

    def __init__( self, path ):
        self.path = os.path.normpath( path )
        self.name =  self.path.split( os.sep )[-1]

    def build_real_color( self ):
        if self.name.startswith( 'LC8' ):
            warnings.warn( 'Landsat 8 may return distorted images as real color.', Warning )
            b1, b2, b3 = 'B6', 'B5', 'B4'
        else:
            b1, b2, b3 = 'B5', 'B4', 'B3'
        B1, B2, B3 = map( imread, [ os.path.join( self.path, '_'.join((self.name,bname))+'.TIF' ) for bname in [ b1, b2, b3 ] ] )
        return np.dstack( ( B1, B2, B3 ) )

    def _set_mask( self ):
        real_color = self.build_real_color()
        white_masks = []
        plt.ioff()
        fig = plt.figure()
        plt.title( 'Press-drag a rectangle for your mask. Close when you are finish.' )
        plt.imshow( real_color, cmap='binary_r' )
        plt.axis('equal')
        x_press = None
        y_press = None
        def onpress(event):
            global x_press, y_press
            x_press = int(event.xdata) if (event.xdata != None) else None
            y_press = int(event.ydata) if (event.ydata != None) else None
        def onrelease(event):
            global x_press, y_press
            x_release = int(event.xdata) if (event.xdata != None) else None
            y_release = int(event.ydata) if (event.ydata != None) else None
            if (x_press != None and y_press != None
                and x_release != None and y_release != None):
                (xs, xe) = (x_press, x_release+1) if (x_press <= x_release) \
                  else (x_release, x_press+1)
                (ys, ye) = (y_press, y_release+1) if (y_press <= y_release) \
                  else (y_release, y_press+1)
                print( "The mask you selected is [{0}:{1},{2}:{3}]".format(
                    xs, xe, ys, ye) )
                white_masks.append( [ ys, ye, xs, xe ] )
                plt.fill( [xs,xe,xe,xs,xs], [ys,ys,ye,ye,ys], 'r', alpha=0.25 )
                event.canvas.draw()
            x_press = None
            y_press = None
        cid_press   = fig.canvas.mpl_connect('button_press_event'  , onpress  )
        cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)
        plt.show()
        return white_masks

    def get_georef( self ):
        bands, GeoTransf = LoadLandsatData( self.path )
        return GeoReference( GeoTransf )

    def _georeference_masks( self, masks, inverse=False ):
        GRF = self.get_georef()
        gmask = []
        for mask in masks:
            [ys, ye, xs, xe] = mask
            X, Y = GRF.RefCurve( np.array([xs,xe]), np.array([ys,ye]), inverse=inverse )
            gmask.append( [ Y[0], Y[1], X[0], X[1] ] )
        return gmask
    
    def georeference( self, masks ):
        gmasks = self._georeference_masks( masks )
        return gmasks

    def dereference( self, gmasks ):
        masks = self._georeference_masks( gmasks, inverse=True )
        return masks

    def __call__( self, *args, **kwargs ):
        inverse = kwargs.pop( 'inverse', False )
        if inverse:
            gmasks = list( sys.argv[1] )
            return self.dereference( gmasks )
        masks = self._set_mask()
        return self.georeference( masks )


def LoadLandsatData( dirname ):
    '''Load Relevant Bands for the Current Landsat Data'''
    if os.path.split(dirname)[-1].startswith('LC8'): bidx = range( 2, 7 )
    else: bidx = range( 1, 6 )
    base = os.path.join( dirname, os.path.basename(dirname) )
    ext = '.TIF'
    bnames = [ ('_B'.join(( base, '%d' % i )))+ext for i in bidx ]
    [ B, G, R, NIR, MIR ] = [ imread( band ) for band in bnames ]
    bands = [ R, G, B, NIR, MIR ]
    geo = gdal.Open( bnames[0] )
    GeoTransf = {    
        'PixelSize' : abs( geo.GetGeoTransform()[1] ),
        'X' : geo.GetGeoTransform()[0],
        'Y' : geo.GetGeoTransform()[3],
        'Lx' : bands[0].shape[1],
        'Ly' : bands[0].shape[0]
        }
    return bands, GeoTransf


class MaskedShow( object ):

    def __init__( self, data ):
        self.data = data

    def get_mask(self, mask):
        self.mask = mask

    def show_masked( self ):
        plt.figure()
        m = plt.imshow( np.ma.masked_where( self.mask, self.data ), cmap=plt.cm.spectral )
        n = plt.imshow( np.ma.masked_where( 1-self.mask, self.data ), cmap=plt.cm.gray )
        plt.colorbar( m )
        plt.show()


class BW( object ):
    '''
    Interactive Black and White Image.
    Allows Interaction with figures in order to modify the image array itself
    by means of event handlings
    '''
    
    def __init__( self, img ):
        self.bw = img.astype( np.uint8 )

    def RemovePoints( self, rm=0 ):
        '''Remove Points by selection'''
        plt.ioff()
        fig = plt.figure()
        if rm==0: plt.title('Click on Pixels you want to remove')
        elif rm==1: plt.title('Click on Pixels you want to add')
        cm = plt.pcolormesh( self.bw, vmin=0, vmax=1, cmap='binary_r' )
        plt.axis('equal')
        def onclick( event ):
            indexx = int(event.xdata)
            indexy = int(event.ydata)
            print("Index ({0},{1}) will be set to {2}".format(
                    indexx, indexy, rm) )
            self.bw[indexy, indexx] = rm
            cm.set_array( self.bw.ravel() )
            event.canvas.draw()
        cid = fig.canvas.mpl_connect( 'button_press_event', onclick )
        plt.show()
        
    def RemoveRectangle( self, rm=0 ):
        '''Remove an Entire Rectangle from bw figure'''
        plt.ioff()
        fig = plt.figure()
        if rm==0: plt.title('Select rectangle you want to remove')
        elif rm==1: plt.title('Select rectangle you want to add')
        cm = plt.pcolormesh( self.bw, vmin=0, vmax=1, cmap='binary_r' )
        plt.axis('equal')
        x_press = None
        y_press = None
        def onpress(event):
            global x_press, y_press
            x_press = int(event.xdata) if (event.xdata != None) else None
            y_press = int(event.ydata) if (event.ydata != None) else None
        def onrelease(event):
            global x_press, y_press
            x_release = int(event.xdata) if (event.xdata != None) else None
            y_release = int(event.ydata) if (event.ydata != None) else None
            if (x_press != None and y_press != None
                and x_release != None and y_release != None):
                (xs, xe) = (x_press, x_release+1) if (x_press <= x_release) \
                  else (x_release, x_press+1)
                (ys, ye) = (y_press, y_release+1) if (y_press <= y_release) \
                  else (y_release, y_press+1)
                print( "Slice [{0}:{1},{2}:{3}] will be set to {4}".format(
                    xs, xe, ys, ye, rm) )
                self.bw[ys:ye, xs:xe] = rm
                cm.set_array( self.bw.ravel() )
                plt.fill( [xs,xe,xe,xs,xs], [ys,ys,ye,ye,ys], 'r', alpha=0.25 )
                event.canvas.draw()
            x_press = None
            y_press = None
        cid_press   = fig.canvas.mpl_connect( 'button_press_event'  , onpress   )
        cid_release = fig.canvas.mpl_connect( 'button_release_event', onrelease )
        plt.show()

    def AddPoints( self ):
        '''Add Points by selection'''
        return self.RemovePoints( rm=1 )

    def AddRectangle( self ):
        '''Remove an Entire Rectangle from bw figure'''
        return self.RemoveRectangle( rm=1 )

class MaskClean( object ):

    '''
    Interactive Black and White Image.
    Allows Interaction with figures in order to modify the image array itself
    by means of event handlings
    '''
    
    def __init__( self, bw, bg=None ):
        self.bw = bw.astype( int )
        self.bg = np.zeros(bw.shape) if bg is None else bg
        
    def __call__( self ):
        #real_color = self.build_real_color()
        white_masks = []
        plt.ioff()
        fig = plt.figure()
        plt.title( 'Press-drag a rectangle for your mask. Close when you are finish.' )
        plt.imshow( self.bg, cmap='binary_r' )
        plt.imshow( self.bw, cmap='jet', alpha=0.5 )
        plt.axis('equal')
        x_press = None
        y_press = None
        def onpress(event):
            global x_press, y_press
            x_press = int(event.xdata) if (event.xdata != None) else None
            y_press = int(event.ydata) if (event.ydata != None) else None
        def onrelease(event):
            global x_press, y_press
            x_release = int(event.xdata) if (event.xdata != None) else None
            y_release = int(event.ydata) if (event.ydata != None) else None
            if (x_press != None and y_press != None
                and x_release != None and y_release != None):
                (xs, xe) = (x_press, x_release+1) if (x_press <= x_release) \
                  else (x_release, x_press+1)
                (ys, ye) = (y_press, y_release+1) if (y_press <= y_release) \
                  else (y_release, y_press+1)
                print( "Slice [{0}:{1},{2}:{3}] will be set to {4}".format(
                    xs, xe, ys, ye, 0) )
                self.bw[ ys:ye,xs:xe ] = 0
                plt.fill( [xs,xe,xe,xs,xs], [ys,ys,ye,ye,ys], 'r', alpha=0.25 )
                event.canvas.draw()
            x_press = None
            y_press = None
        cid_press   = fig.canvas.mpl_connect('button_press_event'  , onpress  )
        cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)
        plt.show()
        return self.bw


def ShowRasterData( data, label='', title='' ):
    '''Return a GridSpec Instance with Raster imshow,
    colorbar and histogram of its values'''

    # Set figure and axes
    w, h = plt.figaspect(0.5)
    fig = plt.figure( figsize=(w,h) )
    gs = GridSpec(100,100,bottom=0.18,left=0.08,right=0.98)
    ax1 = fig.add_subplot( gs[:,:50] ) # Left
    ax2 = fig.add_subplot( gs[:10,55:] ) # Upper Right
    ax3 = fig.add_subplot( gs[40:,55:] ) # Lower Right    
    # Plot Data
    cs = ax1.imshow( data )
    if title != '': ax1.set_title( r'%s' % title )
    fig.colorbar(cs, label=(r'%s values' % label).strip(), ax=ax1, cax=ax2, orientation='horizontal')
    # Values Histogram
    x = np.linspace(np.nanmin(data[np.isfinite(data)]), np.nanmax(data[np.isfinite(data)]), 1000)
    hist, bins = np.histogram( data.flatten(), bins=x, normed=True )
    bins = 0.5*(bins[1:]+bins[:-1])
    y = np.linspace(0, 2*hist.max(), x.size)
    x[0], x[-1] = x[1], x[-2] # Apply a Manual Fix
    X, Y = np.meshgrid( x, y )
    Z = X
    ax3.pcolor( X, Y, Z )
    ax3.fill_between( bins, y.max(),
                      hist, color='w' )
    ax3.set_xlim([x.min(), x.max()])
    ax3.set_ylim([0, 1.2*hist.max()])
    ax3.set_xlabel( (r'%s values' % label).strip() )
    ax3.set_ylabel( r'frequency' )
    return fig



def NaNs( N ):
    '''Create a 1d array of NaNs of size N'''
    return np.full( N, np.nan )

def crossprod2( v1, v2 ):
    '''3rd component of 2d vectors cross product'''
    return v1[0]*v2[1] - v1[1]*v2[0]

def Intersection( P, Q, R, S, return_point=True ):
    '''Check if two segments PQ and RS intersect and where'''
    QP = Q - P
    CRS = crossprod2( R, S ) 
    t = ( QP[0]*S[1] - S[0]*QP[1] ) / CRS
    u = ( QP[0]*R[1] - R[0]*QP[1] ) / CRS
    if ( abs(CRS) > 0 and 0 <= abs(t) <= 1 and 0 <= abs(u) <= 1 ):
        # Segments Intersect!
        if return_point: return True, P + t*R
        else: return True
    if return_point: return False, NaNs(2)
    else: return False

def PolygonCentroid( x, y, return_area=False ):
    if not np.allclose( [x[0], y[0]], [x[-1], y[-1]] ):
        x = np.append( x, x[0] )
        y = np.append( y, y[0] )
    a = x[:-1] * y[1:]
    b = x[1:] * y[:-1]
    A = 0.5 * (a - b).sum()    
    cx = x[:-1] + x[1:]
    cy = y[:-1] + y[1:]
    Xc = np.sum( cx*(a-b) ) / (6*A)
    Yc = np.sum( cy*(a-b) ) / (6*A)
    X = np.array([Xc, Yc])
    if return_area:
        return X, A
    return X
