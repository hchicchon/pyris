from __future__ import division
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import gdal

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
class Line2D( object ):
    # FIX : use decorators
    d = {} # Dictionary
    
    def __init__( self, line ):
        self.x = line[0]
        self.y = line[1]
        self.d['x'] = self.x
        self.d['y'] = self.y
                
    def attr( self, key, val ):
        self.d[key] = val


class GeoReference( object ):
    '''Provide Georeferenced Coordinates for an Image Object'''

    def __init__( self, I, GeoTransf ):
        self.I = I
        self.GeoTransf = GeoTransf

    def RefImage( self ):
        XX = np.arange( 1, self.I.T.shape[0]+1 ) * self.GeoTransf['PixelSize'] + self.GeoTransf['X']
        YY = (np.arange(1, self.I.T.shape[1]+1 )-self.I.T.shape[1]) * self.GeoTransf['PixelSize'] + self.GeoTransf['Y']
        return XX, YY

    def RefCurve( self, X, Y ):
        XX, YY = self.RefImage()
        return X*self.GeoTransf['PixelSize']+XX[0], Y*self.GeoTransf['PixelSize']+YY[0]

def get_dt( name1, name2 ):
    '''Return Time Interval in Years'''
    t1 = os.path.splitext( os.path.split( name1 )[-1] )[0]
    t2 = os.path.splitext( os.path.split( name2 )[-1] )[0]
    [ y1, d1 ] = [ int( i.strip() ) for i in t1.split('_') ]
    [ y2, d2 ] = [ int( i.strip() ) for i in t2.split('_') ]
    T1 = y1 + d1/365
    T2 = y2 + d2/365
    return T2 - T1

def outliers( data, m=3 ):
    d = np.abs( data - np.median(data) )
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return s>m

def find_nearest( array, value ):
    '''Find closest element to value inside array'''
    return array.flat[ np.abs( array - value ).argmin() ]

def argfind_nearest( array, value ):
    '''Find closest element to value inside array'''
    return np.abs( array - value ).argmin()

def smooth( y, freq=200 ):
    '''Lowpass filter (Butterworth)'''
    b, a = signal.butter( 4, 5/(freq/2), btype='low' )
    return signal.filtfilt( b, a, y )

def smooth2( y, fs, cutoff, order=5 ):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter( order, normal_cutoff, btype='low' )
    return signal.filtfilt( b, a, y )

def NaNs( N ):
    '''Create a 1d array of NaNs of size N'''
    return np.full( N, np.nan )

def crossprod2( v1, v2 ):
    '''3rd component of 2d vectors cross product'''
    return v1[0]*v2[1] - v1[1]*v2[0]

def Intersection( P, Q, R, S ):
    '''Check if two segments PQ and RS intersect and where'''
    QP = Q - P
    CRS = crossprod2( R, S ) 
    t = ( QP[0]*S[1] - S[0]*QP[1] ) / CRS
    u = ( QP[0]*R[1] - R[0]*QP[1] ) / CRS
    if ( abs(CRS) > 0 and 0 <= abs(t) <= 1 and 0 <= abs(u) <= 1 ):
        # Segments Intersect!
        return True, P + t*R
    return False, NaNs(2)

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

def PlotWavelet(time, sig, icwt, cwt, period):
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    plt.figure()
    gs = gridspec.GridSpec( 100, 100 )
    power = (np.abs(cwt)) ** 2  # compute wavelet power spectrum
    global_ws = np.mean(power, axis=1) * sig.var()/sig.size
    smax = period[ global_ws.argmax() ]    
    ax1 = plt.subplot( gs[:30,    :70] ) # Signal
    ax2 = plt.subplot( gs[35:85,  :70], sharex=ax1 ) # CWT Power
    ax3 = plt.subplot( gs[35:85:, 75:], sharey=ax2 ) # GWS
    ax4 = plt.subplot( gs[95:,    :70]  ) # cbar    
    ax1.plot( time, sig, 'k', label='original' )
    ax1.plot( time, icwt, 'r', label='filtered', lw=2 )
    ax1.set_ylabel( r'$\mathcal{C}(s)$' )
    ax1.legend( loc='best' )
    ax3.plot(global_ws, period)
    CF = ax2.contourf( time, period, np.log2(power), 50, cmap=plt.cm.Spectral )
    ax2.axhline( y=smax, c='gray', lw=2, linestyle='--' )
    ax2.set_ylabel(r'Period $[km]$')
    ax2.set_xlabel(r'$s [km]$')
    ax2.set_yscale('log', basey=2, subsy=None)
    ax2.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax2.ticklabel_format(axis='y', style='plain')
    ax2.invert_yaxis()
    ax3.hold(True)
    ax3.set_xlabel('GWS')
    ax3.set_yscale('log', basey=2, subsy=None)
    ax3.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax3.ticklabel_format(axis='y', style='plain')
    cbar = plt.colorbar( CF, orientation='horizontal', cax=ax4 )
    cbar.set_label( r'$\log_2(\mathrm{Power})$' )
    xx = ax2.get_yticks()
    ll = [ '%.1f' % a for a in xx ]
    ax2.set_yticks( xx, ll )
    xx = ax3.get_yticks()
    ll = [ '%.1f' % a for a in xx ]
    plt.setp( ax1.get_xticklabels(), visible=False )
    plt.setp( ax3.get_yticklabels(), visible=False )
    plt.axis('tight')
    plt.show()

def LoadLandsatData( dirname ):
    '''Load Relevant Bands for the Current Landsat Data'''
    if os.path.split(dirname)[-1].startswith('LC8'):
        Bawei = gdal.Open( os.path.join(dirname, '%s_B7.TIF' % os.path.split(dirname)[-1]) )
        MIR   = gdal.Open( os.path.join(dirname, '%s_B6.TIF' % os.path.split(dirname)[-1]) )
        NIR   = gdal.Open( os.path.join(dirname, '%s_B5.TIF' % os.path.split(dirname)[-1]) )
        R     = gdal.Open( os.path.join(dirname, '%s_B4.TIF' % os.path.split(dirname)[-1]) )
        G     = gdal.Open( os.path.join(dirname, '%s_B3.TIF' % os.path.split(dirname)[-1]) )
        B     = gdal.Open( os.path.join(dirname, '%s_B2.TIF' % os.path.split(dirname)[-1]) )
    # Landsat 7 (TM), 4-5 (TM) and 1-5 (MSS)
    else:
        Bawei = gdal.Open( os.path.join(dirname, '%s_B6.TIF' % os.path.split(dirname)[-1]) )
        MIR   = gdal.Open( os.path.join(dirname, '%s_B5.TIF' % os.path.split(dirname)[-1]) )
        NIR   = gdal.Open( os.path.join(dirname, '%s_B4.TIF' % os.path.split(dirname)[-1]) )
        R     = gdal.Open( os.path.join(dirname, '%s_B3.TIF' % os.path.split(dirname)[-1]) )
        G     = gdal.Open( os.path.join(dirname, '%s_B2.TIF' % os.path.split(dirname)[-1]) )
        B     = gdal.Open( os.path.join(dirname, '%s_B1.TIF' % os.path.split(dirname)[-1]) )
    # Get Georeference Data
    bands = [ R.ReadAsArray(), G.ReadAsArray(), B.ReadAsArray(), NIR.ReadAsArray(), MIR.ReadAsArray() ]
    GeoTransf = {    
        'PixelSize' : abs( R.GetGeoTransform()[1] ),
        'X' : R.GetGeoTransform()[0],
        'Y' : R.GetGeoTransform()[3],
        'Lx' : mask.bw.shape[0], # TODO : VERIIFICARE (se sbagliato c'e' da scommentare l'overraid in SARA)
        'Ly' : mask.bw.shape[1] # TODO : VERIIFICARE (se sbagliato c'e' da scommentare l'overraid in SARA)
        }
    return bands, GeoTransf


def isRaster( img ):
    if isinstance( img, np.ndarray ) :
        if img.ndim == 2:
            return True
    return False


def isRGB( img ):
    if isinstance( img, np.ndarray ) :
        if img.ndim == 3 and img.shape[2] == 3:
            return True
    return False


def isBW( img ):
    if isRaster( img ) and np.allclose( [0.,1.], np.unique( img ) ):
        return True
    return False


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
                event.canvas.draw()
            x_press = None
            y_press = None
        cid_press   = fig.canvas.mpl_connect('button_press_event'  , onpress  )
        cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)
        plt.show()

    def AddPoints( self ):
        '''Add Points by selection'''
        return self.RemovePoints( rm=1 )

    def AddRectangle( self ):
        '''Remove an Entire Rectangle from bw figure'''
        return self.RemoveRectangle( rm=1 )


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

