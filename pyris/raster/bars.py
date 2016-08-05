from __future__ import division
import os, sys
import numpy as np
from scipy import ndimage, interpolate as scipy_interp
from skimage import morphology as mm
from skimage.feature import peak_local_max
from skimage import measure as sm
from matplotlib import pyplot as plt
from matplotlib import gridspec
from ..raster.segmentation import SegmentationIndex
from ..misc.misc import GeoReference, NaNs


class Unwrapper( object ):

    '''
    Unwrap the river channel by performing coordinate transform
    (x,y) --> (s,n)

    Input
    -----
    data        Output for a single river planform
    GeoTransf   Geospatial Coordinate Transform
    '''

    def __init__( self, data, mig, GeoTransf ):
        '''Read the Georeferenced River Planform'''
        self.data = data
        self.GeoTransf = GeoTransf
        self.x = data[0]
        self.y = data[1]
        self.s = data[2]
        self.theta = data[3]
        self.Cs = data[4]
        self.b = data[5]
        self.Bend = mig[4].astype( int )
        self.NextBend = mig[5].astype( int )
        self.Ipoints = ( np.where( data[6].astype( int )==2 )[0] ).astype( int )
        self.BendIndexes = np.unique( self.Bend[ self.Bend>=0 ] )
        return None

    def unwrap( self, shape, Npts=100 ):

        '''
        Returns the Cartesian Coordinates (XC,YC) together with the associated Intrinsic Coordinates (s, n)
        as MeshGrids
        '''
        x, y, s, b, GeoTransf = self.x, self.y, self.s, self.b, self.GeoTransf
        # Pixel units river planform    
        X = ( x -  ( GeoTransf['X'] ) ) / GeoTransf['PixelSize']
        Y = ( y -  ( (1-shape[0])*GeoTransf['PixelSize'] + GeoTransf['Y'] ) ) / GeoTransf['PixelSize']
        S = s / GeoTransf['PixelSize']
        B = b / GeoTransf['PixelSize']

        # Transverse Axis
        N = np.linspace( -1.0, +1.0, Npts, endpoint=True )
        self.N = N
        # Knots
        self.Xc, self.Yc = np.zeros((S.size, N.size)), np.zeros((S.size, N.size))
        angle = np.arctan2( np.gradient( Y ),  np.gradient( X ) )

        # Create Cartesian Coorinates Array for Intrinsic Coordinate Grid
        for i in xrange( S.size ):
            n = N * B[i] # Pixel Units Transverse Coordinate
            self.Xc[i,:] = X[i] + n[:]*np.cos( angle[i]-np.pi/2 )
            self.Yc[i,:] = Y[i] + n[:]*np.sin( angle[i]-np.pi/2 )

        self.I, self.J = np.arange( S.size ), np.arange( N.size )

        self.XC = self.Xc*GeoTransf['PixelSize'] + GeoTransf['X']
        self.YC = self.Yc*GeoTransf['PixelSize'] + GeoTransf['Y'] + (1-shape[0])*GeoTransf['PixelSize']
        self.Sc, self.Nc = np.meshgrid( s/b.mean(), N )

        return [self.XC, self.YC],  [self.Sc, self.Nc]


    def interpolate( self, band ):
        '''Interpolate a band or a bands combination over the grid'''
        return ndimage.interpolation.map_coordinates( band[::-1,:], [self.Yc, self.Xc] )



class BarFinder( object ):

    '''
    Find Bars using in-channel Pixel Classification
    '''


    def __init__( self, unwrapper ):
        '''
        BarFinder(unwrapper) - Read Coordinate Unwrapper
        '''
        self.unwrapper = unwrapper
        self.BendIndexes = self.unwrapper.BendIndexes
        return None


    def FindBars( self, bands, close=True, remove_small=True ):
        '''
        BarFinder.FindBars(bands) - Uses Otsu's global thresholding
                                    with both MNDWI and NDVI indexes
                                    in order to find channel bars.
                                    A convex hull is applied to the
                                    unwrapper bar objects.

        Args
        ====
        bands: dictionary - must contain R, G, B, NIR, MIR bands

        Kwargs
        ======
        close: boolean - perform binary closing on bar mask

        Returns
        =======
        FindBars: Labeled Channel Bars
        '''
        
        Wbands = { } # Water bands
        Vbands = { } # Vegetation bands

        for nband in [ 'R', 'G', 'B', 'NIR', 'MIR', 'SWIR' ]:
            Wbands[nband] = ndimage.interpolation.map_coordinates(
                bands[nband][::-1,:], [self.unwrapper.Yc, self.unwrapper.Xc] )

        Idx, Bars, otsu_glob = SegmentationIndex(
            R=Wbands['R'], G=Wbands['G'], B=Wbands['B'],
            NIR=Wbands['NIR'], MIR=Wbands['MIR'], SWIR=Wbands['SWIR'],
            index='BAR', method='global' ) ## This must be made locally, otherwise we dont see bars eventually


        #plt.figure()
        #plt.imshow( np.dstack((bands['MIR'],bands['NIR'],bands['R'])) )
        #plt.figure()
        #plt.pcolormesh( self.unwrapper.XC, self.unwrapper.YC, Idx, cmap='Spectral' )
        #plt.axis('equal')
        #plt.figure()
        #plt.pcolormesh( self.unwrapper.XC, self.unwrapper.YC, Bars, cmap='Spectral' )
        #plt.axis('equal')
        #plt.show()
        
        # Apply a Convex Hull to Channel Bars
        Bars = mm.convex_hull_object( Bars )

        if close:
            # 1/8 of the total average channel width is used ( 1/4*mean(b) )
            rad = max( 0, 0.25*self.unwrapper.b.mean()/self.unwrapper.GeoTransf['PixelSize'] )
            Bars = mm.binary_closing( Bars, mm.disk(rad) )

        if remove_small:
            Amin = 0.1*self.unwrapper.N.size/2 * ( self.unwrapper.b.mean() /  (self.unwrapper.s[1]-self.unwrapper.s[0]) )
            mm.remove_small_objects( Bars, 2*Amin, in_place=True ) # Remove small Bars
            mm.remove_small_holes(   Bars,   Amin, in_place=True ) # Remove Internal Spots

        # Identify the Largest Bar on the Bend
        labeled_array, num_features = ndimage.measurements.label( Bars )
        self.Bars = labeled_array
        self.BarIdx = np.arange( num_features, dtype=int )+1
        return self.Bars


    def BarCentroid( self ):
        '''Compute Centroid of Each Bar'''

        num_features = self.BarIdx.max() # Number of Bar Features
        IC, JC = np.zeros(num_features,dtype=int), np.zeros(num_features,dtype=int) # Indexes of Bar Centroids
        # Compute Bar Centroids
        for n in xrange( 1,num_features ): [ IC[n-1], JC[n-1] ] = ndimage.measurements.center_of_mass( self.Bars==(n) )
        self.Centroid = np.vstack((IC, JC))
        return IC, JC


    def BarArea( self ):
        ''''Measure the number of Pixels of each Bar'''
        self.Area = np.zeros( self.BarIdx.size, dtype=int )
        for n in self.BarIdx: self.Area[n-1] = np.sum( self.Bars==n )
        return self.Area


    def BarType( self ):
        '''
        Define if a Bar is either Central, Lateral or Mixed

        ---------- - +1     ^ N
        2222222222 _        |
        1111111111   +0.5   |
        0000000000 -  0    -|--------> S
        1111111111 _ -0.5   |
        2222222222          |
        ---------- - -1     |
        '''
        JC = self.Centroid[1]
        NC = self.unwrapper.N[ JC ]
        TYPE = np.ones( NC.size, dtype=int ) # MiXeD
        TYPE[ np.abs(NC)<0.25 ] = 0 # MidChannel
        TYPE[ np.abs(NC)>0.5 ] = 2 # Lateral        
        self.TYPE = TYPE
        self.Central = self.BarIdx[ TYPE==0 ]
        self.Mixed = self.BarIdx[ TYPE==1 ]
        self.Lateral = self.BarIdx[ TYPE==2 ]
        self.TYPES = [ self.Central, self.Mixed, self.Lateral ]
        return TYPE

    def BarBend( self ):
        '''Return the Bar-Bend Index (to which bend the bar belongs)'''
        self.BBIdx = np.full( self.Centroid[0].size, -1, dtype=int )
        for j, c in enumerate( self.Centroid[0] ):
            for i in self.BendIndexes:
                mask = self.unwrapper.Bend==i
                s = self.unwrapper.s[mask]
                if s[0] < self.unwrapper.s[c] <= s[-1]:
                    self.BBIdx[j] = i
                    break
        return self.BBIdx


    def MainBarTypeBend( self, TYPE=None ):
        '''
        Identify the main bar for a specific bar type for each bend
        None : Any Kind of Bar
        0    : Central
        1    : Mixed
        2    : Lateral
        '''
        if TYPE is None: BBIdx = self.BBIdx
        else: BBIdx = self.TYPES[ TYPE ]

        BIdx = np.unique( BBIdx[ BBIdx>=0 ] )
        BarBendTypeIdx = -np.ones( BIdx.size, dtype=int )
        for i, n in enumerate( BIdx ):
            Areas =  np.asarray( [ (self.Bars==(k+1)).sum() for ki, k in enumerate(self.BarIdx[ BBIdx==n ]) ] )
            Indexes = self.BarIdx[ BBIdx==n ]
            BarBendTypeIdx[i] = Indexes[ Areas.argmax() ]
        return BarBendTypeIdx


    def MainBarBend( self ):
        '''Identify the main bar for each bend'''
        self.BarBendIdx = self.MainBarTypeBend()
        return self.BarBendIdx


    def BarContour( self ):
        '''Compute the contour line of each emerging bar'''
        self.Contours = []
        for n in self.BarIdx:
            contour = sm.find_contours( self.Bars==n, 0.5 )
            I, J = contour[0][:,0], contour[0][:,1]

            if np.abs(I[-1]-I[0])>1:
                _I = np.arange( I[-1], I[0], np.sign(I[0]-I[-1]) )
                _J = np.ones( _I.size ) * J[-1]
                I, J = np.append( I, _I ), np.append( J, _J )

            self.Contours.append( [I, J] )
        return self.Contours            


    def BarProps( self ):
        '''Compute Properties of Channel Bars'''
        self.BarCentroid()
        self.BarArea()
        self.BarType()
        self.BarBend()    
        self.BarContour()
        self.MainBarBend()
        return None



    def Show( self, bands ):

        plt.figure()
        gs = gridspec.GridSpec( 12, 2 )
        ax1 = plt.subplot( gs[:1, :] ) # Colorbar
        ax2 = plt.subplot( gs[2:, 0] ) # Topography
        ax3 = plt.subplot( gs[2:, 1], sharex=ax2, sharey=ax2 ) # RGB        
        GR = GeoReference( self.unwrapper.GeoTransf )
        XC, YC = self.unwrapper.XC, self.unwrapper.YC
        Zc = self.unwrapper.interpolate( bands['B'] )
        ax2.imshow( bands['G'], cmap='binary', extent=GR.extent )
        pcm = ax2.pcolormesh( XC, YC, Zc, cmap='Spectral_r', alpha=0.75 )
        ax2.contour( XC, YC, Zc )
        plt.plot( XC[:,0], YC[:,0], 'k', lw=2 )
        plt.plot( XC[:,-1], YC[:,-1], 'k', lw=2 )    
        Zi = np.dstack( (bands['MIR'], bands['NIR'], bands['R']) )
        ax3.imshow( Zi, extent=GR.extent )
        cb = plt.colorbar( pcm, cax=ax1, orientation='horizontal' )
        plt.axis('equal')

        plt.figure()
        gs = gridspec.GridSpec( 12, 2 )
        ax1 = plt.subplot( gs[:1, :] ) # Colorbar
        ax2 = plt.subplot( gs[2:, 0] ) # Topography
        ax3 = plt.subplot( gs[2:, 1], sharex=ax2, sharey=ax2 ) # RGB        
        GR = GeoReference( self.unwrapper.GeoTransf )
        XC, YC = self.unwrapper.XC, self.unwrapper.YC
        Zc = self.unwrapper.interpolate( bands['B'] )
        ax2.imshow( bands['G'], cmap='binary', extent=GR.extent )
        pcm = ax2.pcolormesh( XC, YC, Zc, cmap='Spectral_r', alpha=0.75 )
        ax2.contour( XC, YC, Zc )
        plt.plot( XC[:,0], YC[:,0], 'k', lw=2 )
        plt.plot( XC[:,-1], YC[:,-1], 'k', lw=2 )    
        # Draw Cross Sections
        for i in xrange(1, self.unwrapper.s.size-1, 10):
            ax2.plot( XC[i,:], YC[i,:], 'k' )
            ax2.text( XC[i,-1], YC[i,-1], 's=%s' % int(self.unwrapper.s[i]/self.unwrapper.b.mean()) )
            ax3.plot( XC[i,:], YC[i,:], 'k' )
            ax3.text( XC[i,-1], YC[i,-1], 's=%s' % int(self.unwrapper.s[i]/self.unwrapper.b.mean()) )
        Zi = np.dstack( (bands['MIR'], bands['NIR'], bands['R']) )
        ax3.imshow( Zi, extent=GR.extent )
        cb = plt.colorbar( pcm, cax=ax1, orientation='horizontal' )
        plt.axis('equal')
    
        plt.figure()
        gs = gridspec.GridSpec( 60, 60 )
        ax1 = plt.subplot( gs[7:11, :] ) # Colorbar
        ax2 = plt.subplot( gs[15:25,:] ) # Surface
        ax3 = plt.subplot( gs[30:40, :], sharex=ax2 ) # Width
        ax4 = plt.subplot( gs[48:58, :], sharex=ax2 ) # Curvature    
        # Surface
        pcm = ax2.pcolormesh( self.unwrapper.s/self.unwrapper.b.mean(), self.unwrapper.N, Zc.T, cmap='Spectral_r' )
        ax2.contour( self.unwrapper.Sc, self.unwrapper.Nc, Zc.T )
        ax2.set_ylabel( r'$n^*/B^*$' )    
        # Colorbar
        cb = plt.colorbar( pcm, cax=ax1, orientation='horizontal' )        
        # Width
        ax3.plot( self.unwrapper.s/self.unwrapper.b.mean(), self.unwrapper.b/self.unwrapper.b.mean() )
        ax3.set_ylabel( r'$B^*/B_0^*$' )    
        # Curvature
        ax4.plot( self.unwrapper.s/self.unwrapper.b.mean(), self.unwrapper.Cs )
        ax4.set_ylabel( r'$\mathcal{C^*}$' )
        ax4.set_xlabel( r'$s^*/B_0^*$' )
        plt.axis('tight')

        plt.show()

    
    def __call__( self, bands, close=True, remove_small=True ):
        self.FindBars( bands, close, remove_small )
        self.BarProps()
        return None



class TemporalBars( object ):

    def __init__( self ):
        '''Perform a Temporal Analysis of Channel Bars'''
        self.T = []
        self.Bars = []
        return None

    def GetFinder( self, T, Finder ):
        '''Read a BarFinder Instance with its time and store it'''
        self.T.append( T )
        self.Bars.append( Finder )
        return None

    def IterData( self ):
        '''Bar Iterator'''
        for ti, ( T, Bar ) in enumerate( zip( self.T, self.Bars ) ):
            yield ti, ( T, Bar )

    def BendIndexes( self, ti ):
        '''Return the sequence of Bend Indexes for a given time index'''
        return self.Bars[ti].BendIndexes

    def IterBends( self, ti=0 ):
        '''Iterate over individual meander bends at a given time index ti'''
        for idx in self.BendIndexes( ti ):
            yield idx

    def CentroidsEvol( self, bend_idx, normalize=True ):
        '''Follow the evolution of the centroid of main bar of an individual meander bend'''

        bend = bend_idx

        centroids_IJ = []
        centroids_XY = []
        centroids_SN = []
        
        for ti, ( T, Bars ) in self.IterData():

            if bend == -1: break # CutOff is Reached
            
            # Find the Current Bend
            bend_indexes = Bars.unwrapper.Bend
            bend_indexes_next = Bars.unwrapper.NextBend
            mask = bend_indexes==bend

            # Find the Dominant Bar of the Current Bend
            try: ibar = Bars.BarBendIdx[ bend ]
            except IndexError: continue ## ??? what's wrong here ???
            
            # Get Bar Properties
            IC, JC = Bars.Centroid[ :,ibar ]
            X = Bars.unwrapper.XC[ IC, JC ]
            Y = Bars.unwrapper.YC[ IC, JC ]
            S = Bars.unwrapper.s[ IC ]
            N = Bars.unwrapper.N[ -JC ]

            if normalize:
                Sbend = Bars.unwrapper.s[ mask ]
                SS = S # Bend-Normalized S
                SS -= Sbend[ int( Sbend.size / 2 ) ] # Relative to "Bend Apex" (FIXME: use proper apex)
                SS /= (0.5*( Sbend[-1] - Sbend[0] )) # Normalize to Bend Half-Length

            centroids_IJ.append( [ IC, JC ] )
            centroids_XY.append( [ X, Y ] )
            centroids_SN.append( [ SS, N ] )

            # Get the Bend Index for the Next Time Step
            bend = bend_indexes_next[ mask ][0]

        self.centroids_IJ, self.centroids_XY, self.centroids_SN = centroids_IJ, centroids_XY, centroids_SN

        return centroids_IJ, centroids_SN, centroids_XY


    def MainBarEvol( self, bend_idx, normalize=True ):
        '''Follow the evolution of the main bar of an individual meander bend'''

        bend = bend_idx

        contours_XY = []
        contours_SN = []
        contours_IJ = []
        
        for ti, ( T, Bars ) in self.IterData():

            if bend == -1: break # CutOff is Reached
            
            # Find the Current Bend
            bend_indexes = Bars.unwrapper.Bend
            bend_indexes_next = Bars.unwrapper.NextBend
            mask = bend_indexes==bend

            # Find the Dominant Bar of the Current Bend
            try: ibar = Bars.BarBendIdx[ bend ]
            except IndexError: continue ## ??? what's wrong here ???
            
            # Get Bar Properties
            contour = Bars.Contours[ ibar ]
            X = Bars.unwrapper.XC[ contour[0].astype(int),contour[1].astype(int) ]
            Y = Bars.unwrapper.YC[ contour[0].astype(int),contour[1].astype(int) ]
            S = Bars.unwrapper.s[ contour[0].astype(int) ]
            N = -Bars.unwrapper.N[ contour[1].astype(int) ]

            if normalize:
                Sbend = Bars.unwrapper.s[mask]
                S -= Sbend[ int( Sbend.size / 2 ) ] # Relative to "Bend Apex" (FIXME: use proper apex)
                S /= ( 0.5*(Sbend[-1] - Sbend[0]) ) # Normalize to Bend Half-Length
            
            contours_IJ.append( [ contour[0], contour[1] ] )
            contours_XY.append( [ X, Y ] )
            contours_SN.append( [ S, N ] )

            # Get the Bend Index for the Next Time Step
            bend = bend_indexes_next[ mask ][0]

        self.contours_IJ, self.contours_XY, self.contours_SN = contours_IJ, contours_XY, contours_SN

        return contours_IJ, contours_SN, contours_XY


    def Show( self, landsat_dirs, geodir, bend=None ):

        for BEND in self.IterBends():
            if bend is not None:
                if not BEND==bend: continue

            if hasattr( self, 'contours_IJ' ): [ contours_IJ, contours_SN, contours_XY ] = [ self.contours_IJ, self.contours_SN, self.contours_XY ]
            else: [ contours_IJ, contours_SN, contours_XY ] = self.MainBarEvol( BEND )

            if hasattr( self, 'centroids_IJ' ): [ centroids_IJ, centroids_SN, centroids_XY ] = [ self.centroids_IJ, self.centroids_SN, self.centroids_XY ]
            else: [ centroids_IJ, centroids_SN, centroids_XY ] = self.CentroidsEvol( BEND )
    
            colors = [ plt.cm.Spectral_r(k) for k in np.linspace(0, 1, len(contours_SN)) ]
            lws = np.linspace( 0.5, 2.5, len(contours_SN) )
        
            smin, smax = 0, 0
            xmin, xmax = contours_XY[0][0].min(), contours_XY[0][0].max()
            ymin, ymax = contours_XY[0][1].min(), contours_XY[0][1].max()
            for i, (Csn, Cxy) in enumerate( zip( contours_SN, contours_XY ) ):
                s = np.append( Csn[0], Csn[0][-1] )
                n = np.append( Csn[1], Csn[1][-1] )
                x = np.append( Cxy[0], Cxy[0][-1] )
                y = np.append( Cxy[1], Cxy[1][-1] )
                smin, smax = min( smin, s.min() ), max( smax, s.max() )
                xmin, xmax = min( xmin, x.min() ), max( xmax, x.max() )
                ymin, ymax = min( ymin, y.min() ), max( ymax, y.max() )

            for i, (Csn, Cxy) in enumerate( zip( contours_SN, contours_XY ) ):
                plt.figure()
                gs = gridspec.GridSpec( 60, 60 )
                ax1 = plt.subplot( gs[5:55,:23] ) # SN
                ax2 = plt.subplot( gs[5:55,28:51] ) # XY

                geofile = sorted(os.listdir(geodir))[i]
                name = ''.join((os.path.splitext( os.path.basename( geofile ) )[0].split('_')))
                found = False
                for landsat_dir in landsat_dirs:
                    lname = os.path.splitext(os.path.split(landsat_dir)[-1])[0][9:16]
                    if name == lname:
                        found = True
                        break
                if found == True:
                    from ..misc import LoadLandsatData
                    [ R, G, B, NIR, MIR ], GeoTransf = LoadLandsatData( landsat_dir )
                    bands = { 'R' : R, 'G' : G, 'B' : B, 'NIR' : NIR, 'MIR' : MIR }                    
                    ax2.imshow( np.dstack( (bands['MIR'], bands['NIR'], bands['R']) ), extent=GeoReference(GeoTransf).extent )

                if i>0 and (centroids_SN[i][0]-centroids_SN[i-1][0]) > 4: continue
                for j, (csn, cxy) in enumerate( zip( contours_SN[:i+1], contours_XY[:i+1] ) ):
                    s = np.append( csn[0], csn[0][-1] )
                    n = np.append( csn[1], csn[1][-1] )
                    x = np.append( cxy[0], cxy[0][-1] )
                    y = np.append( cxy[1], cxy[1][-1] )
                    ax1.plot( s, n, label=r'%s' % int(self.T[j]), lw=4, c=colors[j], alpha=0.75 )
                    ax1.fill( s, n, color=colors[j], alpha=0.5 )
                    ax2.plot( x, y, label=r'%s' % int(self.T[j]), lw=4, c=colors[j], alpha=0.75 )
                    ax2.fill( x, y, color=colors[j], alpha=0.5 )
                ax1.set_xlabel( r'$s/B_0\, [-]$' )
                ax1.set_ylabel( r'$n/B_0\, [-]$' )
                ax2.set_xlabel( r'$x_{\mathrm{UTM}} [\mathrm{m}]$' )
                ax2.set_ylabel( r'$y_{\mathrm{UTM}} [\mathrm{m}]$' )
                dx, dy = xmax-xmin, ymax-ymin
                smax = max( abs(smax), abs(smin) )
                smin = -smax
                ds = smax-smin
                ax1.set_xlim( [ smin-0.2*ds, smax+0.2*ds ] )
                ax1.set_ylim( [ -1, 1 ] )
                ax2.set_xlim( [ xmin-dx, xmax+dx ] )
                ax2.set_ylim( [ ymin-dy, ymax+dx ] )
                ax1.axvline( 0, color='gray', lw=2 )
                ax1.text( 0.05, 0.5, 'bend apex', rotation=90 )
                ax1.text( smin-0.1*ds, 0.93, 'flow' )
                ax1.arrow( smin-0.1*ds, 0.9, 0.2*ds, 0 )
                ax2.legend( loc='center left', bbox_to_anchor=(1,0.5), ncol=2 )
                plt.show()


class FreeTemporalBars( TemporalBars ):

    def AccumulateBends( self ):
        
        '''For each bend we follow its history through indices'''

        BendIdx = self.Bars[0].unwrapper.Bend.astype(int)
        Bends = np.unique(BendIdx[BendIdx>=0]).astype(int)
        NextBend = self.Bars[0].unwrapper.NextBend.astype(int)
        self.BendAccumulator = -np.ones( (Bends.size,len(self.T)), dtype=int )
        for iBend, Bend in enumerate( Bends ):
            self.BendAccumulator[iBend,0] = Bend
            b = NextBend[ BendIdx==Bend ][0]
            for iFinder, Finder in enumerate( self.Bars[1:], 1 ):
                if b == -1: break
                self.BendAccumulator[iBend,iFinder] = int(b)
                b = Finder.unwrapper.NextBend[ Finder.unwrapper.Bend==b ][0]
        return self.BendAccumulator
        

    def CorrelateBars( self ):
        '''For each BarIdx(t) compute BarIdx(t+dt)'''

        accumulator = self.AccumulateBends()
        self.BarAccumulator = -np.ones( (self.Bars[0].BarIdx.size,len(self.T)), dtype=int )
        self.BarAccumulator[:,0] = self.Bars[0].BarIdx
        self.BarsCorr = []
        
        for iBars, (BarsL, BarsR, TL, TR) in enumerate( zip( self.Bars[:-1], self.Bars[1:], self.T[:-1], self.T[1:] ) ):

            BarCorr = []
            dT = TR - TL

            for iBarL, BarL in enumerate( BarsL.BarIdx ):

                # Bar Centroid (t)
                IL, JL = BarsL.Centroid[0,iBarL], BarsL.Centroid[1,iBarL]
                XcL, YcL = BarsL.unwrapper.XC[IL,JL], BarsL.unwrapper.YC[IL,JL]
                NL = BarsL.unwrapper.N[JL] # Transverse Coordinate
                SL = BarsL.unwrapper.s[IL] # Longitudinal Coordinate
                BendL = BarsL.BBIdx[iBarL] # Bend to which Bar(t) belongs
                SL_0 = BarsL.unwrapper.s[ BarsL.unwrapper.Bend==BendL ][0] # Coordinate of Upstream Inflection Point
                RL = SL - SL_0 # Relative Longitudinal Coordinate

                # Bar Centroid (t+dt)
                xR, yR = BarsR.unwrapper.XC[BarsR.Centroid[0,:], BarsR.Centroid[1,:]], BarsR.unwrapper.YC[BarsR.Centroid[0,:], BarsR.Centroid[1,:]]
                NR = BarsR.unwrapper.N[ BarsR.Centroid[1,:] ] # Transverse Coordinate
                SR = BarsR.unwrapper.s[ BarsR.Centroid[0,:] ] # Longitudinal Coordinate
                BendR = BarsL.unwrapper.NextBend[BarsL.unwrapper.Bend==BendL][0] # Relative to the same Bend of BarsL
                SR_0 = BarsR.unwrapper.s[ BarsR.unwrapper.Bend==BendR ][0] # Coordinate of Upstream Inflection Point
                RR = SR - SR_0 # Relative Longitudinal Coordinate

                if any(( BendL<0, BendR<0 )):
                    BarCorr.append( [iBarL, IL, JL, XcL, YcL, -1, -1, -1, np.nan, np.nan, np.nan, np.nan, np.nan] )
                    continue


                # Reference System (0)
                Bend0 = self.Bars[0].unwrapper.Bend[ self.Bars[0].unwrapper.Bend==accumulator[accumulator[:,iBars]==BendL,0] ][0]
                S_0 = self.Bars[0].unwrapper.s[ self.Bars[0].unwrapper.Bend==Bend0 ][0]
                L0 = self.Bars[0].unwrapper.s[self.Bars[0].unwrapper.Bend==Bend0][-1] - S_0 # Bends Length (0)

                lL = BarsL.unwrapper.s[BarsL.unwrapper.Bend==BendL][-1] - BarsL.unwrapper.s[BarsL.unwrapper.Bend==BendL][0] # Bend Length (t)

                mask = np.logical_or.reduce(( np.abs(NR-NL)>0.25,
                                              RR<0,
                                              np.abs(RR-RL)>2*BarsL.unwrapper.b.mean()*dT,
                                              np.sqrt( (xR-XcL)**2 + (yR-YcL)**2 )>2*BarsL.unwrapper.b.mean()*dT ))
                RR[ mask ] = np.nan

                try:
                    iBarR = np.nanargmin( np.abs(RR-RL) )
                    IR, JR = BarsR.Centroid[0,iBarR], BarsR.Centroid[1,iBarR]
                    if iBarL>0 and BarCorr[-1][5] == iBarR:
                        if (RR[iBarR]-RL) >= BarCorr[-1][10]:
                            raise ValueError
                        else:
                            BarCorr[-1][5:12] = [-1, -1, -1, np.nan, np.nan, np.nan, np.nan]

                    # Position of BarL with respect to the Initial Planform

                    # Scale on Bend Elongation (if more than one bend is involved, we account for all of them)
                    if SR[iBarR] > BarsR.unwrapper.s[BarsR.unwrapper.Bend==BendR][-1]: ibend = 1 # The bar has moved to the next bend
                    else: ibend = 0
                    LL0 = self.Bars[0].unwrapper.s[self.Bars[0].unwrapper.Bend==(Bend0+ibend)][-1] - self.Bars[0].unwrapper.s[self.Bars[0].unwrapper.Bend==Bend0][0] # Bends Length (0)
                    LL = BarsL.unwrapper.s[BarsL.unwrapper.Bend==(BendL+ibend)][-1] - BarsL.unwrapper.s[BarsL.unwrapper.Bend==BendL][0] # Bends Length (t)
                    LR = BarsR.unwrapper.s[BarsR.unwrapper.Bend==(BendR+ibend)][-1] - BarsR.unwrapper.s[BarsR.unwrapper.Bend==BendR][0] # Bends Length (t+dt)
                    rR = (RR[iBarR]-RL) * LL/LR / BarsL.unwrapper.b.mean()
                    nR = (NR[iBarR]-NL) * LL/LR
                    BarCorr.append( [iBarL, IL, JL, XcL, YcL, iBarR, IR, JR, xR[iBarR], yR[iBarR], rR, nR, RL*L0/lL+S_0] )
                    self.BarAccumulator[self.BarAccumulator[:,iBars]==BarL,iBars+1] = iBarR
                except ValueError:
                    BarCorr.append( [iBarL, IL, JL, XcL, YcL, -1, -1, -1, np.nan, np.nan, np.nan, np.nan, RL*L0/lL+S_0] )
                    continue


                if False: # Centroids Correlation
                    f = plt.figure()
                    ax1 = f.add_subplot( 211 )
                    ax2 = f.add_subplot( 212, sharex=ax1, sharey=ax1 )
                    ax1.set_title('%d' % (iBarL))
                    ax2.set_title('%d' % (iBarR))
                    p1 = ax1.pcolormesh( BarsL.unwrapper.XC, BarsL.unwrapper.YC, BarsL.Bars, cmap='spectral' )
                    ax1.plot( BarsL.unwrapper.XC[IL,JL], BarsL.unwrapper.YC[IL,JL], 'wo', markersize=16 )
                    p2 = ax2.pcolormesh( BarsR.unwrapper.XC, BarsR.unwrapper.YC, BarsR.Bars, cmap='spectral' )
                    ax2.plot( BarsR.unwrapper.XC[IR,JR], BarsR.unwrapper.YC[IR,JR], 'wo', markersize=16 )
                    ax1.set_aspect('equal')
                    ax2.set_aspect('equal')
                    plt.show()

            # Test Plots!!
            if False: # Bars Arrows
                f = plt.figure()
                plt.pcolormesh( BarsL.unwrapper.XC, BarsL.unwrapper.YC, BarsL.Bars, cmap='Spectral', alpha=0.5 )
                for i in xrange( len(BarCorr) ):
                    if BarCorr[i][5]<0: continue
                    [ x0, y0 ] = BarCorr[i][3:5]
                    [ x1, y1 ] = BarCorr[i][8:10]
                    plt.plot( x0, y0, 'yo' )
                    plt.arrow( x0, y0, x1-x0, y1-y0, facecolor='k', edgecolor='k', head_width=50, head_length=50, width=30 )
                plt.axis('equal')
                plt.show()                            

            self.BarsCorr.append( BarCorr )
        return self.BarsCorr

    
    def CentroidsEvol( self, bend_idx, normalize=True ):
        '''Follow the evolution of the centroid of main bar of an individual meander bend'''

        self.CorrelateBars()

        Zs = []
        self.BarMigRate = []

        for iFinder, (T1, T2, Finder, BarCorr) in enumerate( zip( self.T[:-1], self.T[1:], self.Bars[:-1], self.BarsCorr ) ):
            position = Finder.unwrapper.s
            dT = T2 - T1
            s, n = Finder.unwrapper.s, Finder.unwrapper.N
            NMAX = len( BarCorr )
            I, J = np.zeros(NMAX,dtype=int), np.zeros(NMAX,dtype=int)
            dsi, dni, dxi, dyi, dmi, dzi = NaNs(NMAX), NaNs(NMAX), NaNs(NMAX), NaNs(NMAX), NaNs(NMAX), NaNs(NMAX)

            for i in xrange(NMAX):
                I[i] = BarCorr[i][1]
                J[i] = BarCorr[i][2]
                dsi[i] = BarCorr[i][10]
                dni[i] = BarCorr[i][11]
                dzi[i] = np.sqrt( (dsi[i])**2 + (dni[i])**2 )
                dxi[i] = BarCorr[i][8] - BarCorr[i][3]
                dyi[i] = BarCorr[i][9] - BarCorr[i][4]
                dmi[i] = np.sqrt( (dxi[i])**2 + (dyi[i])**2 )

            if iFinder == 0:
                X, Y = Finder.unwrapper.XC, Finder.unwrapper.YC

            si, ni, xi, yi = s[I]/Finder.unwrapper.b.mean(), n[J], Finder.unwrapper.XC[I,J], Finder.unwrapper.YC[I,J]
            zi = dsi
            mask = np.isfinite( zi ) # Mask out NaNs
            
            b0 = Finder.unwrapper.b.mean()

            #S, N = Finder.unwrapper.Sc*Finder.unwrapper.b.mean(), Finder.unwrapper.Nc
            S, N = Finder.unwrapper.Sc, Finder.unwrapper.Nc

            Z = scipy_interp.griddata( (si[mask]*b0, ni[mask]*b0), zi[mask]*b0, (S*b0,N*b0), method='cubic' ).T / b0 / dT
            Zi = scipy_interp.griddata( (si[mask]*b0, ni[mask]*b0), zi[mask]*b0, (S[int(N.shape[0]/2),:]*b0,N[int(N.shape[0]/2),:]*b0), method='cubic' ).T / b0 / dT
            #Z = np.vstack( [Zi for i in xrange(S.shape[0])] ).T
            # We need a regridded version in a General Reference System for the Average
            Zgrid0 = scipy_interp.griddata( (Finder.unwrapper.XC.flatten(), Finder.unwrapper.YC.flatten()), Z.flatten(), (X, Y), method='linear' ) # GRIDDARE CURVA X CURVA!!!!!
            Zs.append( Zgrid0 )
            self.BarMigRate.append( Zi )

            if False: #True:
                plt.figure()
                plt.pcolor( Finder.unwrapper.XC, Finder.unwrapper.YC, np.ma.array(Z,mask=np.isnan(Z)) )
                #plt.pcolor( Finder.unwrapper.XC, Finder.unwrapper.YC, Finder.unwrapper.Sc.T, cmap='YlGn' )
                plt.colorbar()
                plt.contour( Finder.unwrapper.XC, Finder.unwrapper.YC, Finder.Bars, 1, colors='k' )
                plt.contour( self.Bars[iFinder+1].unwrapper.XC, self.Bars[iFinder+1].unwrapper.YC, self.Bars[iFinder+1].Bars, 1, colors='r' )
                for i in xrange( len(BarCorr) ):
                    if BarCorr[i][5]<0: continue
                    [ x0, y0 ] = BarCorr[i][3:5]
                    [ x1, y1 ] = BarCorr[i][8:10]
                    plt.plot( x0, y0, 'ko' )
                    plt.arrow( x0, y0, x1-x0, y1-y0, facecolor='k', edgecolor='k', head_width=80, head_length=150, width=30 )
                plt.axis('equal')
                #plt.show()

            if False: #True:
                plt.figure(figsize=(10.24, 2.56))
                plt.pcolor( Finder.unwrapper.Sc, Finder.unwrapper.Nc, np.ma.array(Z,mask=np.isnan(Z)).T )
                #plt.pcolor( Finder.unwrapper.Sc, Finder.unwrapper.Nc, Finder.unwrapper.Sc )
                plt.colorbar()
                plt.contour( Finder.unwrapper.Sc, Finder.unwrapper.Nc, Finder.Bars.T, 1, colors='r' )
                #plt.contour( self.Bars[iFinder+1].unwrapper.Sc, self.Bars[iFinder+1].unwrapper.Nc, self.Bars[iFinder+1].Bars.T, 1, colors='r' )
                for i in xrange( len(BarCorr) ):
                    if BarCorr[i][5]<0: continue
                    [ s0, n0 ] = si[i], ni[i]
                    [ ds, dn ] = dsi[i], dni[i]
                    plt.plot( s0, n0, 'ko' )
                    plt.arrow( s0, n0, ds, dn, facecolor='k', edgecolor='k' )
                plt.axis('tight')
                plt.show()

        if True:
            plt.figure(figsize=(10.24, 2.56))
            #plt.pcolor( Finder.unwrapper.Sc, Finder.unwrapper.Nc, np.ma.array(Z,mask=np.isnan(Z)).T )
            plt.pcolor( Finder.unwrapper.Sc, Finder.unwrapper.Nc, Finder.unwrapper.Sc )
            plt.colorbar()
            plt.contour( Finder.unwrapper.Sc, Finder.unwrapper.Nc, Finder.Bars.T, 1, colors='r' )
            #plt.contour( self.Bars[iFinder+1].unwrapper.Sc, self.Bars[iFinder+1].unwrapper.Nc, self.Bars[iFinder+1].Bars.T, 1, colors='r' )
            for (BarCorr, Finder) in zip(self.BarsCorr,self.Bars[:-1]):
                for i in xrange( len(BarCorr) ):
                    if BarCorr[i][5]<0: continue
                    #BarCorr.append( [iBarL, IL, JL, XcL, YcL, iBarR, IR, JR, xR[iBarR], yR[iBarR], rR, nR, RL*L0/lL+S_0] )
                    [ s0, n0 ] = BarCorr[i][12]/Finder.unwrapper.b.mean(), Finder.unwrapper.N[BarCorr[i][2]]
                    [ ds, dn ] = BarCorr[i][10], BarCorr[i][11]
                    plt.plot( s0, n0, 'ko' )
                    plt.arrow( s0, n0, ds, dn, facecolor='k', edgecolor='k' )
            plt.axis('tight')
            plt.show()


        self.BarMigRate.append( NaNs( Finder.unwrapper.s.size ) )
        Z = np.nanmean( np.dstack(Zs), axis=2 )

        if True: #False:
            plt.figure()
            plt.title( 'Average annual longitudinal migration rate of channel bars (%d-%d)' % (int(self.T[0]), int(self.T[-1])) )
            plt.pcolormesh( X, Y, np.ma.array(Z,mask=np.isnan(Z)), cmap='RdYlBu_r', vmin=-1, vmax=3 )
            plt.xlabel( r'$x$' )
            plt.xlabel( r'$y$' )
            plt.colorbar()
            plt.axis( 'equal' )
            plt.show()

        return None


    def AverageBarMigRate( self ):

        '''Compute the Average Migration Rate of Channel Bars over the years'''

        s = self.Bars[0].unwrapper.s
        BendIdx = self.Bars[0].unwrapper.Bend
        Bends = np.unique( BendIdx[BendIdx>=0] )
        AMR = NaNs( (BendIdx.size,len(self.Bars)-1) )
        Cs_vals = []
        MR_vals = []

        for i,Bend in enumerate(Bends):

            Sbend = s[BendIdx==Bend]
            N = Sbend.size

            iBend = Bend

            for iFinder,(Finder,MigRate) in enumerate( zip(self.Bars[:-1],self.BarMigRate[:-1]) ):

                if iBend == -1: break

                mask = (Finder.unwrapper.Bend==iBend)
                n = mask.sum()

                # Interpolate the Bend's Migration Rate over the Initial Points
                f = scipy_interp.interp1d( np.linspace(0,1,n), MigRate[mask], kind='cubic' )
                AMR[BendIdx==Bend,iFinder] = f( np.linspace(0,1,N) )
                
                # Bend Index (t+dt)
                iBend = Finder.unwrapper.NextBend[ Finder.unwrapper.Bend==iBend ][0]

                Cs_vals = Cs_vals + np.abs( Finder.unwrapper.Cs[mask] ).tolist()
                MR_vals = MR_vals + MigRate[mask].tolist()

        AveMigRate = np.nanmean( AMR, axis=1 )
        aCs = np.abs( self.Bars[0].unwrapper.Cs )

        unwrapper = self.Bars[0].unwrapper
        x, y, s = unwrapper.x, unwrapper.y, unwrapper.s
        db = np.ediff1d(BendIdx, to_begin=0)
        idx = np.where(db>0)[0]

        colors = [plt.cm.jet(xx) for xx in np.linspace(0,1,AMR.shape[1])]
        lws = np.linspace(0.5,2.5,AMR.shape[1])

        f1 = plt.figure()
        plt.plot( x, y, 'b' )
        plt.plot( x[idx], y[idx], 'o', c='r' )
        for j in xrange(idx.size):
            plt.text( x[idx[j]], y[idx[j]], str(int(BendIdx[idx[j]])) )
        plt.axis( 'equal' )
        plt.legend( loc='best' )

        f2 = plt.figure()
        ax1 = f2.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.plot( s, AveMigRate, '-b' )
        ax2.plot( s, aCs/aCs.max(), 'r--' )
        for i in idx:
            ax2.axvline(s[i], color='gray')
            ax2.text(s[i],0.9,'%d' % BendIdx[i])

        f3 = plt.figure()
        for j in xrange(AMR.shape[1]):
            plt.plot( s, AMR[:,j], c=colors[j], lw=lws[j], label='%d' % j )
        plt.legend()

        f4 = plt.figure()
        plt.scatter( Cs_vals, MR_vals )
        plt.show()

        return AveMigRate
