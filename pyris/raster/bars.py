from __future__ import division
import os, sys
import numpy as np
from scipy import ndimage, interpolate as scipy_interp
from skimage import morphology as mm
from skimage.feature import peak_local_max
from skimage import measure as sm
from matplotlib import pyplot as plt
from ..raster.segmentation import SegmentationIndex
from ..misc.misc import GeoReference, NaNs


class Unwrapper( object ):

    '''
    Unwrap the river channel by performing coordinate transform
    (x,y) --> (s,n)

    Input
    -----
    data        SARA.py-like output for a single river planform
    GeoTransf   SARA.py-like Geospatial Coordinate Transform
    '''

    def __init__( self, data, GeoTransf ):
        '''Read the Georeferenced River Planform'''
        self.data = data
        self.GeoTransf = GeoTransf
        self.x = data[0]
        self.y = data[1]
        self.s = data[2]
        self.theta = data[3]
        self.Cs = data[4]
        self.b = data[5]
        self.Bend = data[8].astype( int )
        self.NextBend = data[9].astype( int )
        self.Ipoints = ( np.where( data[10].astype( int )==2 )[0] ).astype( int )
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
        N = np.linspace( +1.0, -1.0, Npts, endpoint=True )
        self.N = N
        # Knots
        self.Xc, self.Yc = np.zeros((S.size, N.size)), np.zeros((S.size, N.size))
        dS = np.gradient( S )
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

        for nband in [ 'R', 'G', 'B', 'NIR', 'MIR', 'Bawei' ]:
            Wbands[nband] = ndimage.interpolation.map_coordinates(
                bands[nband][::-1,:], [self.unwrapper.Yc, self.unwrapper.Xc] )

        # Segmentation - Define Water
        Idx, isWater, otsu_glob = SegmentationIndex(
            R=Wbands['R'], G=Wbands['G'], B=Wbands['B'],
            NIR=Wbands['NIR'], MIR=Wbands['MIR'], Bawei=Wbands['Bawei'],
            index='MNDWI', method='global' )
    
        # Segmentation - Define Vegetation
        for nband in [ 'R', 'G', 'B', 'NIR', 'MIR', 'Bawei' ]:
            Vbands[nband] = np.where( ~isWater.bw, Wbands[nband], np.nan )

        Idx, isNotVeg, otsu_glob = SegmentationIndex(
            R=Vbands['R'], G=Vbands['G'], B=Vbands['B'],
            NIR=Vbands['NIR'], MIR=Vbands['MIR'], Bawei=Vbands['Bawei'],
            index='NDVI', method='local' ) ## This must be made locally, otherwise we dont see bars eventually

        # Channel - ( Water + Vegetation ) = Bars
        Bars = np.where( np.bitwise_and(~isWater.bw, isNotVeg.bw), 1, 0 ).astype(np.uint8)
        
        # Apply a Convex Hull to Channel Bars
        Bars = mm.convex_hull_object( Bars )
        if close:
            # 1/8 of the total average channel width is used ( 1/4*mean(b) )
            rad = max( 0, 0.25*self.unwrapper.b.mean()/self.unwrapper.GeoTransf['PixelSize'] )
            Bars = mm.binary_closing( Bars, mm.disk(rad) )

        # FIXME : How do we define the threshold dimension for channel bars in the regridded unwrapped shit?
        if remove_small:
            Amin = self.unwrapper.N.size/2 * ( self.unwrapper.b.mean() / np.ediff1d(self.unwrapper.s[:2])[0] )
            Bars = mm.remove_small_objects( Bars, 2*Amin ) # Remove small Bars
            Bars = mm.remove_small_holes( Bars, Amin ) # Remove Internal Spots

        # Identify the Largest Bar on the Bend
        labeled_array, num_features = ndimage.measurements.label( Bars )
        self.Bars = labeled_array
        self.BarIdx = np.arange( num_features, dtype=int )
        return self.Bars


    def BarCentroid( self ):
        '''Compute Centroid of Each Bar'''

        num_features = self.BarIdx.max() + 1 # Number of Bar Features
        IC, JC = np.zeros(num_features,dtype=int), np.zeros(num_features,dtype=int) # Indexes of Bar Centroids
        # Compute Bar Centroids
        for n in xrange( num_features-1 ): [ IC[n], JC[n] ] = ndimage.measurements.center_of_mass( self.Bars==(n+1) )
        self.Centroid = np.vstack((IC, JC))
        return IC, JC


    def BarArea( self ):
        ''''Measure the number of Pixels of each Bar'''
        self.Area = np.zeros( self.BarIdx.size, dtype=int )
        for n in self.BarIdx: self.Area[n] = np.sum( self.Bars==n )
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
            ##############################
            ## DEBUG
            #mask = self.unwrapper.BendIndexes==n
            #BRS = np.zeros(self.Bars.shape)
            #f1 = plt.figure()
            #f2 = plt.figure()
            #ax1 = f1.add_subplot(111)
            #ax2 = f2.add_subplot(111)
            #for ki, k in enumerate(self.BarIdx[ (BBIdx==n)]):
            #    #BRS[self.Bars==k] = ki
            #    #plt.contourf(np.where(self.Bars==k, ki, np.nan), alpha=0.5)
            #    ax1.plot( self.unwrapper.XC[self.Contours[k][0].astype(int), self.Contours[k][1].astype(int)], self.unwrapper.YC[self.Contours[k][0].astype(int), self.Contours[k][1].astype(int)], lw=4, label=r'%s' % (ki+1) )
            #    print ki, k, np.unique(self.Bars==k, return_counts=True)
            #    #ax2.imshow( self.Bars, cmap='spectral', aspect='auto' )
            #    BRS[self.Bars==k] = np.sqrt((self.Bars==k).sum())
            #print
            #cf = ax2.imshow(BRS, aspect='auto', cmap='spectral')
            #plt.colorbar(cf)
            ##plt.colorbar()
            ###############################
            Areas =  np.asarray( [ (self.Bars==(k+1)).sum() for ki, k in enumerate(self.BarIdx[ BBIdx==n ]) ] )
            Indexes = self.BarIdx[ BBIdx==n ]
            #print Areas
            ##############################
            ## DEBUG
            ##plt.contour( np.where(self.Bars==Indexes[ Areas.argmax() ], 1, 0), [0.5] )
            #ax1.plot( self.unwrapper.XC[self.Contours[Indexes[ Areas.argmax() ]][0].astype(int), self.Contours[Indexes[ Areas.argmax() ]][1].astype(int)],
            #            self.unwrapper.YC[self.Contours[Indexes[ Areas.argmax() ]][0].astype(int), self.Contours[Indexes[ Areas.argmax() ]][1].astype(int)], 'ro',
            #            lw=0.5 )
            #ax1.set_title('%s on %s' % ( Areas.argmax(), len(Areas) ) )
            #ax1.legend()
            #plt.show()
            ###############################
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
            contour = sm.find_contours(self.Bars==n+1, 0.5)
            I, J = contour[0][:,0], contour[0][:,1]
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
            ibar = Bars.BarBendIdx[ bend ]
            
            # Get Bar Properties
            IC, JC = Bars.Centroid[ :,ibar ]
            X = Bars.unwrapper.XC[ IC, JC ]
            Y = Bars.unwrapper.YC[ IC, JC ]
            S = Bars.unwrapper.s[ JC ]
            N = Bars.unwrapper.N[ JC ]

            if normalize:
                Sbend = Bars.unwrapper.s[mask]
                S -= Sbend[ int( Sbend.size / 2 ) ] # Relative to "Bend Apex" (FIXME: use proper apex)
                S /= ( Sbend[-1] - Sbend[0] ) # Normalize to Bend Half-Length

            centroids_IJ.append( [ IC, JC ] )
            centroids_XY.append( [ X, Y ] )
            centroids_SN.append( [ S, N ] )

            # Get the Bend Index for the Next Time Step
            bend = bend_indexes_next[ mask ][0]

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
            ibar = Bars.BarBendIdx[ bend ]
            
            # Get Bar Properties
            contour = Bars.Contours[ ibar ]
            X = Bars.unwrapper.XC[ contour[0].astype(int),contour[1].astype(int) ]
            Y = Bars.unwrapper.YC[ contour[0].astype(int),contour[1].astype(int) ]
            S = Bars.unwrapper.s[ contour[0].astype(int) ]
            N = Bars.unwrapper.N[ contour[1].astype(int) ]

            if normalize:
                Sbend = Bars.unwrapper.s[mask]
                S -= Sbend[ int( Sbend.size / 2 ) ] # Relative to "Bend Apex" (FIXME: use proper apex)
                S /= ( 0.5*(Sbend[-1] - Sbend[0]) ) # Normalize to Bend Half-Length

            contours_IJ.append( [ contour[0], contour[1] ] )
            contours_XY.append( [ X, Y ] )
            contours_SN.append( [ S, N ] )

            # Get the Bend Index for the Next Time Step
            bend = bend_indexes_next[ mask ][0]
        

        # TMP

        return contours_IJ, contours_SN, contours_XY
