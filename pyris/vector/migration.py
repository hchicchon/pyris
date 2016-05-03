from __future__ import division
import numpy as np
from scipy import interpolate
from ..misc import NaNs, Intersection, PolygonCentroid
from .. import HAS_MLPY, MLPYException, MLPYmsg
from .interpolation import InterpPCS
if HAS_MLPY: from .. import wave
import matplotlib.pyplot as plt

class AxisMigration( object ):

    '''
    MigRateBend - Read a List of River Planforms, Locate Individual Bends, compute Migration Rates
    '''

    omega0 = 6 # Morlet Wavelet Parameter
    icwtC = [] # Reconstructed ICWT Filtered Curvature
    I = [] # Inflection Points
    CI1 = [] # Backward Correlated Inflection Points
    CI12 = [] # Forward Correlate Inflection Points
    CI11 = [] # Auto-Correlation
    BI = [] # Bend Indexes
    dx = [] # Local x Migration Rate
    dy = [] # Local y Migration Rate
    dz = [] # Local Migration Rate Magnitude
    
    def __init__( self, Xseries, Yseries ):
        '''Constructor - Get Planforms'''
        self.data = []
        for x, y in zip( Xseries, Yseries ):
            x, y = np.asarray(x), np.asarray(y)
            dx = np.ediff1d( x, to_begin=0 )
            dy = np.ediff1d( y, to_begin=0 )
            ds = np.sqrt( dx**2 + dy**2 )
            s = np.cumsum( ds )
            c = -np.gradient( np.arctan2( dy, dx ), np.gradient(s) )
            self.data.append( { 'x': x, 'y': y, 's': s, 'c':c } )
        return None

    def IterData( self ):
        '''Data Iterator'''
        for i, d in enumerate( self.data ): yield i, d

    def IterData2( self ):
        '''Data Pair Iterator'''
        for i, (d1, d2) in enumerate( zip( self.data[:-1], self.data[1:]) ): yield i, (d1, d2)

    def Iterbends( self, Idx ):
        '''Bend Iterator'''
        for i, (il,ir) in enumerate( zip( Idx[:-1], Idx[1:] ) ): yield i, (il, ir)

    def Iterbends2( self, Idx1, Idx2 ):
        '''Bend Pair Iterator'''
        for i, (i1l,i1r,i2l,i2r) in enumerate( zip( Idx1[:-1], Idx1[1:], Idx2[:-1], Idx2[1:] ) ): yield i, (i1l,i1r,i2l,i2r)

    def FindPeaks( self, arr ):
        '''Make NaN any element that is not a local maximum'''
        arr = np.abs( arr )
        arr[1:-1] = np.where(
            np.logical_and(arr[1:-1]>arr[2:], arr[1:-1]>arr[:-2]), arr[1:-1], np.nan
            )
        arr[0], arr[-1] = np.nan, np.nan
        return arr

    def FilterAll( self, reduction=0.33 ):
        '''Perform ICWT Filtering on all the Data'''
        for i, d in self.IterData():
            self.icwtC.append( self.FilterCWT( d['c'], d['s'], reduction=reduction ) )
        return None

    def FilterCWT( self, *args, **kwargs ):
        '''Use Inverse Wavelet Transform in order to Filter Data'''

        sgnl = args[0]
        time = args[1]
        reduction = kwargs.pop( 'reduction', 0.33 )
        full_output = kwargs.pop( 'full_output', False )

        N = sgnl.size
        dt = time[1] - time[0]
        omega0 = self.omega0
        scales = wave.autoscales( N=N, dt=dt, dj=0.1, wf='morlet', p=omega0 )
        cwt = wave.cwt( x=sgnl, dt=dt, scales=scales, wf='morlet', p=omega0 )
        gws = (np.abs(cwt)**2).sum( axis=1 ) / N
        peaks = np.full( gws.size, np.nan )
        peaks[1:-1] = np.where( np.logical_and(gws[1:-1]>gws[2:],gws[1:-1]>gws[:-2]), gws[1:-1], np.nan )
        for i in xrange( (~np.isnan(peaks)).astype(int).sum() ):
            p = np.nanargmax( peaks )
            peaks[p] = np.nan
            scalemax = scales[ p ] # Fundamental Harmonic
            mask = ( scales >= reduction*scalemax )
            icwt = wave.icwt( cwt[mask, :], dt, scales[mask], wf='morlet', p=omega0 )
            if not np.allclose(icwt, 0): break
        if full_output: return icwt, scales, scalemax
        return icwt
        
    def GetInflections( self, Cs ):
        '''Compute 0-crossings of channel curvature'''
        return np.where( Cs[1:]*Cs[:-1] < 0 )[0]

    def GetAllInflections( self ):
        '''Get Inflection points on Inverse Wavelet Transform for Curvature.'''
        self.I = []
        for i, d in self.IterData():
            self.I.append( self.GetInflections( self.icwtC[i] ) )
        return None

    def CorrelateInflections( self, *args, **kwargs ):
        '''Find the Closest Inflection Points'''

        self.CI1 = [] # Points on the Current Planform
        self.CI12 = [] # Points to which the First Planform Points Converge to the Second Planform
        self.CI11 = [] # Points where the second planform converges into itself to get in the next one (some bends become one bend)
        C1 = self.I[0] # Initial Reference Planform
        # Go Forward
        for i, (d1, d2) in self.IterData2():
            self.CI11.append( C1 )
            C2 = self.I[i+1]
            C12 = np.zeros_like( C1, dtype=int )
            x1, y1 = d1['x'], d1['y']
            x2, y2 = d2['x'], d2['y']
            Cs1 = self.icwtC[i]
            Cs2 = self.icwtC[i+1]
            for ipoint, Ipoint in enumerate( C1 ):
                xi1, yi1 = x1[Ipoint], y1[Ipoint]
                #xC2, yC2 = x2[C2], y2[C2] # Do not care about sign
                xC2 = np.where( Cs2[C2+1]*Cs1[Ipoint+1]<0, np.nan, x2[C2] ) # Take real curvature sign
                yC2 = np.where( Cs2[C2+1]*Cs1[Ipoint+1]<0, np.nan, y2[C2] ) # Take real curvature sign
                # Find the Closest
                C12[ipoint] = C2[ np.nanargmin( np.sqrt( (xC2-xi1)**2 + (yC2-yi1)**2 ) ) ]
            # There are some duplicated points - we need to get rid of them
            unique, counts = np.unique(C12, return_counts=True)
            duplic = unique[ counts>1 ]
            cduplic = counts[ counts > 1 ]
            for idup, (dup, cdup) in enumerate( zip( duplic, cduplic ) ):
                idxs = np.where( C12==dup )[0]
                idx = np.argmin( np.sqrt( (x2[dup]-x1[C1][idxs])**2 + (y2[dup]-y1[C1][idxs])**2 ) )
                idxs = np.delete( idxs, idx )
                C1 = np.delete( C1, idxs )
                C12 = np.delete( C12, idxs )

            # Sometimes inflections are messed up. Sort them out!
            C1.sort()
            C12.sort()

            # Plot for Cutoff-to-NewBend Inflection Correlation (for DEBUG purposes only)
            #plt.figure()
            #plt.plot( x1, y1, 'k' )
            #plt.plot( x2, y2, 'r' )
            #plt.plot( x1[C1], y1[C1], 'ko' )
            #plt.plot( x2[C12], y2[C12], 'ro' )
            #for i in xrange(C1.size):
            #    plt.plot( [x1[C1[i]], x2[C12[i]]], [y1[C1[i]], y2[C12[i]]], 'g' )
            #plt.show()

            self.CI1.append(C1)
            self.CI12.append(C12)
            C1 = C12
        self.CI1.append(C12)
        return None

    def BendUpstreamDownstream( self, I, icwtC ):
        '''Bend Upstream-Downstream Indexes'''
        BUD = NaNs( icwtC.size )
        for i, (il,ir) in self.Iterbends( I ):
            iapex = il + np.abs( icwtC[ il:ir ] ).argmax()
            BUD[ il ] = 2 # Inflection Point
            BUD[ ir+1 ] = 2 # Inflection Point
            BUD[ iapex ] = 0 # Bend Apex
            BUD[ il:iapex ] = -1  # Bend Upstream
            BUD[ iapex+1:ir ] = +1 # Bend Downstream
        return BUD

    def AllBUDs( self ):
        '''Bend Upstream-Downstream Indexes for All Planforms'''
        self.BUD = []
        for i, d in self.IterData():
            self.BUD.append( self.BendUpstreamDownstream( self.CI1[i], self.icwtC[i] ) )
        return None

    def GetBends( self, c ):
        '''Returns Inflection Points, Bend Indexes'''
        Idx = self.GetInflections( c )
        BIDX = self.LabelBends( c.size, Idx )
        return BIDX, Idx

    def LabelBends( self, *args, **kwargs ):
        'Bend label for each point of the planform'
        N = args[0] if isinstance( args[0], int ) else args[0].size
        Idx = args[1]
        labels = -np.ones( N, dtype=int )
        for i, (il, ir) in self.Iterbends( Idx ):
            labels[il:ir] = i
        return labels

    def LabelAllBends( self, *args, **kwargs ):
        '''Apply Bend Labels to Each Planform'''
        self.BI = []
        for i, d in self.IterData():
            self.BI.append( self.LabelBends( d['s'].size, self.CI1[i] ) )
        return None

    def CorrelateBends( self, *args, **kwargs ):
        '''Once Bends are Separated and Labeled, Correlate Them'''
        self.B12 = []
        for i, (d1, d2) in self.IterData2():
            B1 = self.BI[i]
            B2 = self.BI[i+1]
            B12 = -np.ones( B1.size, dtype=int )
            I1 = self.CI1[i]
            I2 = self.CI1[i+1]
            I12 = self.CI12[i]
            x1, y1 = d1['x'], d1['y']
            x2, y2 = d2['x'], d2['y']

            # X il momento tengo la correlazione tra gli inflections
            for i, (i1l, i1r, i2l, i2r) in self.Iterbends2( I1, I12 ):
                vals, cnts = np.unique( B2[i2l:i2r], return_counts=True )
                if len( vals ) == 0:
                    B12[i1l:i1r] = -1
                else:
                    B12[i1l:i1r] = vals[ cnts.argmax() ]

            # for DEBUG purposes
            #for i, (il, ir) in self.Iterbends( I1 ):
            #    b1 = slice(il,ir)
            #    b2 = B2==B12[il]
            #    print B1[il], B12[il]
            #    if B12[il] < 0: continue
            #    xb1, yb1 = x1[b1], y1[b1]
            #    xb2, yb2 = x2[b2], y2[b2]
            #    plt.figure()
            #    plt.plot( x1, y1, 'k' )
            #    plt.plot( x2, y2, 'r' )
            #    plt.plot( xb1, yb1, 'k', lw=4 )
            #    plt.plot( xb2, yb2, 'r', lw=4 )
            #    plt.show()

            self.B12.append( B12 )
        self.B12.append( -np.ones( x2.size ) ) # Add a Convenience -1 Array for the Last Planform

        # for DEBUG purposes
        #B = 14
        #plt.figure()
        #for i, d in self.IterData():
        #    xi, yi = d[0], d[1]
        #    if i == 0: plt.plot(xi, yi, 'k')
        #    X = xi[self.BI[i]==B]
        #    Y = yi[self.BI[i]==B]
        #    plt.plot(X, Y, lw=3)
        #    B = ( self.B12[i][ self.BI[i]==B ] )[0]
        #plt.show()

        return None        

    def FindOrthogonalPoint( self, data1, data2, i2, L=None ):
        '''Find the orthogonal point to second line on the first one'''
        [ x1, y1, s1 ] = data1['x'], data1['y'], data1['s']
        [ x2, y2, s2 ] = data2['x'], data2['y'], data2['s']
        if L is None: L = 10*np.gradient( s1 ).mean()
        a = np.arctan2( ( y2[i2+1] - y2[i2-1] ), ( x2[i2+1] - x2[i2-1] ) ) - np.pi/2 # Local Perpendicular Angle
        P = np.array( [ x2[i2], y2[i2] ] )
        R = np.array( [ np.cos(a), np.sin(a) ] ) * L
        hits = []
        for i in xrange( 1, x1.size ):
            Q = np.array( [ x1[i-1], y1[i-1] ] )
            S = np.array( [ x1[i], y1[i] ] ) - Q
            segments_intersect, (xi, yi) = Intersection( P, Q, R, S )
            if segments_intersect:
                hits.append( np.sqrt( (x1-xi)**2 + (y1-yi)**2 ).argmin() )
        if hits == []: return None
        return np.min( hits )

    def MigrationRates( self, data1, data2, I1, I12, B1, B2, B12,
                        recall_on_cutoff=True  ):
        '''Compute Local Migration Rates by connected individual bends'''

        [ x1, y1, s1 ] = data1['x'], data1['y'], data1['s']
        [ x2, y2, s2 ] = data2['x'], data2['y'], data2['s']
        [ dx, dy, dz]  = [ NaNs( x1.size ), NaNs( x1.size ), NaNs( x1.size ) ]
        Ictf = I1.copy()
        dzmax = 5*np.gradient( s1 ).mean() # XXX : check if this makes sense!!
        for i, (il,ir) in self.Iterbends( I1 ):
            # Isolate Bend
            mask1 = np.full( s1.size, False, dtype=bool ); mask1[il:ir]=True
            mask2 = B2==B12[il]
            if B12[il] < 0: continue # Bend Is not Correlated
            bx1, by1, N1 = x1[mask1], y1[mask1], mask1.sum() # Bend in First Planform
            bx2, by2, N2 = x2[mask2], y2[mask2], mask2.sum() # Bend in Second Planform
            # In order to apply a PCS to the Second Planform, it cannot has more points than the first one
            if N1 <=1 or N2<=1: continue # FIXME: this shouldn't happen but sometimes it does
            if N2 > N1: # Remove Random Points from Second Bend
                idx = np.full( N2, True, bool )
                idx[ np.random.choice( np.arange(1,N2-1), N2-N1, replace=False ) ] = False
                bx2 = bx2[ idx ]
                by2 = by2[ idx ]
                N2 = bx2.size
            # ReInterpolate Second Planform (Parametric Cubic Spline)
            if N1 <= 3 or N2 <= 3: kpcs=1 # If we have too few points, use linear interpolation
            else: kpcs=3
            bx2, by2 = InterpPCS( bx2, by2, N=N1, s=N2, k=kpcs, with_derivatives=False )
            # Compute Migration Rates for the whole bend
            dxb = bx2 - bx1
            dyb = by2 - by1
            dzb = np.sqrt( dxb**2 + dyb**2 )
            [ dxr, dyr, dzr] = map( np.copy, [ dxb, dyb, dzb ] )
            # If the Migration Rate is too high, maybe it is wrong. Put a NaN
            dxb[ dzb > dzmax ] = np.nan # Bind Local Migration Rate to a Maximum
            dyb[ dzb > dzmax ] = np.nan # Bind Local Migration Rate to a Maximum
            dzb[ dzb > dzmax ] = np.nan # Bind Local Migration Rate to a Maximumx

            # Where more than 50% of the migration rates are NaNs, assume a CutOff has occurred
            if (np.isnan(dzb)).sum() / dzb.size > 0.5:
                dxb[:] = np.nan
                dyb[:] = np.nan
                dzb[:] = np.nan
                if recall_on_cutoff:
                    ictfl = self.FindOrthogonalPoint( data1, data2, I12[i] )
                    ictfr = self.FindOrthogonalPoint( data1, data2, I12[i+1] )
                    Ictf[i] = ictfl if ictfl is not None else Ictf[i]
                    Ictf[i+1] = ictfr if ictfr is not None else Ictf[i+1]
            else: # Otherwise, restore migration values
                dxb = dxr
                dyb = dyr
                dzb = dzr
            # Set Migration Rate into Main Arrays
            dx[ mask1 ] = dxb
            dy[ mask1 ] = dyb
            dz[ mask1 ] = dzb

        if recall_on_cutoff:
            Ictf = np.unique( np.asarray( Ictf ) )
            return self.MigrationRates( data1, data2, Ictf, I12, B1, B2, B12,
                                        recall_on_cutoff=False )
        # for DEBUG purposes only
        #plt.figure()
        #plt.plot(x1, y1, 'k')
        #plt.plot(x2, y2, 'r')
        #plt.plot(x1[Ictf], y1[Ictf], 'ko')
        #plt.plot(x1[I1], y1[I1], 'mo')
        #plt.plot(x2[I12], y2[I12], 'ro')
        #for i in xrange(x1.size):
        #    plt.arrow(x1[i], y1[i], dx[i], dy[i], fc='g', ec='g', head_length=20, head_width=20)
        #plt.show()
        return dx, dy, dz

    def AllMigrationRates( self, recall_on_cutoff=True ):
        '''Apply Migration Rates Algorithm to the whole set of planforms'''
        self.dx = []
        self.dy = []
        self.dz = []
        
        for i, (d1, d2) in self.IterData2():
            I1, I12 = self.CI1[i], self.CI12[i]
            B1, B2, B12 = self.BI[i], self.BI[i+1], self.B12[i]
            dxi, dyi, dzi = self.MigrationRates( d1, d2, I1, I12, B1, B2, B12,
                                                 recall_on_cutoff=recall_on_cutoff )
            # for DEBUG purposes only
            #x1, y1, x2, y2 = d1[0], d1[1], d2[0], d2[1]
            #Ictf = I1
            #plt.figure()
            #plt.plot(x1, y1, 'k')
            #plt.plot(x2, y2, 'r')
            #plt.plot(x1[Ictf], y1[Ictf], 'ko')
            #plt.plot(x1[I1], y1[I1], 'mo')
            #plt.plot(x2[I12], y2[I12], 'ro')
            #for i in xrange(x1.size):
            #    plt.arrow(x1[i], y1[i], dxi[i], dyi[i], fc='g', ec='g', head_length=20, head_width=20)
            #plt.show()
            self.dx.append( dxi ), self.dy.append( dyi ), self.dz.append( dzi )
        N = ( d2['s'] ).size
        self.dx.append( NaNs( N ) ), self.dy.append( NaNs( N ) ), self.dz.append( NaNs( N ) )
        return None

    def __call__( self, filter_reduction=0.33, return_on_cutoff=True ):
        self.FilterAll( reduction=filter_reduction )
        self.GetAllInflections()
        self.CorrelateInflections()
        self.LabelAllBends()
        self.AllBUDs()
        self.CorrelateBends()
        self.AllMigrationRates( recall_on_cutoff=True )
        return self.dx, self.dy, self.dz, self.icwtC, self.BI, self.B12, self.BUD

