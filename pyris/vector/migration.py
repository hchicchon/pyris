from __future__ import division
import numpy as np
from scipy import interpolate
from ..misc import NaNs, Intersection, PolygonCentroid
from .. import HAS_MLPY, MLPYException, MLPYmsg
from .interpolation import InterpPCS
if HAS_MLPY: from .. import wave
import matplotlib.pyplot as plt

class MigRateBend( object ):

    '''
    MigRateBend - Read a List of River Planforms, Locate Individual Bends, compute Migration Rates
    '''

    omega0 = 6 # Morlet Wavelet Parameter
    icwtC1 = [] # 1st Harmonic ICWT Filtered Curvature
    icwtC3 = [] # 3st Harmonic ICWT Filtered Curvature    
    T = [] # Times
    DT = [] # Time Steps
    I = [] # Inflection Points
    CI1 = [] # Backward Correlated Inflection Points
    CI12 = [] # Forward Correlate Inflection Points
    CI11 = [] # Auto-Correlation
    BI = [] # Bend Indexes
    dx = [] # Local x Migration Rate
    dy = [] # Local y Migration Rate
    dz = [] # Local Migration Rate Magnitude
    
    def __init__( self, data, T, **kwargs ):
        '''Constructor - Get Planforms'''
        self.data = data
        self.T = np.asarray( T )
        self.SetDT( self.T )
        self.use_wavelets = kwargs.pop( 'use_wavelets' , HAS_MLPY )
        self.Bmult = kwargs.pop( 'Bmult', None )
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

    def FilterAll( self, reduction=1 ):
        '''Perform ICWT Filtering on all the Data'''
        self.icwtC1, self.icwtC3 = [], []
        for i, d in self.IterData():
            icwtC1, icwtC3 = self.FilterCWT( d[4], d[2], reduction=reduction )
            self.icwtC1.append( icwtC1 )
            self.icwtC3.append( icwtC3 )
        return None

    def FilterCWT( self, *args, **kwargs ):
        '''Use Inverse Wavelet Transform in order to Filter Data'''

        if not self.use_wavelets: raise MLPYException, MLPYmsg
        
        sgnl = args[0]
        time = args[1]
        reduction = kwargs.pop( 'reduction', 1 )
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
            scale1 = scales[ p ] # Fundamental Harmonic
            scale3 = scales[ np.abs(scales-scale1/3).argmin() ] # Up to Third Harmonic
            mask1 = (scales>=reduction*scale1)
            mask3 = (scales>=scale3)
            icwt1 = wave.icwt( cwt[mask1, :], dt, scales[mask1], wf='morlet', p=omega0 )
            icwt3 = wave.icwt( cwt[mask3, :], dt, scales[mask3], wf='morlet', p=omega0 )
            if not np.allclose(icwt1, 0): break
        if full_output: return icwt1, icwt3, scales, scale1, scale3
        return icwt1, icwt3
        
    def GetInflections( self, Cs ):
        '''Compute 0-crossings of channel curvature'''
        return np.where( Cs[1:]*Cs[:-1] < 0 )[0]

    def GetAllInflections( self ):
        '''Get Inflection points on Inverse Wavelet Transform for Curvature.'''
        self.I = []
        for i, d in self.IterData():
            self.I.append( self.GetInflections( self.icwtC1[i] ) )
        return None

    def CorrelateInflections( self, *args, **kwargs ):
        '''Find the Closest Inflection Points'''

        self.CI1 = [] # Points on the Current Planform
        self.CI12 = [] # Points of which the First Planform Points Converge to the Second Planform
        self.CI11 = [] # Points where the second planform converges into itself to get in the next one (some bends become one bend)
        C1 = self.I[0] # Initial Reference Planform
        # Go Forward
        for i, (d1, d2) in self.IterData2():
            self.CI11.append( C1 )
            C2 = self.I[i+1]
            C12 = np.zeros_like( C1, dtype=int )
            x1, y1 = d1[0], d1[1]
            x2, y2 = d2[0], d2[1]
            Cs1 = self.icwtC1[i]
            Cs2 = self.icwtC1[i+1]
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
                idxs = np.where(C12==dup)[0]
                idx = np.argmin( np.sqrt( (x2[dup]-x1[C1][idxs])**2 + (y2[dup]-y1[C1][idxs])**2 ) )
                idxs = np.delete( idxs, idx )
                C1 = np.delete( C1, idxs )
                C12 = np.delete( C12, idxs )

            # Plot for Cutoff-to-NewBend Inflection Correlation
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

    def BendUpstreamDownstream( self, d, I, icwtC ):
        '''Bend Upstream-Downstream Indexes'''
        BUD = np.full( d[0].size, np.nan )
        for i, (il,ir) in self.Iterbends( I ):
            iapex = il + icwtC[ il:ir ].argmax()
            BUD[ il ] = 2 # Inflection Point
            BUD[ ir+1 ] = 2 # Inflection Point
            BUD[ iapex ] = 0 # Bend Apex
            BUD[ il:iapex ] = -1  # Bend Upstream
            BUD[ iapex+1:ir ] = +1 # Bend Downstream
        return BUD

    def AllBUDs( self, harm=1 ):
        '''Bend Upstream-Downstream Indexes for All Planforms'''
        self.BUD = []
        if harm==1: icwtC = self.icwtC1
        elif harm==3: icwtC = self.icwtC3
        for i, d in self.IterData():
            self.BUD.append( self.BendUpstreamDownstream( d, self.CI1[i], icwtC[i] ) )
        return None


    def PlotRealFiltered( self, *args, **kwargs ):
        '''Plot the original planform and the Wavelet-Reconstructed one'''
        idx = args[0] # Which planform
        harm = kwargs.pop( 'harm', 1 ) # Which Recontruction (to which harmonic)
        d = self.data[idx]
        x0, y0, s0, theta0, Cs0 = d[0], d[1], d[2], d[3], d[4]
        if harm == 1: Cs1 = self.icwtC1[idx]
        elif harm == 3: Cs1 = self.icwtC3[idx]
        else: raise ValueError, 'harm must be either 1 or 3'
        ds = np.ediff1d( s0, to_begin=0 )
        theta1 = np.arctan2(y0[1]-y0[0], x0[1]-x0[0]) - np.cumsum( Cs1*ds )
        x1 = x0[0] + np.cumsum( ds*np.cos(theta1) )
        y1 = y0[0] + np.cumsum( ds*np.sin(theta1) )
        f = plt.figure()
        plt.plot(x0, y0)
        plt.plot(x1, y1)
        return f

    def SetDT( self, times ):
        '''Compute DT steps in years from time names'''
        years = np.array([ float(t.split('_')[0].strip()) for t in times ])
        jdays = np.array([ float(t.split('_')[1].strip())/365 for t in times ])
        self.T = jdays + years
        self.DT = np.gradient( self.T )
        return None

    def GetDT( self ):
        return self.DT

    def GetBends( self, Cs ):
        '''Returns Inflection Points, Bend Indexes'''
        Idx = self.GetInflections( Cs )
        BIDX = self.LabelBends( Cs.size, Idx )
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
            self.BI.append( self.LabelBends( d[0].size, self.CI1[i] ) )
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
            x1, y1 = d1[0], d1[1]
            x2, y2 = d2[0], d2[1]

            # X il momento tengo la correlazione tra gli inflections
            for i, (i1l, i1r, i2l, i2r) in self.Iterbends2( I1, I12 ):
                vals, cnts = np.unique( B2[i2l:i2r], return_counts=True )
                if len( vals ) == 0:
                    B12[i1l:i1r] = -1
                else:
                    B12[i1l:i1r] = vals[ cnts.argmax() ]

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

    def FindOrthogonalPoint( self, data1, data2, i2, DT=1 ):
        '''Find the orthogonal point to second line on the first one'''
        [ x1, y1, s1, theta1, Cs1, W1 ] = data1[:6]
        [ x2, y2, s2, theta2, Cs2, W2 ] = data2[:6]
        L = 4 * W1.mean() * DT # Segment Length
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
                        DT=1, Bmult=8, recall_on_cutoff=True  ):
        '''Compute Local Migration Rates by connected individual bends'''

        [ x1, y1, s1, theta1, Cs1, W1 ] = data1[:6]
        [ x2, y2, s2, theta2, Cs2, W2 ] = data2[:6]
        [ dx, dy, dz]  = [ NaNs( s1.size ), NaNs( s1.size ), NaNs( s1.size ) ]
        Ictf = I1.copy()
        for i, (il,ir) in self.Iterbends( I1 ):
            # Isolate Bend
            #mask1 =  B1==B1[il]
            mask1 =  np.full( s1.size, False, dtype=bool ); mask1[il:ir]=True
            mask2 = B2==B12[il]
            if B12[il] < 0: continue # Band Is not Correlated
            bx1, by1, bs1, N1 = x1[mask1], y1[mask1], s1[mask1], mask1.sum() # Bend in First Planform
            bx2, by2, bs2, N2 = x2[mask2], y2[mask2], s2[mask2], mask2.sum() # Bend in Second Planform
            # In order to apply a PCS to the Second Planform, it cannot has more points than the first one
            if N2 > N1: # Remove Random Points from Second Bend
                idx = np.full( N2, True, bool )
                idx[ np.random.choice( np.arange(1,N2-1), N2-N1, replace=False ) ] = False
                bx2 = bx2[ idx ]
                by2 = by2[ idx ]
            # ReInterpolate Second Planform (Parametric Cubic Spline)
            if N1 <= 3: kpcs=1 # If we have too few points, use linear interpolation
            else: kpcs=3
            bx2, by2 = InterpPCS( bx2, by2, N=N1, s=N2, k=kpcs, with_derivatives=False )
            # Compute Migration Rates for the whole bend
            dxb = bx2 - bx1
            dyb = by2 - by1
            dzb = np.sqrt( dxb**2 + dyb**2 )
            [ dxr, dyr, dzr] = map( np.copy, [ dxb, dyb, dzb ] )
            # If the Migration Rate is too high, maybe it is wrong. Put a NaN
            dxb[ dzb > W1.mean()*Bmult*DT ] = np.nan # Bind Local Migration Rate to a Maximum
            dyb[ dzb > W1.mean()*Bmult*DT ] = np.nan # Bind Local Migration Rate to a Maximum
            dzb[ dzb > W1.mean()*Bmult*DT ] = np.nan # Bind Local Migration Rate to a Maximumx

            # Where more than 50% of the migration rates are NaNs, assume a CutOff has occurred
            if (np.isnan(dzb)).sum() / dzb.size > 0.5:
                dxb[:] = np.nan
                dyb[:] = np.nan
                dzb[:] = np.nan
                if recall_on_cutoff:
                    ictfl = self.FindOrthogonalPoint( data1, data2, I12[i], DT=DT )
                    ictfr = self.FindOrthogonalPoint( data1, data2, I12[i+1], DT=DT )
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
            return self.MigrationRates( data1, data2, Ictf, I12, B1, B2, B12, DT=1,
                                        Bmult=Bmult, recall_on_cutoff=False )
        #plt.figure()
        #plt.plot(x1, y1, 'k')
        #plt.plot(x2, y2, 'r')
        #plt.plot(x1[Ictf], y1[Ictf], 'ko')
        #plt.plot(x1[I1], y1[I1], 'mo')
        #plt.plot(x2[I12], y2[I12], 'ro')
        #for i in xrange(x1.size):
        #    plt.arrow(x1[i], y1[i], dx[i], dy[i], fc='g', ec='g', head_length=20, head_width=20)
        #plt.show()
        return dx, dy, dz/DT

    def AllMigrationRates( self, recall_on_cutoff=True ):
        '''Apply Migration Rates Algorithm to the whole set of planforms'''
        self.dx = []
        self.dy = []
        self.dz = []
        
        for i, (d1, d2) in self.IterData2():
            I1, I12 = self.CI1[i], self.CI12[i]
            B1, B2, B12 = self.BI[i], self.BI[i+1], self.B12[i]
            if self.DT == []: DT = 1
            else: DT = self.DT[i]
            dxi, dyi, dzi = self.MigrationRates( d1, d2, I1, I12, B1, B2, B12,
                                                 DT=DT, recall_on_cutoff=recall_on_cutoff )
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
        N = ( d2[0] ).size
        self.dx.append( NaNs( N ) ), self.dy.append( NaNs( N ) ), self.dz.append( NaNs( N ) )
        return None

    def MigrationStyles( self, data1, data2, I1, I12, B1, B2, B12, dz, BUD1, BUD2 ):
        '''Define the Style of Migration of Each Bend between two Planforms '''
        # Create Arrays
        rot = np.full( data1[0].size, np.nan ) # Bend Rotation
        Dx = np.full( data1[0].size, np.nan ) # Bend Translation x
        Dy = np.full( data1[0].size, np.nan ) # Bend Translation y
        dA = np.full( data1[0].size, np.nan ) # Bend Area Growth
        dL = np.full( data1[0].size, np.nan ) # Bend Elongation
        dS = np.full( data1[0].size, np.nan ) # Bend Elongation
        # Read Input Arguments
        [ x1, y1, s1, theta1, Cs1, W1 ] = data1[:6] # First Planform Geometry
        [ x2, y2, s2, theta2, Cs2, W2 ] = data2[:6] # Second Planform Geometry
        # COmpute Migration Styles for each Bend
        for i, (il, ir) in self.Iterbends( I1 ):
            mask1 = B1 == B1[il]
            mask2 = B2 == B12[il]
            if B12[il] < 0 or np.isnan( dz[mask1] ).all(): continue # Band Is not Correlated or a CutOff Occurred
            bx1, by1, bs1, N1 = x1[mask1], y1[mask1], s1[mask1], mask1.sum() # Bend in First Planform
            bx2, by2, bs2, N2 = x2[mask2], y2[mask2], s2[mask2], mask2.sum() # Bend in Second Planform
            # Bend Apex
            iapex1 = np.where(BUD1[mask1]==0) # icwtC_1[ mask1 ].argmax()
            iapex2 = np.where(BUD2[mask2]==0) # icwtC_2[ mask2 ].argmax()
            # MidPoint between Inflection Points
            x0_1 = 0.5 * (bx1[0] + bx1[-1])
            y0_1 = 0.5 * (by1[0] + by1[-1])
            x0_2 = 0.5 * (bx2[0] + bx2[-1])
            y0_2 = 0.5 * (by2[0] + by2[-1])
            # Slope of the bend Axes
            m_1 = (by1[iapex1] - y0_1) / (bx1[iapex1] - x0_1)
            m_2 = (by2[iapex2] - y0_2) / (bx2[iapex2] - x0_2)
            # Sinuosities
            sinuosity1 = np.diff(bs1).sum() / np.sqrt( (bx1[-1]-bx1[0])**2 + (by1[-1]-by1[0])**2 )
            sinuosity2 = np.diff(bs2).sum() / np.sqrt( (bx2[-1]-bx2[0])**2 + (by2[-1]-by2[0])**2 )
            # Bend Rotation
            [X1, Y1], A1 = PolygonCentroid( bx1, by1, return_area=True )
            [X2, Y2], A2 = PolygonCentroid( bx2, by2, return_area=True )
            rot[ mask1 ] = np.arctan( m_2 ) - np.arctan( m_1 )
            Dx[ mask1 ] = X2 - X1
            Dy[ mask1 ] = Y2 - Y1
            dA[ mask1 ] = A2 - A1
            dL[ mask1 ] = np.diff(bs2).sum() - np.diff(bs1).sum()
            dS[ mask1 ] = sinuosity2 - sinuosity1
        return rot, Dx, Dy, dA, dL, dS

    def AllMigrationStyles( self, harm=1 ):
        '''Compute The Migration Style for each Bend on each Planform'''
        # Create Migration Styles Lists
        self.rot = []
        self.Dx = []
        self.Dy = []
        self.dA = []
        self.dL = []
        self.dS = []
        # Call MigrationStyles on every Planform
        for i, (d1, d2) in self.IterData2():
            # Create input Arguments
            I1, I12 = self.CI1[i], self.CI12[i]
            B1, B2, B12 = self.BI[i], self.BI[i+1], self.B12[i]
            dz = self.dz[i]
            BUD1, BUD2 = self.BUD[i], self.BUD[i+1]
            DT = 1 if self.DT==[] else self.DT[i]
            # Call MigrationStyles
            rot, Dx, Dy, dA, dL, dS = self.MigrationStyles( d1, d2, I1, I12, B1, B2, B12, dz, BUD1, BUD2 )
            self.rot.append( rot ), self.Dx.append( Dx ), self.Dy.append( Dy )
            self.dA.append( dA ), self.dL.append( dL ), self.dS.append( dS )
        # Append NaN Arrays for the Last Planform
        N = ( d2[0] ).size
        self.rot.append( NaNs( N ) ), self.Dx.append( NaNs( N ) ), self.Dy.append( NaNs( N ) )
        self.dA.append( NaNs( N ) ), self.dL.append( NaNs( N ) ), self.dS.append( NaNs( N ) )
        return None

    def BuildFullDict( self ):

        '''Bind all Planform Migration Propertied to a Dict'''

        FullDict = {
            'times' : self.T,
            'DT' : self.DT,
            'Csf1' : self.icwtC1,
            'Csf3' : self.icwtC3,
            'B1' : self.BI,
            'B12' : self.B12,
            'BUD' : self.BUD,
            'dx' : self.dx,
            'dy' : self.dy,
            'dz' : self.dz,
            'rot' : self.rot,
            'dA' : self.dA,
            'dL' : self.dL,
            'dS' : self.dS,
            'Dx' : self.Dx,
            'Dy' : self.Dy
            }

        return FullDict

    def __call__( self, filter_reduction=1, return_on_cutoff=True, bud_harm=1 ):
        self.FilterAll( reduction=0.5 )
        self.GetAllInflections()
        self.CorrelateInflections()
        self.LabelAllBends()
        self.AllBUDs( harm=bud_harm )
        self.CorrelateBends()
        self.AllMigrationRates( recall_on_cutoff=True )
        self.AllMigrationStyles()
        return self.BuildFullDict()

