from __future__ import division
import numpy as np
from scipy import interpolate
from numpy.lib import stride_tricks
from ..misc import GeoReference, Line2D

class AxisReader( object ):
    '''
    Read Pruned and Skeletonized Axis/Distance Image and
    extract vector data.
    '''

    primitives = ([[1,0,0],
                   [0,1,0],
                   [0,0,0]],
                  [[0,1,0],
                   [0,1,0],
                   [0,0,0]])

    def __init__( self, I, first_point=None, start_from=None, method='width', verbose=True, call_depth=0, jidx=[] ):
        '''Constructor'''

        self.I = I # Image
        # Apply A Filter on Border in order to avoid errors
        self.I[0,:] = 0
        self.I[:,0] = 0
        self.I[-1,:] = 0
        self.I[:,-1] = 0
        self.BI = np.where( I>0, 1, 0 ) # Binary
        self.hits = self.BI.copy() # For binary search
        self.first_point = first_point
        self.start_from = 't'
        if start_from is not None: self.start_from = start_from
        self.verbose = verbose
        self.method = method
        self.call_depth = call_depth
        self.jidx = jidx

    def GetJunction( self, idx ):
        '''Junction Indexes List'''
        if len( self.jidx ) > 0: idx += self.jidx[-1]
        self.jidx.append( idx )

    def BuildStrides( self ):
        '''Build cache-friendly Strides Array'''
        n = 3
        i = 1 + self.BI.shape[0] - 3
        j = 1 + self.BI.shape[1] - 3
        self.strides = stride_tricks.as_strided( self.BI, (i,j,n,n),
                                                 strides=2*self.BI.strides )


    def GetFirstPoint( self ):

        ## TODO :: fix inflow direction

        if self.first_point is not None:
            self.i0, self.j0 = self.first_point
            return None

        if self.start_from == 't':
            for i in xrange( self.BI.shape[0]-1, 0, -1 ):
                if np.all( self.BI[i,:] == 0 ): continue
                for j in xrange( 1, self.BI.shape[1]-1 ):
                    for primitive in self.primitives:
                        for iSide in xrange( 4 ):
                            seed = np.rot90( primitive, iSide )
                            if ( self.strides[i-1,j-1] == seed ).all():
                                self.i0, self.j0 = i, j
                                return None

        elif self.start_from == 'b':
            for i in xrange( 1, self.BI.shape[0] ):
                if np.all( self.BI[i,:] == 0 ): continue
                for j in xrange( 1, self.BI.shape[1]-1 ):
                    for primitive in self.primitives:
                        for iSide in xrange( 4 ):
                            seed = np.rot90( primitive, iSide )
                            if ( self.strides[i-1,j-1] == seed ).all():
                                self.i0, self.j0 = i, j
                                return None

        elif self.start_from == 'l':
            for j in xrange( 1, self.BI.shape[1]-1 ):
                if np.all( self.BI[:,j] == 0 ): continue
                for i in xrange( 1, self.BI.shape[0]-1 ):
                    for primitive in self.primitives:
                        for iSide in xrange( 4 ):
                            seed = np.rot90( primitive, iSide )
                            if ( self.strides[i-1,j-1] == seed ).all():
                                self.i0, self.j0 = i, j
                                return None

        elif self.start_from == 'r':
            for j in xrange( self.BI.shape[1]-1, 0, -1 ):
                if np.all( self.BI[:,j] == 0 ): continue
                for i in xrange( 1, self.BI.shape[0]-1 ):
                    for primitive in self.primitives:
                        for iSide in xrange( 4 ):
                            seed = np.rot90( primitive, iSide )
                            if ( self.strides[i-1,j-1] == seed ).all():
                                self.i0, self.j0 = i, j
                                return None

        raise IndexError, 'First Point Not Found!'    


    def NeiDist( self, idx1, idx2 ):
        '''Cartesian Distance between pixel cells'''
        i1, j1 = idx1
        i2, j2 = idx2
        return np.sqrt( (i1-i2)**2 + (j1-j2)**2 )


    def Vectorize( self, MAXITER=100000 ):

        '''Find Indexes and Points'''

        I, J = [ self.i0 ], [ self.j0 ]
        N = 0
        ijunct = 0 # Junction Index
        junct_found = False
        for ITER in xrange( MAXITER ):
            i0, j0 = I[-1], J[-1]            
            self.hits[i0,j0] = 0
            seed = self.hits[i0-1:i0+2, j0-1:j0+2]
            pos = zip( *np.where( seed > 0 ) )
            if len( pos ) == 0:
                break # End Point Found
            elif len( pos ) == 1:
                # Put Coordinates in Global Reference System
                i, j = pos[0]
                i += i0 - 1
                j += j0 - 1    
                I.append(i), J.append(j)
                N += 1
                self.offset = self.hits.shape[1] - j # for GeoReferencing
            elif len( pos ) > 1:
                jdist = self.NeiDist( pos[0], pos[1] ) 
                if len( pos ) == 2 and np.abs(jdist-1) < 1.e-08:
                    # Two Neighboring cells are found. Just take the closest one
                    dist = np.zeros( len(pos) )
                    for ipos, p in enumerate(pos):
                        dist[ipos] = np.sqrt( (1 - p[0])**2 + (1 - p[1])**2 )
                    idist = dist.argmin()
                    pos = [ pos[idist] ]
                    # Put Coordinates in Global Reference System
                    i, j = pos[0]
                    i += i0 - 1
                    j += j0 - 1
                    I.append(i), J.append(j)
                    N += 1
                    self.offset = self.hits.shape[1] - j # for GeoReferencing

                else: # We find a junction between two or more branches
                    # Recursively Compute the Longest Path to the end of the channel
                    # By Recursively Removing Branch Junction Points
                    print '   Found Channel Junction at ', i0, j0, 'n branches %d. ' % len( pos ), \
                        'Level of recursion: %d' % ( self.call_depth )

                    jncsl = np.zeros( len( pos ) ) # Total Lengths of the Following Branches at Junction
                    jncsw = np.zeros( len( pos ) ) # Average Width of the Following Branches at Junction
                    rdepths = np.zeros( len( pos ), dtype=int )
                    self.GetJunction( N )                    
                    jhits = self.hits.copy()
                    axijs = []

                    for ij in xrange( len(pos) ):
                        # For each of the Junctions is created a recursive instance
                        # with a maximum iteration number of 50 cells
                        # the one width the maximum average width is chosen
                        first_point = ( pos[ij][0]+i0-1, pos[ij][1]+j0-1 ) # Initial Point of the Local Branch
                        jhits[ pos[abs(ij-1)][0]+i0-1, pos[abs(ij-1)][1]+j0-1  ] = 0 # Remove the other ones
                
                        if self.method == 'width': ITER=200
                        else: ITER=MAXITER
                        
                        axr = AxisReader( self.I*jhits, first_point=first_point,
                                          verbose=True, method=self.method,
                                          call_depth=self.call_depth+1, jidx=self.jidx )
                        axij = axr( MAXITER=ITER )

                        axijs.append( axij )
                        jncsl[ij] = axij[2].size # Total Path Length
                        jncsw[ij] = axij[2].mean() # Total Path Average Width
                        rdepths[ij] = axr.call_depth # Total Level of Recursion of Class Instance

                    # Now Remove Narrower Branch Iteratively
                    if self.method == 'width':
                        for ij in xrange( len(pos)-1 ):
                            # Remove Narrower Branches from the Hit&Miss Matrix
                            self.hits[ axijs[jncsw.argmin()][1][0], axijs[jncsw.argmin()][0][0] ] = 0 # Delete Averagely Narrowest Path
                            jncsw = np.delete( jncsw, jncsw.argmin() )

                    elif self.method == 'length':
                        # Recursively Append the Following Reach
                        _J, _I, _ = axijs[ jncsl.argmax() ] # Longest Path
                        self.call_depth += rdepths[ jncsl.argmax() ]
                        I.extend( _I ), J.extend( _J )
                        break

                    elif self.method == 'std':
                        # Length Control
                        for ij in xrange( len(pos) ):
                            jmin = jncsl.argmin()
                            jmax = jncsl.argmax()
                            if jncsl[jmin]<0.75*jncsl[jmax]:
                                # If a branch is much shorter than another one, forget about it
                                jncsl = np.delete( jncsl, jmin )
                                jncsw = np.delete( jncsw, jmin )
                                axijs = np.delete( axijs, jmin )
                                rdepths = np.delete( rdepths, jmin )
                        # Take the Widest between the remaining branches
                        _J, _I, _ = axijs[ jncsw.argmax() ] # Widest Branch
                        self.call_depth = rdepths[ jncsw.argmax() ]
                        I.extend( _I ), J.extend( _J )
                        del axijs, axij, axr # Free some Memory
                        break

        if ITER == MAXITER-1 and self.verbose:
            print 'WARNING: Maximum number of iteration reached in axis extraction!'
        Xpx, Ypx = np.asarray( I ), np.asarray( J )
        Bpx = self.I[I, J]
        # For some reason they are inverted
        # TODO: check code
        return [ Ypx, Xpx, Bpx ]

    def __call__( self, MAXITER=100000 ):
        self.BuildStrides()
        self.GetFirstPoint()
        return self.Vectorize( MAXITER=MAXITER )

                


def ReadAxisLine( I, GeoTransf, flow_from=None, method='std' ):

    '''
    Convenience function for AxisReader class.
    Return a Line2D instance with width as attribute.
    
    Args
    ====
    I : Image array of the channel axis over black background
    GeoTransf : Dict Containing Lower Left Corner Coordinates and
                Pixel Size
                - X
                - Y
                - PixelSize
    Return
    ======
    Line2D with axis coordinates, intrinsic coordinate
    and local width distribution dictionary attributes
    '''
    
    r = AxisReader( I, start_from=flow_from, method=method )
    [ Xpx, Ypx, Bpx ] = r()
    print 'Axis Read with a Recursion Level of %s' % r.call_depth
    
    # Cut Borders (there are some issues sometimes)
    Xpx = Xpx[10:-10]
    Ypx = Ypx[10:-10]
    Bpx = Bpx[10:-10]

    # GeoReference
    # ------------
    GR = GeoReference( I, GeoTransf )
    X, Y = GR.RefCurve( Xpx, Ypx )
    B = Bpx * GeoTransf['PixelSize']

    line = Line2D( [ X, Y ] )
    line.attr( 'B', B )
    dx = np.ediff1d( X, to_begin=0 )
    dy = np.ediff1d( Y, to_begin=0 )
    ds = np.sqrt( dx**2 + dy**2 )
    s = np.cumsum( ds )
    line.attr( 's', s )
    line.attr( 'L', line.d['s'][-1] )

    return line




