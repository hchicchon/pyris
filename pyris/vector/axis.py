from __future__ import division
import numpy as np
from scipy import interpolate
from numpy.lib import stride_tricks
from skimage.measure import regionprops
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

        # Skeleton
        self.I = I
        self.I[0,:] = 0
        self.I[:,0] = 0
        self.I[-1,:] = 0
        self.I[:,-1] = 0
        self.hits = np.where( I>0, 1, 0 ).astype( int ) # Hit & Miss Matrix
        self.first_point = first_point # Initial Point if known (used in recursion)
        self.start_from = 'b' if start_from is None else start_from # Where flow comes from
        self.method = method # Method used for multithread reaches
        self.verbose = verbose
        self.call_depth = call_depth # Level of recursion
        self.jidx = jidx # Indexes of multothread junctions
        [ xl, yl, xr, yr ] = [ int(b) for b in regionprops( self.hits )[0].bbox ]
        self.Lmin = int( min( abs(xr-xl), abs(yr-yl) ) ) # Maximum cartesian size of the channel in pixels

    def GetJunction( self, idx ):
        '''List of multithread junctions indexes'''
        if len( self.jidx ) > 0: idx += self.jidx[-1]
        self.jidx.append( idx )

    def BuildStrides( self ):
        '''Build cache-friendly Strides Array'''
        n = 3
        i = 1 + self.hits.shape[0] - 3
        j = 1 + self.hits.shape[1] - 3
        return stride_tricks.as_strided( self.hits, (i,j,n,n), strides=2*self.hits.strides )

    def GetFirstPoint( self ):
        '''Look for a 3x3 primitive in the image corresponding to the channel starting point'''

        if self.first_point is not None:
            self.i0, self.j0 = self.first_point
            return None

        strides = self.BuildStrides()

        if self.start_from == 't':
            for i in xrange( self.hits.shape[0]-1, 0, -1 ):
                if np.all( self.hits[i,:] == 0 ): continue
                for j in xrange( 1, self.hits.shape[1]-1 ):
                    for primitive in self.primitives:
                        for iSide in xrange( 4 ):
                            seed = np.rot90( primitive, iSide )
                            if ( strides[i-1,j-1] == seed ).all():
                                self.i0, self.j0 = i, j
                                return None

        elif self.start_from == 'b':
            for i in xrange( 1, self.hits.shape[0] ):
                if np.all( self.hits[i,:] == 0 ): continue
                for j in xrange( 1, self.hits.shape[1]-1 ):
                    if self.hits[i,j] == 0: continue
                    for primitive in self.primitives:
                        for iSide in xrange( 4 ):
                            seed = np.rot90( primitive, iSide )
                            if ( strides[i-1,j-1] == seed ).all():
                                self.i0, self.j0 = i, j
                                return None

        elif self.start_from == 'l':
            for j in xrange( 1, self.hits.shape[1]-1 ):
                if np.all( self.hits[:,j] == 0 ): continue
                for i in xrange( 1, self.hits.shape[0]-1 ):
                    for primitive in self.primitives:
                        for iSide in xrange( 4 ):
                            seed = np.rot90( primitive, iSide )
                            if ( strides[i-1,j-1] == seed ).all():
                                self.i0, self.j0 = i, j
                                return None

        elif self.start_from == 'r':
            for j in xrange( self.hits.shape[1]-1, 0, -1 ):
                if np.all( self.hits[:,j] == 0 ): continue
                for i in xrange( 1, self.hits.shape[0]-1 ):
                    for primitive in self.primitives:
                        for iSide in xrange( 4 ):
                            seed = np.rot90( primitive, iSide )
                            if ( strides[i-1,j-1] == seed ).all():
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

        I, J = [ self.i0 ], [ self.j0 ] # Lists of channel points
        N = 0 # Counter
        ijunct = 0 # Junction Index

        for ITER in xrange( MAXITER ):
            i0, j0 = I[-1], J[-1] # Previous Point
            self.hits[i0,j0] = 0 # Set it to 0 in the Hit&Miss Matrix
            seed = self.hits[i0-1:i0+2, j0-1:j0+2] # 3x3 neighboring element
            pos = zip( *np.where( seed > 0 ) ) # Positive neighbors

            if len( pos ) == 0: # End Point of the channel found
                break

            elif len( pos ) == 1: # Next Point identified
                i, j = pos[0]
                i += i0 - 1 # Reference system
                j += j0 - 1 # Reference system    
                I.append(i), J.append(j)
                N += 1

            elif len( pos ) > 1: # More neighboring points
                jdist = self.NeiDist( pos[0], pos[1] )
                if len( pos ) == 2 and np.abs(jdist-1) < 1.e-08:
                    # Two Neighboring cells are found
                    # Pattern:
                    #          - * *    o=current cell
                    #          - o -    *=neighboring cells
                    #          - - -    -=0 cells
                    dist = np.zeros( len(pos) )
                    # Choose the closest positive cell
                    for ipos, p in enumerate(pos):
                        dist[ipos] = np.sqrt( (1 - p[0])**2 + (1 - p[1])**2 )
                    idist = dist.argmin()
                    pos = [ pos[idist] ]
                    i, j = pos[0]
                    i += i0 - 1
                    j += j0 - 1
                    I.append(i), J.append(j)
                    N += 1

                else: # Multithread channel junction
                    if self.call_depth==0:
                        print 'channel junction at ', i0, j0, 'n branches %d - ' % len( pos ), \
                            'starting recursion (this may require some time)...'
                    elif self.call_depth > 0 and self.verbose:
                        print 'channel junction at ', i0, j0, 'n branches %d - ' % len( pos ), \
                            'level of recursion: %d' % ( self.call_depth )

                    jncsl = np.zeros( len( pos ) ) # Total Lengths of the Following Branches at Junction
                    jncsw = np.zeros( len( pos ) ) # Average Width of the Following Branches at Junction
                    rdepths = np.zeros( len( pos ), dtype=int )
                    self.GetJunction( N )                    
                    axijs = []

                    for ij in xrange( len(pos) ):
                        # For each of the Junctions is created a recursive instance
                        first_point = ( pos[ij][0]+i0-1, pos[ij][1]+j0-1 ) # Initial Point of the Local Branch
                         # Temporally remove other branches
                        removed_indexes = [ ij-1, (ij+1)%len(pos) ]
                        for idx in removed_indexes: self.hits[ pos[idx][0]+i0-1, pos[idx][1]+j0-1 ] = 0
                        # Set the maximum number of iteration
                        if self.method == 'fast': ITER = self.Lmin
                        else: ITER = MAXITER                        
                        # Recursive call
                        axr = AxisReader( self.I*self.hits, first_point=first_point, method=self.method,
                                          call_depth=self.call_depth+1, jidx=self.jidx )
                        axij = axr( MAXITER=ITER )

                        # Set back the other branches
                        for idx in removed_indexes: self.hits[ pos[idx][0]+i0-1, pos[idx][1]+j0-1 ] = 1

                        axijs.append( axij )
                        jncsl[ij] = axij[2].size # Total Path Length
                        jncsw[ij] = axij[2].mean() # Total Path Average Width
                        rdepths[ij] = axr.call_depth # Total Level of Recursion of Class Instance

                    if self.method == 'std':
                        # Length Control
                        for ij in xrange( len(pos) ):
                            jmin = jncsl.argmin()
                            jmax = jncsl.argmax()
                            if jncsl[jmin]<0.75*jncsl[jmax]:
                                # If a branch is much shorter than another one, forget about it
                                del axijs[ jmin ] # This is a list
                                jncsl = np.delete( jncsl, jmin )
                                jncsw = np.delete( jncsw, jmin )
                                rdepths = np.delete( rdepths, jmin )
                        # Take the Widest between the remaining branches
                        _J, _I, _ = axijs[ jncsw.argmax() ] # Widest Branch
                        self.call_depth = rdepths[ jncsw.argmax() ]
                        I.extend( _I ), J.extend( _J )
                        del axijs, axij, axr # Free some Memory
                        break

                    elif self.method == 'fast':
                        if np.all( jncsl ) < MAXITER: # Planform is finished
                            _J, _I, _ = axijs[ jncsw.argmax() ] # Take the widest branch
                            self.call_depth = rdepths[ jncsw.argmax() ]
                            I.extend( _I ), J.extend( _J )
                            del axijs, axij, axr # Free some memory
                            break
                        elif np.any( jncsl ) < MAXITER: # Some of the branches are shorter
                            idxs_to_rm = []
                            # Look for the short branches
                            for idx in xrange( len( pos ) ):
                                if jncsl[idx] < MAXITER:
                                    self.hits[ axijs[idx][1][0], axijs[idx][0][0] ] = 0
                                    idxs_to_rm.append( idx )
                            # Remove the short branches
                            axijs = [ _i for _j, _i in enumerate(axijs) if not _j in idxs_to_rm ]
                            jncsl = np.delete( jncsl, idxs_to_rm )
                            jncsw = np.delete( jncsw, idxs_to_rm )
                            rdepths = np.delete( rdepths, idxs_to_rm )
                        # Remove from Hit&Miss the main branch
                        self.hits[ axijs[jncsw.argmax()][1], axijs[jncsw.argmax()][0] ] = 0
                        # Now append
                        _J, _I, _ = axijs[ jncsw.argmax() ] # Take the widest branch
                        self.call_depth = rdepths[ jncsw.argmax() ]
                        I.extend( _I ), J.extend( _J )
                        del axijs, axij, axr # Free some memory
                        continue

        if ITER == MAXITER-1 and not self.method == 'fast':
            print 'WARNING: Maximum number of iteration reached in axis extraction!'
        Xpx, Ypx = np.asarray( I ), np.asarray( J )
        Bpx = self.I[I, J]
        # TODO: check code - For some reason they are inverted
        return [ Ypx, Xpx, Bpx ]

    def __call__( self, MAXITER=100000 ):
        self.GetFirstPoint()
        return self.Vectorize( MAXITER=MAXITER )

                


def ReadAxisLine( I, flow_from=None, method='std' ):

    '''
    Convenience function for AxisReader class.
    Return a Line2D instance with width as attribute.
    
    Args
    ====
    I : Image array of the channel axis over black background
    Return
    ======
    Line2D with axis coordinates, intrinsic coordinate
    and local width distribution dictionary attributes
    '''
    
    r = AxisReader( I, start_from=flow_from, method=method )
    [ Xpx, Ypx, Bpx ] = r()
    print 'axis read with a recursion level of %s' % r.call_depth
    
    # Pixelled Line
    # -------------
    line = Line2D( x=Xpx, y=Ypx, B=B )
    return line

    ## # GeoReferenced Line
    ## # ------------------
    ## GR = GeoReference( I, GeoTransf )
    ## X, Y = GR.RefCurve( Xpx, Ypx )
    ## B = Bpx * GeoTransf['PixelSize']
    ## geoline = Line2D( [ X, Y ] )
    ## geoline.attr( 'B', B )
    ## dx = np.ediff1d( X, to_begin=0 )
    ## dy = np.ediff1d( Y, to_begin=0 )
    ## ds = np.sqrt( dx**2 + dy**2 )
    ## s = np.cumsum( ds )
    ## geoline.attr( 's', s )
    ## geoline.attr( 'L', line.d['s'][-1] )
    #return line, geoline




