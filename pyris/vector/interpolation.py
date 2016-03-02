from __future__ import division
import numpy as np
from scipy import interpolate

def InterpPCS( x, y, N=1000, s=None, with_derivatives=True, k=3 ):
    '''
    PCS(x, y, N=1000, s=1) - Apply Parametric Cubic Spline Interpolation
    to the centerline axis of a meandering channel, in order to obtain
    a continuous representation
    '''
    tckp, u = interpolate.splprep( [x, y], s=s, k=k ) # Parametric Representation
    u_PCS = np.linspace( 0, 1, N ) # ArcLength
    x_PCS, y_PCS = interpolate.splev( u_PCS, tckp, der=0 )
    if with_derivatives:
        d1x_PCS, d1y_PCS = interpolate.splev( u_PCS, tckp, der=1 )
        d2x_PCS, d2y_PCS = interpolate.splev( u_PCS, tckp, der=2 )
        return x_PCS, y_PCS, d1x_PCS, d1y_PCS, d2x_PCS, d2y_PCS
    else:
        return x_PCS, y_PCS

def CurvaturePCS( *args, **kwargs ):
    '''
    Compute curvature of the PCS.
    Methods:
     * 1 - Finite differences
     * 2 - Guneralp and Rhoads 2007 (requires spline derivatives)
     * 3 - Schwenk et al. 2015
    '''
    method = kwargs.pop( 'method', 1 )
    return_diff = kwargs.pop( 'return_diff', False )
    apply_filter = kwargs.pop( 'apply_filter', True )
    x = args[0]
    y = args[1]
    dx = np.ediff1d( x, to_begin=0 )
    dy = np.ediff1d( y, to_begin=0 )
    ds = np.sqrt( dx**2 + dy**2 )
    s = np.cumsum( ds )
    theta = np.arctan2( dy, dx )
    for i in xrange(1,theta.size):
        if theta[i] - theta[i-1] > np.pi: theta[i] -= 2*np.pi
        elif theta[i] - theta[i-1] < -np.pi: theta[i] += 2*np.pi
    if method == 1:
        Cs = -np.gradient( theta, np.gradient(s) )
    elif method == 2:
        d1x = args[2]
        d1y = args[3]
        d2x = args[4]
        d2y = args[5]
        Cs = - ( d1x*d2y - d1y*d2x ) / ( d1x**2 + d1y**2 )**(3/2)
    elif method == 3:
        ax = x[1:-1] - x[:-2]
        bx = x[2:] - x[:-2]
        cx = x[2:] - x[1:-1]
        ay = y[1:-1] - y[:-2]
        by = y[2:] - y[:-2]
        cy = y[2:] - y[1:-1]
        Cs = 2 * (ay*bx - ax*by) / \
          np.sqrt( (ax**2+ay**2) * (bx**2+by**2) * (cx**2+cy**2) )
        # Fix First and Last Point with finite difference
        Cs = np.concatenate((np.array( -(theta[1]-theta[0])/(s[1]-s[0] ) ), Cs,
                             np.array( -(theta[-1]-theta[-2])/(s[-1]-s[-2]) )))
    if apply_filter:
        Cs[1:-1] = (Cs[:-2] + 2*Cs[1:-1] + Cs[2:]) / 4
    if return_diff:
        return s, theta, Cs, dx, dy, ds
    else:
        return s, theta, Cs

    
def WidthPCS( s, B, sPCS, kind='linear' ):
    f = interpolate.interp1d( s, B, kind=kind )
    return f( sPCS )
