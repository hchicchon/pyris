from __future__ import division
import numpy as np
from skimage import morphology as mm
from ..misc import BW, isBW


def CleanIslands( bw, size, conn=1 ):
    '''CleanIslands( bw, size, conn=1 ) - Remove Islands inside channel'''
    if not isBW( bw ):
        raise TypeError, \
          'Input must be a Black&White image'
    return BW( mm.remove_small_objects( bw==False, size, conn ) == False )


def RemoveSmallObjects( bw, size, conn=1 ):
    '''RemoveSmallObjects( bw, size, conn=1 )'''
    if not isBW( bw ):
        raise TypeError, \
          'Input must be a Black&White image'
    return BW( bw * mm.remove_small_objects(
        bw.astype(bool), size, conn ).astype(np.uint8) )

    
def Skeletonize( bw ):
    '''Combine Skeletonization with Isles Removal.'''
    if not isBW( bw ):
        raise TypeError, \
            'Input must be a Black&White image'
    skel, dist = mm.medial_axis( bw, return_distance=True )
    skel = mm.skeletonize( bw ) # Overwrite with Skeletonize Algorithm
    skeldist = skel*dist
    return skel, dist, skeldist
