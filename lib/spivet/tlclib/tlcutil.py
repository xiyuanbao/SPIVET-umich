"""
Filename:  tlcutil.py
Copyright (C) 2007-2010 William Newsome
 
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details, published at 
http://www.gnu.org/copyleft/gpl.html

/////////////////////////////////////////////////////////////////
Description:
  Module containing temperature field utilities.

Contents:
  tlcmask()
  xtblkhue()

"""

from numpy import *

from spivet import pivlib
from spivet.pivlib import pivutil
import tlctc
from spivet import compat

#################################################################
#
def tlcmask(
    dhsic,rbndx,bsize,bgth=0.01,sth=0.01,bcth=15.,hxrng=None,hxbpft=0.5,show=False
):
    """
    ----

    dhsic             # List of dewarped HSI channels [hue, sat, int].
    rbndx             # Analyzed region boundary index (2x2 array).
    bsize             # Block size ([y,x] pixels).
    bgth=0.01         # Intensity image background threshold.
    sth=0.01          # Saturation image threshold.
    bcth=15.          # Bubble convolution threshold.
    hxrng=None        # Hue range to exclude (2-element list).
    hxbpft=0.5        # Limiting fraction for hue exclusion.
    show=False        # Flag to show results.

    ----

    tlcmask() takes the hue, saturation, and intensity channels 
    of an image, breaks them into blocks of bsize for processing, 
    and creates a mask representing the location of liquid crystals.  
    Large bright regions are excluded as these are often bubbles 
    or other debris.
    
    Note that tlcmask returns an image the same size as the input 
    channels however particles will only be identified for the region 
    specified by rbndx.  The image border outside of rbndx will be set 
    to zero in the mask.

    For tlcmask(), the value of rbndx[:,0] can be (0,0), but the value 
    of bsize should be kept around (32,32).  rbndx will be subdivided
    into the largest possible number of integral blocks.

    Bubble exclusion works by convolving a normalized version of the 
    input intensity channel with a 5x5 kernel.  The resulting convolved 
    image is then thresholded to determine the locations of bubbles.  
    The thresholding is controlled by bcth which should be in the range 
    0.<= bcth <=21.  Small values of bcth will classify more input
    image features as bubbles.  Large values of bcth will only exclude
    larger items (up to a max of 5x5).  If bcth > 21., no bubbles
    will be excluded.

    After bubble exclusion is performed, candidate particles are
    removed if the intensity or saturation is less than bgth or sth,
    respectively.  Additional thresholding is performed on these channels
    using the channel statistics.  See code for details.

    Particles are then subjected to yet another test, hue exclusion.
    The motivation for hue exclusion follows from our lab setup where
    the working fluid is not colorless.  In this case, white secondary
    tracers used to provide PIV capabilities when the TLC's aren't 
    visible (ie, when the TLC's are too hot or too cold) don't appear
    white, but instead take on the color of the fluid.  Since these
    secondary tracers are still visible when the TLC's are, they need
    to be removed from valid TLC particles or the computed TLC hue
    for the block will be erroneous.  One final caveat: depending on
    the color of the secondary tracers, a situation exists where the
    color of the secondary tracers could match that of TLC's at a given
    temperature.  In this case, the secondary tracers should not be
    excluded from the valid particle mask (otherwise, valid TLC's will
    be thrown away as well).  With this motivation, the technique
    proceeds as follows.  If hxrng is not None, then any pixel with a 
    hue such that 
        hxrng[0] <= hue <= hxrng[1]
    will be excluded from valid particles provided that the fraction
    of pixels within the block having the exclusionary hue does not
    exceed hxbpft.  As an example, consider a standard 32 x 32 block
    having 1024 pixels and suppose that hxrng = [0.1,0.1], hxbpft = 0.5.
    If the number of pixels in the block with hue = 0.1 is greater than
    512 (hxbpft*1024), then no pixels will be excluded.

    If show = True, a plot is created of the mask at various stages
    of creation.

    Returns a mask of the same size as imin.  Mask value is one if over an
    area deemed to be a particle and zero otherwise.
    """
    import pylab

    # Initialization.
    chnlh = dhsic[0]
    chnls = dhsic[1]
    chnli = dhsic[2]

    imdim = chnls.shape
    rbndx = array(rbndx)

    pbsndx = zeros(2,dtype=int)
    pbendx = pbsndx.copy()

    [rsize,lbso,hbso,lnblks,hnblks] = \
        pivutil.getblkprm(rbndx, bsize, [0,0], 1)

    cmap = pylab.gray()

    hxmxp = hxbpft*bsize[0]*bsize[1]

    # Tuning parameters.
    #    ssigsf ----- Scale factor for saturation standard deviation.  Let
    #                 save represent the average saturation for the block and
    #                 ssig represent the block saturation standard deviation.
    #                 Any pixel in the block having a saturation less than
    #                     save + ssigsf*ssig
    #                 will be eliminated from the valid particle mask.
    #    iavesf ----- Scale factor for intensity average.  Let iave represent
    #                 the average intensity for the block.  Any pixel in the
    #                 block having an intensity less than iavesf*iave will
    #                 be eliminated from the valid particle mask.
    ssigsf = 0.1 
    iavesf = 0.1

    # Grab a copy of the input image and set border to black.
    mci = zeros(imdim,dtype=float)
    mci[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]] = \
        chnli[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]]

    if ( show ):
        pylab.figure()
        pylab.subplot(221)
        pylab.imshow(mci,cmap=cmap)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Phase 0: Raw Image')

    # Get bubble mask.
    [bmci,bmask] = pivutil.bblmask(mci,rbndx,bsize,bgth,bcth)

    # Invert mask and apply to modified image.
    bmask = 1. -bmask
    mci   = bmask*mci
    
    if ( show ):
        pylab.subplot(222)
        pylab.imshow(bmask,cmap=cmap)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Phase 1: Large Bubbles (Black)')

        pylab.subplot(223)
        pylab.imshow(mci,cmap=cmap)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Phase 2: Particle Finder Input')

    # Create particle mask.
    for m in range(lnblks[0]):
        pbsndx[0] = rbndx[0,0] +m*lbso[0]
        pbendx[0] = pbsndx[0] +bsize[0]
        for n in range(lnblks[1]):
            pbsndx[1] = rbndx[1,0] +n*lbso[1]
            pbendx[1] = pbsndx[1] +bsize[1]

            sblock = chnls[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
            ssth   = max( sblock.mean() +ssigsf*sblock.std(), sth )

            iblock = mci[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
            sbgth  = max( iavesf*iblock.mean(), bgth )

            iblock = where(iblock<sbgth,0.,1.)            
            iblock = where(sblock<ssth,0.,iblock)

            if ( not compat.checkNone(hxrng) ):
                hblock = chnlh[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
                msk    = ( hblock >= hxrng[0] )*( hblock <= hxrng[1] )
                
                if ( msk.sum() <= hxmxp ):
                    iblock = (-msk)*iblock

            mci[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]] = iblock

    if ( show ):
        pylab.subplot(224)
        pylab.imshow(mci,cmap=cmap)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Phase 3: Particle Mask (White)')            

    return mci


#################################################################
#
def xtblkhue(
    chnlh, pmask, rbndx, nblks, bsize, bso 
):
    """
    ----

    chnlh             # Hue channel.
    pmask             # Valid particle mask from tlcmask().
    rbndx             # Analyzed region boundary index (2x2 array).
    nblks             # Number of blocks ([y,x]).
    bsize             # Block size ([y,x] pixels).
    bso               # Block start offset ([y,x] pixels).

    ----

    Extracts hue values from the hue channel by subdividing the channel
    into blocks and computing the average hue using only valid particles.

    bso should be set equal to bsize -bolap

    Returns [bh,hinac].  s,t below are the number of blocks in the y and
    x directions, respectively.
        bh ----- s x t array containing the computed hue values.
        hinac -- s x t array containing the hue inaccuracy flag.  Will be
                 set to 1 if no valid particles are found within the
                 block.
    """
    # Initialization. 
    rbndx = array(rbndx)

    pbsndx = zeros(2,dtype=int)
    pbendx = pbsndx.copy()

    bsoy = bso[0]
    bsox = bso[1]

    nyblks = nblks[0]
    nxblks = nblks[1]

    # Get the hue values.
    bh    = zeros((nyblks,nxblks),dtype=float)
    hinac = zeros((nyblks,nxblks),dtype=int)
    for m in range(nyblks):
        pbsndx[0] = rbndx[0,0] +m*bsoy
        pbendx[0] = pbsndx[0] +bsize[0]
        for n in range(nxblks):
            pbsndx[1] = rbndx[1,0] +n*bsox
            pbendx[1] = pbsndx[1] +bsize[1]

            pmblock = pmask[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
            hblock  = chnlh[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
            
            msk = pmblock > 0.
            if ( not msk.any() ):
                hinac[m,n] = 1
            else:
                msk    = msk.reshape(msk.size)
                hblock = hblock.reshape(hblock.size)

                hblock = compress(msk,hblock)

                bh[m,n]    = hblock.mean()
                hinac[m,n] = 0

    return [bh,hinac]

