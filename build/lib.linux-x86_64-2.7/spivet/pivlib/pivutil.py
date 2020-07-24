"""
Filename:  pivutil.py
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
  Module containing utility routines for PivLIB.

Contents:
  Classes:
    tpswarp
  
  Functions:
    bblmask()
    bimedfltr()
    bldfltr()
    bldgbkern()
    ccxtrct()
    constrch()
    crcht()
    drwcrc()
    drwlin()
    esttpswarp()
    getblkprm()
    getpivcmap()
    gfit()
    grad_fo()
    grad_socd()
    imread()
    imshift()
    imtpswarp()
    imwrite()
    linht()
    padfltr()
    padimg()
    pkfind()
    prtmask()
    pxshift()
    rgb2hsi()
    rmvstat()

"""

from PIL import Image, ImageFilter, ImageOps
import pivlibc, pivtpsc
import string
from numpy import *
from scipy import linalg, ndimage, stats
from spivet import compat

class tpswarp():
    """
    The tpwswarp class represents a coordinate mapping using
    thin-plate splines.  The approach is that of Bookstein:1989.
    """ 
    def __init__(self,tfpts,sfpts,lmda=0.):
        """
        ----
        
        tfpts      # lx2 array of [y,x] template image feature points.
        sfpts      # lx2 array of [y,x] search image feature points.
        lmda=0.    # Regularization parameter.
        
        ----
        
        tfpts is an lx2 array of template image feature points.  The 
        template image can be thought of as Frame 1.  sfpts is an lx2
        array of search image feature points.  The search image can be
        thought of as Frame 2.  The constructed tpswarp objected will
        then map coordinates from the template frame to the search frame.
        l represents the number of points, and all points must be ordered
        as [y,x].
        
        lmda specifies a regularization parameter for the computation
        of the TPS warp.  The larger lmda, the smoother the the TPS warp
        will be.  If lmda = 0., the TPS warp will be an exact interpolation
        of the feature points.
        """
        if ( tfpts.shape[0] != sfpts.shape[0] ):
            raise ValueError("Number of tfpts and sfpts must be equal.")

        # Store the template points.
        tfpts = array(tfpts)
        if ( tfpts.size == 2 ):
            tfpts = tfpts.reshape((1,2))
        npts = tfpts.shape[0]
        
        self.m_kn    = npts
        self.m_tfpts = tfpts.copy()
        self.m_sfpts = sfpts.copy()
    
        # Build the forward system matrix.
        lmat  = self.__bldtpsmat__(tfpts,lmda,True)
        ilmat = linalg.inv(lmat)
        
        bvec = zeros((npts+3,2))
        bvec[0:npts,:] = sfpts
        
        self.m_wghts = dot(ilmat,bvec)
        
        # Build the inverse system matrix.  Note, this is not the true
        # inverse.  See documentation for ixfrm() and aixfrm().
        lmat  = self.__bldtpsmat__(sfpts,lmda,False)
        ilmat = linalg.inv(lmat)
        
        bvec = zeros((npts+3,2))
        bvec[0:npts,:] = tfpts
        
        self.m_iwghts = dot(ilmat,bvec)

    def __bldtpsmat__(self,pts,lmda,frwd=True):
        """
        ----
        
        pts        # lx2 array of [y,x] points.
        lmda       # Regularization parameter.
        frwd=True  # Flag indicating how matrix should map coordinates.
        
        ----
        
        Constructs the system matrix for the thin-plate spline.  This
        matrix is matrix L in Bookstein:1989.
        
        If frwd is True, then the L matrix can be used to map template
        coordinates to search coordinates.  If frwd is False, the reverse
        mapping is returned.
        
        If l = kn, where kn is the number of points used to create the
        tpswarp object, then lmda specifies a regularization parameter.
        lmda will be ignored otherwise.
        
        Returns lmat.
        """
        if ( frwd ):
            lmat = pivtpsc.bldtpsmat(pts,lmda,self.m_tfpts,self.m_kn)
        else:
            lmat = pivtpsc.bldtpsmat(pts,lmda,self.m_sfpts,self.m_kn)
        
        return lmat

    def aixfrm(self,pts):
        """
        ----
        
        pts        # lx2 array of [y,x] points.
        
        ----
        
        Computes an approximation to the inverse warp (ie, from search to 
        template).  The inverse warp should be used to compute coordinates 
        for warping the template image using pivutil.pxshift().
        
        An exact inverse of the forward warp is not analytically tractable.
        Instead the true inverse must be computed using an iterative 
        procedure (eg, Newton's method) to solve for each template point
        that minimizes the following error for the specified pts[i]
            e = | pts[i] -xfrm(tfpts) |^2
        
        The method used here, however, is to approximate the inverse 
        transform.  For a given set of template and search feature points, 
        tfpts and sfpts, an approximate inverse warp should be that given 
        by creating a tpswarp instance with sftps sent in as the tfpts
        and vice versa.
        
        Returns xpts, an lx2 array of inverse warped [y,x] points.
        """
        lmat = self.__bldtpsmat__(pts, 0.,False)
        npts = lmat.shape[0] -3
        
        xpts = dot(lmat[0:npts,:],self.m_iwghts)

        return xpts
    
    def ixfrm(self,pts):
        """
        ----
        
        pts        # lx2 array of [y,x] points.
        
        ----
        
        Computes the inverse warp (ie, from search to template).  The 
        inverse warp should be used to compute coordinates for warping the 
        template image using pivutil.pxshift().
        
        An exact inverse of the forward warp is not analytically tractable.
        Instead the true inverse must be computed using an iterative 
        procedure (eg, Newton's method) to solve for each template point
        that minimizes the following error for the specified pts[i]
            e = | pts[i] -xfrm(tfpts) |^2
        
        This function uses Newton's method to compute a better 
        approximation to the inverse mapping of search points to template 
        points.
        
        Returns xpts, an lx2 array of inverse warped [y,x] points.
        """
        # Initialization.
        pts = array(pts)
        if ( pts.size == 3 ):
            pts = pts.reshape((1,3))
        npts   = pts.shape[0]        
        
        maxits = 10
        eps    = 1.e-3
        
        # Get an approximate inverse.
        tpts = self.aixfrm(pts)
        
        # Main loop.  Let f() represents xfrm(), and f()[*] 
        # represent xfrm()[*].  g represents the following
        # system
        #     g[i] = (pts[:,0] -f(tpts)[:,0])*df(tpts)[:,0]d[i] 
        #           +(pts[:,1] -f(tpts)[:,1])*df(tpts)[:,1]d[i]
        # where df()[*]d[i] represents the partial derivative of f()[*]
        # with respect to the i'th axis (starting at y).
        #jmat = empty((2,2),dtype=float)
        jmat = zeros(6*npts,dtype=float)
        gvec = empty(2*npts,dtype=float)
        for i in range(maxits):
            spts    = self.xfrm(tpts)
            
            spts_py = self.xfrm(tpts +array([1,0]))
            spts_px = self.xfrm(tpts +array([0,1]))
        
            dfydy = spts_py[:,0] -spts[:,0]
            dfydx = spts_px[:,0] -spts[:,0]
            dfxdy = spts_py[:,1] -spts[:,1]
            dfxdx = spts_px[:,1] -spts[:,1]
        
            err = pts -spts
        
            g1 = (err[:,0])*dfydy +(err[:,1])*dfxdy
            g2 = (err[:,0])*dfydx +(err[:,1])*dfxdx

            # This is an approximation to the derivatives of g.  
            err_py = pts -spts_py
            err_px = pts -spts_px
            
            g1_py = (err_py[:,0])*dfydy +(err_py[:,1])*dfxdy
            g1_px = (err_px[:,0])*dfydy +(err_px[:,1])*dfxdy
            g2_py = (err_py[:,0])*dfydx +(err_py[:,1])*dfxdx
            g2_px = (err_px[:,0])*dfydx +(err_px[:,1])*dfxdx
                    
            # Build the 2x2 Jacobians.  Note that a band solver is
            # used.  The equivalent for-loop over pts follows. 
            """
            hleps = True
            for p in range(npts):
                jmat[0,0] = g1_py[p] -g1[p]
                jmat[0,1] = g1_px[p] -g1[p]
                jmat[1,0] = g2_py[p] -g2[p]
                jmat[1,1] = g2_px[p] -g2[p]
            
                h = linalg.solve(jmat,-array([g1[p],g2[p]]))
            
                if ( (h*h).sum() > eps ):
                    hleps = False
            
                tpts[p,:] = tpts[p,:] +h
            """
            jmat[1::6] = g1_py -g1  # jmat[0,0]
            jmat[2::6] = g2_py -g2  # jmat[1,0]
            jmat[3::6] = g1_px -g1  # jmat[0,1]
            jmat[4::6] = g2_px -g2  # jmat[1,1]
    
            gvec[0::2] = -g1
            gvec[1::2] = -g2
            
            h = linalg.solve_banded((1,1),
                                    jmat.reshape((2*npts,3)).transpose(),
                                    gvec)
            
            h    = h.reshape((npts,2))
            tpts = tpts +h
            
            h = (h*h).sum(1)
                
            if ( ( h < eps ).all() ):
                break
            
        return tpts
        

    def xfrm(self,pts):
        """
        ----
        
        pts        # lx2 array of [y,x] points.
        
        ----
        
        Computes the forward warp (ie, from template to search).  The
        forward warp should be used to compute coordinates for de-warping
        the search image using pivutil.pxshift().
        
        Returns xpts, an lx2 array of warped [y,x] points.
        """
        lmat = self.__bldtpsmat__(pts, 0.,True)
        npts = lmat.shape[0] -3
        
        xpts = dot(lmat[0:npts,:],self.m_wghts)

        return xpts


#################################################################
#
def bblmask(imin,rbndx,bsize,bgth=0.3,bcth=10.):
    """
    ----
    
    imin           # Input image (greyscale mxn array).
    rbndx          # Analyzed region boundary index (2x2 array).
    bsize          # Block size ([y,x] pixels).
    bgth=0.3       # Background threshold.
    bcth=10.       # Bubble convolution threshold.
   
    ----
    
    Identifies location of large, contiguous intensity highs 
    (usually bubbles) in an intensity image.

    Note that bblmask returns an image the same size as imin, however
    bubbles will only be identified for the region specified by
    rbndx.  The image border outside of rbndx will be set to zero
    in the mask.

    Prior to any processing of the input image, the image is
    thresholded by bgth to remove any background noise.  If imin < bgth,
    the pixel will be set to 0.

    Bubble exclusion works by convolving a normalized version of the 
    input image with a 5x5 kernel.  The resulting convolved image is
    then thresholded to determine the locations of bubbles.  The 
    thresholding is controlled by bcth which should be in the range 
    0.<= bcth <=21.  Small values of bcth will classify more input
    image features as bubbles.  Large values of bcth will only exclude
    larger items (up to a max of 5x5).  If bcth > 21., no bubbles
    will be excluded.

    Returns [mim,bmask] where
        mim --- Modified input image of same size as imin.  Image
                within rbndx will be background thresholded and mean
                intensity will be removed.
        bmask - Bubble mask of same size as imin.  Values of 1.0 represent
                bubbles.
    """

    # Initialization.
    imdim = imin.shape
    rbndx = array(rbndx)

    pbsndx = zeros(2,dtype=int)
    pbendx = pbsndx.copy()

    [rsize,lbso,hbso,lnblks,hnblks] = \
        getblkprm(rbndx,bsize,[0,0],1)

    ck = ones((5,5),dtype=float)
    ck[0,0] = ck[0,4] = ck[4,0] = ck[4,4] = 0.

    # Floor the background.
    mim = zeros(imdim,dtype=float)
    mim[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]] = \
        imin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]]

    mim = where(mim<bgth,0.,mim)

    # Normalize the intensity image prior to convolution.
    for m in range(lnblks[0]):
        pbsndx[0] = rbndx[0,0] +m*lbso[0]
        pbendx[0] = pbsndx[0] +bsize[0]
        for n in range(lnblks[1]):
            pbsndx[1] = rbndx[1,0] +n*lbso[1]
            pbendx[1] = pbsndx[1] +bsize[1]

            block = mim[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
            bave  = block.mean()

            block = block -bave
            block = where(block<0.,0.,block)

            bmax = block.max()
            if (bmax > 0.):
                block = block/bmax

            mim[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]] = block

    # Locate large bubbles.
    cnv = ndimage.convolve(mim,ck,mode='constant')
    cnv = where(cnv>bcth,1.,0.)
    cnv = ndimage.convolve(cnv,ck,mode='constant')
    cnv = where(cnv>0.1,1.,0.)

    return [mim,cnv]


#################################################################
#
def bimedfltr(imin,rbndx,bsize,rthsf=2.):
    """
    ----
    
    imin           # Greyscale, floating point input image
    rbndx          # Analyzed region boundary index (2x2 array).
    bsize          # Block size ([y,x] pixels).
    rthsf=2.       # Residual threshold scale factor.
    
    ----
        
    Breaks an image into blocks of size bsize, then applies a
    median filter to each block.  This median filter is very
    effective at removing salt and pepper noise from an image.

    The median filter application is an adaptation of the method
    used in pivpost (the medfltr() function which is based off of
    an outlier detection scheme of Westerweel).  The algorithm
    functions as follows:
        - Image region specified by rbndx is broken into blocks
          of size bsize.
        - The median of a block is computed.
        - The residuals, res, for each pixel within the block are
          computed as: res = abs( block - median(block) ).
        - The median of the residuals is computed.
        - Any pixel residual within the block that exceeds the median 
          residual by a factor of rthsf is replaced by the block median.

    Returns mfim, the median filtered image.
    """
    # Initialization.
    imdim = imin.shape
    rbndx = array(rbndx)

    pbsndx = zeros(2,dtype=int)
    pbendx = pbsndx.copy()

    [rsize,lbso,hbso,lnblks,hnblks] = getblkprm(rbndx,bsize,[0,0],1)

    mfim = zeros(imdim,dtype=float)
    frac = zeros(lnblks,dtype=float)

    rnp = 1./(bsize[0]*bsize[1])

    # Filter the image.
    for m in range(lnblks[0]):
        pbsndx[0] = rbndx[0,0] +m*lbso[0]
        pbendx[0] = pbsndx[0] +bsize[0]
        for n in range(lnblks[1]):
            pbsndx[1] = rbndx[1,0] +n*lbso[1]
            pbendx[1] = pbsndx[1] +bsize[1]

            block = imin[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]

            med  = median(block.reshape(block.size))
            res  = abs(block -med)
            rmed = median(res.reshape(res.size))

            msk = res > rthsf*rmed
            block[msk] = med

            frac[m,n] = rnp*msk.sum()

            mfim[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]] = block

    print "bimedfltr - Average fraction filtered: %g" % ( frac.mean() )
    return mfim
    

#################################################################
#
def bldfltr(bsize,fnf):
    """
    ----

    bsize          # Block size of filter.
    fnf            # Cutoff frequency of filter.
    
    ----
    
    Consructs a symmetric gaussian filter with standard deviation 
    of fnf.

    fnf is the cutoff frequency as a fraction of the the Nyquist
    frequency.  An ideal filter (ie, the step function) that has
    no impact on the data would have an fnf of 1.0.  An fnf of
    0.0 would obliterate the entire data set.

    Returns a filter with dimensions of bsize.
    """

    # Initialization.
    fltr = zeros((bsize[0],bsize[1]), dtype=float)

    bcoy = bsize[0]/2.
    bcox = bsize[1]/2.

    rn = indices((bsize[0],bsize[1]))
    ry = rn[0,:,:]
    rx = rn[1,:,:]

    ry = ry.reshape(ry.size)
    rx = rx.reshape(rx.size)

    cry = (ry -bcoy)/bcoy
    crx = (rx -bcox)/bcox

    r = sqrt(cry*cry +crx*crx)

    fltr[ry,rx] = exp(-r**2/(2.*fnf**2))

    return fltr


#################################################################
#
def bldgbkern(gbsd):
    """
    ----
    
    gbsd           # Gaussian standard deviation along each axis.
    
    ----

    Utility function to create an n-dimensional gaussian blur kernel 
    for spatial domain processing.
    
    gbsd is a list of standard deviations, one for each axis.  

    The kernel will have dimensions of 

        2*int( 2.5*gbsd.max() ) +1 

    along all axes.
    
    Returns a kernel.
    """
    # Initialization.
    try:
        ndim = len(gbsd)
    except:
        gbsd = [gbsd]

    gbsd = array(gbsd)
    ndim = gbsd.size

    # Determine dimensions of the kernel.
    kcen = int(2.5*gbsd.max())
    kdim = 2*kcen +1
    kdim = array([kdim]).repeat(ndim)

    ndxm        = indices(kdim,dtype=float) -kcen
    ndxm[0,...] = ndxm[0,...]**2/(2.*gbsd[0]**2)
    for i in range( 1, ndim ):
        ndxm[0,...] = ndxm[0,...] +ndxm[i,...]**2/(2.*gbsd[i]**2)
    
    krnl = exp(-ndxm[0,...])    
    krnl = krnl/krnl.sum()

    return krnl


#################################################################
#
def ccxtrct(imin,rbndx,dtheta,drho,show=True):
    """
    ----
    
    imin           # Greyscale, floating point input image (see notes).
    rbndx          # Analyzed region boundary index (2x2).
    dtheta         # theta precision (rad).
    drho           # rho precision (pixels).
    show=True      # Flag to show results.
    
    ----    
    
    This function extracts the location of line intersections for
    photogrammetric calibration.

    Lines are expected to be white on a black background.

    Returns [nscts, params, lhrtg] where: 
        nscts is an lx2 array with l being the number of intersections 
        found and
            nscts[:,0] ----- y-coordinate
            nscts[:,1] ----- x-coordinate

        params is an mx2 array with m being the number of lines found
        by the Hough tranform and (see linht() for more details)
            params[:,0] ----- theta
            params[:,1] ----- rho

        lhrtg is an m-element list of sublists.  Each sublist corresponds
        to one of the m lines of params.  The entries in the sublists are
        row indices of intersections that belong to that line. 
    """

    print "STARTING: ccxtrct"
    import pylab

    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]

    cmat = zeros((2,2),dtype=float)
    bvec = zeros(2,dtype=float)

    teps = 1./min(rsize)

    # Get the params.
    print " | Calling linht."
    params = linht(imin,rbndx,dtheta,drho,show)

    lhrtg = [[]]
    for i in range(1,params.shape[0]):
        lhrtg.append([])

    # Find intersections.
    print " | Extracting intersections."
    count = 0
    nscts = []
    for i in range(params.shape[0]):
        for j in range(i,params.shape[0]):

            if ( abs(params[i,0] -params[j,0]) < teps ):
                continue

            cmat[0,0] = sin(params[i,0])
            cmat[0,1] = cos(params[i,0])
            cmat[1,0] = sin(params[j,0])
            cmat[1,1] = cos(params[j,0])

            bvec[0] = params[i,1]
            bvec[1] = params[j,1]

            try:
                ns = linalg.solve(cmat, bvec)
            except linalg.LinAlgError:
                continue

            if ( ( ns[0] > rsize[0] ) or ( ns[0] < 0. ) ):
                continue
            if ( ( ns[1] > rsize[1] ) or ( ns[1] < 0. ) ):
                continue

            lhrtg[i].append(count)
            lhrtg[j].append(count)
            nscts.append(ns)

            count = count +1

    nscts = array(nscts)
    print " | Intersections found: " +str(nscts.shape[0])

    if (show):
        cp        = zeros((nscts.shape[0],3),dtype=int)
        cp[:,0:2] = (nscts.round()).astype(int)
        cp[:,2]   = 4

        cimg = drwcrc(rsize,cp)
        timg = imin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]].copy()
        timg = where(cimg > 0.,2.,timg)

        pylab.figure()
        pylab.imshow(timg,interpolation="nearest",cmap=pylab.cm.gray)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title("Intersections")
        
    print " | EXITING: ccxtrct"

    return [nscts,params,lhrtg]


#################################################################
#
def constrch(imin,rbndx,cutoff):
    """
    ----
    
    imin           # Image to be contrast stretched (mxn greyscale image).
    rbndx          # Analyzed region boundary index (2x2 array).
    cutoff         # Percent of tails pixels to disregard.
    
    ----
        
    Stretches the contrast of imin within the region specified by
    rbndx.  The darkest and lightest cutoff percent of pixels will
    be ignored, and the rest will be stretched to fill the full
    range of intensity values from 0.0 - 1.0.

    This approach is also known as contrast normalization.

    Returns csim, the contrast stretched region.
    """
    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]

    imrg = imin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]]
    imrg = (imrg*255.).astype("UInt8")

    im = Image.frombuffer(
        "L",
        (rsize[1],rsize[0]),
        imrg,
        "raw",
        "L",
        0,
        1
        )

    im = ImageOps.autocontrast(im,cutoff)

    csim = array(im.getdata()).astype('float')
    csim = csim/255.

    return csim.reshape( (rsize[0],rsize[1]) )


#################################################################
#
def crcht(bimin,rbndx,radmn,radmx,show=True):
    """
    ----
    
    bimin          # Binary, floating point input image.
    rbndx          # Analyzed region boundary index (2x2).
    radmn          # Minimum acceptable circle radius.
    radmx          # Maximum acceptable circle radius.
    show=True      # Flag to show results.
    
    ----
        
    Computes a Hough transform (HT) for circles.  Potential centers are
    restricted to values that are within phimx radians of the edge
    gradient direction, and are further limited to radius values between
    radmn and radmx.  Votes for all valid parameters are collected in a 
    3D array.  The 3D array is then collapsed along the radius to create 
    a center vote map of dimensions equal to rbndx.  The projected HT is 
    then divided into blocks of size 2*radmn, and the local maxima within 
    each block is selected as a candidate hit.  The candidate centers are
    then evaluated a second time again using blocks of size 2*radmn, however
    the blocks are shifted such that the candidate centers lie in the middle
    of the block.  If the previously selected candidate maximum remains the
    local maximum within the shifted block, the candidate center coordinates
    are deemed a valid hit and stored.

    Only integer values of circle center are considered.

    If show = True, two diagnostic plots are produced of various HT
    stages.

    NOTE: Returned y0,x0 coordinates are with respect to analyzed region
    boundaries.  In other words, the origin of the coordinate system 
    used for the Hough transform is set to (rbndx[0,0],rbndx[1,0]).

    Returns an lx3 array, params, where l is the number of circles
    found and
        params[:,0] ----- y0, circle center y-coordinate.
        params[:,1] ----- x0, circle center x-coordinate.
        params[:,2] ----- r, circle radius.
    """
    import pylab

    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]
    rlen  = rsize[0]*rsize[1]

    eps = 0.00001

    ndxmat = indices(rsize)

    radmn = int(radmn)
    radmx = int(radmx)

    # Construct mask of valid center offsets.
    comat = indices( ( 2*radmx +1, 2*radmx +1 ) )
    yco = comat[0,:,:]
    xco = comat[1,:,:]

    yco = yco.reshape(yco.size)
    xco = xco.reshape(xco.size)

    yco = yco -radmx
    xco = xco -radmx

    rd  = sqrt(yco*yco +xco*xco)
    msk = rd <= (radmx +eps)
    yco = compress(msk, yco)
    xco = compress(msk, xco)
    rd  = compress(msk, rd)

    msk = rd >= (radmn -eps)
    yco = compress(msk, yco)
    xco = compress(msk, xco)
    rd  = compress(msk, rd)

    # Sort the radii and determine a unique set.
    srd  = sort(rd)
    nrdk = [ srd[0] ]
    ndx  = 0
    for i in range(srd.size):
        if (srd[i] == srd[ndx]):
            continue
        
        ndx = i
        nrdk.append( srd[i] )

    nrdk  = array(nrdk)

    # Get the gradient and extract edge indices and gradient direction.
    [gmag,gtheta] = grad_socd(bimin,rbndx)
    if ( show ):
        pylab.figure()
        pylab.subplot(221)
        pylab.imshow(bimin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]],
                     cmap=pylab.cm.gray,interpolation="nearest")
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Hough Transform Raw Image')

        pylab.subplot(223)
        pylab.imshow(gmag,cmap=pylab.cm.jet,interpolation="nearest")
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.colorbar()
        pylab.title('| Gradient |')

        pylab.subplot(224)
        pylab.imshow(gtheta,cmap=pylab.cm.jet,interpolation="nearest")
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.colorbar()
        pylab.title('Gradient Orientation [rad]')

    gmag   = gmag.reshape(rlen)
    gtheta = gtheta.reshape(rlen)

    msk   = gmag > 0.1
    eyndx = ndxmat[0,:,:].reshape(rlen)
    eyndx = compress(msk, eyndx)

    exndx = ndxmat[1,:,:].reshape(rlen)
    exndx = compress(msk, exndx)

    etheta = compress(msk, gtheta)

    htmat = pivlibc.crcht_core(rsize,eyndx,exndx,etheta,yco,xco,rd,nrdk)

    # Collapse the HT along the radius parameter.
    phtmat = htmat.sum(2)

    if ( show ):
        pylab.figure()
        pylab.subplot(221)
        pylab.imshow(phtmat,cmap=pylab.cm.jet,interpolation="nearest")
        pylab.colorbar()
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Hough Transform Center Votes')

    # Get centers.
    pkbs   = (2*radmn,2*radmn)
    htthold = pi*pkbs[0]/2.
    [ccntr,wcntr] = pkfind(phtmat,pkbs,htthold,pkbs,False)

    # Grab the winning radii.
    wrad = []
    for i in range(wcntr.shape[0]):
        rarg = htmat[wcntr[i,0],wcntr[i,1],:].argmax()
        wrad.append(nrdk[rarg])

    # Tidy up.
    params = zeros((wcntr.shape[0],3))
    params[:,0] = wcntr[:,0]
    params[:,1] = wcntr[:,1]
    params[:,2] = wrad
    
    if ( show ):
        timg = bimin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]].copy()
        timg[ccntr[:,0],ccntr[:,1]] = 2.
        pylab.subplot(223)
        pylab.imshow(timg,cmap=pylab.cm.gray,interpolation="nearest")
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Candidate Centers (White)')

        timg = bimin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]].copy()
        cimg = drwcrc(rsize,params)
        timg = where(cimg > 0., 2., timg)
        pylab.subplot(224)
        pylab.imshow(timg,cmap=pylab.cm.gray,interpolation="nearest")
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Final Centers (White)')
        
    return params


#################################################################
#
def drwcrc(imdim,params):
    """
    ----
    
    imdim          # Tuple containing the (y,x) image dimensions.
    params         # Array of circle parameters (yc, xc, r).
    
    ----
        
    Utility function to create a binary image of circles and
    correspoding centers.  The function is primarily intended to
    be used after a call to crcht().

    Note: params should be an lx3 array, where l is the number
    of circles to draw.

    Returns a binary, floating point image of size imdim.
    """
    
    # Initialization.
    params = array(params)

    eps = 0.00001

    im = zeros(imdim, dtype=float)

    # Main loop.
    for i in range(params.shape[0]):
        ir  = ceil(params[i,2])
        nde = 2*ir +1

        ndxmat = indices((nde,nde))
        ndxmat = ndxmat -ir

        yov = ndxmat[0,:,:]
        xov = ndxmat[1,:,:]

        nrsq = yov**2 +xov**2
        crsq = params[i,2]**2

        rerr   = abs(nrsq -crsq)
        mnerry = rerr.min(0)
        mnerrx = rerr.min(1)
        mnnmy  = where(mnerry < eps,1.,mnerry)
        mnnmx  = where(mnerrx < eps,1.,mnerrx)
        rerry  = rerr/mnnmy -1.
        rerry  = where(rerry < 0., 0., rerry)
        rerrx  = (rerr.transpose()/mnnmx).transpose() -1.
        rerrx  = where(rerrx < 0., 0., rerrx)

        cb = where(rerry < eps, 1., 0.)
        cb = where(rerrx < eps, 1., cb)
        cb = cb.reshape(cb.size)

        yv = yov +params[i,0]
        yv = yv.astype(int)
        yv = yv.reshape(yv.size)
        xv = xov +params[i,1]
        xv = xv.astype(int)
        xv = xv.reshape(xv.size)

        msky = ( yv > 0 ) * ( yv < imdim[0] ) 
        mskx = ( xv > 0 ) * ( xv < imdim[1] )
        msk  = msky*mskx
        yv   = compress(msk,yv)
        xv   = compress(msk,xv)
        cb   = compress(msk,cb)

        im[yv,xv] = cb
        im[params[i,0],params[i,1]] = 1.

    return im


#################################################################
#
def drwlin(imdim,params):
    """
    ----
    
    imdim          # Tuple containing the (y,x) image dimensions.
    params         # Array of line parameters (theta, rho).
    
    ----
        
    Utility function to create a binary image of lines.  The 
    function is primarily intended to be used after a call to lin
    ht().

    Note: params should be an lx2 array, where l is the number
    of lines to draw.

    NOTE: drwlin expects 0 +/- eps <= theta < pi.

    Returns a binary, floating point image of size imdim.
    """

    # Initialization.
    im  = zeros(imdim,dtype=float)

    params = array(params)
    
    # Main loop.
    yv = array(range(imdim[0]))
    xv = array(range(imdim[1]))
    for i in range(params.shape[0]):
        if ( abs(params[i,0] <= pi/4. ) ):
            fxv = (params[i,1] -yv*sin(params[i,0]))/cos(params[i,0])
            fxv = fxv.round()

            fyv = yv
        elif ( abs(params[i,0] <=  3.*pi/4.) ):
            fyv = (params[i,1] -xv*cos(params[i,0]))/sin(params[i,0])
            fyv = fyv.round()

            fxv = xv
        else:
            fxv = (params[i,1] -yv*sin(params[i,0]))/cos(params[i,0])
            fxv = fxv.round()

            fyv = yv

        # Make sure line segments are within bounds of image.
        msky = ( fyv >= 0 ) * ( fyv < imdim[0] )
        mskx = ( fxv >= 0 ) * ( fxv < imdim[1] )
        msk  = msky * mskx

        fyv = compress(msk, fyv)
        fxv = compress(msk, fxv)

        im[fyv.astype(int),fxv.astype(int)] = 1.

    return im


#################################################################
#
def esttpswarp(f1,f2,rbndx,csrp=0.1,ithp=95,wsize=[5,5],sdmyf=0.1,alpha=0.9,beta=1.5,csize=7,nits=30,scit=10,annl=0.98):
    """
    ----
    
    f1             # Frame 1 (greyscale mxn array).
    f2             # Frame 2 (greyscale mxn array).
    rbndx          # Region boundary index (2x2, referenced to frame 1).
    csrp=0.1       # Contrast stretching rejection percentile.
    ithp=95        # Intensity threshold percentile for feature points.
    wsize=[5,5]    # Feature point extraction window.
    sdmyf=0.1      # Synthetic dummy point fraction.
    alpha=0.9      # Intensity similarity weighting for cost matrix.
    beta=1.5       # Weighting of spring force for cost matrix.
    csize=7        # Cluster size.
    nits=30        # Number of iterations.    
    scit=10        # Run shape context every scit iterations.
    annl=0.98      # Controls rate of annealing for TPS regularization.
    
    ----
    
    Computes an estimate of the thin-plate spline transform for warping 
    the template region, f1, to the search region, f2.
    
    Image features are selected based on intensity using the following
    procedure.  First, the region of interest in both frames is contrast 
    stretched (pivutil.constrch()) with the upper and lowermost csrp 
    percentile pixels being rejected.  After contrast stretching, a 
    threshold is set at the ithp percentile intensity.  Any pixels with
    intensity below this threshold will not be considered further.  
    Increasing ithp will increasing the threshold intensity value and
    reject more pixels.  If ithp is too low, noise will be considered as
    valid feature points.  Once a threshold is available, feature points 
    (particles) are extracted using pivutil.pkfind() and a window of size
    wsize (see pkfind() documentation for more details).
    
    NOTE: ithp is the parameter that essentially controls the number of
    extracted feature points in each frame.
    
    To warp one frame to another, two problems must be solved:
    1) the correspondence problem that determines which feature points
    in Frame 1 correspond to those of Frame 2, and 2) computation of the
    actual warping function that maps coordinates from Frame 1 to Frame 2
    based on the feature point correspondences.  Solving these two
    problems separately is challenging, but solving them together is much
    easier.  The iterative technique used here derives predominantly from
    the work of Belongie:2002 and Okamoto:1995, and was also influenced by 
    Chui:2003 (annealing).  
    
    The basic algorithm is as follows:
        a) Compute an estimate of the correspondence.
        b) Use the correspondence to construct an estimated warp.
        c) Warp the template feature points.
        d) Update the correspondence. 
        e) Loop to b.
    
    The overall measure of correspondence is based on 3 components:
        1) The similarity between the shape contexts of Belongie:2002. 
        For a given feature point in a collection of feature points, the 
        shape context provides a coarse, 2D histogram of the location of 
        all other feature points in the collection relative to the 
        selected point.  The histogram axes are spanned by log10(r) and
        theta, where r is the radius from the selected feature point (pt0)
        to another feature point (ptj) in the collection, and theta is the
        angle between the x-axis and vector from pt0 to ptj.
        
        Comparing the shape context of a feature point from Frame 1 to
        those of all feature points from Frame 2, one can construct a 
        metric that measures the degree of similarity between the two
        shape contexts (and hence the probability of feature point
        correspondence).  
        
        2) The similarity between the intensity profiles of feature
        points.  The assumption here is very straightforward: a bright
        particle in Frame 1 will be similarly bright in Frame 2.  For
        each feature point, a 3x3 window is constructed, and the
        intensity values in those windows are compared between frames.
        The importance of the intesity mismatch is controlled by alpha.
        As alpha increases, more emphasis will be placed on ensuring that
        'corresponding' feature points have similar appearance.  Disable
        by setting alpha to 0.0.
        
        3) In fluid flows, small clusters of particles should move
        in a somewhat uniform manner.  Although the particles in the 
        cluster may deform slightly, any gross deformation might be 
        assumed to be an indicator of a particle mismatch.  Okamoto:1995
        used these concepts to build a registration measure that 
        considers a cluster of particles to be connected to each other by
        a set of springs.  The force in the spring network then becomes
        a indicator of feature point mismatching, with lower forces
        suggesting a better match.  The number of feature points in a
        cluster is set with the csize parameter.

    With seperate measures (ie, costs) for matching the feature points 
    available, some means of combining the individual costs into a single, 
    aggregate cost is necessary.  Unfortunately, this is somewhat difficult
    since the measures are based on entirely different principles.  In
    other words, how does one compare a cost of 23 units from a shape 
    context similarity measure to a spring model force of 6 units?  Although 
    esttpswarp() does combine the three measures into a single aggregate
    cost (more on this below), the iterative algorithm alternately
    optimizes two seperate cost functions based on a user specified 
    schedule, scit.  The first aggregate cost is given by
        cmat = scmat +alpha*(scmat.max()/icmat.max())*icmat 
                     +beta*(scmat.max()/spmat.max())*spmat 
    where scmat is the cost matrix from the shape contexts, icmat the
    cost from intensity comparison, spmat the cost matrix from the spring
    model, and alpha, beta are user-specified weighting paramters.  As
    can be seen, all three measures are combined in cmat.  The second
    cost function that is minimized is that from the spring model alone
    (ie, spmat).  cmat will be minimized every scit iterations, while
    spmat will be minimized for all other iterations.  NOTE: The initial
    estimate of point correspondences is computed using cmat with
    beta = 0.0.

    Once the cost matrices, cmat or spmat, have been computed, the
    optimum pairing of feature points that minimizes the total overal cost 
    must be determined.  Such optimizations are known as linear assignment
    problems or bipartite graph matching problems.  The algorithm (LAPJV)
    used by esttpswarp() is that of Jonker:1987.

    The warping function used here is that of thin-plate splines (see
    Bookstein:1989).  Thin plate splines combine the affine transformation
    with an energy minimizing set of basis functions that can be used
    to warp one coordinate system to another in a much more general way.
    During early iterations, the correspondences between Frame 1 and 
    Frame 2 feature points is poor, so the warping function is regularized 
    to prevent construction of a mapping that exhibits excessive (and 
    spurious) wiggling.  As the iterative method progresses and the 
    correspondences become more sound, the amount of regularization is 
    reduced according to an annealing schedule.  At the start of the 
    algorithm, the regularization parameter, regp, is set equal to the 
    square of the maximum dimension of the region of interest window.
    At the end of each iteration, regp is reduced as
        regp = annl*regp
    where annl is the rate of annealing parameter.  The smaller annl,
    the faster the TPS method will begin to exactly reproduce the point
    correspondences.  But remember: for early iterations, these
    f1 to f2 point pairings are likely to contain errors.  If annl is
    too small, the TPS warp will produce a coordinate mapping that
    fully displaces f1 points to 'matched' f2 points regardless of the 
    quality of the point match.  The end result is that the overall 
    iterative algorithm can become unstable and produce garbage.  Generally
    speaking, the larger annl, the slower the method converges, and the
    better the final point matching results will be.  Reducing annl
    below 0.9 for all but a few iterations will almost certainly result
    in the method becoming unstable.
    
    One final paramter, sdmyf, remains to be discussed.  When extracting
    feature points from two image frames taken at different times, there 
    is no guarantee that both frames will produce the same number of 
    feature points.  And even if the same number of feature points are
    extracted, there is no guarantee that the exact same feature
    points will be extracted in both frames.  The latter issue can lead
    to feature points from f1 being assigned to incorrect feature points
    from f2 simply because the correct f2 feature point was not extracted
    (using the technique described above).  sdmyf provides a mechanism
    to address this second issue.  For an image containing n feature
    points (the min of f1 or f2 is actually used), an additional 
    sdmyf*n dummy feature points will be added to the mix.  These synthetic
    dummy feature points have a matching cost of 0.0 and will be paired to
    real feature points that don't have a good match otherwise.  Set
    sdmyf = 0.0 to disable sythetic dummy points.
    
    Returns tpw, the tpswarp object for the mapping.
      
    """
    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]
    
    rmax = rsize.max()
    
    mnpl = 5   # If points < mnpl, will raise ValueError.
    
    wrbndx = array([[0,rsize[0]],[0,rsize[1]]])
    
    trgn = f1[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]]   # Template.
    srgn = f2[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]]   # Search.
    
    trgn = trgn -trgn.min()
    trgn = trgn/trgn.max()
    
    srgn = srgn -srgn.min()
    srgn = srgn/srgn.max()
    
    # Extract feature points.  The images are first contrast stretched.
    # Once the threshold has been computed using ithp, the feature
    # points are determined using pivutil.pkfind().  Note: Setting the
    # second pkfind() window to [3,3] seems to produce better results.
    trgn = constrch(trgn,wrbndx,csrp)
    srgn = constrch(srgn,wrbndx,csrp)
    
    tthold = stats.scoreatpercentile(trgn.reshape(trgn.size),ithp)
    sthold = stats.scoreatpercentile(srgn.reshape(srgn.size),ithp)

    [cpks,tfpts] = pkfind(trgn,wsize,tthold,[3,3],False)
    [cpks,sfpts] = pkfind(srgn,wsize,sthold,[3,3],False)
    
    if ( tfpts.size == 0 ):
        raise ValueError("No points in template region.")
    if ( sfpts.size == 0 ):
        raise ValueError("No points in search region.")
    
    tfpts = tfpts.round().astype(int)
    sfpts = sfpts.round().astype(int)

    ntfpts = tfpts.shape[0]
    nsfpts = sfpts.shape[0]
    
    nsdmy = int( ceil( sdmyf*min(ntfpts,nsfpts) ) )
    
    tfndx = arange(ntfpts)  # Indices into the tfpts.
    
    # Setup spring model for template points.
    trmat = empty((ntfpts,ntfpts),dtype=float)
    for t in range(ntfpts):
        trad = tfpts[t,:] -tfpts
        trad = sqrt( ( trad*trad ).sum(1) )

        trmat[t,:] = trad
        
    nbrndx = trmat.argsort(1)[:,1:csize]
    
    # Build shape contexts.
    ltfpts = tfpts.copy().tolist()
    lsfpts = sfpts.copy().tolist()
    
    tcxa = []
    for i in range( ntfpts ):
        pt = ltfpts.pop(0)
        tcxa.append( shpctxt(pt,ltfpts,rmax) )
        
        ltfpts.append(pt)

    tcxa   = array(tcxa)
    ntfpts = tfpts.shape[0]

    scxa = []        
    for i in range( nsfpts ):
        pt = lsfpts.pop(0)
        scxa.append( shpctxt(pt,lsfpts,rmax) )
        
        lsfpts.append(pt)
        
    scxa = array(scxa)
    
    # Build the cost matrix and optimize.
    cmat   = pivtpsc.sccost(tcxa,scxa,nsdmy)
    cmatmx = cmat.max()
    
    if ( alpha > 0. ):
        icmat = pivtpsc.iscost(trgn,tfpts,srgn,sfpts,nsdmy)
        cmat  = cmat +alpha*cmatmx/max(icmat.max(),1.)*icmat 
        
    [cst,rcmap,crmap] = pivtpsc.lapjv(cmat)

    # ----- MAIN LOOP -----
    # Compute the thin-plate spline warp transformation and update the 
    # correpsondences.  In what follows, several 'types' of points are
    # used:
    #    ctfpts ---- Candidate template feature points.  A candidate
    #                point is a point that has NEVER moved beyond
    #                the rbndx perimeter.  The full set of candidate points
    #                will be used for correspondence matching each iteration.
    #    vtfpts ---- Valid template feature points.  A point is considered
    #                valid if it is a candidate and not paired to a dummy.
    #                The set of valid tfpts is constructed from ctfpts
    #                each iteration and is used to construct the 
    #                warp estimate.
    #    cwtfpts --- Candidate feature points that have been warped using
    #                the latest estimate of the TPS warp.  These points
    #                are used to compute an update of the shape contexts.
    ctfpts  = tfpts     # Candidate template points.
    nctfpts = ntfpts

    iregp = rsize.max()**2
    regp  = iregp
    for j in range(nits):
        # If ntfpts > nsfpts, then columns will be filled with dummy points.
        # We don't want to use any match to a dummy point.
        dmymsk = rcmap[0:nctfpts] < nsfpts
        dtfpts = ctfpts[dmymsk,:]
        drcmap = rcmap[0:nctfpts][dmymsk]
        
        # Compute the transformation for warping template coordinates
        # to search image coordinates using the valid template feature
        # points from above.
        tpw = tpswarp(dtfpts,sfpts[drcmap,:],regp)
        if ( j == nits -1 ):
            break
         
        # Warp the candidate template feature points.  Some of the        
        # the candidate template feature points will be outside the
        # search region.  So they should be permanently dropped.
        wtfpts = tpw.xfrm(ctfpts)
        wtfpts = wtfpts.round().astype(int)

        bndmsk = ( wtfpts[:,0] >= 0 )*( wtfpts[:,0] < rsize[0] )\
                *( wtfpts[:,1] >= 0 )*( wtfpts[:,1] < rsize[1] )

        ctfpts  = ctfpts[bndmsk,:]
        cwtfpts = wtfpts[bndmsk,:]
        rcmap   = rcmap[0:nctfpts][bndmsk]
        tfndx   = tfndx[bndmsk]

        nctfpts = tfndx.size 
        nsdmy   = int( ceil( sdmyf*min(nctfpts,nsfpts) ) )

        if ( nctfpts < mnpl ):
            raise ValueError("Number of candidate points below %i." % mnpl)

        # Compute the cost matrix using the spring model.
        spmat = pivtpsc.spcost(trmat,nbrndx,tfndx,rcmap,cwtfpts,sfpts,nsdmy)

        if ( mod(j+1,scit) == 0 ):
            # Compute updated shape contexts and build the new cost matrix.
            ltfpts = cwtfpts.copy().tolist()
    
            tcxa = []
            for i in range( nctfpts ):
                pt = ltfpts.pop(0)
                tcxa.append( shpctxt(pt,ltfpts,rmax) )
            
                ltfpts.append(pt)
    
            tcxa   = array(tcxa)
            cmat   = pivtpsc.sccost(tcxa,scxa,nsdmy)
            cmatmx = cmat.max()
            
            # Penalize the cost matrix based on difference in feature point
            # intensity.
            if ( alpha > 0. ):
                # icmat can have some dummy columns if ntfpts > nsfpts.
                # So column slice must span 0:cmat.shape[1].
                cicmat = icmat[tfndx,0:cmat.shape[1]]                 
                cmat[0:nctfpts,:] = cmat[0:nctfpts,:] \
                                   +alpha*cmatmx/max(cicmat.max(),1.)*cicmat       

            # Add penalty from spring model and optimize.
            cmat  = cmat +beta*cmatmx/max(spmat.max(),1.)*spmat
            [cst,rcmap,crmap] = pivtpsc.lapjv(cmat)
        else:
            [cst,rcmap,crmap] = pivtpsc.lapjv(spmat)
            
        regp = annl*regp
    
    return tpw


#################################################################
#
def getblkprm(rbndx,bsize,bolap,bsdiv):
    """
    ----
    
    rbndx          # Analyzed region boundary index (2x2 array).
    bsize          # Block size ([y,x] pixels).
    bolap          # Block overlap ([y,x] pixels).
    bsdiv          # Block subdivision factor.
        
    ----
    
    Computes block parameters for driver functions like ofcomp().
    The arguments to getblkprm should be those that are generally
    part of the pivdict object.
    
    Low resolution blocks are those that do not use block subdivision.
    This naming convention comes from the fact that block overlap uses
    a single size block for all analysis, while block subdivision uses
    a smaller block size for part of the analysis.
    
    Returns [rsize, lbso, hbso, lnblks, hnblks], where
        rsize ---- Array containing the [y,x] region size.
        lbso ----- Low resolution block start offset [y,x].
        hbso ----- High resolution block start offset [y,x].
        lnblks --- Low resolution number of blocks [y,x].
        hnblks --- High resolution number of blocks [y,x].
    """
    # Initialization.
    rbndx = array(rbndx)
    bsize = array(bsize)
    bolap = array(bolap)
    
    # Compute the parameters.
    rsize = rbndx[:,1] -rbndx[:,0]

    if ( bsdiv < 1 ):
        raise ValueError( "bsdiv must be >= 1" )

    nzbolap = ( bolap[0] > 0 ) or ( bolap[1] > 0 )
    if ( ( nzbolap ) and ( bsdiv > 1 ) ):
        raise ValueError( "bolap > 0 and bsdiv > 1 cannot be used simultaneously." )

    lbso   = bsize -bolap
    hbso   = bsize/bsdiv
    lnblks = ( rsize/lbso ).astype(int) -( bsize/lbso ).astype(int) +1
    hnblks = bsdiv*lnblks

    if ( mod(rsize[0],bsize[0]) != 0 ):
        print ' | WARNING: Region not spanned by an integer number of blocks in y.'
    if ( mod(rsize[1],bsize[1]) != 0 ):
        print ' | WARNING: Region not spanned by an integer number of blocks in x.'

    if ( mod(bsize[0],lbso[0]) != 0 ):
        print ' | WARNING: gp_bsize[0]/(gp_bsize[0]-gp_bolap[0]) not an integer.' 
    if ( mod(bsize[1],lbso[1]) != 0 ):
        print ' | WARNING: gp_bsize[1]/(gp_bsize[1]-gp_bolap[1]) not an integer.'

    return [rsize,lbso,hbso,lnblks,hnblks]


#################################################################
#
def getpivcmap(mthd=0):
    """
    ----
    
    mthd=0          # Method to use for hue splitting.

    ----
    
    Utility function to build a fully saturated colormap suitable 
    for use in generating colorbars for pylab.imshow() plots.  

    Colormap created corresponds to either of the two hue splitting
    methods employed by imread().
        0 ----- Standard hue map computed using method of
                Ledley:1990. Hue = 0 is red, and map progresses 
                ROYGBIVR.
        1 ----- Rotated hue map similar to that used by Dabiri:1991.  
                Hue 0 is a reddish blue given by (R,G,B) = (0.5,0,1).  
                Map progresses IBGYORVI.

    Returns a LinearSegmentedColormap instance that can be passed
    to pylab.imshow() via the argument 'cmap'.
    """

    from matplotlib import colors
    
    # Initialization.

    ns = 37  # (ns -1) should be divisible by 4.

    tns = (ns -1)*3

    chnlr = zeros(tns,dtype=float)
    chnlg = chnlr.copy()
    chnlb = chnlr.copy()

    fracv = array(range(tns)).astype('float')/(tns -1)

    rd = zeros((tns,3),dtype=float)
    rd[:,0] = fracv

    gd = rd.copy()
    bd = rd.copy()

    # Sector 1.
    qty  = ns
    hqty = (ns -1)/2 +1
    fracv = array(range(hqty)).astype('float')/(hqty -1.)
    chnlb[0:qty] = 0.

    chnlr[0:hqty]       = 1.
    chnlr[(hqty-1):qty] = 1. -fracv

    chnlg[0:hqty]       = fracv
    chnlg[(hqty-1):qty] = 1.

    ndx = qty -1

    # Sector 2.
    chnlr[ndx:(ndx+qty)] = 0.

    chnlg[ndx:(ndx+hqty)]         = 1.
    chnlg[(ndx+hqty-1):(ndx+qty)] = 1. -fracv

    chnlb[ndx:(ndx+hqty)]         = fracv
    chnlb[(ndx+hqty-1):(ndx+qty)] = 1.

    ndx = ndx +qty -1

    # Sector 3.
    chnlg[ndx:tns] = 0.

    chnlb[ndx:(ndx+hqty)]   = 1.
    chnlb[(ndx+hqty-1):tns] = 1. -fracv[0:(hqty-1)]

    chnlr[ndx:(ndx+hqty)]   = fracv
    chnlr[(ndx+hqty-1):tns] = 1.

    if ( mthd == 1 ):
        qqty  = (ns -1)/4 +1
        cqqty = ns -qqty

        tchnlr = chnlr[(tns-cqqty):tns].copy()
        tchnlg = chnlg[(tns-cqqty):tns].copy()
        tchnlb = chnlb[(tns-cqqty):tns].copy()

        chnlr[cqqty:tns] = chnlr[0:(tns-cqqty)]
        chnlg[cqqty:tns] = chnlg[0:(tns-cqqty)]
        chnlb[cqqty:tns] = chnlb[0:(tns-cqqty)]

        chnlr[0:cqqty] = tchnlr
        chnlg[0:cqqty] = tchnlg
        chnlb[0:cqqty] = tchnlb

        chnlr = list(chnlr)
        chnlg = list(chnlg)
        chnlb = list(chnlb)

        chnlr.reverse()
        chnlg.reverse()
        chnlb.reverse()

        chnlr = array(chnlr)
        chnlg = array(chnlg)
        chnlb = array(chnlb)
    
    # Build the data vectors.
    rd[:,1] = rd[:,2] = chnlr
    gd[:,1] = gd[:,2] = chnlg
    bd[:,1] = bd[:,2] = chnlb

    pivmap_data = { 
        'red':   rd,
        'green': gd,
        'blue':  bd
        }

    pivmap = colors.LinearSegmentedColormap('pivmap',pivmap_data,256)
    
    return pivmap


#################################################################
#
def gfit(pts,maxits,eps):
    """
    ----
    
    pts            # Three element array containing values to fit.
    maxits         # Maximum iterations.
    eps            # Desired accuracy.
    
    ----
        
    Fits a gaussian to three points using Newton-Raphson.  

       f = a exp( -(x-mu)^2/b^2 )

    The three points will be assigned coordinates [-1, 0, 1].

    The initial values of [a, mu, b] will be [pts[1], 0., 1.]

    Returns [a, mu, b].  
    """

    return pivlibc.gfitcore(pts,maxits,eps)


#################################################################
#
def grad_fo(imin,rbndx):
    """
    ----
    
    imin           # Input image (greyscale mxn array).
    rbndx          # Analyzed region boundary index (2x2 array).
    
    ----    
    
    Computes a first order approximation to the gradient of the
    image using forward difference operators.

        grad = sqrt(Iy^2 +Ix^2)

    where Iy is the first order derivative in y, and Ix is the first
    order derivative in x.

    Returns [gmag,gtheta] where
        gmag ----- rsize array of gradient magnitudes.
        gtheta --- rsize array of gradient orientations.  
                   -pi <= gtheta <= pi.  
    """

    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]

    gy = zeros(rsize,dtype=float)
    gx = gy.copy()

    gy[0:(rsize[0]-1),0:rsize[1]] = \
        imin[(rbndx[0,0]+1):rbndx[0,1],rbndx[1,0]:rbndx[1,1]] \
       -imin[rbndx[0,0]:(rbndx[0,1]-1),rbndx[1,0]:rbndx[1,1]] 

    gx[0:rsize[0],0:(rsize[1]-1)] = \
        imin[rbndx[0,0]:rbndx[0,1],(rbndx[1,0]+1):rbndx[1,1]] \
       -imin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:(rbndx[1,1]-1)]

    gmag   = sqrt( gy*gy +gx*gx )
    gtheta = arctan2(gy,gx)

    return [gmag, gtheta]


#################################################################
#
def grad_socd(imin,rbndx):
    """
    ----
    
    imin           # Input image (greyscale mxn array).
    rbndx          # Analyzed region boundary index (2x2 array).
    
    ----
        
    Computes a second order approximation to the gradient of the
    image using central difference operators.

        grad = sqrt(Iy^2 +Ix^2)

    where Iy is the second order derivative in y, and Ix is the second
    order derivative in x.

    Returns [gmag,gtheta] where
        gmag ----- rsize array of gradient magnitudes.
        gtheta --- rsize array of gradient orientations.  
                   -pi <= gtheta <= pi.  
    """

    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]

    gy = zeros(rsize,dtype=float)
    gx = gy.copy()

    gy[1:(rsize[0]-1),0:rsize[1]] = \
        imin[ (rbndx[0,0]+2):rbndx[0,1], rbndx[1,0]:rbndx[1,1] ] \
       -imin[ rbndx[0,0]:(rbndx[0,1]-2), rbndx[1,0]:rbndx[1,1] ] 

    gx[0:rsize[0],1:(rsize[1]-1)] = \
        imin[ rbndx[0,0]:rbndx[0,1], (rbndx[1,0]+2):rbndx[1,1] ] \
       -imin[ rbndx[0,0]:rbndx[0,1], rbndx[1,0]:(rbndx[1,1]-2) ]

    gmag   = 0.5*sqrt( gy*gy +gx*gx )
    gtheta = arctan2(gy,gx)

    return [gmag, gtheta]


#################################################################
#
def imread(impath,mthd=0,rgb=False):
    """
    ----
    
    impath          # Path to image file.
    mthd=0          # Method to use for hue splitting. 
    rgb=False       # Return RGB channels instead of HSI.
    
    ----
        
    Simple utility function to read an image file from disk and
    extract the hue, saturation, and intensity.  Hue is computed
    using either of the following two methods.  Note: The 
    formulation for the two methods are identical, but the direction 
    of increasing hue and the color for hue=0 are different.
        0 ----- Standard hue map computed using method of
                Ledley_1990. Hue = 0 is red, and map progresses 
                ROYGBIVR.
        1 ----- Rotated hue map similar to that used by Dabiri:1991.  
                Hue 0 is a reddish blue given by (R,G,B) = (0.5,0,1).  
                Map progresses IBGYORVI.

    If rgb = False, returns [chnlh, chnls, chnli] where
       chnlh ----- mxn array containing the hue channel.
                   Range: 0.0 .. 1.0
       chnls ----- mxn array containing the saturation channel.
                   Range: 0.0 .. 1.0
       chnli ----- mxn array containing the intensity channel.
                   Range: 0.0 .. 1.0

    If rgb = True, returns [chnlr, chnlg, chnlb] where
       chnlr ----- mxn array containing the red channel.
                   Range: 0.0 .. 1.0
       chnlg ----- mxn array containing the green channel.
                   Range: 0.0 .. 1.0
       chnlb ----- mxn array containing the blue channel.
                   Range: 0.0 .. 1.0

    Note: m is the number of rows and n is the number of columns.

    NOTE: imread() currently converts all images to a per channel bit depth 
    of 8 bits.  This conversion may fail for some image types with an 
    "illegal conversion" error.  If so, the user should convert the
    images using ImageMagick or the like and retry.  This shortcoming
    is due to limitations in the Python Imaging Library.

    The Python Image Library v1.1.6 seems to have issues with bmp files.
    Use png or tif instead!
    """
    print "STARTING: imread"
    # Initialization.

    if ( string.upper(impath[-3:]) == "BMP"):
        print " | WARNING: bmp files are not handled correctly."

    im    = Image.open(impath)
    immod = im.mode
    im    = im.convert("RGB")  # Force to 8 bit RGB.
    imc   = im.split()
    imdim = im.size

    print " | Path: " + impath
    print " | Mode: " + immod
    print " | Size: " + str(imdim[::-1])

    chnlr = array(imc[0].getdata()).astype('float')
    chnlg = array(imc[1].getdata()).astype('float')
    chnlb = array(imc[2].getdata()).astype('float')

    chnlr = reshape(chnlr,(imdim[1],imdim[0]))
    chnlg = reshape(chnlg,(imdim[1],imdim[0]))
    chnlb = reshape(chnlb,(imdim[1],imdim[0]))

    # Normalize the image such that the channels span 0.0 - 1.0.
    # PIL doesn't provide a convenient mechanism to determine the 
    # image bit depth, so we have to guess.  Actually, PIL use should
    # probably be replaced entirely with a more complete package
    # such as GraphicsMagick.
    rmax = chnlr.max()
    gmax = chnlg.max()
    bmax = chnlb.max()
    cmax = max(rmax,gmax,bmax)
    if ( cmax > 1. ):
        sf    = 1./255.
        chnlr = sf*chnlr
        chnlg = sf*chnlg
        chnlb = sf*chnlb

    if ( rgb ):
        print " | EXITING: imread"
        return [chnlr, chnlg, chnlb]

    # Extract HSI.
    [chnlh,chnls,chnli] = rgb2hsi([chnlr,chnlg,chnlb],mthd)

    print " | EXITING: imread"
    return [chnlh,chnls,chnli]


#################################################################
#
def imshift(imin,rbndx,p,mthd='C',edge=0):
    """
    ----
    
    imin           # Image to be shifted (greyscale mxn array).
    rbndx          # Analyzed region boundary index (2x2 array).
    p              # Array containing displacements ([dy,dx] pixels).
    mthd='C'       # Interpolation method for shift.
    edge=0         # Edge treatment for bicubic interpolation.
    
    ----
    
    Shifts a single region of imin using interpolation of the type
    speficied by methd.  methd can be one of two values:
        "L" - Bilinear interpolation.
        "C" - Bicubic interpolation.

    The underlying bicubic interpolation mechanism uses central
    differences to approximate the first derivatives of the image.
    As such and without special edge handling techniques, the image 
    being analyzed with bicubic interpolation needs to be 1 pixel 
    larger in all directions than the output image (this constraint 
    applies AFTER the shift is applied).  edge specifies how these 
    edge pixels of the output image should be generated.  
        0 - No special edge treatment.  Bicubic interpolation will
            be used for the entire output image.  Therefore, input 
            image is guaranteed to be 1 pixel larger than output 
            in all directions after shift is applied.
        1 - Use nearest neighbor for edge pixels of output.  Input
            image is guaranteed to be padded sufficiently for shift
            only (i.e., the extra 1-pixel pad necessary for bicubic
            interpolation is no longer needed).

    NOTE: p represents image displacements (not camera)!

    edge will be ignored if methd = "L".

    Returns shifted image of size bsize.
    """
    # Initialization.
    rbndx = array(rbndx)

    umthd = string.upper(mthd)

    if  ( umthd == 'L' ):
        return pivlibc.imsblicore(imin, rbndx.reshape(4), tuple(p))
    elif ( umthd == 'C' ):
        return pivlibc.imsbcicore(imin, rbndx.reshape(4), tuple(p), edge)
    else:
        raise ValueError("Invalid interpolation method " +umthd)
    
    
#################################################################
#
def imtpswarp(imin,rbndx,tpw,worig=None,approx=True):
    """
    ----
    
    imin           # The image (greyscale mxn array) to be warped.
    rbndx          # Region boundary index (2x2).
    tpw            # The tpswarp object.
    worig=None     # 2-element array specifying the [y,x] warp origin.
    approx=True    # Whether image warp should be approximate.
    
    ----
    
    Warps imin using the thin-plate spline warping of tpw, an instance
    of the tpswarp class.  
    
    NOTE: tpw should be the warping object that maps imin coordinates
    to the warped frame (not the other way arround).  
    
    As just noted, the twp object maps imin coordinates into the warped
    image frame.  To warp the input image itself, however, imtpswarp must
    first compute a mapping of coordinates from the warped to input image.
    Unfortunately, this inverse thin-plate spline mapping is not easily 
    computed.  As a result, two options for warping imin are provided.  
    If approx is True, then a computationally cheaper but crude 
    approximation to warping will be computed.  If approx is False, a more 
    expensive and accurate warp will be determined.  See documentation on 
    the tpswarp class for more details.
    
    If worig is None, the tpw object transformation origin will be assumed
    to be rbndx[:,0], otherwise the warping origin will be set to worig
    (ie, coordinates for the warp will be computed relative to worig).
        
    Returns wimg, a warped image of the same size as rbndx.
    """
    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]
    
    if ( compat.checkNone(worig) ):
        worig = rbndx[:,0]
    else:
        worig = array(worig)
    
    imdim = array(imin.shape)
    
    npts = rsize[0]*rsize[1]
    
    ndxmat = indices(rsize)
    yvec   = ndxmat[0,...].reshape(npts)
    xvec   = ndxmat[1,...].reshape(npts)
    
    wpxca = array([yvec,xvec]).transpose()  # Coordinates for ROI.
    apxca = wpxca +rbndx[:,0]  # Unwarped, full image coordinates.
    
    # Get awmat inverse and compute the pixel coordinates in the
    # unshifted image that correspond to the shifted pixels.
    rpxca = apxca -worig  # Relative to worig.
    if ( approx ):
        spxca = tpw.aixfrm(rpxca)
    else:
        spxca = tpw.ixfrm(rpxca)
    
    spxca = spxca +worig  # Warped, but back in full image coordinates.
    
    # We use bicubic interpolation, so we need a 1 pixel pad.
    msk = ( spxca[:,0] >= 1 )*( spxca[:,0] < imdim[0] -1 ) \
         *( spxca[:,1] >= 1 )*( spxca[:,1] < imdim[1] -1 )

    wpxca = wpxca[msk,:]
    spxca = spxca[msk,:]
    apxca = apxca[msk,:]    
    
    # pa gives the shift necessary to move a pixel in the warped frame
    # to the unwarped frame. 
    pa = spxca -apxca
    
    # Form the warped image.
    spx = pxshift(imin,apxca,-pa)
    
    wimg                        = zeros(rsize,dtype=float)
    wimg[wpxca[:,0],wpxca[:,1]] = spx
    
    return wimg
    

#################################################################
#
def imwrite(imin,impath,cmap=None,vmin=None,vmax=None):
    """
    ----
    
    imin            # Image data.
    impath          # Path to output image file.
    cmap=None       # Pylab colormap.
    vmin=None       # Min channel value.
    vmax=None       # Max channel value.
    
    ----
        
    Dumps an array to an image file.

    imin can be either a 2D array of greyscale pixel values
    or an array of RGB values.  For the greyscale case, a Pylab 
    colormap can be specified, or the image will be stored as greyscale 
    if cmap=None.  

    For the RGB case, imin must have the shape

        (nc,ny,nx)

    where ny, nx are the number of pixels in the y and x dimensions,
    respectively.  nc represents the number of channels in the image,
    which must be 3 for an RGB image.

    Because an array can contain floating point values that span a 
    variety of ranges, imwrite() must make some assumptions when
    generating an image from the data.  If vmin and vmax are None,
    then imwrite() will evaluate each channel and scale the channel
    such that its minimum value corresponds to an intensity of 0 and
    the channel's max value corresponds to an intensity of 255.  
    Setting vmin and vmax to another value will result in all values
    less than or equal to vmin being set to zero, and all values
    greater than or equal to vmax being set to 255.

    The file type of the image is determined from the extension of impath.

    Note: cmap, is ignored if an RGB image is passed.
    """
    # Initialization.
    imin = array(imin).astype(float)
    imin = imin.squeeze()
    if ( imin.ndim == 2 ):
        imin = imin.reshape((1,imin.shape[0],imin.shape[1]))
    elif ( imin.ndim == 1 ):
        imin = imin.reshape((1,1,imin.shape[0]))

    ncmp  = imin.shape[0]
    imdim = imin.shape[1:3]
    npix  = imdim[0]*imdim[1]

    if ( ( ncmp != 1 ) and ( ncmp != 3 ) ):
        raise ValueError("imin must have either 1 or 3 channels.")

    # Scale the image so that it runs from 0 .. 255.
    for i in range(ncmp):
        if ( not compat.checkNone(vmin) ):
            imn = vmin
        else:
            imn = imin[i,...].min()

        if ( not compat.checkNone(vmax) ):
            imx = vmax
        else:
            imx = imin[i,...].max()

        ntrval = imx -imn

        imin[i,...] = 255.*(imin[i,...] -imn)/ntrval

    imin = imin.clip(0.,255.)

    # Reshape the data for PIL.
    imin = imin.reshape(ncmp,npix)
    imin = imin.transpose()
    imin = imin.reshape(ncmp*npix)

    # Get the PIL image.
    if ( ncmp == 3 ):
        mode = "RGB"
    else:
        mode = "L"

    img = Image.frombuffer(
        mode,
        (imdim[1],imdim[0]),
        imin.astype("UInt8"),
        "raw",
        mode,
        0,
        1 )

    # Get PIL colormap from Pylab colormap.
    if ( ( not compat.checkNone(ncmp == 1 ) and ( cmap) ) ):
        icmap = 255.*cmap( arange(256) )
        icmap = icmap[:,0:3]
        icmap = icmap.reshape( icmap.size )

        img.putpalette(icmap.astype("UInt8"))

    img.save(impath)
    
    
#################################################################
#
def padfltr(fltr,pfdim,decntr=True):
    """
    ----
    
    fltr           # mxn floating point array.
    pfdim          # List of the padded dimensions, [y,x].
    decntr=True    # Specifies whether fltr should be decentered.
    
    ----
    
    Pads a frequency domain filter, fltr, with zeros such that the 
    resulting filter has dimensions of fdim.  The padding is
    accomplished in the following manner:
        * fltr is decentered if decntr is True.
        * Inverse FFT of fltr is taken to form a spatial domain
          kernel.
        * Real part of kernel is extracted.
        * Real kernel is centered.
        * Centered kernel is padded on right and bottom edges with 
          zeros to dimensions of fdim.
        * Forward FFT is taken.
    
    Note: A centered filter is one where the filter DC value is
    is in the middle of the image of the filter.  This is often
    the way FFT's and filters are viewed.  If decntr is True,
    padfltr() will take care of decentering fltr prior to padding.    
    
    Returns pfltr, the padded filter.  pfltr is not centered (DC is
    pixel 0,0).
    """
    # Initialization.
    pfltr = zeros(pfdim,dtype=float)
    fdim  = fltr.shape
    FLTR  = fltr
    
    # Build the padded filter.
    if ( decntr ):
        FLTR = fft.ifftshift(FLTR)

    fltr = real( fft.fftshift( fft.ifft2(FLTR) ) )
    
    pfltr[0:fdim[0],0:fdim[1]] = fltr
    
    PFLTR = fft.fft2( pfltr )
    
    return PFLTR


#################################################################
#
def padimg(imin,pimdim,ptype=0):
    """
    ----
    
    imin           # Greyscale, floating point input image.
    pimdim         # List of the padded dimensions, [y,x].
    ptype=0        # Type of padding.

    ----
    
    Pads imin based on ptype to dimensions of pimdim.  Padding
    of the image (and potentially the filter) is useful for frequency 
    domain filtering to avoid wrap-around errors.  In such cases,
    set pimdim = 2*imin.shape.  See Gonzalez and Woods (Gonzalez:2002)
    for more information on frequency domain filtering.
    
    Two types of padding are supported via ptype.
        0 ---- Pad the right and bottom edges of the image with zeros.
        1 ---- Pad the right and bottom edges of the image with the
               mean value of imin.  Depending on the image and filter, 
               mean padding can significantly reduce edge effects such
               as ringing.
               
    Returns pimin, the padded image.
    """
    # Initialization.
    pimin = zeros(pimdim,dtype=float)
    imdim = imin.shape
    
    # Build the padded image.
    if ( ptype == 0 ):
        pimin[0:imdim[0],0:imdim[1]] = imin
    elif ( ptype == 1 ):
        pimin[:,...] = imin.mean()
        pimin[0:imdim[0],0:imdim[1]] = imin

    return pimin


#################################################################
#
def linht(imin,rbndx,dtheta,drho,show=True):
    """
    ----
    
    imin           # Greyscale, floating point input image (see notes).
    rbndx          # Analyzed region boundary index (2x2).
    dtheta         # theta precision (rad).
    drho           # rho precision (pixels).
    show=True      # Flag to show results.
    
    ----
        
    Computes a Hough transform (HT) for lines using the parameterization

        rho = x*cos(theta) + y*sin(theta)

    where rho is the magnitude of the vector originating from the origin
    and perpendicular to the candidate line, and theta is the angle 
    between the x-axis and the vector rho.  In general, 0 <= theta < pi, 
    however discretization error may cause theta to be slightly negative
    instead of 0.

    Parameter space is divided into cells of size dtheta x drho.
    For 768 x 1024 images, the recommended values for dtheta and drho
    are pi/2048 and 0.5 respectively.

    linht() uses a greyscale image as input and weights the voting results
    by the intensity of the image.

    If show = True, two diagnostic plots are produced of various HT
    stages.

    NOTE: The origin of the coordinate system used for the Hough 
    transform is set to (rbndx[0,0],rbndx[1,0]).

    NOTE: lines in imin should be white on a black background and should 
    not be thicker than 7 pixels.

    WARNING: linht() is very memory intensive when using the recommended
    values for dtheta and drho.

    Returns an lx2 array, params, where l is the number of lines
    found and
        params[:,0] ----- theta
        params[:,1] ----- rho       
    """

    print "STARTING: linht"
    import pylab

    # Initialization.
    rbndx  = array(rbndx)
    rsize  = rbndx[:,1] -rbndx[:,0]
    rlen   = rsize[0]*rsize[1]
    rimin  = imin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]]
    ndxmat = indices(rsize)

    ntheta = int(round(pi/dtheta)) +1
    dtheta = pi/(ntheta -1.)

    rhomx = linalg.norm(rsize,2)
    nrho  = 2*int(round(rhomx/drho)) +1    # rho can be negative.
    drho  = 2.*rhomx/(nrho -1.)
    rco   = (nrho -1)/2

    print " | Realized dtheta: " +str(dtheta)
    print " | Realized drho: " +str(drho)

    # Tuning parameters.
    gbkrad = 1.5     # Base std dev of gaussian for peak sharpening.
    gbkpow = 4       # Square of std dev reduction factor. 

    bgth = (rimin.max() -rimin.mean())/2.       # Background threshold.

    dtmx = pi/10.    # Angular variation of theta for each x,y point.

    htthd = 5.0    # htthold = htmat.max()/htthd
    #gmthd = 4.     # gmthold = gmag.max()/gmthd

    # Sharpen the peaks in the input image.  This step helps accomodate 
    # thick lines.
    ck = bldgbkern([gbkrad,gbkrad])
    ck = pow(ck,gbkpow)
    ck = ck/ck.sum()

    fimin = ndimage.convolve(rimin,ck,mode='constant')    

    # Get theta.
    [gmag,theta] = grad_fo(imin,rbndx)

    if ( show ):
        pylab.figure()
        pylab.subplot(221)
        pylab.imshow(rimin,interpolation="nearest",cmap=pylab.cm.gray)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Hough Transform Raw Image')

        pylab.subplot(222)
        pylab.imshow(fimin,interpolation="nearest",cmap=pylab.cm.jet)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.colorbar()
        pylab.title('Peak-Sharpened Weights')
        
        pylab.subplot(223)
        pylab.imshow(gmag,cmap=pylab.cm.jet,interpolation="nearest")
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.colorbar()
        pylab.title('| Gradient |')

        pylab.subplot(224)
        pylab.imshow(theta,cmap=pylab.cm.jet,interpolation="nearest")
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.colorbar()
        pylab.title('Gradient Orientation [rad]')

    # Collapse weights, indices, and theta.
    fimin = fimin.reshape(rlen)
    eyndx = ndxmat[0,:,:].reshape(rlen)
    exndx = ndxmat[1,:,:].reshape(rlen)
    theta = theta.reshape(rlen)
    gmag  = gmag.reshape(rlen)
    
    #gmth  = gmag.max()/gmthd
    msk   = rimin > bgth 
    msk   = msk.reshape(rlen)
    fimin = compress(msk,fimin)
    eyndx = compress(msk,eyndx)
    exndx = compress(msk,exndx)
    theta = compress(msk,theta)
    gmag  = compress(msk,gmag)

    # Following Duda:1975, shift theta so that 0 <= theta <= pi.  I don't
    # believe this is necessary.
    theta = where(theta < 0., theta +pi, theta)

    # Setup relative theta.
    rtheta = indices([2*ceil(dtmx/dtheta) +1])
    rtheta = rtheta.reshape(rtheta.size)
    rtheta = dtheta*(rtheta -ceil(dtmx/dtheta))
    
    # Setup relative rho to vote for the rho values roughly spanning 
    #    rho - 1 <= rho <= rho + 1
    rrhosz = max(3,2*round(2./drho) +1)  # In case drho > 1.
    rrho   = indices([rrhosz],dtype=float).reshape(rrhosz)
    rrho   = rrho -(rrhosz -1.)/2.
    rrho   = drho*rrho

    # Initialize htmat.  Note: htmat is periodic in theta.
    ntpad = (rtheta.size -1)/2
    htmat = zeros((ntheta +ntpad,nrho),dtype=float)

    print " | HT base dimensions: (" +str(ntheta) +", " +str(nrho) +")"

    # Compute the Hough transform.
    tpc = 0
    print " | Investigating " +str(fimin.size) +" points ..." 
    for i in range (fimin.size):
        vtheta = rtheta +theta[i]
        vtheta = where(vtheta < 0., vtheta +pi, vtheta)
        vtheta = where(vtheta >= pi, vtheta -pi, vtheta)

        vrho = eyndx[i]*sin(vtheta) +exndx[i]*cos(vtheta)

        prrho  = repeat(rrho.reshape((1,rrhosz)),vtheta.size,axis=0)
        prrho  = prrho.reshape(prrho.size)
        vtheta = repeat(vtheta,rrhosz)
        vrho   = repeat(vrho,rrhosz)
        vrho   = vrho +prrho

        vtndx = ((vtheta/dtheta).round()).astype(int) +ntpad
        vrndx = ((vrho/drho).round()).astype(int) +rco

        htmat[vtndx,vrndx] = htmat[vtndx,vrndx] +fimin[i]*gmag[i]

        if ( int(10*i/fimin.size) > tpc ):
            tpc = tpc +1
            print " |  " + str(tpc*10) + "% complete"

    if ( show ):
        pylab.figure()
        if ( ( ntheta < 3000 ) and ( nrho < 3000 ) ): 
            pylab.subplot(221)
            pylab.imshow(htmat[ntpad:,:],
                         cmap=pylab.cm.jet,
                         interpolation="nearest"
                         )
            pylab.colorbar()
            pylab.title("Hough Transform Votes")
            pylab.setp(pylab.gca(),xticks=[],yticks=[])
            pylab.ylabel("pi <-- theta <-- 0")
            pylab.xlabel("0 --> rho --> %g" % rhomx)

    # Enforce periodic boundary conditions.
    htmat[0:ntpad,:] = htmat[ntheta-1:ntheta+ntpad-1,::-1]
    
    # Get line parameters.
    pkbs    = (rtheta.size,2*int(round(ck.shape[0]/drho)))
    htthold = htmat.max()/htthd
    [cline,params] = pkfind(htmat[0:ntheta,:],pkbs,htthold,pkbs,False)

    # Convert indices to theta and rho.
    params[:,0] = (params[:,0] -ntpad)*dtheta
    params[:,1] = (params[:,1] -rco)*drho

    print " | Lines found: " +str(params.shape[0])

    if ( show ):
        pylab.subplot(222)
        pylab.plot((cline[:,1]-rco)*drho,(cline[:,0]-ntpad)*dtheta,'kx')
        pylab.title("Candidate Parameters")
        pylab.ylabel("theta [rad]")
        pylab.xlabel("rho")
        pylab.grid()

        pylab.subplot(223)
        pylab.plot(params[:,1],params[:,0],'kx')
        pylab.title("Final Parameters")
        pylab.ylabel("theta [rad]")
        pylab.xlabel("rho")
        pylab.grid()

        timg = rimin.copy()
        limg = drwlin(rsize,params)
        timg = where(limg > 0., 2., timg)
        pylab.subplot(224)
        pylab.imshow(timg,interpolation="nearest",cmap=pylab.cm.gray)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title("Reconstructed Lines (White)")

    print " | EXITING: linht"

    return params


#################################################################
#
def pkfind(data,p1bsize,p1thold,p2bsize,show=True):
    """
    ----
    
    data           # Floating point input data.
    p1bsize        # Phase I block size [y,x].
    p1thold        # Phase I threshold.
    p2bsize        # Phase II block size [y,x].
    show=True      # Flag to show results.
    
    ----
        
    Utility function that finds peak indices in a 2D scalar dataset using a 
    two step method adapted from Kierkegaard:1992.  Data does not need to 
    be bounded between 0. and 1.

    In Phase I, the dataset is divided into blocks of size p1bsize, and
    the maximum of each block is retained as a candidate peak as long as
    the candidate is greater than p1thold.

    In Phase II, blocks of size p2bsize are centered at each candidate peak
    from Phase I.  If the previously selected candidate peak from Phase I 
    remains the local maximum within the shifted block, the candidate 
    peak is deemed a valid peak and stored.

    If show = True, two diagnostic plots are produced.

    Returns [cpks,pks], where: 
        cpks ---- An lx2 array of candidate peak indices identified in Phase I.
        pks ----- An mx2 array of valid peaks from Phase II.
    """
    import pylab
    
    # Initialization.
    rsize = data.shape

    pbsndx = zeros(2,dtype=int)
    pbendx = pbsndx.copy()

    # Peak Extraction Round 1: Identify candidates.
    bsoy = p1bsize[0]
    bsox = p1bsize[1]

    nyblks = int( ceil( 1.*rsize[0]/bsoy ) )
    nxblks = int( ceil( 1.*rsize[1]/bsox ) )

    cpks = []
    for m in range(nyblks):
        pbsndx[0] = m*bsoy
        pbendx[0] = min(pbsndx[0] +bsoy,rsize[0])
        for n in range(nxblks):
            pbsndx[1] = n*bsox
            pbendx[1] = min(pbsndx[1] +bsox,rsize[1])

            block = data[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
            bmax  = block.max()

            if ( bmax <= p1thold ):
                continue
          
            absize = block.shape
            arg    = argmax(block)
            yarg   = int(arg/absize[1])
            xarg   = arg -yarg*absize[1]

            cpks.append([pbsndx[0]+yarg,pbsndx[1]+xarg])

    cpks = array(cpks)

    # Peak Extraction Round 2: Indentify winners.
    wyndx = []
    wxndx = []

    bsoy = p2bsize[0]
    bsox = p2bsize[1]

    hbsoy = int(bsoy/2)
    hbsox = int(bsox/2)
    for i in range(len(cpks)):
        pbsndx[0] = cpks[i,0] -hbsoy
        pbendx[0] = pbsndx[0] +bsoy

        pbsndx[1] = cpks[i,1] -hbsox
        pbendx[1] = pbsndx[1] +bsox

        # Make sure the block is within htmat.
        pbsndx[0] = max(pbsndx[0],0)
        pbendx[0] = min(pbendx[0],rsize[0])
        pbsndx[1] = max(pbsndx[1],0)
        pbendx[1] = min(pbendx[1],rsize[1])

        block = data[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]

        # Get the coordinates for the new max.
        absize = block.shape
        arg    = argmax(block)
        yarg   = int(arg/absize[1])
        xarg   = arg -yarg*absize[1]

        yarg = pbsndx[0] +yarg
        xarg = pbsndx[1] +xarg

        # If the new max coordinates don't match the old, we're probably on
        # the slope of a peak.
        if ( ( yarg != cpks[i,0] ) or ( xarg != cpks[i,1] ) ):
            continue

        # Store the winning parameters.
        wyndx.append( yarg )
        wxndx.append( xarg )

    # Tidy up.
    wyndx = array(wyndx)
    wxndx = array(wxndx)
    pks   = zeros((wyndx.size,2))

    pks[:,0] = wyndx
    pks[:,1] = wxndx

    if ( show ):
        pylab.figure()
        pylab.subplot(221)
        pylab.imshow(data,cmap=pylab.cm.jet,interpolation='nearest')
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.colorbar()
        pylab.title('Input Data')

        pylab.subplot(223)
        ax = pylab.gca()
        #ax.xaxis.set_ticks_position('top')
        pylab.plot(cpks[:,1],cpks[:,0],'kx')
        pylab.setp(ax,ylim=[rsize[0],0],xlim=[0,rsize[1]])
        pylab.title("Candidate Peaks")
        pylab.ylabel("yndx")
        pylab.xlabel("xndx")
        pylab.grid()

        pylab.subplot(224)
        ax = pylab.gca()
        #ax.xaxis.set_ticks_position('top')
        pylab.plot(wxndx,wyndx,'kx')
        pylab.setp(ax,ylim=[rsize[0],0],xlim=[0,rsize[1]])
        pylab.title("Winning Peaks")
        pylab.ylabel("yndx")
        pylab.xlabel("xndx")
        pylab.grid()

    return [cpks,pks]


#################################################################
#
def prtmask(imin,rbndx,bsize,bgth=0.3,bcth=10.,show=True):
    """
    ----
    
    imin           # Input image (greyscale mxn array).
    rbndx          # Analyzed region boundary index (2x2 array).
    bsize          # Block size ([y,x] pixels).
    bgth=0.3       # Background threshold.
    bcth=10.       # Bubble convolution threshold.
    show=True      # Flag to show results.
    
    ----    
    
    prtmask() takes the intensity channel of an image, breaks it
    into blocks of bsize for processing, and creates a mask 
    representing the location of particles.  Large bright regions 
    are excluded as these are often bubbles or other debris.
    
    Mask value is zero if not over an area deemed to be a particle 
    and one otherwise.

    Note that prtmask returns an image the same size as imin, however
    particles will only be identified for the region specified by
    rbndx.  The image border outside of rbndx will be set to zero
    in the mask.

    For prtmask(), the value of rbndx[:,0] can be (0,0), but the value 
    of bsize should be kept around (32,32).  rbndx will be subdivided
    into the largest possible number of integral blocks.

    Bubble exclusion works by convolving a normalized version of the 
    input image with a 5x5 kernel.  The resulting convolved image is
    then thresholded to determine the locations of bubbles.  The 
    thresholding is controlled by bcth which should be in the range 
    0.<= bcth <=21.  Small values of bcth will classify more input
    image features as bubbles.  Large values of bcth will only exclude
    larger items (up to a max of 5x5).  If bcth > 21., no bubbles
    will be excluded.

    If show = True, a plot is created of the mask at various stages
    of creation.

    Returns a mask of the same size as imin.
    """
    import pylab

    # Initialization.
    imdim = imin.shape
    rbndx = array(rbndx)

    pbsndx = zeros(2,dtype=int)
    pbendx = pbsndx.copy()

    [rsize,lbso,hbso,lnblks,hnblks] = getblkprm(rbndx,bsize,[0,0],1)

    cmap = pylab.gray()

    # Tuning parameters.
    #    iavesf ----- Scale factor for intensity average.  Let iave represent
    #                 the average intensity for the block.  Any pixel in the
    #                 block having an intensity less than iavesf*iave will
    #                 be eliminated from the valid particle mask.
    iavesf = 0.5

    # Grab a copy of the input image and set border to black.
    mim = zeros(imdim,dtype=float)
    mim[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]] = \
        imin[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]]

    if ( show ):
        pylab.figure()
        pylab.subplot(221)
        pylab.imshow(mim,cmap=cmap)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Phase 0: Raw Image')

    # Get bubble mask.
    [mim,bmask] = bblmask(mim,rbndx,bsize,bgth,bcth)

    # Invert mask and apply to modified image.
    bmask = where(bmask>0.1,0.,1.)
    mim = bmask*mim

    if ( show ):
        pylab.subplot(222)
        pylab.imshow(bmask,cmap=cmap)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Phase 1: Large Bubbles (Black)')

        pylab.subplot(223)
        pylab.imshow(mim,cmap=cmap)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Phase 2: Particle Finder Input')

    # Create particle mask.
    for m in range(lnblks[0]):
        pbsndx[0] = rbndx[0,0] +m*lbso[0]
        pbendx[0] = pbsndx[0] +bsize[0]
        for n in range(lnblks[1]):
            pbsndx[1] = rbndx[1,0] +n*lbso[1]
            pbendx[1] = pbsndx[1] +bsize[1]

            
            block = mim[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
            bave  = block.mean()

            if ( bave <= 0. ):
                bave = 1.

            block = where(block<iavesf*bave,0.,1.)

            mim[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]] = block

    if ( show ):
        pylab.subplot(224)
        pylab.imshow(mim,cmap=cmap)
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        pylab.title('Phase 3: Particle Mask (White)')            

    return mim


#################################################################
#
def pxshift(imin,pxca,pa):
    """
    ----
    
    imin           # Image to be shifted (greyscale mxn array).
    pxca           # lx2 array of integer pixel coordinates (y,x).
    pa             # lx2 array of shift components (dy,dx). 
    
    ----
        
    pxshift() uses bicubic interpolation to interpolate between
    four neighboring pixels.  In essence, pxshift() applies a 
    shift of magnitude pxca to each pixel in pxca.  The function
    is very similar to imshift(), except that pxshift() allows
    individual shift vectors to be applied to individual pixels
    (whereas imshift() applies one shift vector to the entire
    image).

    NOTE: pxca must be passed as an array of integers.  No forced
    type casting is done by this function, and an error will be
    generated otherwise.

    pa is equivalent to p in imshift() -- it defines the direction
    the image should be shifted.

    PivLIB's bicubic interpolation implementation uses central
    difference approximations to the first derivatives of the
    image data.  As a result, if floor(pxca -pa) lies on the 
    outermost one-pixel wide border of the image, pxshift() will
    use nearest neigbor interpolation for those pixels.

    Returns pixa, an l-element array of new pixel values. 
    """
    return pivlibc.pxsbcicore(imin,pxca,pa)


#################################################################
#
def rgb2hsi(rgbc,mthd=0):
    """
    ----
    
    rgbc           # List of red, green, blue channels.
    mthd=0         # Method used for hue splitting.
    
    ----
        
    Takes a list of red, green, and blue channels and computes the
    corresponding hue, saturation, and intensity channels.  Hue is 
    computed using either of the following two methods.  Note: The 
    formulation for the two methods are identical, but the direction 
    of increasing hue and the color for hue=0 are different.
        0 ----- Standard hue map computed using method of
                Ledley:1990. Hue = 0 is red, and map progresses 
                ROYGBIVR.
        1 ----- Rotated hue map similar to that used by Dabiri:1991.  
                Hue 0 is a reddish blue given by (R,G,B) = (0.5,0,1).  
                Map progresses IBGYORVI.

    Each of the red, green, and blue channels must be an mxn image
    normalized to run from 0.0 - 1.0.  Note: m is the number of rows 
    and n is the number of columns.

    Returns [chnlh,chnls,chnli], where
       chnlh ----- mxn array containing the hue channel.
                   Range: 0.0 .. 1.0
       chnls ----- mxn array containing the saturation channel.
                   Range: 0.0 .. 1.0
       chnli ----- mxn array containing the intensity channel.
                   Range: 0.0 .. 1.0
    """
    # Initialization.
    chnlr = rgbc[0]
    chnlg = rgbc[1]
    chnlb = rgbc[2]

    # Extract HSI (Ledley_1990).
    chnli = (chnlr +chnlg +chnlb)/3.

    # Note:  Dabiri:1991's definition for tchnlb differs from
    # the following.  Ledley:1990's definition has been used instead
    # for both methods.
    tchnla = (1./sqrt(6.))*(2.*chnlr -chnlg -chnlb)
    tchnlb = (1./sqrt(2.))*(chnlg -chnlb) 

    if ( mthd == 0 ):
        chnlh = arctan2(tchnlb,tchnla)
        chnlh = where(chnlh<0.,2*pi+chnlh,chnlh)
        chnlh = chnlh/(2.*pi)
    elif ( mthd == 1 ):
        chnlh = arctan2(tchnla,tchnlb)
        chnlh = (chnlh +pi)/(2.*pi)
    else:
        raise ValueError("Invalid mthd " + str(mthd))

    chnls = sqrt(tchnla**2 +tchnlb**2)

    return [chnlh,chnls,chnli]


#################################################################
#
def shpctxt(pt,fpts,rmax,rbins=5,tbins=12):
    """
    ----
    
    pt             # Feature point for shape context. 
    fpts           # Surrounding feature points.
    rmax           # Maximum possible radius.
    rbins=5        # Number of bins for spanning log10(r) space.
    tbins=12       # Number of bins for spanning theta space.
    
    ----
    
    Constructs a shape context as described in Belongie:2002.
    A shape context is a coarse, 2D histogram of the location
    of all other interest points, fpts, relative to pt.  The
    histogram axes are spanned by log10(r) and theta,
    where r is the radius from pt to a point in fpts, and theta
    is the angle between the x-axis and a vector from pt to the point
    in fpts.
    
    pt is a 2 element array of feature point coordinates ordered 
    as [y,x].  fpts is an lx2 array of coordinates for the feature points
    that surround pt (ie, fpts does not include pt), where l is the
    number of points to consider.  Each entry in fpts must be ordered
    as [y,x].
    
    rmax specifies the maximum possible radius and will be used to
    construct the histogram bins.  rmax should be the same between
    all calls to shpctxt, otherwise comparing the shape context for
    one point to another will have no meaning.  A good starting value for
    rmax is the maximum anticipated rmax for the pointset.
    
    Returns sctxt, an rbins x tbins integer array.
    """
    return pivtpsc.shpctxt(pt,fpts,rmax,rbins,tbins)


