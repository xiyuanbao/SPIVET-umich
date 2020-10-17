"""
Filename:  pivpgcal.py
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
  Module containing photogrammetric calibration routines.

    Cameras are modeled as distortion-bearing pinhole cameras 
    in PivLIB.  For an overview of pinhole models, consult 
    Tsai:1987, Willert:2006, Heikkila:1997, Trucco:1998, and 
    Weng:1992.  The PivLIB model adopts the following spatial 
    orientation convention:
        - For this module, pivpgcal, image coordinates (ie, those 
          coordinates expressed in units of pixels) will often be 
          represented as v, u instead of y, x to avoid confusion 
          with spatial coordinates (anything with units of mm).
        - As with all of PivLIB, the origin for image coordinates 
          is the top left corner of the image:
              *---> u
              |
              V v
        - Camera coordinates are centered at the camera pinhole
          with increasing z_c passing through the image plane at 
          roughly its center (the exact coordinates will be determined 
          as part of the optimization) and pointing toward the object 
          being viewed.  The image plane lies between the pinhole and the
          object being viewed.  The camera x_c and  y_c axes are parallel 
          with image u and v axes, respectively. 

    In order to move back and forth from image to world coordinates, 
    PivLIB makes use of three intermediate coordinate systems 
    (Camera, Undistorted Sensor, and Distorted Sensor).  In moving 
    from  world to image coordinates, the following progression 
    occurs:
        World              --> Camera
        Camera             --> Undistorted Sensor
        Undistorted Sensor --> Distorted Sensor
        Distorted Sensor   --> Image

    PivLIB's optimized camera model includes first and second 
    order radial, first order tangential, and first order thin 
    prism distortion. Unfortunately, the complexity of the distortion 
    model makes the equations relating Distorted to Undistorted 
    Sensor coordinates coupled and non-linear.  As a result, Willson 
    (http://www.cs.cmu.edu/~rgw/TsaiCode.html) and Tsai's approach 
    for the relation between distorted and undistorted sensor 
    coordinates is used here.  Namely
        y_d f(y_d,x_d) = y_u
        x_d g(y_d,x_d) = x_u
    where f and g are the non-linear distortion functions.  

    PivLIB employs a two phase camera calibration procedure.  During 
    the first phase, the camera is calibrated using a distortionless 
    direct linear transform (ref. Heikkila:1997 and Trucco:1998).  
    The DLT calibration is then used as an initial guess for an 
    optimized calibration that includes the effects of the three 
    types of distortion.  Altogether, the camera is characterized by 
    16 calibrated parameters: 6 external (Rz, Ry, Rx, Tz, Ty, Tx) 
    and 10 internal (f, sx, v0, u0, k1, k2, p1, p2, s1, s2).

    Camera calibration parameters are stored in a dictionary with
    the following entries (call initcamcal() to initialize):
        camcal['dpx'] ----- /Float/ Effective dimension of CCD pixel
                            parallel to u-axis [mm].
        camcal['dpy'] ----- /Float/ Effective dimension of CCD pixel
                            parallel to v-axis [mm].
        camcal['u0'] ------ /Float/ u-intercept of camera z-axis 
                            with sensor in the image coordinate 
                            system.  Corresponds to Cx in Willson's 
                            code.  The u-axis is aligned with the 
                            large dimension of the sensor.  
                            [pixels]
        camcal['v0'] ------ /Float/ v-intercept of camera z-axis 
                            with sensor in the image coordinate 
                            system.  Corresponds to Cy in Willson's 
                            code.  The v-axis is aligned with the 
                            smaller dimension of the sensor. 
                            [pixels]
        camcal['sx'] ------ /Float/ Pixel scale factor. 
        camcal['f'] ------- /Float/ Effective focal length of camera 
                            lens [mm].
        camcal['k1'] ------ /Float/ First order radial distortion 
                            coefficient [1/mm^2].
        camcal['k2'] ------ /Float/ Second order radial distortion 
                            coefficient [1/mm^4].
        camcal['p1'] ------ /Float/ Tangential distortion coefficient 
                            [1/mm^2].
        camcal['p2'] ------ /Float/ Tangential distortion coefficient 
                            [1/mm^2].
        camcal['s1'] ------ /Float/ Prism distortion coefficient 
                            [1/mm^2].
        camcal['s2'] ------ /Float/ Prism distortion coefficient
                            [1/mm^2].
        camcal['T'] ------- /Float/ Three element array defining the 
                            (z,y,x) translation between World and 
                            Camera coordinates. [mm]
        camcal['R'] ------- /Float/ Three-element array of angles for 
                            rotating from the World to Camera
                            coordinate system. (Rz,Ry,Rx) [rad]
        camcal['Rmat'] ---- /Float/ Full 3x3 rotation matrix for 
                            moving from World to Camera coordinates.

Contents:
  calibrate()
  camcal2prma()
  ccpts()
  dcorim()
  dltsolv()
  dsuds()
  d_dsuds()
  im2wrld()
  importwcc()
  initcamcal()
  initknowns()
  lmfun()
  loadcamcal()
  loadccpa()
  optimsolv()
  prma2camcal()
  rvec2rmat()
  savecamcal()
  uds2ds()
  wrld2cam()
  wlrd2uds()
  wrld2im()
"""
import matplotlib.pyplot as plt
from numpy import *
from scipy import ndimage
from scipy import linalg
from scipy import optimize

import pivutil
import pivlinalg
import pivpickle
from spivet import compat

#################################################################
#
def calibrate(ccpa,camcal,maxits=1000):
    """
    ----
    
    ccpa            # lx5 array of camera calibration points. 
    camcal          # Camera calibration dictionary.
    maxits=1000     # Maximum number of optimization iterations.
    
    ----
        
    Primary driver routine for camera calibration.  Users
    simply need to pass in a properly initialized camcal
    dictionary and an array of camera calibration points
    from ccpts().
    
    Returns [ocamcal,oerr]
        ocamcal ---- Optimized camera calibration. 
        oerr ------- (v,u) error computed by projecting world points
                     of ccpa onto the image plane using ocamcal.
    """

    print "STARTING: calibration"

    # Do the calibration.
    print " | Solving for direct linear transformation."
    dcamcal = dltsolv(ccpa,camcal)

    print " | Performing calibration optimization."
    ocamcal = optimsolv(ccpa,dcamcal,maxits)

    # Compute error by projecting the world points onto the image plane.
    dcip  = wrld2im(ccpa[:,0:3],dcamcal)
    derr  = dcip -ccpa[:,3:5]
    derrm = sqrt(sum(derr*derr,1))

    ocip  = wrld2im(ccpa[:,0:3],ocamcal)
    oerr  = ocip -ccpa[:,3:5]
    oerrm = sqrt(sum(oerr*oerr,1))

    # Print some diagnostic information.
    print " | ---------- CALIBRATION PARAMETERS ----------"
    print " |        DLT            OPTIMIZED"
    print " | dpy   %13e %13e" % ( dcamcal['dpy'],       ocamcal['dpy'] )
    print " | dpx   %13e %13e" % ( dcamcal['dpx'],       ocamcal['dpx'] )
    print " | f     %13e %13e" % ( dcamcal['f'],         ocamcal['f'] )
    print " | sx    %13e %13e" % ( dcamcal['sx'],        ocamcal['sx'] )
    print " | v0    %13e %13e" % ( dcamcal['v0'],        ocamcal['v0'] )
    print " | u0    %13e %13e" % ( dcamcal['u0'],        ocamcal['u0'] )
    print " | k1    %13e %13e" % ( dcamcal['k1'],        ocamcal['k1'] )
    print " | k2    %13e %13e" % ( dcamcal['k2'],        ocamcal['k2'] )
    print " | p1    %13e %13e" % ( dcamcal['p1'],        ocamcal['p1'] )
    print " | p2    %13e %13e" % ( dcamcal['p2'],        ocamcal['p2'] )
    print " | s1    %13e %13e" % ( dcamcal['s1'],        ocamcal['s1'] )
    print " | s2    %13e %13e" % ( dcamcal['s2'],        ocamcal['s2'] )
    print " | Tz    %13e %13e" % ( dcamcal['T'][0],      ocamcal['T'][0] )
    print " | Ty    %13e %13e" % ( dcamcal['T'][1],      ocamcal['T'][1] )
    print " | Tx    %13e %13e" % ( dcamcal['T'][2],      ocamcal['T'][2] )
    print " | Rz    %13e %13e" % ( dcamcal['R'][0],      ocamcal['R'][0] )
    print " | Ry    %13e %13e" % ( dcamcal['R'][1],      ocamcal['R'][1] )
    print " | Rx    %13e %13e" % ( dcamcal['R'][2],      ocamcal['R'][2] )
    print " | R00   %13e %13e" % ( dcamcal['Rmat'][0,0], ocamcal['Rmat'][0,0] )
    print " | R01   %13e %13e" % ( dcamcal['Rmat'][0,1], ocamcal['Rmat'][0,1] )
    print " | R02   %13e %13e" % ( dcamcal['Rmat'][0,2], ocamcal['Rmat'][0,2] )
    print " | R10   %13e %13e" % ( dcamcal['Rmat'][1,0], ocamcal['Rmat'][1,0] )
    print " | R11   %13e %13e" % ( dcamcal['Rmat'][1,1], ocamcal['Rmat'][1,1] )
    print " | R12   %13e %13e" % ( dcamcal['Rmat'][1,2], ocamcal['Rmat'][1,2] )
    print " | R20   %13e %13e" % ( dcamcal['Rmat'][2,0], ocamcal['Rmat'][2,0] )
    print " | R21   %13e %13e" % ( dcamcal['Rmat'][2,1], ocamcal['Rmat'][2,1] )
    print " | R22   %13e %13e" % ( dcamcal['Rmat'][2,2], ocamcal['Rmat'][2,2] )
    print " | ----- CALIBRATION IMAGE PLANE ERROR [PIXELS] -----"
    print " |        DLT            OPTIMIZED"
    print " | MEAN  %13e %13e" % ( derrm.mean(), oerrm.mean() )
    print " | MAX   %13e %13e" % ( derrm.max(),  oerrm.max() )
    print " | MIN   %13e %13e" % ( derrm.min(),  oerrm.min() )
    print " | STDEV %13e %13e" % ( derrm.std(),  oerrm.std() )
    print " |"

    print " | EXITING: calibration"
    return [ocamcal,oerr]


#################################################################
#
def camcal2prma(camcal):
    """
    ----
    
    camcal         # Camera calibration dictionary.
    
    ----
        
    Converts camera calibration dictionary into an array of
    parameters to be calibrated.
    """
    # Initialization.
    prma = zeros(16,dtype=float)

    # Set up the array.
    prma[0]  = camcal['v0']
    prma[1]  = camcal['u0']
    prma[2]  = camcal['sx']
    prma[3]  = camcal['f']
    prma[4]  = camcal['k1']
    prma[5]  = camcal['k2']
    prma[6]  = camcal['p1']
    prma[7]  = camcal['p2']
    prma[8]  = camcal['T'][0]
    prma[9]  = camcal['T'][1]
    prma[10] = camcal['T'][2]
    prma[11] = camcal['R'][0]
    prma[12] = camcal['R'][1]
    prma[13] = camcal['R'][2]
    prma[14] = camcal['s1']
    prma[15] = camcal['s2']

    return prma


#################################################################
#
def ccpts(imsin,rbndx,knowns,ccptune,ofpath=None,show=True):
    """
    ----
    
    imsin          # List of intensity images for camera.
    rbndx          # Analyzed region boundary index (2x2).
    knowns         # Array of known (z,y,x) intersections [mm].
    ccptune        # Dictionary of tuning parameters.
    ofpath=None    # Path to output file of calibration data.
    show=True      # Flag to show results.
    
    ----
        
    Prepares an array of correlated world and image points for use
    in camera calibration.

    The points corresponding to the knowns are determined from 
    imaging a test target consisting of a series of intersecting 
    lines (think gridded graph paper).  The test target images input 
    to ccpts() should be white lines against a black background (so 
    input images may need to be inverted prior to calling ccpts()).  
    ccpts() then proceeds to determine the intersections of the lines 
    using cross correlation (for high accuracy) and a Hough transform
    (to assist ordering the intersections).  The image coordinates of
    the intersections will be incremented by rbndx[:,0].  

    An output file is created at ofpath, if specified, that contains 
    the image intersections and the knowns.  Data in the output file
    will be ordered such that it is compatible with Willson's calibration
    routines.  Namely, each line of the output will represent one
    set of calibration points
        x_w y_w z_w u v
    Note: providing a valid ofpath is recommended as running ccpts() is 
    time consuming.  Call loadccpa() to read in the stored data and
    order it for PivLIB.

    ccpts() implicitly assumes that the sequence of images
    contained in the input list are of a single planar target
    positioned at known intervals along the z axis (ie, perpendicular
    to the light sheet).  The actual z-positions of the target are
    specified in the knowns array, the ordering of which must coincide 
    with the order of images in the input list.  NOTE: The z-position of 
    the target in the middle of the lightsheet should be set to 0.0 
    (the z-origin).

    knowns is an l x 3 array, where l is the number of knowns.  knowns
    must be ordered first in increasing z, then in increasing y, and 
    then in increasing x according to the coordinate system convention 
    of PivLIB.

    Note: ccpts() may fail for images with severe radial distortion, since
    target lines become curves in the input images under these conditions.
    The severe curvature of the target lines may cause the Hough transform
    to fail (or to generate spurious intersections).

    ccpts() makes use of the following tuning parameters:
        ccptune['pp_exec'] ------ /Bool/ Specifies whether image preprocessing
                                  should be performed prior to the Hough
                                  transform and convolution.  If so, the image
                                  will be broken into blocks of pp_bsize,
                                  the intensity values of each block will
                                  be raised to the power pp_pow, the mean of
                                  each block will be subtracted, and finally
                                  the block intensities will be divided by
                                  the maximum block value.
        ccptune['pp_bsize'] ----- /Int/ Two element tuple specifying the 
                                  (y,x) block size for image pre-processing.
                                  Recommended value: (32,32).
        ccptune['pp_p1pow'] ----- /Float/ Exponent for first pass power law 
                                  tranformation during pre-processing.  
                                  pp_p1pow > 1 will help compensate for 
                                  over-exposed or low contrast images.  
                                  pp_p1pow < 1 will accentuate low intensity 
                                  values.
        ccptune['pp_p2pow'] ----- /Float/ Exponent for second pass power law 
                                  transformation. This power law transformation 
                                  will be applied after pp_p1pow.  Generally, 
                                  a value of 0.8 is good as it tends to beef 
                                  up the lines slighly so the Hough transform 
                                  works well.
        ccptune['ht_dtheta'] ---- /Float/ Hough transform theta discretization
                                  size.  See pivutil.linht() for more details.
                                  Recommended value: pi/2048.
        ccptune['ht_drho'] ------ /Float/ Hough transform rho discretization
                                  size.  See pivutil.linht() for more details.
                                  Recommended value: 0.5
        ccptune['ht_show'] ------ /Bool/ Flag to show various images from
                                  Hough transform.  Once debugging is complete,
                                  should be set to False.
        ccptune['cc_krnl'] ------ /Float/ m x m array containing the 
                                  convolution kernel used for cross-
                                  correlation.  m should be an odd number.
                                  At present, ccpts() makes use of convolution 
                                  for the cross-correlation.  As a result, the 
                                  kernel should be symmetric.
        ccptune['cc_eps'] ------- /Float/ Epsilon for gaussian fit of cross-
                                  correlation peaks.  
                                  Recommended value: 1.3e-3
        ccptune['cc_maxits'] ---- /Int/ Maximum iterations for gaussian fit
                                  algorithm.  
                                  Recommended value: 100
        ccptune['cc_pf1bs'] ----- /Int/ Two element tuple specifying the 
                                  block size for Phase I of post-correlation
                                  peak finding. See pivutil.pkfind() for 
                                  more details.  Generally, the Phase I
                                  block size should be less than half the width
                                  between lines in images of the calibration
                                  target.
        ccptune['cc_thold'] ----- /Float/ Value of minimum threshold for
                                  Phase I peak finding.  See pivutil.pkfind()
                                  for more details.  If thold is too large,
                                  too many Phase I peaks will be discarded.
        ccptune['cc_pf2bs'] ----- /Int/ Two element tuple specifying the
                                  block size for Phase II of post-correlation
                                  peak finding. See pivutil.pkfind() for 
                                  more details.  Generally, the Phase II
                                  block size should be very nearly but
                                  slightly less than twice the width between 
                                  lines in images of the calibration target.
        ccptune['cc_subht'] ----- /Bool/ Flag to substitute HT intersection
                                  coordinates for any missing cross-correlation
                                  intersections.  The Hough transform is more 
                                  robust to missing intersections than cross-
                                  correlation. If True and cross-correlation
                                  fails to identify an intersection (that was
                                  obscured by an air bubble on the target, 
                                  say), the Hough transform results will be
                                  substituted.  The missing CC intersection
                                  will be identified in a plot with a double 
                                  circle.  ccpts() should ideally be run
                                  with cc_subht = False.  Only set to True if
                                  necessary (HT coordinates may not be
                                  as accurate as those from cross-correlation).
        ccptune['cc_show'] ------ /Bool/ Flag to show images from 
                                  pivutil.pkfind().  After debugging is
                                  complete, should be set to False.

    Returns an l x 5 array, ccpa, where l is the total number of knowns and
        ccpa[:,0:3] -- (z,y,x) coordinates of knowns [mm]
        ccpa[:,3:5] -- (v,u) coordinates of corresponding image intersections
                       [pixels].
    """

    print "STARTING: ccpts"
    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]

    pbsndx = zeros(2,dtype=int)
    pbendx = pbsndx.copy()

    mrbndx      = zeros((2,2),dtype=int)
    mrbndx[:,1] = rsize

    pp_exec   = ccptune['pp_exec']
    pp_bsize  = ccptune['pp_bsize']
    pp_p1pow  = ccptune['pp_p1pow']
    pp_p2pow  = ccptune['pp_p2pow']
    ht_dtheta = ccptune['ht_dtheta']
    ht_drho   = ccptune['ht_drho']
    ht_show   = ccptune['ht_show']
    cc_krnl   = ccptune['cc_krnl']
    cc_eps    = ccptune['cc_eps']
    cc_maxits = ccptune['cc_maxits']
    cc_pf1bs  = ccptune['cc_pf1bs']
    cc_thold  = ccptune['cc_thold']
    cc_pf2bs  = ccptune['cc_pf2bs']
    cc_subht  = ccptune['cc_subht']
    cc_show   = ccptune['cc_show']
    
    ccmxsrv = min(cc_pf2bs)/2.

    nscts = zeros((knowns.shape[0],2),dtype=float)
    ccpa  = zeros((knowns.shape[0],5),dtype=float)

    ccpa[:,0:3] = knowns

    peps = pi/6

    # Get location of intersections.
    tni = 0
    pni = 0 
    print " | Extracting intersections..."
    for i in range(len(imsin)):
        print " |  vvvvv Image: %i vvvvv " % i 
        mim = imsin[i][rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]].copy()
        
        # Preprocess the image.
        if ( pp_exec ):
            bsoy = pp_bsize[0]
            bsox = pp_bsize[1]

            nyblks = int( ceil( 1.*rsize[0]/bsoy ) ) 
            nxblks = int( ceil( 1.*rsize[1]/bsox ) ) 

            # Normalize the intensity image prior to convolution.
            for m in range(nyblks):
                pbsndx[0] = m*bsoy
                pbendx[0] = min(pbsndx[0] +pp_bsize[0],rsize[0])
                for n in range(nxblks):
                    pbsndx[1] = n*bsox
                    pbendx[1] = min(pbsndx[1] +pp_bsize[1],rsize[1])
        
                    block = mim[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]]
                    block = pow(block,pp_p1pow)
                    block = block/block.max()
        
                    npix = sum(where(block>0.,1.,0.))
                    npix = max(npix,1)
                    bave = sum(block)
                    bave = bave/npix
    
                    block = block -bave
                    block = where(block<0.,0.,block)
            
                    bmax = block.max()
                    if (bmax > 0.):
                        block = block/bmax
            
                    mim[pbsndx[0]:pbendx[0],pbsndx[1]:pbendx[1]] = block

            # Now apply the second power law scaling.
            mim = pow(mim,pp_p2pow)

        # Show intermediate results.
        if ( ( i == 4 ) and ( show ) ):
            plt.figure()
            plt.imshow(mim,cmap=plt.cm.gray,interpolation='nearest')
            plt.setp(plt.gca(),xticks=[],yticks=[])
            plt.title("Input Image after Specified Preprocessing")
            plt.show()

        # Get intersections via Hough transform.
        [hnscts,params,lhrtg] = pivutil.ccxtrct(mim,
                                                mrbndx,
                                                ht_dtheta,
                                                ht_drho,
                                                ht_show)

        ni = hnscts.shape[0]
        print " | Image: " +str(i) +" Intersections: " +str(ni) 

        # Check that the number of intersections for current image
        # matches the number for the previous image.
        #Xiyuan
	if (i==-1):
		plt.figure(figsize=(16,9))
		plt.imshow(mim,cmap=plt.cm.gray,interpolation='nearest')
		plt.scatter(hnscts[:,1],hnscts[:,0],s=3,c="red")
		plt.show()
		return
		#ni=pni
		#print "hnscts=",hnscts,"\nshape=",hnscts.shape
        if ( ( i != 0 ) and ( ni != pni ) ):
            print " | ERROR: Inter-frame intersection count mismatch."
            return

        # Get intersections using cross-correlation.
        cnv = ndimage.convolve(mim,cc_krnl,mode='constant')
        [cpks,pks] = pivutil.pkfind(cnv,cc_pf1bs,cc_thold,cc_pf2bs,cc_show)
        #Xiyuan
        pks = pks.astype(int)
        if ( ( pks.shape[0] != ni ) and not ( cc_subht ) ): 
            print " | ERROR: Inter-method intersection count mismatch."  
            print " |        Consider setting cc_subht = True."
            print " | --> HT: %i CC: %i" % (ni,pks.shape[0])
            return
        elif ( pks.shape[0] < ni ):
            print " | WARNING: Inter-method intersection count mismatch.  Will"
            print " |          substitute HT intersections." 
            print " | --> HT: %i CC: %i" % (ni,pks.shape[0])
        elif ( pks.shape[0] > ni ):
            print " | ERROR: Inter-method intersection count mismatch.  CC"
            print " |        intersections exceed HT intersections."
            print " | --> HT: %i CC: %i" % (ni,pks.shape[0])
            return

        # Perform a gaussian fit on the cross-correlation results.
        cnscts = pks.astype(float).copy()
        for m in range(pks.shape[0]):
            # Fit to gaussian.
            if ( ( pks[m,0] == 0 ) or ( pks[m,0] >= rsize[0] ) ):
                continue
            else:
		#Xiyuan
		#print pks[m,0]-1,pks[m,0]+2,pks[m,1]
                svec = pivutil.gfit(
                    cnv[(pks[m,0]-1):(pks[m,0]+2),pks[m,1]],
                    cc_maxits,
                    cc_eps)
                if ( compat.checkNone(svec) ):
                    continue
                else:
                    cnscts[m,0] = pks[m,0] +svec[1]

            if ( ( pks[m,1] == 0 ) or ( pks[m,1] >= rsize[1] ) ):
                continue
            else:
                svec = pivutil.gfit(
                    cnv[pks[m,0],(pks[m,1]-1):(pks[m,1]+2)],
                    cc_maxits,
                    cc_eps)
                if ( compat.checkNone(svec) ):
                    continue
                else:
                    cnscts[m,1] = pks[m,1] +svec[1]

        tni = tni +ni 
        pni = ni

        if ( tni > knowns.shape[0] ):
            print " | ERROR: Number of intersections exceeds knowns."
            return

        # Associate image intersections with knowns.  We need to find
        # the set of horizontal lines.
        lndx   = array(range(params.shape[0]))
        msk    = abs(params[:,0] -pi/2) < peps

        hln    = compress(msk,params[:,1])
        lndx   = compress(msk,lndx)

        shlndx = argsort(hln)
        lndx   = lndx[shlndx]

        # Now we can order the image intersections.
        indx = []
        for j in range(len(lndx)):
            indx.extend( lhrtg[ lndx[j] ] )
        indx = array(indx)

        onscts = hnscts[indx,:]

        # Extract the corresponding intersections from cross
        # correlation. 
        cmsng = []
        pkn   = range(cnscts.shape[0])   
        for m in range(ni):
            yv = cnscts[pkn,0] -onscts[m,0]
            xv = cnscts[pkn,1] -onscts[m,1]
            rv = yv*yv +xv*xv
#Xiyuan
            if (len(rv)<1):
		print " | WARNING: EMPTY rv. Using HT results for intersection at:"
                print " | --> (%f, %f)" % (onscts[m,0],onscts[m,1])
                cmsng.append(onscts[m,:])
            elif ( sqrt( rv.min() ) > ccmxsrv ):
                print " | WARNING: Using HT results for intersection at:"
                print " | --> (%f, %f)" % (onscts[m,0],onscts[m,1])
                cmsng.append(onscts[m,:])
            else:
                rvn = rv.argmin()
                onscts[m,:] = cnscts[pkn[rvn],:]
                del pkn[rvn]

        nscts[(tni-pni):tni,:] = onscts
        cmsng = array(cmsng)

        # Show intermediate results.  
        if ( ( ( i == 0 ) and ( show ) ) or ( cmsng.size > 0 ) ):
            tc = (onscts +10)
            #tc[:,0] = rsize[0] -tc[:,0]

            cp        = zeros((ni,3),dtype=int)
            cp[:,0:2] = (onscts.round()).astype(int)
            cp[:,2]   = 4
            cimg      = pivutil.drwcrc(rsize,cp)

            tstr = ""
            if ( cmsng.size > 0 ):
                tstr = " (Count Mismatch)"

                cp        = zeros((cmsng.shape[0],3),dtype=int)
                cp[:,0:2] = (cmsng.round()).astype(int)
                cp[:,2]   = 6

                cimg = cimg +pivutil.drwcrc(rsize,cp)
                cimg = cimg.clip(0.,1.)

            timg = imsin[i][rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]].copy()
            timg = where(cimg > 0.,2.,timg)
            #Xiyuan

            plt.figure(figsize=(16,9))
            plt.imshow(timg,interpolation="nearest",cmap=plt.cm.gray)
            plt.setp(plt.gca(),xticks=[],yticks=[])
            plt.title("Intersections" +tstr)

            dni = 1
            if ( ni > 200 ):
                dni = 10
            for i in range(0,ni,dni):
                plt.text(tc[i,1],tc[i,0],str(i),color='w',size=6)
	    '''    
            plt.figure()
            plt.imshow(timg,interpolation="nearest",cmap=plt.cm.gray)
            plt.setp(plt.gca(),xticks=[],yticks=[])
            plt.title("Intersections" +tstr)

            dni = 1
            if ( ni > 200 ):
                dni = 10
            for i in range(0,ni,dni):
                plt.text(tc[i,1],tc[i,0],str(i),color='w',size=6)
	    '''    
        print " | "

    if ( nscts.shape[0] != knowns.shape[0] ):
        print " | ERROR: Number of intersections and knowns do not match."
        return

    # Increment intersections by rbndx[:,0].
    nscts[:,0] = nscts[:,0] +rbndx[0,0]
    nscts[:,1] = nscts[:,1] +rbndx[1,0]

    ccpa[:,3:5] = nscts

    # Write out the results ordered appropriately for Willson's functions
    # (ie, x,y,z).
    if ( not compat.checkNone(ofpath) ):
        print " | Writing results to " +ofpath
        fh = open(ofpath, 'w')
        for i in range(nscts.shape[0]):
            fh.write( str(knowns[i,2]) +' '
                      +str(knowns[i,1]) +' '
                      +str(knowns[i,0]) +' '
                      +str( nscts[i,1]) +' '
                      +str( nscts[i,0]) +'\n')

        fh.close()

    
    print " | EXITING: ccpts"
    return ccpa


#################################################################
#
def dcorim(imin,camcal):
    """
    ----
    
    imin
    camcal
    
    ----    
    
    Corrects non-projective distortion in an image by mapping from image
    coordinates to undistorted sensor coordinates and then directly from
    undistorted back to image (ie, skipping distorted sensor).

    Note: dcorim performs no interpolation on the resulting image so
    images may possess tears.
    """
    # Initialization.
    v0  = camcal['v0']
    u0  = camcal['u0']
    dpy = camcal['dpy']
    dpx = camcal['dpx']
    sx  = camcal['sx']

    imdim  = imin.shape
    imsize = imin.size
    uimin = zeros(imdim,dtype=float)

    ndxmat = indices(imdim)
    yndx   = ndxmat[0,:,:].reshape(imsize)
    xndx   = ndxmat[1,:,:].reshape(imsize)

    dyndx = yndx.copy()
    dxndx = xndx.copy()

    # Convert to sensor coordinates. 
    yndx = (yndx -v0)*dpy
    xndx = (xndx -u0)*dpx/sx

    dspt = array([yndx,xndx]).transpose()

    # Convert from distorted to undistorted sensor coordinates.
    udspt = ds2uds(dspt,camcal)
    
    # Covert back to image coordinates.
    yndx = udspt[:,0]/dpy +v0
    xndx = sx*udspt[:,1]/dpx +u0

    yndx = yndx.round().astype(int) 
    xndx = xndx.round().astype(int)

    msk   = ( yndx < imdim[0] )*( yndx > 0 )
    yndx  = compress(msk,yndx)
    dyndx = compress(msk,dyndx)
    xndx  = compress(msk,xndx)
    dxndx = compress(msk,dxndx)

    msk   = ( xndx < imdim[1] )*( xndx > 0 )
    yndx  = compress(msk,yndx)
    dyndx = compress(msk,dyndx)
    xndx  = compress(msk,xndx)
    dxndx = compress(msk,dxndx)

    # Undistort.
    uimin[yndx,xndx] = imin[dyndx,dxndx]

    return uimin


#################################################################
#
def dltsolv(ccpa,camcal):
    """
    ----
    
    ccpa           # lx5 array of camera calibration points. 
    camcal         # Camera calibration dictionary.
    
    ----
        
    Computes the transformation matrix for the direct linear
    transform calibration technique and returns an updated camcal.

    NOTE: Modifying this routine without a firm handle on the 
    coordinate system used in PivLIB will cause endless consternation.

    ccpa should be an lx5 array of camera calibration parameters
    from ccpts().
    """
    # Initialization.
    dpy = camcal['dpy']
    dpx = camcal['dpx']

    ncamcal = camcal.copy()

    npts = ccpa.shape[0]

    lmat = zeros((2*npts,12),dtype=float)
    rmat = zeros((3,3),dtype=float)
    rvec = zeros(3,dtype=float)
    tvec = zeros(3,dtype=float)

    zv = ccpa[:,0]
    yv = ccpa[:,1]
    xv = ccpa[:,2]

    vv = ccpa[:,3]
    uv = ccpa[:,4]

    # Build the system matrix.
    for i in range(npts):
        rndx = i*2
        lmat[rndx,0] = xv[i]
        lmat[rndx,1] = yv[i]
        lmat[rndx,2] = zv[i]
        lmat[rndx,3] = 1.

        lmat[rndx,8:12]   = -lmat[rndx,0:4]*uv[i]
        lmat[rndx+1,4:8]  =  lmat[rndx,0:4]
        lmat[rndx+1,8:12] = -lmat[rndx,0:4]*vv[i]
        
    # Compute the solution to L m = 0.
    [u,s,vh] = linalg.svd(lmat,full_matrices=0)
    mmat     = vh.transpose()[:,11]
    mmat     = mmat.reshape((3,4)) 

    # Extract camera parameters.
    gamma = sqrt( sum( mmat[2,0:3]*mmat[2,0:3] ) )
    mmat  = mmat/gamma

    sig = sign(mmat[2,3])  # The World origin is in front of the camera ...
    Tz  = sig*mmat[2,3]    # therefore, Tz should be positive.

    rmat[0,0] = sig*mmat[2,2]
    rmat[0,1] = sig*mmat[2,1]
    rmat[0,2] = sig*mmat[2,0]

    v0 = sum(mmat[1,0:3]*mmat[2,0:3])
    u0 = sum(mmat[0,0:3]*mmat[2,0:3])

    fy = sqrt( sum(mmat[1,0:3]*mmat[1,0:3]) -v0**2 ) 
    fx = sqrt( sum(mmat[0,0:3]*mmat[0,0:3]) -u0**2 ) 

    f  = fy*dpy
    sx = fx*dpx/f

    rmat[1,0] = sig*(-v0*mmat[2,2] +mmat[1,2])/fy
    rmat[1,1] = sig*(-v0*mmat[2,1] +mmat[1,1])/fy
    rmat[1,2] = sig*(-v0*mmat[2,0] +mmat[1,0])/fy

    rmat[2,0] = sig*(-u0*mmat[2,2] +mmat[0,2])/fx
    rmat[2,1] = sig*(-u0*mmat[2,1] +mmat[0,1])/fx
    rmat[2,2] = sig*(-u0*mmat[2,0] +mmat[0,0])/fx
 
    Ty = sig*(-v0*mmat[2,3] +mmat[1,3])/fy
    Tx = sig*(-u0*mmat[2,3] +mmat[0,3])/fx

    tvec[:] = [Tz,Ty,Tx]

    # PivLIB uses Heikkila:1997's rotation convention (also same as Willson).
    # Rotate first about x, then y, then z.  See rvec2rmat() for equations.
    rvec[0] = arctan2(rmat[1,2],rmat[2,2])
    rvec[1] = arctan2(-rmat[0,2],rmat[2,2]*cos(rvec[0]) +rmat[1,2]*sin(rvec[0]))
    rvec[2] = arctan2(rmat[2,0]*sin(rvec[0]) -rmat[1,0]*cos(rvec[0]),
                      rmat[1,1]*cos(rvec[0]) -rmat[2,1]*sin(rvec[0]))

    # Recompute rmat with rvec.
    rmat = rvec2rmat(rvec)

    # Copy paramters into camcal.
    ncamcal['u0']   = u0
    ncamcal['v0']   = v0
    ncamcal['sx']   = sx
    ncamcal['f']    = f
    ncamcal['T']    = tvec
    ncamcal['R']    = rvec
    ncamcal['Rmat'] = rmat

    return ncamcal


#################################################################
#
def ds2uds(dspt,camcal):
    """
    ----

    dspt           # Point in distorted sensor space (y_d,x_d) coordinates.
    camcal         # Camera calibration dictionary.
    
    ----    
    
    Utility function to convert from distorted sensor to undistorted
    sensor coordinates.

    Returns an lx2 array of points in the undistorted sensor coordinate
    system.
    """

    # Initialization.
    k1  = camcal['k1']
    k2  = camcal['k2']
    p1  = camcal['p1']
    p2  = camcal['p2']
    s1  = camcal['s1']
    s2  = camcal['s2']

    dpa = array(dspt)
    if ( dpa.size == 2 ):
       dpa = dpa.reshape((1,2)) 

    # Compute the coordinates.
    yd  = dpa[:,0]
    yds = yd*yd
    xd  = dpa[:,1]
    xds = xd*xd
    
    rds = yds +xds

    rcor = 1. +k1*rds +k2*rds*rds 

    prd  = 2.*xd*yd
    yu = yd*rcor +p1*(rds +2.*yds) +p2*prd +s1*rds 
    xu = xd*rcor +p2*(rds +2.*xds) +p1*prd +s2*rds 

    udspt = array([yu,xu]).transpose()
    
    return udspt


#################################################################
#
def d_ds2uds(dspt,camcal):
    """
    ----
    
    dspt           # Point in distorted sensor space (y_d,x_d) coordinates.
    camcal         # Camera calibration dictionary.
    
    ----
        
    Utility function to compute the derivative of the functions used
    to convert from distorted sensor to undistorted sensor coordinates.

    Returns an lx array of 2x2 Jacobians, where l is the number of points.  
    Let f represent the distortion function for y_d and g represent 
    the distortion function for x_d.  The Jacobian matrix will be ordered 
    as
        | f_y f_x |
        | g_y g_x |
    """

    # Initialization.
    k1  = camcal['k1']
    k2  = camcal['k2']
    p1  = camcal['p1']
    p2  = camcal['p2']
    s1  = camcal['s1']
    s2  = camcal['s2']

    dpa = array(dspt)
    if ( dpa.size == 2 ):
       dpa = dpa.reshape((1,2)) 

    # Compute the Jacobian.
    yd  = dpa[:,0]
    yds = yd*yd
    xd  = dpa[:,1]
    xds = xd*xd

    rds = yds +xds
    rd  = sqrt(rds)
    rdc = pow(rd,3)

    rcor  = 1. +k1*rds +k2*rds*rds 
    drcor = 4.*(k1*rd +2.*k2*rdc)
    pdr   = xd*yd*drcor
    rdp1  = 4.*(rd +1.)
    rdy   = 4.*rd*yd
    rdx   = 4.*rd*xd

    fy = rcor +yds*drcor +yd*p1*rdp1 +2.*p2*xd +s1*rdy
    fx = pdr +(p1 +s1)*rdx +2.*p2*yd

    gy = pdr +(p2 +s2)*rdy +2.*p1*xd
    gx = rcor +xds*drcor +xd*p2*rdp1 +2.*p1*yd +s2*rdx

    jmat = array([[fy,gy],[fx,gx]]).transpose()
    
    return jmat


#################################################################
#
def im2wrld(imspt,camcal):
    """
    ----
    
    imspt          # Point in image space (v,u) coordinates.
    camcal         # Camera calibration dictionary.
    
    ----
        
    Transforms a point in image-space to World-space.  In the
    notation of Willert:2006, im2wrld converts from uv-space to
    XYZ_w-space.  When backprojecting onto the World plane, Z_w
    must be specified and it is assumed to be zero here.

    imspt can be an lx2 array of points where l is the number of 
    points to be converted.  Each point must be ordered as (v,u)

    Returns an lx3 array of world points ordered as (z,y,x).
    """
    # Initialization.
    tvec = camcal['T']
    rmat = matrix(camcal['Rmat'])
    f    = camcal['f']
    sx   = camcal['sx']
    dpx  = camcal['dpx']
    dpy  = camcal['dpy']
    u0   = camcal['u0']
    v0   = camcal['v0']

    ipa = array(imspt)
    if ( ipa.size == 2 ):
       ipa = ipa.reshape((1,2)) 

    bvec = ones((ipa.shape[0],3),dtype=float)

    # Convert from image space to distorted sensor coordinates.
    bvec[:,1] = (ipa[:,0] -v0)*dpy
    bvec[:,2] = (ipa[:,1] -u0)*dpx/sx

    # Convert from distorted to undistorted sensor coordinates.
    bvec[:,1:3] = ds2uds(bvec[:,1:3],camcal)

    # Convert from undistorted sensor to World coordinates.
    mrmat = rmat.copy()
    mrmat[1:3,:] = mrmat[1:3,:]*f
    mrmat[0,0]   =   tvec[0]
    mrmat[1,0]   = f*tvec[1]
    mrmat[2,0]   = f*tvec[2]

    try:
        wrldpt = linalg.solve(mrmat,bvec.transpose())
    except linalg.LinAlgError:
        return None

    wrldpt      = wrldpt.transpose()   # (1/z_c, Y_w/z_c, X_w/z_c)
    wrldpt[:,0] = 1./wrldpt[:,0]
    wrldpt[:,1] = wrldpt[:,0]*wrldpt[:,1]
    wrldpt[:,2] = wrldpt[:,0]*wrldpt[:,2]

    wrldpt[:,0] = 0.    # Force Z_w = 0.

    return wrldpt


#################################################################
#
def importwcc(ifpath,ofpath=None):
    """
    ----
    
    ifpath,         # Path to input file.
    ofpath=None     # Path to output pivpickle file.
    
    ----
        
    Willson's camera calibration procedure is very similar to
    PivLIB's (PivLIB's coordinate system and equation formulation
    are essentially identical).  As a result, his code is a good 
    reference (http://www.cs.cmu.edu/~rgw/TsaiCode.html) and makes
    a very nice, working benchmark to ensure PivLIB code mods
    aren't doing something strange.

    importwcc() reads a camera calibration output file produced by 
    Willson's code and returns the data in a camcal dictionary.  
    importcc() also pickles the dictionary using the pivpickle 
    module (i.e., stores it to a portable file) for future use if 
    ofpath is supplied. 

    Note: Willson's dy and dx parameters are assumed equal to dpy
    and dpx.  As a result, dy and dx are not retained in camcal.
    """
    # Initialization.
    tvec = zeros(3,dtype=float)
    rvec = zeros(3,dtype=float)
    rmat = zeros((3,3),dtype=float)

    camcal = {}

    # Get the data.
    fh = open(ifpath,'r')
    dmy           =   int( round( float( fh.readline().strip() ) ) )
    dmy           = float( fh.readline().strip() )
    dmy           = float( fh.readline().strip() )
    dmy           = float( fh.readline().strip() )
    camcal['dpx'] = float( fh.readline().strip() )
    camcal['dpy'] = float( fh.readline().strip() )
    camcal['u0']  = float( fh.readline().strip() )
    camcal['v0']  = float( fh.readline().strip() )
    camcal['sx']  = float( fh.readline().strip() )
    camcal['f']   = float( fh.readline().strip() )
    camcal['k1']  = float( fh.readline().strip() )

    camcal['k2'] = 0.
    camcal['p1'] = 0.
    camcal['p2'] = 0.
    camcal['s1'] = 0.
    camcal['s2'] = 0.

    tvec[2]       = float( fh.readline().strip() )    
    tvec[1]       = float( fh.readline().strip() )
    tvec[0]       = float( fh.readline().strip() )
    camcal['T']   = tvec

    rvec[2]       = float( fh.readline().strip() )
    rvec[1]       = float( fh.readline().strip() )
    rvec[0]       = float( fh.readline().strip() )
    camcal['R']   = rvec

    fh.close()

    rmat[0,0] =  cos(rvec[2])*cos(rvec[1])
    rmat[0,1] =  cos(rvec[1])*sin(rvec[2])
    rmat[0,2] = -sin(rvec[1])
    rmat[1,0] =  cos(rvec[2])*sin(rvec[1])*sin(rvec[0]) \
                -cos(rvec[0])*sin(rvec[2])
    rmat[1,1] =  sin(rvec[2])*sin(rvec[1])*sin(rvec[0]) \
                +cos(rvec[2])*cos(rvec[0])
    rmat[1,2] =  cos(rvec[1])*sin(rvec[0])
    rmat[2,0] =  sin(rvec[2])*sin(rvec[0]) \
                +cos(rvec[2])*cos(rvec[0])*sin(rvec[1])
    rmat[2,1] =  cos(rvec[0])*sin(rvec[2])*sin(rvec[1]) \
                -cos(rvec[2])*sin(rvec[0])
    rmat[2,2] =  cos(rvec[1])*cos(rvec[0])
    camcal['Rmat'] = rmat

    if ( not compat.checkNone(ofpath) ):
        savecamcal(camcal,ofpath)

    return camcal


#################################################################
#
def initcamcal(dpy,dpx):
    """
    ----
    
    dpy            # Physical pixel height parallel to v-axis [mm].
    dpx            # Physical pixel width parallel to u-axis [mm].
    
    ----
        
    Initializes a camcal dictionary with the only two user-specified
    parameters.  Consult the camera specifications for pixel
    dimensions.
    """
    # Initialization.
    tvec = zeros(3,dtype=float)
    rvec = zeros(3,dtype=float)
    rmat = zeros((3,3),dtype=float)

    camcal = {}

    # Setup the dictionary.
    camcal['dpy'] = dpy
    camcal['dpx'] = dpx

    camcal['u0'] = 0.
    camcal['v0'] = 0.
    camcal['sx'] = 0.
    camcal['f']  = 0.
    camcal['k1'] = 0.
    camcal['k2'] = 0.
    camcal['p1'] = 0.
    camcal['p2'] = 0.
    camcal['s1'] = 0.
    camcal['s2'] = 0.

    camcal['T']    = tvec
    camcal['R']    = rvec
    camcal['Rmat'] = rmat

    return camcal


#################################################################
#
def initknowns(nknowns,kspc,ccrot=0):
    """
    ----
    
    nknowns   # Tuple containing number of knowns, ordered as (w, v, u).
    kspc      # Tuple or list containing world spacing of knowns (z, y, x).
    ccrot=0   # Camera coordinate system rotation flag.
    
    ----
        
    Convenience function to construct an array of knowns based
    on the number of knowns in each dimension and the spacing
    between the knowns (in world coordinates).  

    nknowns should be ordered as (w, v, u) to match the
    order in which the intersections will be extracted from the images.
    Using this convention, w will always correspond to the number
    of z-values, while v and u will correspond to the number of
    knowns along the vertical and horizontal axes of the image,
    respectively.  

    The (y,x) origin of the knowns will be taken consistent with 
    pivlib coordinate conventions.  The z-origin will be set to
    the middle z-plane (unless kspc[0] is an array).

    The z entry for kspc can be an array of of exact values for
    z-locations instead of a scalar spacing.  If the z entry is an
    array, the number of values must match that specified in nknowns.
    The z-locations themselves should be arranged in increasing order.

    initknowns() is capable of handling 4 basic orientations of the 
    camera coordinates with respect to world coordinates.  This 
    functionality permits the cameras to be rotated a multiple of 
    pi/2 while automatically compensating for the rotation (so that
    the orientation of world coordinates is preserved).  The camera 
    rotation angle will taken to be taken to be ccrot*pi/2, specified
    in a counter-clockwise manner about the z-axis in camera
    coordinates.  

    NOTE: kspc values do not need to be altered for camera rotation
    (ie, keep kspc positive).

    NOTE: Do not forget that positive z points away from the camera
    in PivLIB's coordinate system.

    NOTE: kspc should be in world units (e.g., mm).

    NOTE: Although not completely necessary, for symmetry, the number 
    of knowns in the z-direction should be odd. 

    Returns knowns, an lx3 array of knowns.  Knowns is ordered first
    by z, then y, then x.
    """
    # Initialization
    tnkn   = nknowns[0]*nknowns[1]*nknowns[2]
    knowns = zeros((tnkn,3),dtype=float)
    
    print "ccrot %i" % ccrot

    # Create the array of knowns.
    ndxm = indices((nknowns[0],nknowns[1],nknowns[2]))
    zndx = ndxm[0,:,:].reshape(tnkn)
    if ( ccrot == 0 ):
        yndx = ndxm[1,:,:].reshape(tnkn)
        xndx = ndxm[2,:,:].reshape(tnkn)
    elif ( ccrot == 1):
        yndx = ndxm[2,:,:].reshape(tnkn)        
        xndx = ndxm[1,:,:].reshape(tnkn)[::-1]
    elif ( ccrot == 2 ):
        yndx = ndxm[1,:,:].reshape(tnkn)[::-1]
        xndx = ndxm[2,:,:].reshape(tnkn)[::-1]
    elif ( ccrot == 3 ):
        yndx = ndxm[2,:,:].reshape(tnkn)[::-1]  
        xndx = ndxm[1,:,:].reshape(tnkn)                    
    else:
        print "ERROR: ccrot must be 0, 1, 2, or 3."
        return None

    # If kspc[0] is an array, use the values provided.
    zspc = array(kspc[0])
    if ( zspc.size > 1 ):
        zndx = array(zndx)
        kndx = array(range(tnkn))
        for i in range(nknowns[0]):
            msk = ( zndx == i )
            msk = compress(msk,kndx)
            knowns[msk,0] = zspc[i]
    else:
        zndx = zndx -int(nknowns[0]/2)

        knowns[:,0] = zndx*kspc[0]

    knowns[:,1] = yndx*kspc[1]
    knowns[:,2] = xndx*kspc[2]

    return knowns


#################################################################
#
def lmfun(prma,ccpa,camcal):
    """
    ----
    
    prma        # Camcal as a vector.
    ccpa        # lx5 array of camera calibration points.
    camcal      # Camera calibration dictionary.
    
    ----
        
    Utility function for Levenberg-Marquardt optimization.
    lmfun() takes prma, puts it in a dictionary, and then 
    converts the world and image points of ccpa to undistorted
    sensor coordinates (where the error for the Levenberg-
    Marquardt method is computed). 

    Returns the L2 norm of the error for each point in ccpa.
    """
    # Initialization.
    ncamcal = prma2camcal(prma,camcal)

    v0  = ncamcal['v0']
    u0  = ncamcal['u0']
    sx  = ncamcal['sx']
    dpy = ncamcal['dpy']
    dpx = ncamcal['dpx']

    # Compute the error in the undistorted coordinate system.
    w2u = wrld2uds(ccpa[:,0:3],ncamcal)

    i2dy = (ccpa[:,3] -v0)*dpy
    i2dx = (ccpa[:,4] -u0)*dpx/sx
    i2d  = array([i2dy,i2dx]).transpose()
    i2u  = ds2uds(i2d,ncamcal)

    dlta = i2u -w2u
    dlta = sqrt(sum(dlta*dlta,1))

    return dlta


#################################################################
#
def loadcamcal(ifpath):
    """
    ----
    
    ifpath         # Path to pivpickled calibration dictionary.
    
    ----
        
    Loads a pivpickled version of the camcal dictionary.
    """
    camcal = pivpickle.pklload(ifpath)

    return camcal


#################################################################
#
def loadccpa(ifpath):
    """
    ----
    
    ifpath         # Path to stored ccpa data.
    
    ----
        
    Convenience function to load calibration data stored in a
    text file.

    The data is expected to be ordered appropriately for use
    by Willson's functions -- as x_w y_w z_w u v.  Output
    stored by ccpts() is correctly ordered.
    """
    fh = open(ifpath)
    cc = loadtxt(fh)
    fh.close()

    ccpa = zeros((cc.shape[0],5),dtype=float)

    wpts = cc[:,0:3]
    ipts = cc[:,3:5]
    
    wpts = wpts[:,::-1]
    ipts = ipts[:,::-1]

    ccpa[:,0:3] = wpts
    ccpa[:,3:5] = ipts

    return ccpa


#################################################################
#
def optimsolv(ccpa,camcal,nits = 1000):
    """
    ----
    
    ccpa           # lx5 array of camera calibration points. 
    camcal         # Camera calibration dictionary.
    nits = 1000    # Maximum number of iterations.
    
    ----
        
    Computes an optimized camera calibration using SciPy's
    leastsq() (which is a wrapper around MINPACK's lmdif()).

    camcal should be initialized to some reasonable values
    (eg, by calling dltsolv()).

    ccpa should be an lx5 array of camera calibration parameters
    from ccpts().

    Returns an updated camcal.
    """
    # Initialization.
    prma = camcal2prma(camcal)

    # Do the optimization.
    [prma,ier] = optimize.leastsq(lmfun,prma,(ccpa,camcal),maxfev=nits)

    # Update the camcal.
    ncamcal = prma2camcal(prma,camcal)

    return ncamcal


#################################################################
#
def prma2camcal(prma,camcal):
    """
    ----
    
    prma           # Camera calibration array.
    camcal         # Camera calibration dictionary.
    
    ----
        
    Converts camera calibration array into a dictionary of
    parameters and inserts those parameters into a copy of
    camcal.
    """
    # Initialization.
    ncamcal = camcal.copy()
    tvec    = zeros(3, dtype=float)
    rvec    = zeros(3, dtype=float) 

    # Set up the dictionary.
    tvec[0] = prma[8] 
    tvec[1] = prma[9] 
    tvec[2] = prma[10] 

    rvec[0] = prma[11] 
    rvec[1] = prma[12] 
    rvec[2] = prma[13]

    rmat = rvec2rmat(rvec)

    ncamcal['v0']   = prma[0] 
    ncamcal['u0']   = prma[1] 
    ncamcal['sx']   = prma[2] 
    ncamcal['f']    = prma[3] 
    ncamcal['k1']   = prma[4] 
    ncamcal['k2']   = prma[5]
    ncamcal['p1']   = prma[6]
    ncamcal['p2']   = prma[7]
    ncamcal['s1']   = prma[14]
    ncamcal['s2']   = prma[15]

    ncamcal['T']    = tvec
    ncamcal['R']    = rvec
    ncamcal['Rmat'] = rmat

    return ncamcal


#################################################################
#
def rvec2rmat(rvec):
    """
    ----
    
    rvec           # Vector of Euler angles.
    
    ----
        
    Builds rotation matrix from Euler angles.  Rvec should be
    ordered as (Rz, Ry, Rx).

    PivLIB uses the rotation convention used by Heikkila:1997
    and Willson.  Rotations are applied first about x, then y,
    then z.
    """
    # Initialization.
    rmat = zeros((3,3),dtype=float)

    # Compute the rotation matrix.
    rmat[0,0] =  cos(rvec[2])*cos(rvec[1])
    rmat[0,1] =  cos(rvec[1])*sin(rvec[2])
    rmat[0,2] = -sin(rvec[1])
    rmat[1,0] =  cos(rvec[2])*sin(rvec[1])*sin(rvec[0]) \
                -cos(rvec[0])*sin(rvec[2])
    rmat[1,1] =  sin(rvec[2])*sin(rvec[1])*sin(rvec[0]) \
                +cos(rvec[2])*cos(rvec[0])
    rmat[1,2] =  cos(rvec[1])*sin(rvec[0])
    rmat[2,0] =  sin(rvec[2])*sin(rvec[0]) \
                +cos(rvec[2])*cos(rvec[0])*sin(rvec[1])
    rmat[2,1] =  cos(rvec[0])*sin(rvec[2])*sin(rvec[1]) \
                -cos(rvec[2])*sin(rvec[0])
    rmat[2,2] =  cos(rvec[1])*cos(rvec[0])

    return rmat


#################################################################
#
def savecamcal(camcal,ofpath):
    """
    ----
    
    camcal
    ofpath         # Path to pivpickled calibration dictionary.
    
    ----
        
    Saves a pivpickled version of the camcal dictionary.
    """
    pivpickle.pkldump(camcal,ofpath)


#################################################################
#
def uds2ds(udspt,camcal):
    """
    ----
    
    udspt          # Point in undistorted sensor space (y_u,x_u) coordinates.
    camcal         # Camera calibration dictionary.
    
    ----
        
    Utility function to convert from undistorted sensor to distorted
    sensor coordinates.  

    PivLIB's distortion model results in a non-linear coupled system
    of equations.  Therefore for each point in the undistorted
    sensor plane, a 2x2 system of equations must be solved iteratively
    using Newton-Raphson.

    Returns an lx2 array of points in the distorted sensor coordinate
    system.
    """

    # Initialization.
    eps   = 1.e-16
    nreps = 1.e-8    # Should be much smaller than dpy and dpx.  

    maxits = 20

    upa = array(udspt)
    if ( upa.size == 2 ):
       upa = upa.reshape((1,2)) 

    pts   = upa.shape[0]
    ptndx = array(range(pts))

    dspt = upa.copy()

    hvn = zeros(pts,dtype=float)

    # Start of Newton-Raphson loop.
    for n in range(maxits):
        jmat = d_ds2uds(dspt[ptndx,:],camcal)
        fun  = upa[ptndx,:] -ds2uds(dspt[ptndx,:],camcal)

        # Solve each system of equations using map.
        hv = map(pivlinalg.dsolve,list(jmat),list(fun))
        hv = array(hv)

        dspt[ptndx,:] = dspt[ptndx,:] +hv

        hvn = sqrt(sum(hv*hv,1))

        # Remove those points that have converged from further iteration.
        msk   = hvn > nreps
        ptndx = compress(msk,ptndx)
        hvn   = compress(msk,hvn)

        if ( ptndx.size == 0 ):
            break

    return dspt


#################################################################
#
def wrld2cam(wrldpt,camcal):
    """
    Converts from world to intermediate camera coordinates.

    wrldpt can be an lx3 array of points where l is the number of 
    points to be converted.  Each point must be ordered as (z,y,x).

    Returns an lx3 array of camera points ordered as (z_c,y_c,x_c).
    """
    # Initialization.
    tvec = camcal['T']
    rmat = matrix(camcal['Rmat'])

    wpa = array(wrldpt)
    if ( wpa.size == 3 ):
       wpa = wpa.reshape((1,3)) 

    # Convert to intermediate camera coordinate system.
    xc      = ( rmat*wpa.transpose() ).transpose()
    xc      = array(xc)
    xc[:,0] = xc[:,0] +tvec[0]
    xc[:,1] = xc[:,1] +tvec[1]
    xc[:,2] = xc[:,2] +tvec[2]

    return xc

#################################################################
#
def wrld2uds(wrldpt,camcal):
    """
    Converts from world to undistorted sensor coordinates.

    wrldpt can be an lx3 array of points where l is the number of 
    points to be converted.  Each point must be ordered as (z,y,x)

    Returns an lx2 array of undistorted sensor points ordered 
    as (y_u,x_u).    
    """
    # Initialization.
    f = camcal['f']
  
    wpa = array(wrldpt)
    if ( wpa.size == 3 ):
       wpa = wpa.reshape((1,3)) 

    udspt = zeros((wpa.shape[0],2),dtype=float)

    # Convert from world to intermediate camera coordinates.
    xc = wrld2cam(wpa,camcal)
  
    # Convert to undistorted sensor coordinates.
    fdzc = f/xc[:,0]

    udspt[:,0] = fdzc*xc[:,1]
    udspt[:,1] = fdzc*xc[:,2]

    return udspt


#################################################################
#
def wrld2im(wrldpt,camcal):
    """
    ----
    
    wrldpt         # Point in world space (z,y,x) coordinates.
    camcal         # Camera calibration dictionary.
    
    ----
        
    Transforms a point in World-space to image-space.  In the
    notation of Willert:2006, wrld2im converts from XYZ_w-space to
    uv-space.

    NOTE: Because of the distortion model used by PivLIB, 
    conversion of world to image coordinates is time consuming.

    wrldpt can be an lx3 array of points where l is the number of 
    points to be converted.  Each point must be ordered as (z,y,x)

    Returns an lx2 array of image points ordered as (v,u).
    """
    # Initialization.
    sx   = camcal['sx']
    dpx  = camcal['dpx']
    dpy  = camcal['dpy']
    u0   = camcal['u0']
    v0   = camcal['v0']

    wpa = array(wrldpt)
    if ( wpa.size == 3 ):
       wpa = wpa.reshape((1,3)) 

    # Convert to undistorted sensor coordinates.
    imspt = wrld2uds(wpa,camcal)

    # Convert to distorted sensor coordinates.  
    imspt = uds2ds(imspt,camcal)

    # Convert to image coordinates.
    imspt[:,0] = v0 +imspt[:,0]/dpy
    imspt[:,1] = u0 +sx*imspt[:,1]/dpx

    return imspt


