"""
Filename:  __init__.py
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
  PivLIB provides a computational framework for Particle Image
  Velocimetry.  

  PivLIB assumes all images are m rows by n columns (a 640x480
  image has 480 rows and 640 columns) with xy-coordinate system
  oriented at the upper left corner of the image.
    *----> x-axis
    |
    |
    v  y-axis

  This coordinate system convention is used by graphics cards and 
  other image processing software/hardware.  NOTE:  Care must be taken
  to ensure that coordinate system conventions of post-processing
  software (eg, ParaView) are understood.  (ParaView initially flips 
  the image about the horizontal axis).
  
  The arrays used by PivLIB ALWAYS reference the first array index to
  the y-axis and the second index to the x-axis.  This convention is
  not necessarily used by external image processing libraries.

  In the event that PivLIB processes 3D data, the positive z-axis will be 
  taken to point into the screen (or, equivalently, away from the 
  camera).

  PivLIB expects all images to be normalized with pixel values ranging
  from 0. - 1.  

  The overall operation of several core PivLIB functions are governed by 
  a user-specified dictionary containing various configuration parameters.  
  Parameters are grouped by application using the scheme
      gp ---- Global Parameter
      ir ---- Image Registration
      of ---- Optical Flow
      pg ---- Photogrammetric
      tc ---- Thermochromic
  All parameters must be specified.  The format of the dictionary is as 
  follows:
    pivdict['gp_rbndx'] ----- /Int/ 2x2 array giving the bounding indices
                              for the region to be analyzed (following the
                              Python convention for slice bounds).  Values are
                              as follows:
                                  gp_rbndx[0,0] -- ystart
                                  gp_rbndx[0,1] -- ystop +1
                                  gp_rbndx[1,0] -- xstart
                                  gp_rbndx[1,1] -- xstop +1
                              Example:  To analyze a region starting at pixel
                              (10,10) and extending 10 pixels in each direction,
                              gp_rbndx = [[10,20],[10,20]].
    pivdict['gp_bsize'] ----- /Int/ Two element tuple giving [y,x] block size 
                              used for optical flow and temperature extraction
                              in pixels.  Input images will be tiled into 
                              blocks of bsize.  
    pivdict['gp_bolap'] ----- /Int/ Two element tuple giving the [y,x] overlap
                              between blocks in pixels.  If bolap = 0, 
                              there will be no overlap between blocks.
                              NOTE:  0 <= bolap < bsize.
    pivdict['gp_bsdiv'] ----- /Int/ Block subdivision factor.  If gp_bsdiv > 1,
                              ofcomp() will compute the flow using blocks 
                              of size gp_bsize first.  Then these coarse flow
                              results will be used as initial values, and
                              the flow is recomputed using blocks of size
                              gp_bsize/gp_bsdiv.
                              NOTE: Either gp_bsdiv can be set greater than 1
                              or gp_bolap can be set greater than zero.  Both
                              cannot be used.
                              NOTE: gp_bsize must be evenly divisible by
                              gp_bsdiv.
    pivdict['ir_eps'] ------- /Float/ Termination threshold for iterative 
                              loops.  Set to 0.003 for most work.
    pivdict['ir_maxits'] ---- /Int/ Maximum number if iterations for iterative
                              loops.
    pivdict['ir_mineig'] ---- /Float/ Sets the minimum acceptable eigenvalue 
                              for the system matrix used by the Lucas-Kanade 
                              algorithm.  This is semi-equivalent to the 
                              effects of a regularization parameter, but here, 
                              matrices with deficient eigenvalues won't be 
                              registered at all.  A reasonable value is 
                              ~0.05 for PIV work.
    pivdict['ir_imthd'] ----- /String/ Interpolation method.  Possible values:
                                  'C' -- Bicubic interpolation.
                                  'L' -- Bilinear interpolation.
                              Bicubic interpolation produces higher quality
                              images with continuous first and second
                              derivatives.  Recommended value is 'C'.
    pivdict['ir_iedge'] ----- /Int/ Specifies edge treatment technique for
                              bicubic interpolation.  Possible values:
                                  0 - Use bicubic interpolation to generate
                                      entire output image.
                                  1 - Use nearest neighbor to generate edge
                                      pixels of output image.
                              Set ir_iedge = 0.
    pivdict['ir_tps_csrp'] -- /Float/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Percentile of pixels to reject during contrast
                              stretching.  Recommended value is 0.1.
    pivdict['ir_tps_ithp'] -- /Float/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Intensity threshold percentile for feature points.
                              Recommended value is 95.                              
    pivdict['ir_tps_wsize'] - /Int/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              2-element list specifying the [y,x] feature point
                              extraction window size in pixels.  Recommended 
                              value is [5,5].
    pivdict['ir_tps_sdmyf'] - /Float/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Synthetic dummy point fraction.  Recommended 
                              value is 0.1.                              
    pivdict['ir_tps_alpha'] - /Float/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Intensity similarity weighting for cost matrix.
                              Recommended value is 0.9.
    pivdict['ir_tps_beta'] -- /Float/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Weighting of spring force for cost matrix.
                              Recommended value is 1.5.
    pivdict['ir_tps_csize'] - /Int/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Feature point cluster size.  Recommended
                              value is 7.
    pivdict['ir_tps_nits'] -- /Int/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Number of registration iterations.  Recommended
                              value is 30.    
    pivdict['ir_tps_scit'] -- /Int/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Schedule to run shape context correspondence.
                              Recommended value is 10.    
    pivdict['ir_tps_annl'] -- /Float/ Auxiliary registration with TPS.  See
                              esttpswarp() documentation.
                              Controls rate of annealing for TPS regularization.
                              Recommended value is 0.98.
    pivdict['of_maxdisp'] --- /Float/ Two element tuple giving the magnitude of 
                              the maximum acceptable incremental displacement
                              computed during each stage of ofcomp().  Should 
                              be less than or equal to border specified
                              with gp_rbndx.
    pivdict['of_rmaxdisp'] -- /Float/ Two element tuple giving the magnitude 
                              of the reduced maxdisp.  After Stage I
                              registration, Stage II and Stage III will
                              attempt to refine Stage I results.  The change
                              to Stage I results will limited to of_rmaxdisp.
                              Recommended value is [5,5].
    pivdict['of_highp'] ----- /Boolean/ High precision flag.  If set to 'True',
                              a third stage of image registration will 
                              proceed using Lukas-Kanade.  Lukas-Kanade
                              can reduce absolute errors in displacement
                              magnitude and direction by a factor of 5
                              or more for some blocks.
    pivdict['of_hrsc'] ------ /Boolean/ Flag to enable irsctxt() usage with
                              subdivided blocks.  If of_hrsc is False,
                              irsctxt() will only be used for coarse blocks
                              when the normalized cross correlation coefficient
                              is below of_nccth.  If of_hrsc is True, 
                              irsctxt() can be called for coarse and fine
                              blocks as needed.  Recommended value is True.
    pivdict['of_nccth'] ----- /Float/ Minimum acceptable normalized cross
                              correlation coefficient before auxiliary 
                              measures are taken.  If low resolution (ie, 
                              gp_bsdiv = 1) Stage II registration fails,
                              the image blocks will be registered using 
                              point correspondences and thin-plate splines
                              (TPS).  Valid values are 0.0 - 1.0. Larger 
                              values will result in more blocks being 
                              processed using the auxiliary system.  To 
                              disable these auxiliary techniques completely,
                              set of_nccth = -1.  Recommended value is 0.7.
    pivdict['of_bsdcmx'] ---- /Float/ Enforces quality monitoring of
                              normalized cross-correlation coefficient during
                              block subdivision.  The difference between the NCC
                              coefficient produced for the coarse block and 
                              during block subdivision must be less than or 
                              equal to of_bsdcmx otherwise the coarse block
                              results will be retained.  This parameter helps
                              ensure that block subdivision does not degrade
                              coarse results.  If of_bsdcmx = 0.0, then 
                              subdivided NCC coefficient must always be at
                              least as large as coarse results to retain the 
                              subdivided value.  Set to 1.0 to keep subdivided
                              results regardless of NCC coefficient. Recommended
                              value is 0.0.
    pivdict['pg_camcal'] ---- /Object/ List of camera calibration 
                              dictionaries ordered according to camera
                              number.
    pivdict['pg_wicsp'] ----- /Object/ List of world to image correspondence
                              dictionaries ordered according to camera.
    pivdict['tc_tlccal'] ---- /Object/ Thermochromic calibration dictionary.
                              Only used by tlclib.tfcomp().
    pivdict['tc_tlccam'] ---- /Int/ Index into pg_camcal indicating the 
                              photogrammetric calibration dictionary of 
                              the camera used during thermochromic calibration.
    pivdict['tc_ilvec'] ----- /Float/ Three element tuple giving the 
                              illumination unit vector [z,y,x].  This vector
                              points along the direction of propagation for 
                              unscattered photons from the lightsheet source.
    pivdict['tc_interp'] ---- /Boolean/ Interpolate blocks instead of 
                              subdividing.  If interp is True, 'gp_bsdiv' is not
                              used during hue to temperature mapping.  Instead, 
                              blocks of size 'gp_bsize' are used throughout and 
                              the results are interpolated onto a fine grid (as 
                              though gp_bsdiv had been used) after hue has been
                              converted to temperature.  If interp is False, hue 
                              will be computed for blocks of size 
                              'gp_bsize'/'gp_bsdiv'.  See documentation for 
                              tlclib.tfcomp().  Recommended value is True.
"""

import pivdata
from pivdata import PIVVar, PIVEpoch, PIVData, cpivvar, loadpivdata

import pivir

import pivof
from pivof import ofcomp, tcfrecon

import pivutil
from pivutil import imread, imwrite, getpivcmap, constrch, bimedfltr, rgb2hsi,\
    padfltr, padimg

import pivpgcal
from pivpgcal import calibrate, ccpts, im2wrld, initcamcal, initknowns,\
    loadcamcal, loadccpa, savecamcal, wrld2im

import pivpg
from pivpg import dscwrld, loadwicsp, prjim2wrld, wrld2imcsp

import pivlinalg

import pivpickle
from pivpickle import pkldump, pklload

import pivpost
from pivpost import divfltr, medfltr, gsmooth

import pivsim
from pivsim import *

import pivcolor as cm