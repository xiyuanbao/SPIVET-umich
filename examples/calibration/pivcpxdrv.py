"""
Filename:  pivcpxdrv.py
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
  Script for extracting PIV calibration points.
  
  The calibration procedure images a grid like target with known line
  spacing at multiple positions spanning the full thickness of the 
  lightsheet.  This script analyzes those images and extracts the pixel 
  coordinates of the intersections using two methods:
     1) Hough transform
     2) cross-correlation
  These points are then stored in files for later processing by the
  pivpgcdrv.py script.
     
  More information on the calibratin technique can be found in the
  documentation for the pivlib.ccpts() function.  In essence,
  the Hough transform is used principally to order the intersections,
  while cross-correlation is used to extract the coordinates of the
  intersections.  Periodically, an intersection can't be found by
  cross-correlation and the location for the intersection as computed
  by the Hough transform will be used instead (debug images are generated
  that show where this occurs).
  
  The recommended way for running this script is to do so within an
  active python session using Python's import command.  This way the
  user can actually view the debug images.
"""

from spivet import pivlib
import pylab
import math
from numpy import *
from PIL import Image

import os,sys

# >>>>> ANALYSIS SETUP <<<<<

# Calibration plane world z-coordinates [mm].  These are the positions
# of the calibration target relative to the middle of the lightsheet.
kzv = (4.68,3.51,2.27,1.01,0.,-0.94,-2.11,-2.79,-3.60)

# Known calibration point world [y,x] spacing [mm].
ccpwsp = [6.35,6.35] 

# Number of calibration points [z,y,x].  
ccpdim = [len(kzv),27,37]     

# Camera rotation factor for initknowns().  Camera will be rotated 
# ccrot*pi/2 counterclockwise about the z-axis in camera coordinates,
# where ccrot is either 0, 1, 2, or 3.
ccrot = 0

# Base path to images.
bpath  = "/Volumes/globalstore/Projects/PIV/Datasets/SYRUPCAL-04192008/DATA"

# Base output file name for calibration points.
bccpdata = 'SYRUPCAL-04192008_CC_OUT_'

# Setup the ccptune dictionary.
#
# NOTE: The Hough transform implementation of SPIVET is crankier than it
# should be.  There's a reasonable chance that the ccpts() function
# below will fail completely for one camera or the other and abort
# processing.  When this happens, first try doubling the denominator of
# of ht_dtheta in ccptune (so use 2048.).  If it fails again, then halve
# the denominator (ie, use 512.).   
rbndx0 = ((26,768),(0,1012))     # CAM0
rbndx1 = ((24,768),(0,1007))     # CAM1

ccptune = {
    'pp_exec':True,
    'pp_bsize':(64,64),
    'pp_p1pow':1.6,
    'pp_p2pow':0.6,
    'ht_dtheta':pi/1024., 
    'ht_drho':1.,
    'ht_show':False,    
    'cc_eps':1.e-3,    
    'cc_maxits':100,   
    'cc_pf1bs':(16,16),
    'cc_thold':10.,     
    'cc_pf2bs':(40,40),  
    'cc_subht':True,
    'cc_show':False
}

# >>>>> END USER MODIFIABLE CODE <<<<<

# Turn on interactive pylab so the debug images get displayed.
pylab.ion()

# Load the individual images for each camera.
dl = os.listdir(bpath)
dl.sort()

cam0ims = []
cam1ims = []
nim     = len(dl)
for i in range(nim):
    print "Loading: %s" % dl[i]
    [imch,imcs,imci] = pivlib.imread(bpath +"/" +dl[i])
    nimci = 1. -imci

    fnl = dl[i].rsplit('_',3)
    cam = fnl[0]
    cam = cam[len(cam)-2:len(cam)]
    if ( cam == 'C0' ):
        cam0ims.append(nimci)
        print "Stored for CAM0"
        print ""
    else:
        cam1ims.append(nimci)
        print "Stored for CAM1"
        print ""

print "CAM0 file count: %i" % len(cam0ims)
print "CAM1 file count: %i" % len(cam1ims)

# Set up the cross-correlation kernel.
cc_krnl = zeros((7,7),dtype=float)     # Cross-correlation kernel.
cc_krnl[2:5,:] = 1.
cc_krnl[:,2:5] = 1.

ccptune['cc_krnl'] = cc_krnl

# Construct the knowns.
knowns = pivlib.initknowns(ccpdim,[kzv,ccpwsp[0],ccpwsp[1]],ccrot)

# Call ccpts() to extract the intersections and match them with
# the knowns.
print "----- CAMERA 0 -----"
pivlib.ccpts(cam0ims,rbndx0,knowns,ccptune,bccpdata+"CAM0")
print
print "----- CAMERA 1 -----"
pivlib.ccpts(cam1ims,rbndx1,knowns,ccptune,bccpdata+"CAM1")

