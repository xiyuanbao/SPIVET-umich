"""
Filename:  pivpgcdrv.py
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
  Photogrammetric camera calibration script.
  
  This script reads the calibration point files created with
  pivcpxdrv and performs camera calibration.
"""
from spivet import pivlib
import pylab
from numpy import *

# >>>>> ANALYSIS SETUP <<<<<
imdim  = [768,1024]         # Camera image dimensions.
ccpdim = [27,37]            # Calibration points per plane [y,x].
dpy    = 4.65e-3            # Camera pixel height parallel to v-axis [mm].
dpx    = dpy                # Camera pixel width parallel to u-axis [mm].

# Base file name for calibration point data.
bccpdata  = "SYRUPCAL-04192008_CC_OUT_"     

# >>>>> END USER MODIFIABLE CODE <<<<<

# Turn on interactive pylab.
pylab.ion()

# Set plot size for some diagnostic plots (dimensions are in inches).
fsize = (17,11)

# Set planar calibration point quantities.
ccpp   = ccpdim[0]*ccpdim[1]

cstr = ["CAM0","CAM1"]
for cam in range(2):
    print "///// " +cstr[cam] +" \\\\\\\\\\\\"
    # Load in data stored by ccpts().
    ccpa = pivlib.loadccpa(bccpdata +cstr[cam])

    # Initialize the camera calibration dictionary.
    ncamcal = pivlib.initcamcal(dpy,dpx)

    # Calibrate the camera and save the calibration.
    [ocamcal,oerr] = pivlib.calibrate(ccpa,ncamcal,1000)
    pivlib.savecamcal(ocamcal,"CAMCAL_" +cstr[cam])

    # Determine number of z planes in calibration dataset.
    nzplns = oerr.shape[0]/ccpp

    # Display some diagnostic plots.  Can handle up to 9 subplots in
    # a 3x3 layout.
    oerrm = sqrt(sum(oerr*oerr,1))

    pylab.figure(figsize=fsize)
    for i in range(nzplns):
        pylab.subplot(3,3,i+1)
        pylab.imshow(oerrm[i*ccpp:(i+1)*ccpp].reshape(ccpdim),
                     interpolation='nearest')
        pylab.title('|err|_2 Plane ' +str(i))
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        c = pylab.colorbar(shrink=0.75)
        c.set_label("err [pixels]")
    pylab.draw()
    pylab.savefig("OPTIM_IMGCAL_ERRMAG_" +cstr[cam] +".png")

    pylab.figure(figsize=fsize)
    for i in range(nzplns):
        pylab.subplot(3,3,i+1)
        pylab.imshow(oerr[i*ccpp:(i+1)*ccpp,0].reshape(ccpdim),
                     interpolation='nearest')
        pylab.title('verr Plane ' +str(i))
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        c = pylab.colorbar(shrink=0.75)
        c.set_label("err [pixels]")
    pylab.draw()
    pylab.savefig("OPTIM_IMGCAL_VERR_" +cstr[cam] +".png")

    pylab.figure(figsize=fsize)
    for i in range(nzplns):
        pylab.subplot(3,3,i+1)
        pylab.imshow(oerr[i*ccpp:(i+1)*ccpp,1].reshape(ccpdim),
                     interpolation='nearest')
        pylab.title('uerr Plane ' +str(i))
        pylab.setp(pylab.gca(),xticks=[],yticks=[])
        c = pylab.colorbar(shrink=0.75)
        c.set_label("err [pixels]")
    pylab.draw()
    pylab.savefig("OPTIM_IMGCAL_UERR_" +cstr[cam] +".png")

    # Discretize world space and select the finest discretization.
    rbndx = [ [0,imdim[0]], [0,imdim[1]] ]
    twdsc = pivlib.dscwrld(rbndx, ocamcal)

    if ( ( cam == 0 ) or ( twdsc['mmpx'] < wdsc['mmpx'] ) ):
        wdsc = twdsc

# Create the world to image correspondence dictionary (used to 
# efficiently discretize world space and project images onto the 
# world plane).
for cam in range(2):
    camcal = pivlib.loadcamcal("CAMCAL_" +cstr[cam])

    pivlib.wrld2imcsp(rbndx,camcal,wdsc,"WICSP_" +cstr[cam])
