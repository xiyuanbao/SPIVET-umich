"""
Filename:  tlccpxdrv.py
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
  Driver file for extracting thermochromic calibration points.
"""

from spivet import pivlib, tlclib
import pylab
import os, sys
from numpy import *
import csv

# >>>>> ANALYSIS SETUP <<<<<
s0zcoord  = 0.                            # z-coordinate of sequence 0. [mm]
zcellsz   = -52.5                         # z Cell size [mm].
ilvec     = [0.,0.,-1.]                   # Illumination vector.
ifbpath   = '../HDATA'                    # Path to input images.
pcamcal1  = '../CALIBRATION/CAMCAL_CAM1'  # Path to CAMCAL_CAM1
pwicsp1   = '../CALIBRATION/WICSP_CAM1'   # Path to WICSP_CAM1
btcpdata  = 'TLCCAL-09282008_TC_OUT_'     # Base output name for cal points.

# drbndx will be added to [ [0,imdim[0]], [0,imdim[1]] ] to create rbndx.
# NOTE: imdim in this case represents the dimensions of the projected image.
drbndx1 = array([[70,-71],[35,-35]])  # CAM1

# Setup the known temperatures corresponding to the bath temp for each dataset.
tmpa = {
"25_0DEGC":24.975,
"25_2DEGC":25.103,
"25_4DEGC":25.317,
"25_6DEGC":25.495,
"25_8DEGC":25.757,
"26_0DEGC":25.886,
"26_2DEGC":26.085,
"26_4DEGC":26.283,
"26_6DEGC":26.489,
"26_8DEGC":26.639,
"27_0DEGC":26.840 }

# Setup hue extraction parameters.
bsize  = (32,32)    # Block size for image analysis.
rthsf  = 2.         # Median filter residual threshold scale factor.
pmbgth = 0.01       # Particle mask background threshold (for tlcmask()).
pmbcth = 15.        # Partilce mask bubble threshold (for tlcmask()).
hsmthd = 0          # Hue separation method.
ishow  = False      # Show intermediate results.

# >>>>> END USER MODIFIABLE CODE <<<<<


#################################################################
#
def parsefn(fn):
    """
    Parses the file name.
    """
    fnl = fn.rsplit('-',2)
    tmp = fnl[1]
    fnl = fnl[2].rsplit('_')
    
    ts  = fnl[ len(fnl) -1 ]
    xnd = ts.rfind('.')
    ts  = int( ts[0:xnd] )

    cam = frm = seq = epc = 0
    for i in range( len(fnl) -1 ):
        fnc = fnl[i]
        idc = fnc[0]
        cpv = int( fnc[1:len(fnc)] )
        if ( idc == 'C' ):        
            cam = cpv
        elif ( idc == 'F' ):
            frm = cpv
        elif ( idc == 'S' ):
            seq = cpv
        elif ( idc == "E" ):
            epc = cpv
            
    return {'TS':ts,
            'C':cam,
            'F':frm,
            'S':seq,
            'E':epc,
            'TMP':tmp}

# Turn on interactive pylab and get a copy of the colormap.
pylab.ion()
cmap = pivlib.getpivcmap(hsmthd)

# Load the camera calibration.
camcal1 = pivlib.loadcamcal(pcamcal1)

# Load the world to image correspondences.
wicsp1 = pivlib.loadwicsp(pwicsp1)

pimdim = wicsp1['wpxdim']
rbndx1 = array([[0,pimdim[0]],[0,pimdim[1]]]) +drbndx1

# Build list of filenames.
dl  = os.listdir(ifbpath)
nim = len(dl)

# Protect for Mac's metadata files.
ndrop = 0
for i in range(nim):
    if ( dl[i-ndrop][0] == '.' ):
        dl.pop(i)
        ndrop = ndrop +1
nim = len(dl)

dl.sort()

# Get camera images.
camimgs = []
for file in dl:
    fnp = parsefn(file)
    if ( fnp["C"] != 1 ):
        continue

    camimgs.append(file)
ncim = len(camimgs)

# Create the knowns.
rsize  = rbndx1[:,1] -rbndx1[:,0]
nyblks = rsize[0]/bsize[0]
nxblks = rsize[1]/bsize[1]
tnblks = nyblks*nxblks

if ( mod(rsize[0],bsize[0]) != 0 ):
    print "WARNING: Region not spanned by an integer number of blocks in y."
if ( mod(rsize[1],bsize[1]) != 0 ):
    print "WARNING: Region not spanned by an integer number of blocks in x."

print "nyblks %i" % nyblks
print "nxblks %i" % nxblks

blkcoord = indices((1,nyblks,nxblks),dtype=float)
bsize    = array(bsize)
cellsz   = bsize*wicsp1['mmpx']
origin   = (rbndx1[:,0] +bsize/2.)*wicsp1['mmpx'] +wicsp1['wos']

blkcoord[1,...] = blkcoord[1,...]*cellsz[0] +origin[0]
blkcoord[2,...] = blkcoord[2,...]*cellsz[1] +origin[1]

wpta  = blkcoord.reshape((3,tnblks)).transpose()
theta = tlclib.tlctc.viewangle(wpta,camcal1,ilvec)

tcpa   = zeros((ncim*tnblks,4),dtype=float)
krbndx = 0
for file in camimgs:
    print "Processing: %s" % file
    bfname = file[0:-4]
    fnp    = parsefn(file)
    s      = fnp["S"]
    temp   = tmpa[ fnp["TMP"] ]

    wzcoord = s*zcellsz +s0zcoord

    krendx = krbndx +tnblks

    [ch,cs,ci] = pivlib.imread("%s/%s" % (ifbpath,file),hsmthd)
    prh        = pivlib.prjim2wrld(ch,wicsp1)
    prs        = pivlib.prjim2wrld(cs,wicsp1)
    pri        = pivlib.prjim2wrld(ci,wicsp1)
        
    # Median filter the hue image to remove secondary tracer particles.
    prh = pivlib.bimedfltr(prh,rbndx1,bsize,rthsf)

    pivlib.imwrite(prh,"%s-MEDF.png" % bfname,cmap,
                   vmin=0.,vmax=1.)

    # Get particle mask and extract hue values.
    pmask = tlclib.tlcmask([prh,prs,pri],rbndx1,bsize,pmbgth,pmbcth,show=ishow)

    [bh,hinac] = tlclib.tlcutil.xtblkhue(prh,pmask,rbndx1,[nyblks,nxblks],
                                         bsize,bsize)

    tcpa[krbndx:krendx,0] = temp
    tcpa[krbndx:krendx,1] = wzcoord
    tcpa[krbndx:krendx,2] = theta
    tcpa[krbndx:krendx,3] = bh.reshape(tnblks)

    bh = bh.reshape((nyblks,nxblks))
    
    bh = repeat(bh,bsize[0],axis=0)
    bh = repeat(bh,bsize[1],axis=1)

    pivlib.imwrite(bh,"%s-AVEHUE.png" % bfname,cmap,vmin=0.,vmax=1.)

    krbndx = krbndx +tnblks

pivlib.pkldump(tcpa,btcpdata+"CAM1")

