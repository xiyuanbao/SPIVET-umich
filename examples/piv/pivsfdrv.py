"""
Filename:  pivsfdrv.py
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
  Simple driver file for non-parallel stereo PIV.  Image files
  to be processed must be specified in FRMLST (see below for
  details).  Does not extract temperature field.

  Calibration files (CAMCAL_CAM0, CAMCAL_CAM1, WICSP_CAM0, and 
  WICSP_CAM1) must be stored in the same path as pivsfdrv.py.
  
  This script does not utilize the SPIVET.steps framework,but
  instead interfaces with PivLIB directly.  
"""
from spivet import pivlib
import pylab
import math
from numpy import *
from PIL import Image
import time
import pickle
import os

# >>>>> ANALYSIS SETUP <<<<<
mfdim     = None                          # Median filter size.
rthsf     = 4.                            # Median threshold scale factor.
mfits     = 3                             # Median filter iterations.
pdesc     = 'DRYCAL2B: TEST RUN'          # pivData description of data.
ex2ofname = 'PIVDATA'                     # File name for ExodusII output.
pcamcal0  = '../CALIBRATION/CAMCAL_CAM0'  # Path to CAMCAL_CAM0.
pcamcal1  = '../CALIBRATION/CAMCAL_CAM1'  # Path to CAMCAL_CAM1
pwicsp0   = '../CALIBRATION/WICSP_CAM0'   # Path to WICSP_CAM0
pwicsp1   = '../CALIBRATION/WICSP_CAM1'   # Path to WICSP_CAM1

# drbndx will be added to [ [0,imdim[0]], [0,imdim[1]] ] to create rbndx.
# NOTE: imdim in this case represents the dimensions of the projected image.
drbndx = array([[55,-55],[54,-54]])       

# Setup the pivlib dictionary.  Details on the contents of pivdict can
# be found in the help for the pivlib module.
pivdict={
    'gp_bsize':(32,32),
    'gp_bolap':(0,0),
    'gp_bsdiv':1,    
    'ir_eps':0.003,
    'ir_maxits':100, 
    'ir_mineig':0.05,
    'ir_imthd':'C',
    'ir_iedge':0,
    'of_maxdisp':(24,24), 
    'of_rmaxdisp':(5,5), 
    'of_highp':False,
    'tf_pmbgth':0.3,
    'tf_pmbcth':8.,
    'tf_pmshow':True,
    'tf_inacval':-1000.
}

# >>>>> END USER MODIFIABLE CODE <<<<<

# Setup worker functions. 
def flowstats(
    ofDisp,
    ofINAC,
    label='OFCOMP STATISTICS'
):
    """
    Displays flow stats.
    """
    ofinac = ofINAC.squeeze()

    imsk = ofinac > 1.
    inac = sum(where(imsk, 1, 0))

    print "--- %s ---" % label
    print "INAC CELLS:  " + str(inac)
    print 
    
    ofDisp.printStats()
    print "---"

def flowcomp(
    camfrms,
    pivdict
):
    """
    Computes two-component optical flow for a single camera.
    """
    # Get optical flow results and perform some statistical analysis.
    a   = time.time()
    [ofDisp,ofINAC] = pivlib.ofcomp(camfrms[0][2],camfrms[1][2],pivdict)
    a   = time.time() -a
    print "OFCOMP RUN TIME: " + str(a)

    flowstats(ofDisp,ofINAC)

    return [ofDisp,ofINAC]

# Set the hue separation method (0 for standard, 1 for modified Dabiri).
hsmthd = 0

# Turn on interactive pylab.
pylab.ion()

# Load the camera calibration.
camcal0 = pivlib.loadcamcal(pcamcal0)
camcal1 = pivlib.loadcamcal(pcamcal1)

# Load the world to image correspondences.
wicsp0 = pivlib.loadwicsp(pwicsp0)
wicsp1 = pivlib.loadwicsp(pwicsp1)

# Load the camera frames listed in the file FRMLST.  Only the first 4 entries
# of FRMLST will be read.  Each line must be of the form:
#     path_to_image
# where path_to_image is the full path to the image file for that frame.
# The first two entries will be stored for CAM0, and the second two for CAM1.
fh = open('FRMLST','r')
    
cf0 = []
cf0.append(fh.readline().strip())
cf0.append(fh.readline().strip())

cf1 = []
cf1.append(fh.readline().strip())
cf1.append(fh.readline().strip())
    
fh.close()

camfrms0 = []
camfrms0.append( pivlib.imread(cf0[0],hsmthd) ) # [hue, sat, int]
camfrms0.append( pivlib.imread(cf0[1],hsmthd) )

for i in range(3):
    camfrms0[0][i] = pivlib.prjim2wrld(camfrms0[0][i],wicsp0)
    camfrms0[1][i] = pivlib.prjim2wrld(camfrms0[1][i],wicsp0)  

camfrms1 = []
camfrms1.append( pivlib.imread(cf1[0],hsmthd) ) # [hue, sat, int]
camfrms1.append( pivlib.imread(cf1[1],hsmthd) )

for i in range(3):
    camfrms1[0][i] = pivlib.prjim2wrld(camfrms1[0][i],wicsp1)
    camfrms1[1][i] = pivlib.prjim2wrld(camfrms1[1][i],wicsp1)

# Append rbndx, camcal and wicsp to the pivdict.
imdim = camfrms0[0][0].shape
rbndx = array([[0,imdim[0]],[0,imdim[1]]]) +drbndx

pivdict['gp_rbndx']  = rbndx
pivdict['pg_camcal'] = [camcal0,camcal1]
pivdict['pg_wicsp']  = [wicsp0,wicsp1]

# Keep these parameters handy for use in the driver file.
bsize   = array(pivdict['gp_bsize'])
bolap   = array(pivdict['gp_bolap'])
bsdiv   = pivdict['gp_bsdiv']
inacval = pivdict['tf_inacval']

# Get a copy of the colormap matching hsmthd.
if (hsmthd == 0):
    mymap = pivlib.getpivcmap()
else:
    mymap = pivlib.getpivcmap(1)

# Compute two-component flow for each camera.
[ofDisp0,ofINAC0] = flowcomp(camfrms0,pivdict)
[ofDisp1,ofINAC1] = flowcomp(camfrms1,pivdict)

ofDisp0.setName("OFDISP2D-C0")
ofINAC0.setName("OFINAC2D-C0")
ofDisp1.setName("OFDISP2D-C1")
ofINAC1.setName("OFINAC2D-C1")

# Compute three-component flow and store the results in a PIVData
# object.
ofDisp = pivlib.tcfrecon(ofDisp0,ofDisp1,pivdict)

if ( mfdim != None ):
    # In many ways, removing spurious vectors on 2D results is better
    # than on 3D results because a good value for reps is more readily
    # determined (the RMS uncertainty from image registration, which is
    # generally about 0.1 pixels).  When working with 3D vectors, the
    # 2D uncertainty is spread across the 3D components in a very complex
    # way.  Nevertheless, we'll apply the filter here and set reps=0.
    # (so this is an aggressive filter that will smooth a lot of cells
    # where the inferred displacements are very tiny).
    [ofDisp,fltrd] = pivlib.medfltr(ofDisp,mfdim,rthsf,reps=0.,nit=mfits)

    numfltrd = where(fltrd > 0, 1, 0)
    print "MEDIAN FILTERED: %i" % numfltrd.sum()

ofINAC = ofINAC0 +ofINAC1
ofINAC.setAttr("OFINACFLAG3D","NA")

cellsz   = (bsize -bolap)*wicsp0['mmpx']/bsdiv
cellsz   = [1.,cellsz[0],cellsz[1]]
origin   = ( rbndx[:,0] +bsize/(2.*bsdiv) )*wicsp0['mmpx'] +wicsp0['wos']
origin   = (0.,origin[0],origin[1])
if ( mfdim != None ):
    pivData  = pivlib.PIVData(cellsz,origin,pdesc,[ofDisp,ofINAC,fltrd])
else:
    pivData  = pivlib.PIVData(cellsz,origin,pdesc,[ofDisp,ofINAC])

# Compute some statistics.
flowstats(ofDisp,ofINAC,'THREE-COMPONENT FLOW STATISTICS')

# Dump the flow results for analysis with Paraview.
pivData.save(ex2ofname)


