"""
Filename:  simb.py
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
  Driver for generating synthetic images of a planar starscape-like
  image using pivsim.  Once the 'images' have been created, 
  displacement vectors can be extracted.

  The raytracing setup mirrors the actual geometry of the lab.  
  Inner and outer tanks are included, as is the water bath and 
  corn syrup.

  The starscape target is placed at the back of the tank and moved
  dz = -1.2, dx = 0.6 mm.  Camera moves 0.6712*dz (this movement is
  necessary for the lab camera to maintain focus, so the motion is
  repeated here even though a pinhole camera has an infinite depth of
  field).
"""

from spivet import pivlib
from numpy import *
import sys
from PIL import Image
# Set a small gap between objects of different refractive index.  This
# ensures that the raytracer finds the different materials without
# having an adverse impact on accuracy of the simulation.  Units are
# in mm.
epsgap = 1.E-6

# Refractive indices of experimental environment.
pgior = 1.49  # Acrylic index of refraction.
csior = 1.50  # Syrup index of refraction.
aior  = 1.0   # Air index of refraction.
wior  = 1.32  # Water index of refraction.

# Start building the simulation.

# Set up the SimEnv.
se = pivlib.SimEnv((1500.,200.,200.),aior)

# Outer tank front surface.
zloc = 9.525/2.
otfs = pivlib.SimRectangle((zloc,0.,0.),
                           (0.,0.,0.),
                           (9.525/2.,396.35/2.,368.3/2.),
                           pgior)
se.addObject(otfs)

# Water bath.
zloc = zloc +9.525/2. +32.6/2. +epsgap
wb   = pivlib.SimRectangle((zloc,0.,0.),
                           (0.,0.,0.),
                           (32.6/2.,396.35/2.,368.3/2.),
                           wior)
se.addObject(wb)

# Inner tank front surface.
zloc = zloc +32.6/2 +9.525/2. +epsgap
itfs = pivlib.SimRectangle((zloc,0.,0.),
                           (0.,0.,0.),
                           (9.525/2.,396.35/2.,368.3/2.),
                           pgior)
se.addObject(itfs)

# Syrup.
cszloc = zloc +9.525/2. +265./2. +epsgap
svol   = pivlib.SimRectangle((cszloc,0.,0.),
                             (0.,0.,0.),
                             (265./2.,396.35/2.,368.3/2.),
                             csior)
se.addObject(svol)

# Camera common parameters.  Note that the lightline has effectively moved
# 113.6 mm from mid-tank (the 3.6 mm comes from the fact that calibration 
# projects onto the middle location of the 5-position calibration sequence). 
cprm  = [15.,768,4.65E-3,1024,4.65E-3]
czloc = -550. +0.6712*113.6

# CAM0.
cam0 = pivlib.SimCamera((czloc,0.,-150.),
                        (pi/2.,(9.+3.1)*pi/180.,0.),
                        cprm)
se.addCamera(cam0)

# CAM1.
cam1 = pivlib.SimCamera((czloc,0.,150.),
                        (pi/2.,-(9.+3.1)*pi/180.,0.),
                        cprm)
se.addCamera(cam1)


# Target.  Each target position will be installed in front of this one.
hysz = 204.809/2.
hxsz = 268.309/2.

for i in range(2):
    tzloc = zloc +9.525/2. +1. +4.5 +260. -1.2*i

    img  = pivlib.imread("SimTarget.png")
    trgt = pivlib.SimRectangle((tzloc,0.,i*0.6),
                               (0.,pi,0.),
                               (1.,hysz,hxsz),
                               None,
                               img[2])
    se.addObject(trgt)

    # Image the scene.
    se.image()
    bmp = cam0.bitmap
    pivlib.imwrite(bmp,"SIMB-E0_C0_F%i_S0_%i.png" % (i,i),vmin=0.,vmax=1.)
    bmp = cam1.bitmap
    pivlib.imwrite(bmp,"SIMB-E0_C1_F%i_S0_%i.png" % (i,i),vmin=0.,vmax=1.)


