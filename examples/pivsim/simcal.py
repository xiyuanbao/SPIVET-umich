"""
Filename:  simcal.py
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
  Driver for generating synthetic images of a calibration target
  using pivsim.  The same grid-type target used for actual lab 
  calibration is used here.  Once the 'images' have been created,
  standard photogrammetric calibration can be performed using them.

  The raytracing setup mirrors the actual geometry of the lab.  
  Inner and outer tanks are included, as is the water bath and 
  corn syrup.

  The simulation uses a random number generator to vary the
  location of the target in the tank to capture estimated
  uncertainties of target positioning from the actual lab 
  calibration.  Y-Variability: +/- 0.05 mm.  Z-Variability: 
  +/- 0.2 mm.  Note: coordinate system is the standard system:
  Y-axis points down, and Z-axis points away from cameras.
"""

from spivet import pivlib
from numpy import *
from PIL import Image

# Set a small gap between objects of different refractive index.  This
# ensures that the raytracer finds the different materials without
# having an adverse impact on accuracy of the simulation.  Units are
# in mm.
epsgap = 1.E-6

# Set the maximum variability for parameters that control the location
# of the calibration target.  These are for target z-axis, theta Euler
# angle, and psi Euler angle.  A uniform distribution will be used
# to simulate variation.   
zsig = 0.2     # Gives z-variability [mm].
tsig = 1.7e-4  # Gives z-variability and x-variability [rad].
psig = 3.e-4   # Gives y-variability and x-variability [rad].

# Refractive indices of experimental environment.
pgior = 1.49  # Acrylic index of refraction.
csior = 1.50  # Syrup index of refraction.
aior  = 1.0   # Air index of refraction.
wior  = 1.32  # Water index of refraction.

# Start building the simulation.

# Set up the SimEnv.  The SimEnv hold the full simulation (cameras and
# refractive objects).  When the camera launches rays, any rays that
# strike the boundary of the SimEnv environment are dropped, and the
# pixel intensity for that ray is set to zero.  Hence the SimEnv acts as
# a sort of 'clipping' box.
se = pivlib.SimEnv((1500.,200.,200.),aior)

# Outer tank front surface.  Note that the lateral extent of the outer
# tank is larger than the SimEnv above.  This is fine.  The SimEnv
# will clip any rays that try to leave the environment as described above.
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

# Inner tank front surface.  Note that the real inner tank and syrup are not
# 368 mm wide and 396 mm tall.  But the cameras don't view all the way
# to the top or sides of the inner tank, so we can use the lateral dimensions
# of the outer tank for simplicity.
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

# Camera common parameters.
cprm  = [15.,768,4.65E-3,1024,4.65E-3]
czloc = -550.

# CAM0.  We take CAM0 to be the left camera and CAM1 to be the right
# when standing behind the cameras and looking toward the tank (ie,
# in the same direction as the cameras' gaze).
cam0 = pivlib.SimCamera((czloc,0.,-150.),
                        (pi/2.,(9.+3.1)*pi/180.,0.),
                        cprm)
se.addCamera(cam0)

# CAM1.
cam1 = pivlib.SimCamera((czloc,0.,150.),
                        (pi/2.,-(9.+3.1)*pi/180.,0.),
                        cprm)
se.addCamera(cam1)


# Target.  Each target position will be installed in front of this one
# (objects cannot be removed from the SimEnv environment, so new objects
# to be imaged must be placed in front of old ones).  We need to image
# the target 5 times, with the target moved toward the camera by 1.8 mm
# (ignoring random variability) after each imaging.
#
# The target lateral dimensions are chosen such that the spacing between
# lines of the CalTarget2 image is 6.35 mm (same as in the real lab target).
hysz = 204.809/2.
hxsz = 268.309/2.

random.seed(1)
zsvec = zsig*(2.*random.rand(5) -1.)

random.seed(2)
tsvec = tsig*(2.*random.rand(5) -1.)

random.seed(3)
psvec = psig*(2.*random.rand(5) -1.)

random.seed(None)
for i in range(5):
    tzloc = zloc +9.525/2. +1. +4.5 +150. -1.8*i +zsvec[i]

    img  = pivlib.imread("CalTarget2.png")
    trgt = pivlib.SimRectangle((tzloc,0.,0.),
                               (pi/2.,pi+tsvec[i],pi/2.+psvec[i]),
                               (1.,hysz,hxsz),
                               None,
                               img[2])
    se.addObject(trgt)

    # Image the scene and extract the newly created bitmaps from each 
    # camera object.
    se.image()
    bmp = cam0.bitmap
    pivlib.imwrite(bmp,"SIMCAL-E0_C0_F0_S0_%i.png" % i,vmin=0.,vmax=1.)
    bmp = cam1.bitmap
    pivlib.imwrite(bmp,"SIMCAL-E0_C1_F0_S0_%i.png" % i,vmin=0.,vmax=1.)

    # Dump out the full simulation environment (with rays) to be visualized
    # by paraview.  This is a good way to get an idea of what the UM lab
    # looks like.
    #se.dump2vtk("SIMENV.vtk")

