"""
Filename:  sync.py
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
  This recipe is used to post process results generated with the
  precipe script.  It's included as an example just to demonstrate
  how steps can be used independently of the loop_plane and 
  loop_epoch container steps discussed in precipe.  The steps 
  below each operate on the full contents of a PIVData object.
  
  To execute the recipe after modifying as needed, just type
      python sync.py
  into a terminal.
"""
from spivet import pivlib, steps
from numpy import *

# Again, the recipe starts by creating configuration dictionaries that
# will be used to tell each of the steps how to operate.

# CONFIG: conf_pzmedfltr 
# Stereoscopic PIV setups tend to have larger errors in the out of plane
# displacement (velocity) component than the two in plane components.
# This nature is a consequence of the geometry of stereoscopic systems and
# error propagation.  The filter below is the standard spurious vector 
# removal tool that was used to process the planar results in the precipe 
# script.  Here it's being run again to process just the z-component 
# of velocity to catch any stragglers that may have gotten through.  To 
# use it, a sensible value for reps must be established or the filter will
# adjust way too many cells.  As before, we set reps the expected RMS
# variation in the data being filtered (here, the z-component of velocity).
# For the University of Michigan setup, an estimate of an upper bound on 
# out of plane error is 0.2 mm RMS.  We are applying this filter to 
# velocities, however, so we need to divide the RMS error by the time 
# between frames (which for the data processed with this script was 
# 7.5 sec).  
zreps = 0.2/7.5

conf_pzmedfltr = {'varnm':'U',
                  'planar':True,
                  'rthsf':1.7,
                  'reps':zreps,
                  'mfits':3,
                  'mfdim':5,
                  'cndx':0,
                  'pdname':'pivdata'}

# CONFIG: synchronize
# The University of Michigan SPIV system scans a tank taking images of
# different z-planes in succession.  That means data for each plane is
# technically valid only at the time it was taken, with other planes
# representing a flow field that is either more or less evolved.  It's
# very convenient, however, to be able to refer to the entire 3D flow
# field at one instant in time (that is the purpose of PIVEpochs after all).
# The synchronize step temporally splines the entire data set and then 
# adjusts the data such that all z-planes within each Epoch are valid at 
# a single point in time (ie, the synchronized data are as though
# all planes for an Epoch were taken instantaneously at the exact same
# time).
#
# NOTE: The synchronize step is destructive and throws away the last
# Epoch in the PIVData object.  See the steps documentation for more
# details.
conf_synchronize = {'varnm':['U-MF','T-MF'],
                    'pdname':'pivdata'}


# CONFIG: gsmooth
# This step applies a very weak Gaussian to smooth the results.  The
# Gaussian is used here to tie the iteratively improved results (refine2dof
# step of precipe) back into the rest of the dataset.
conf_gsmooth = {'pdname':'pivdata',
                'varnm':['U-MF-SN','T-MF-SN'],
                'gbsd':0.7}

# Now the carriage can be set up.  Each of the above steps expects to
# see a PIVData object stored on the carriage.  We load the one generated
# by precipe.  The steps will then operate on this PIVData object and
# add the modified variables.
#
# Note that the loadpivdata() function returns a fully functional PIVData
# object from the ExodusII file in which it was stored.  
pd = pivlib.loadpivdata('PIVDATA.ex2')
carriage = {'pivdata':pd}

# Each step can now be instantiated, passed a configuration object, and
# given the carriage on which it operates.
t = steps.medfltr()
t.setConfig(conf_pzmedfltr)
t.setCarriage(carriage)
t.execute()

t = steps.synchronize()
t.setConfig(conf_synchronize)
t.setCarriage(carriage)
t.execute()

t = steps.gsmooth()
t.setConfig(conf_gsmooth)
t.setCarriage(carriage)
t.execute()

# Get the modified PIVData object and store it.  We want to save this
# PIVData object under a different name for two reasons.  First, the
# original PIVData object stored by precipe should be treated with great
# care.  Generating it consumed a lot of time, and we don't want to
# ever take the chance of accidentally corrupting it.  Second, the
# synchronize step is destructive (it throws away the last Epoch).
pd = carriage['pivdata']
pd.save("PIVDATA-SYNC.ex2")

