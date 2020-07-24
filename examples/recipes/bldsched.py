"""
Filename:  bldsched.py
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
  This recipe configures the steps.bldsched class and builds a 
  'schedule' of images to process.  All images in the specified 
  directory are parsed, ordered, and a series of EFRMLST files are 
  created, one for each Epoch.  The EFRMLST files are structured 
  such that the remaining SPIVET steps know how to process the 
  images (ie, which sets of images correspond to a given z-plane).
  
  For bldsched to work, the image files must be named using the SPIVET
  file naming scheme.  See the help for steps.bldsched.
  
  When processing image sets, this script is generally the first 
  script the user will run.  It's also simple enough to get a good 
  handle on how SPIVET steps actually function.  As discussed in
  the steps module help, the whole set of steps function by performing
  operations on a single shared object, the carriage, that is passed 
  between all steps.  Each step accesses data on the carriage, performs
  work, and then terminates.  Other steps are then free to do their
  own work on the carriage.  Most steps also take some sort of
  user-specified configuration data that explains how the step should
  operate on carriage data.  These two dictionary objects, the
  carriage and the configuration dictionary, are all the user
  passes to a step.  The step's execute() member function is what
  triggers the step to do work.  Each of these elements are demonstrated
  in this bldsched recipe.
  
  To execute the recipe after modifying as needed, just type
      python bldsched.py
  into a terminal. 
"""
from spivet import steps
import os

# CONFIG: bldsched
# Setup the step configuration.  Details on the configuration parameters
# a step can use are provided in the step's help.  For the bldsched
# step, the help can be accessed using 
#    help(steps.bldsched)
# after importing the steps module.  Optional configuration parameters
# are denoted with an (O) in the help, while mandatory configuration 
# parameter are shown with an (M).
#
# The bldsched step generates a schedule that other steps will act
# on later.  Here we specify the URL from which other, later steps should
# retrieve images (the bldsched step itself does not need to connect
# to the URL).
bfileurl = "ftp://10.45.77.2/globalstore/Projects/PIV/Datasets/PLUME-80_0DEGC-B25_2-04022009/DATA"
conf_bldsched = {'bfileurl':bfileurl}

# That's it for configuration.  Now we need to build the carriage.
# The bldsched step requires that the carriage contain a list of the image 
# filenames that later steps are to process.  We read in those files now 
# and construct a carriage dictionary to hold the file list.
files = os.listdir('../DATA')
c = 0
for i in range(len(files)):
    f = files[c]
    if ( f[-4::] != ".tif" ):
        files.pop(c)
    else:
        c = c +1

carriage = {'files':files}

# Instantiate a bldsched step, pass it the configuration dictionary and
# the carriage, and tell it to execute.
t = steps.bldsched()
t.setConfig(conf_bldsched)
t.setCarriage(carriage)
t.execute()
