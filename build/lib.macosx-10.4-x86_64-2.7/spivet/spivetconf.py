"""
Filename:  spivetconf.py
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
  SPIVET configuration information.

Contents:
  get_local_spivet_dir()
  get_nwsconf()
  get_spivet_skel_dir()
  get_user_steps_dir()
  setup_local_spivet_dir()
"""

from distutils.sysconfig import get_python_lib
import os, shutil, imp

_local_spivet_dir = ".spivet"
_user_steps_dir   = "usersteps"
_spivet_skel_dir  = "%s/spivet/skel" % get_python_lib()  

#################################################################
#
def get_local_spivet_dir():
    """
    Returns the path to the local spivet directory.  This directory
    is named .spivet and is located in the user's home directory.  If
    the directory doesn't exist, an attempt is made to create it and 
    copy defaults.

    Returns the path or None if the directory didn't exist and couldn't
    be created.
    """
    hpath = os.path.expanduser('~')
    lsd   = "%s/%s" % (hpath,_local_spivet_dir)

    if ( not os.path.exists(lsd) ):
        try:
            setup_local_spivet_dir()
        except:
            return None

    return lsd


#################################################################
#
def get_nwsconf():
    """
    Returns a reference to the nwsconf module located in _local_spivet_dir,
    if available.  None otherwise.
    """
    lsd = get_local_spivet_dir()

    try:
        [fh,pn,desc] = imp.find_module("nwsconf",[lsd])
        nwsconf      = imp.load_module("nwsconf",fh,"nwsconf.py",desc)
        return nwsconf
    except:
        return None


#################################################################
#
def get_spivet_skel_dir():
    """
    Checks that _spivet_skel_dir exists and returns that value.
    Raises an IOError exception otherwise.
    """
    if ( not os.path.exists(_spivet_skel_dir) ):
        raise IOError("Could not find skeleton directory %s" % \
                          _spivet_skel_dir)

    return _spivet_skel_dir


#################################################################
#
def get_user_steps_dir():
    """
    Returns the path to the directory containing user-defined steps.
    This directory is located in the local_spivet_dir.  

    Returns the path or None if the directory didn't exist.
    """
    lsd = get_local_spivet_dir()
    usd = "%s/%s" % (lsd,_user_steps_dir)
    if ( not os.path.exists(usd) ):
        return None
    
    return usd


#################################################################
#
def setup_local_spivet_dir():
    """
    Copies the defaults from _spivet_skel_dir to the _local_spivet_dir.
    If _local_spivet_dir doesn't exist, it will be created.

    No return value.
    """
    ssd   = get_spivet_skel_dir()
    hpath = os.path.expanduser('~')
    lsd   = "%s/%s" % (hpath,_local_spivet_dir)

    if ( not os.path.exists(lsd) ):
        os.mkdir(lsd)

    dl = os.listdir(ssd)
    for file in dl:
        ifp = "%s/%s" % (ssd,file)
        ofp = "%s/%s" % (lsd,file)

        # Check if a similar local file exists.
        if ( os.path.exists( file ) ):
            shutil.copy(ifp,"%s-default" % ofp)
        else:
            shutil.copy(ifp,ofp)
