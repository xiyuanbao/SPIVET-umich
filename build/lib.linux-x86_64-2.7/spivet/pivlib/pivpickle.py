"""
Filename:  pivpickle.py
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
  pivpickle contains wrapper functions for creating compressed
  pickled objects.

Contents:
  pkldump()
  pklload()
"""

from numpy import *
import cPickle, gzip
from spivet import compat
#################################################################
#
def pkldump(obj,ofpath):
    """
    ----
    
    obj            # Object to be pickled.
    ofpath         # Output file path for storage.
    
    ----    
    
    Stores a compressed pickled version of obj in the file at 
    ofpath.  The compressed file is a gzip file.
    """
    ps = cPickle.dumps(obj)

    fh = gzip.GzipFile(ofpath,'wb')
    fh.write(ps)
    fh.close()


#################################################################
#
def pklload(ifpath):
    """
    ----
    
    ifpath         # Input file path for object to load.
    
    ----
    
    Loads a pickled object stored at ifpath.

    Returns obj, the unpickled object.
    """
    fh = None
    try:
        fh = gzip.GzipFile(ifpath,'rb')
        ps = fh.read()
        fh.close()

        obj = cPickle.loads(ps)

    except:
	if ( not compat.checkNone(fh) ):
            fh.close()

        fh  = open(ifpath,'r')
        obj = cPickle.load(fh)
        fh.close()

    return obj
