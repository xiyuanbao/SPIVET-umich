"""
Filename:  steputil.py
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
  Common utility functions for SPIVET steps.

Contents:
  _fretrieve()
  _ftpcon()
  _ftpgetfile()
  _parsefn()

"""
from numpy import *
from ftplib import FTP
import sys, os, time, urlparse, shutil, traceback
from spivet import compat

#################################################################
#
def _fretrieve(
    rfurl, lbpath, nretry=10, rdelay=60
    ):
    """
    ----

    rfurl           # Full URL to remote file.  Can be a list of URL's.
    lbpath          # Local base path in which to store the retrieved files.
    nretry=10       # Number of retrieve retries.
    rdelay=60       # Delay time between retrieve attempts [sec].

    ----

    Retrieves the file at rfurl and places a copy of it in lbpath.  If rfurl
    is a list of files, then all files in the list will be retrieved.
    
    _fretrieve() supports two scheme's: file, and ftp.

    If the scheme is ftp and a list of URL's is passed, then all URL's must
    have the same network location (ie, come from the same server).
    
    No return value.  Raises an exception if unsuccessful.
    """
    rfurl = array(rfurl)
    rfurl = rfurl.reshape( rfurl.size )

    if ( not os.path.exists( lbpath ) ):
        os.mkdir(lbpath)

    ftp = None
    for url in rfurl:
        urlcomp = urlparse.urlparse(url)

        if ( urlcomp[0] == 'file' ):
            shutil.copy(urlcomp[2],lbpath)
        elif ( urlcomp[0] == 'ftp' ):
            if ( compat.checkNone(ftp) ):
                ftp = _ftpcon(urlcomp[1],nretry,rdelay)

            _ftpgetfile(ftp,
                        urlcomp[2],
                        "%s/%s" % (lbpath,os.path.basename(urlcomp[2])),
                        nretry,
                        rdelay)
        else:
            raise ValueError("Invalid rfurl scheme %s" % urlcomp[0])

    if ( not compat.checkNone(ftp) ):
        ftp.quit()


#################################################################
#
def _ftpcon(
    ftphost, nretry=10, rdelay=60
    ):
    """
    ----

    ftphost         # Host name/IP address for host.
    nretry=10       # Number of retries for connection attempt.
    rdelay=60       # Delay time between connection attempts [sec].

    ----

    Opens an FTP connection to ftphost and return a Python ftplib ftp 
    object.

    NOTE: The user is responsible for calling the ftp object's quit() 
    method after the connection is no longer needed. 
    """
    dbglev = 0

    # Issue an explicit connect() later instead of connecting through
    # FTP().  This will enable the debug level to be set on a 
    # valid ftp object should an error occur.
    ftp = FTP()
 
    for t in range(nretry):
        try:
            print " | Connecting to %s" % ftphost
            ftp.connect(ftphost)
            ftp.login()
            break
        except:
            sys.stderr.write(
                "vvvvv Connect %s FAILED: Try %i vvvvv\n" % (ftphost,t +1)
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("^^^^^ ^^^^^\n")

            print " | | FAILED: Try %i" % (t +1)  

            if ( dbglev == 0 ):
                print " | | Temporarily increasing debug level."
                dbglev = 1
                ftp.set_debuglevel(dbglev)
                      
            if ( t < ( nretry -1 ) ):                    
                time.sleep(rdelay)
            else:
                ftp.set_debuglevel(0)
                raise IOError, "FTP CONNECT failed."
                
    if ( dbglev > 0 ):
        ftp.set_debuglevel(0)
        
    return ftp


#################################################################
#
def _ftpgetfile(
    ftpob, rfpath, lfpath, nretry=10, rdelay=60 
    ):
    """
    ----

    ftpob,          # FTP object.
    rfpath,         # Remote file path.
    lfpath,         # Local file path to store the returned object.
    nretry=10,      # Number of retries.
    rdelay=60       # Delay between retries [sec].

    ----

    Gets a remote file via an open ftp session.
    """
    dbglev = 0
    for t in range( nretry ):
        try:
            print " | Retrieving %s" % rfpath 
            fh = open(lfpath,"wb")
            ftpob.retrbinary("RETR %s" % rfpath,fh.write)
            fh.close()
            break
        except:
            sys.stderr.write(
                "vvvvv Retrieve %s FAILED: Try %i vvvvv\n" % (rfpath,t +1)
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("^^^^^ ^^^^^\n")
            
            print " | | FAILED: Try %i" % (t +1)
            fh.close()
            
            if ( dbglev == 0 ):
                print " | | Temporarily increasing debug level."
                dbglev = 1
                ftpob.set_debuglevel(dbglev)

            if ( t < ( nretry -1 ) ):                    
                time.sleep(rdelay)
            else:
                ftpob.set_debuglevel(0)
                raise IOError, "FTP failed."

    if ( dbglev > 0 ):
        ftpob.set_debuglevel(0)
    

#################################################################
#
def _parsefn(fn):
    """
    ----

    fn              # Filename.

    ----

    Worker function to parse filenames of the following format:
  
        BASENAME-Ew_Cx_Fy_Sz_TIMESTAMP.ext

    where:
        w ---- Epoch number.  0 <= w <= w_max.  Format: %i
        x ---- Camera number. 0 <= x < 2.       Format: %i
        y ---- Frame number.  0 <= y <= y_max.  Format: %i
        z ---- Plane number.  0 <= z <= z_max.  Format: %i.

    Returns the fnc dictionary.
    """
    fnl = fn.rsplit('-',1)
    fnl = fnl[1].rsplit('_')
    
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
            'E':epc}
