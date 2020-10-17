"""
Filename:  steps.py
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
  Provides default steps for SPIVET use.  From a high level, a
  step is merely a sequence of operations performed on PIV data 
  that have been grouped together for easy reuse.  

  From an implementation perspective, all steps are subclasses
  of a base class, spivetstep.  Each step is passed a carriage
  object (a dictionary) from which the step will extract input 
  data (eg, the hue channel of a image) and store output data. 
  That carriage should then be passed to the next step.

  In addition to the carriage, each step may consult a 
  configuration dictionary.  The concept here being that the
  configuration dictionary contains parameters that are constant
  between calls to the step's execute() function, while the
  carriage contains the data the step actually processes for
  the given execute() call.
  
  Steps are the preferred mechanism for building analytical 
  toolsets from the underlying SPIVET libraries. 
  
  Mandatory parameters passed to a step are denoted with an (M)
  in the step documentation, while optional parameters are shown
  with an (O).
  
  The user may specify custom steps in addition to the defaults
  provided.  User-defined steps can be stored in one of two places:
      1) In a module (or modules) residing in the same directory 
         from which the python interpreter was launched, or
      2) In the special usersteps sub-directory of the user's
         spivet config directory (/home/theuser/.spivet/usersteps or 
         something similar).  The user can then conveniently use these 
         custom step modules after calling
             spivet.steps.enable_user_steps()
"""

from numpy import *
from scipy import interpolate
from spivet import pivlib, tlclib, spivetconf
from spivet.steputil import _parsefn, _fretrieve
import sys, os, urlparse
from spivet import compat

"""
Distinction between configuration and carriage parameters:
  - The following conceptual ideas govern the logic behind whether a step
    should be designed to have a specific parameter passed via the config
    dictionary or the carriage.  
    
    The carriage should be used to pass information between steps and
    to hold data that a step will operate on.  The config dictionary
    should be used to pass information that the step requires for proper
    operation, but otherwise has a sense of permanence (ie, won't be
    modified by the step, and could conceivably be used for consecutive
    calls to the step).
"""


#################################################################
#
def enable_user_steps():
    """
    Utility function that adds the usersteps directory to the 
    Python module path.  If the user wants to run custom steps,
    enable_usr_steps() should be called first. 
    """
    usd = spivetconf.get_user_steps_dir()
    if ( not compat.checkNone(usd) ):
        sys.path.insert(1,usd)

#################################################################
#
def url2local(url,lbpath):
    """
    ----
    
    url             # Single or list of URL strings to file(s).
    lbpath          # Local path to store file if retrieved.

    ----

    Takes a URL string or list of URL strings and fetches a local
    copy if necessary.  Can currently handle two schemes: file and
    ftp.  

    Any URL with the file scheme is assumed to already
    reside locally.  The path of the file will be extracted 
    from the URL and returned.

    If the URL scheme is ftp, a copy of the file will be retrieved
    using anonymous ftp and stored in lbpath.  The path to this
    local copy will be returned.

    Returns lfpath, a list (always) of the path(s) to the local
    version of the file.
    """
    # Initialization.
    if ( not isinstance(url,list) ):
        url = [ url ]

    # Get the files if necessary.
    lfpath = []
    for u in url:
        urlcomp = urlparse.urlparse(u)
        if ( urlcomp[0] == 'ftp' ):
            _fretrieve(u,lbpath)
            fname = os.path.basename(urlcomp[2])
            fpath = "%s/%s" % (lbpath,fname)
        else:
            fpath = urlcomp[2]

        lfpath.append( fpath )

    return lfpath


#################################################################
#
def chkpivdict(pivdict,lbpath='CALIBRATION'):
    """
    ----
    
    pivdict         # The pivdict object.
    lbpath          # Base path for storing any retrieved files.
    
    ----
    
    Simple function to check the pivdict for URL's and load the
    calibration files if necessary.
    """
    if ( pivdict.has_key('pg_camcal') ):
        camcal = pivdict['pg_camcal']
        if ( not isinstance(camcal,list) ):
            camcal = [camcal]
            pivdict['pg_camcal'] = camcal
        
        for i in range( len(camcal) ):
            if ( isinstance(camcal[i],basestring) ):
                pth       = url2local(camcal[i],lbpath)
                camcal[i] = pivlib.loadcamcal(pth[0])
        
    if ( pivdict.has_key('pg_wicsp') ):
        wicsp = pivdict['pg_wicsp']
        if ( not isinstance(wicsp,list) ):
            wicsp = [wicsp]
            pivdict['pg_wicsp'] = wicsp
            
        for i in range( len(wicsp) ):
            if ( isinstance(wicsp[i],basestring) ):
                pth      = url2local(wicsp[i],lbpath)
                wicsp[i] = pivlib.loadwicsp(pth[0])
                
    if ( pivdict.has_key('tc_tlccal') ):
        tlccal = pivdict['tc_tlccal']

        if ( isinstance(tlccal,basestring) ):
            pth = url2local(tlccal,lbpath)
            pivdict['tc_tlccal'] = tlclib.loadtlccal(pth[0])
        

#################################################################
#
def csznorgn(zcellsz,pivdict,padpix,usewicsp=True):
    """
    ----
    
    pivdict        # The pivdict dictionary.
    padpix         # Number of pad pixels in the image.
    usewicsp       # Use wicsp if available.
    
    ----
    
    Utility function to compute cellsz and origin.
    
    Returns [cellsz, origin].
    """
    rbndx = array(pivdict['gp_rbndx'])
    bsize = array(pivdict['gp_bsize'])
    bolap = array(pivdict['gp_bolap'])
    bsdiv = pivdict['gp_bsdiv']
    pado  = array([padpix,padpix])
    
    if ( usewicsp and pivdict.has_key('pg_wicsp') ):
        wicsp  = pivdict['pg_wicsp']
        wpxdim = wicsp[0]['wpxdim']
        cellsz = (bsize -bolap)*wicsp[0]['mmpx']/bsdiv
        cellsz = [zcellsz,cellsz[0],cellsz[1]]
        origin = ( rbndx[:,0] -pado +bsize/(2.*bsdiv) )*wicsp[0]['mmpx']\
                  +wicsp[0]['wos']
        origin = [0.,origin[0],origin[1]]
    else:
        cellsz = (bsize -bolap)/bsdiv
        cellsz = [zcellsz,cellsz[0],cellsz[1]]
        origin = rbndx[:,0] -pado +bsize/(2.*bsdiv)
        origin = [0.,origin[0],origin[1]]

    return [cellsz,origin]


#################################################################
#
class spivetstep:
    """
    Base class for all SPIVET steps.  A user-defined step must
    inherit from spivetstep.
    """
    def __init__(self,carriage={},config={}):
        self.m_carriage = carriage
        self.m_config   = config

    def __getattr__(self,attr):
        if ( attr == 'carriage' ):
            return self.m_carriage
        elif ( attr == 'config' ):
            return self.m_config
        else:
            raise AttributeError("Unknown attribute %s" % attr)

    def execute(self):
        pass

    def setCarriage(self,carriage):
        self.m_carriage = carriage
        
    def setConfig(self,config):
        self.m_config = config


#################################################################
#
class bldsched(spivetstep):
    """
    STEP: Build schedule

    Imports a list of image files and constructs an analysis schedule
    from them.  This analysis schedule is stored in a series of EFRMLST 
    files.  Step is currently configured to work with filenames having
    the following format:
  
        BASENAME-Ew_Cx_Fy_Sz_TIMESTAMP.ext

    where:
        w ---- Epoch number.  0 <= w <= w_max.  Format: %i
        x ---- Camera number. 0 <= x < 2.       Format: %i
        y ---- Frame number.  0 <= y <= y_max.  Format: %i
        z ---- Plane number.  0 <= z <= z_max.  Format: %i
        
    The U of M hardware acquires images from two cameras.  The world
    z-axis is aligned with the plane number, so a Plane is a set
    of image frames corresponding to a particular z-plane in the tank.
    To calculate optical flow, two image Frames from each Camera are
    required for each z-plane.  Additionally, the flow is captured for the 
    entire tank at discrete intervals through time.  These time intervals
    are Epochs.
    
    A single EFRMLST file contains the names of all images necessary to
    process one Epoch.  If more than 2 frames are detected when analyzing
    the image filenames on the carriage, multiple sets of EFRMLST files
    will be created.  One set will consist of images from Frame 0 and 1, 
    the next set will consist of images from Frame 0 and 2, and so on.  

    ---- carriage inputs ----
    files ---- (M) List of image filenames to process.  Each of these
               files will be expected to be found at the location pointed
               to by bfileurl.    

    ---- carriage ouptuts ----
    lbfpath -- Same as config.    

    ---- config dictionary contents ----
    bfileurl - (M) URL to the base directory where the image files are 
               stored and can be retrieved later.  
               Eg:
                   file://path/to/files
                   ftp://path/to/files
                   
               So the full URL to image 0 of the carriage files parameter
               would be:
                   bfileurl/files[0]
    lbfpath -- (O) Base path to store the EFRMLST files.  These files
               contain the data necessary to process each Epoch.  Defaults
               to EFRMLSTS.    

    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        files    = carriage['files']
        bfileurl = self.m_config['bfileurl']
        if ( self.m_config.has_key('lbfpath') ):
            lbfpath = self.m_config['lbfpath']
        else:
            lbfpath = 'EFRMLSTS' 

        nim = len( files )

        # Determine number of epochs, frames, and planes.  
        nepcs = 0
        nplns = 0
        nfrms = 0
        cams  = [] 
        tsn   = []
        tsv   = []
        for i in range(nim):
            fnc = _parsefn( os.path.basename(files[i]) )
            if ( fnc['E'] > nepcs ):
                nepcs = fnc['E']
            if ( fnc['F'] > nfrms ):
                nfrms = fnc['F']
            if ( fnc['S'] > nplns ):
                nplns = fnc['S']

            try:
                ndx = cams.index(fnc['C'])
            except:
                cams.append(fnc['C'])
            
            if ( ( fnc['S'] == 0 ) and ( fnc['F'] == 0 ) ):
                try:
                    ndx = tsn.index(fnc['E'])
                except:
                    tsn.append(fnc['E'])
                    tsv.append(fnc['TS'])

        ncams = len(cams)
        nepcs = nepcs +1
        nplns = nplns +1
        nfrms = nfrms +1

        # Successful sort depends on these items being integers.
        cams.sort()
        tsn.sort()
        tsv.sort()    

        print "Epochs:  %i" % nepcs
        print "Cameras: %i" % ncams
        print "Frames:  %i" % nfrms
        print "Planes:  %i" % nplns

        # Make sure the time stamp didn't wrap around.
        for i in range(nepcs -1):
            if ( tsv[i+1] <= tsv[i] ):
                print "ERROR: Time stamp wrapped."
                print "TS[i] %i TS[i+1] %i" % (tsv[i],tsv[i+1])
                sys.exit()

        # Sort the images.  Images will be ordered in increasing
        # camera number.
        camims = []
        for c in range( ncams ):
            camims.append([])
            for t in range(nepcs):
                camims[c].append([])
                for i in range(nfrms):
                    camims[c][t].append([])

        for s in range(nplns):
            for i in range(nim):
                fname = os.path.basename( files[i] )
                fnc   = _parsefn(fname)

                epc = fnc['E']
                cam = fnc['C']
                frm = fnc['F']
                pln = fnc['S']

                if ( pln != s ):
                    continue
                
                cndx = cams.index(cam)
                camims[cndx][epc][frm].append(fname)

        # Make sure the EFRMLST base path exists.
        if ( not( os.path.exists(lbfpath) ) ):
            os.mkdir(lbfpath)

        # Build the EFRMLST files.
        for t in range(nepcs):
            for f in range(1,nfrms):
                frmlst = "EFRMLST-E%i-F0_%i" % (t,f)

                ffh = open("%s/%s" % (lbfpath,frmlst),'w')
                ffh.write("%i\n" % t )
                ffh.write("%i\n" % ncams )
                ffh.write("%i\n" % 2 ) # Number of frames
                ffh.write("%i\n" % nplns)
                ffh.write("%s\n" % bfileurl)
                ffh.write("0,%i\n" % f)

                for s in range(nplns):
                    sstr = ""
                    for c in range(ncams):
                        sstr = sstr \
                            +str( camims[c][t][0][s] ) + " " \
                            +str( camims[c][t][f][s] ) + " "

                    sstr = "%s%s\n" % (sstr,tsv[t])
                    ffh.write(sstr)

                ffh.close()

        carriage['lbfpath'] = lbfpath


#################################################################
#
class loadimg(spivetstep):
    """
    STEP: Load images
    
    Loads a group of images.  Step is currently configured to work with 
    filenames having the following format:
  
        BASENAME-Ew_Cx_Fy_Sz_TIMESTAMP.ext

    where:
        w ---- Epoch number.  0 <= w <= w_max.  Format: %i
        x ---- Camera number. 0 <= x < 2.       Format: %i
        y ---- Frame number.  0 <= y <= y_max.  Format: %i
        z ---- Plane number.  0 <= z <= z_max.  Format: %i

    ---- carriage inputs ----
    imgurls -- (M) List of image URL's to open.  If the URL points to a 
               remote file using the ftp scheme, the file will be
               retrieved using anonymous ftp and stored in lbpath.  

    ---- carriage outputs ----
    imgchnls - List containing extracted image channels for each image 
               (ie, a list of rank 2).  Extracted images will be ordered
               as [h,s,i] (or [r,g,b] if the rgb config flag is True).
    imgprms -- Dictionary containing image identification parameters extracted
               from the filename as described above.  The image parameters
               are epoch number ('E'), camera number ('C'), frame 
               number ('F'), plane number ('S'), and timestamp ('TS'), where
               the strings in parentheses are the dictionary keys.
   
    ---- config dictionary contents ----
    lbpath --- (O) Base path to store files retrieved using anonymous FTP.
               Defaults to DATA.
    hsmthd --- (O) Hue separation method.  Defaults to 0.  See documentation
               on pivlib.imread() for more on hue separation methods. 
    rgb ------ (O) Boolean flag to indicate whether RGB channels should
               be returned instead of HSI.  Defaults to False (ie, returns
               HSI).
    
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage

        imgurls = array(carriage['imgurls'])
        imgurls = imgurls.reshape( imgurls.size )

        if ( self.m_config.has_key('lbpath') ):
            lbpath = self.m_config['lbpath']
        else:
            lbpath = 'DATA'

        if ( self.m_config.has_key('hsmthd') ):
            hsmthd = self.m_config['hsmthd']
        else:
            hsmthd = 0

        if ( self.m_config.has_key('rgb') ):
            rgb = self.m_config['rgb']
        else:
            rgb = False

        # Get a copy of the images if necessary.
        urlcomp = urlparse.urlparse(imgurls[0])
        if ( urlcomp[0] == 'ftp' ):
            _fretrieve(imgurls,lbpath)

        # Load the channels.
        imgchnls = []
        imgprms  = []
        for url in imgurls:
            urlcomp = urlparse.urlparse(url)
            fname   = os.path.basename(urlcomp[2])
            if ( urlcomp[0] == 'ftp' ):
                fpath = "%s/%s" % (lbpath,fname)
            else:
                fpath = urlcomp[2]
                
            imgprms.append( _parsefn(fname) )
            imgchnls.append( pivlib.imread(fpath,hsmthd,rgb) )

        carriage['imgchnls'] = imgchnls
        carriage['imgprms']  = imgprms


#################################################################
#
class dewarpimg(spivetstep):
    """
    STEP: Dewarp images

    Dewarps a group of images.  The appropriate WICSP objects from 
    photogrammetric calibration are chosen based on the camera number
    as specified in imgprms.
    
    If only one camera is used, then the camera number for all images
    MUST be the same.
    
    The number of world pixels in the dewarped images is expected to
    be equal between cameras.
    
    ---- carriage inputs ----
    imgchnls - (M) List of image channels for each image (ie, a list of 
               rank 2).
    imgprms -- (M) Dictionary of image parameters.  Must contain:
                   'C' ----- Camera number

    ---- carriage outputs ----
    imgchnls - List of dewarped image channels for each image 
               ordered according to:
                   imgchnls[CAM][Frame][Channel]
               Note: Frame number from the filename is not consulted
               during ordering.  Images are stored as 'Frames' in
               the order in which they are stored in imgchnls for
               a given camera.
    imgprms -- List of image parameters sorted as:
                   imgprms[CAM][Frame]
               padpix parameter will be added to dictionary 
               (key = 'padpix').  Note: Frame number from the filename 
               is not consulted during ordering.  Images are stored as 
               'Frames' in the order in which they are stored in imgchnls 
               for a given camera.

    ---- config dictionary contents ----
    wicsp  --- (M) List of WICSP objects from calibration ordered 
               according to camera [CAM0, CAM1].  If only one camera
               is used, send a list with only one WICSP object.

               wicsp can also be a list of URL's to WICSP objects.
    padpix --- (O) Number of pad pixels to be placed at the image border
               of dewarped image.  
               
               When computing optical flow on PIV data, the image region
               of interest (ROI) specified in the pivdict dictionary 
               (parameter gp_rbndx) must leave a border around the ROI
               at least as large as the total max permitted displacement
               (specified with of_maxdisp + of_rmaxdisp).  Flow vectors
               cannot be computed for image regions within this border
               zone, so data is 'lost.'  Pad pixels can be used to reduce,
               but not eliminate, the data loss due to the required border. 

               Added pad pixels will have a value of 0.  Defaults to 0.
    lbpath --- (O) Path to store WICSP object(s) if necessary.  Only
               needed if wicsp objects are retrieved.  Defaults to
               'CALIBRATION'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        imgprms  = carriage['imgprms']
        imgchnls = carriage['imgchnls']
        wicsp    = self.m_config['wicsp']
        if ( self.m_config.has_key('lbpath') ):
            lbpath = self.m_config['lbpath']
        else:
            lbpath = 'CALIBRATION'

        for i in range( len(wicsp) ):
            if ( isinstance(wicsp[i],basestring) ):
                pth      = url2local(wicsp[i],lbpath)
                wicsp[i] = pivlib.loadwicsp(pth[0])

        if ( len(imgprms) != len(imgchnls) ):
            raise ValueError("Length of imgprms and imgchnls must be equal.")

        if ( self.m_config.has_key('padpix') ):
            padpix = self.m_config['padpix']
        else:
            padpix = 0

        ncam = len(wicsp)
        if ( ncam == 1 ):
            snglcam  = imgprms[0]['C']    

        nchnls = len(imgchnls[0])

        # Project the images to the world plane.
        simgchnls = []
        simgprms  = []
        for i in range(ncam):
            simgchnls.append([])
            simgprms.append([])

        nim = len(imgprms)
        for i in range(nim):
            cam = imgprms[i]['C']
            if ( ncam == 1 ):
                if ( cam != snglcam ):
                    raise ValueError("Camera number mismatch for a single camera.")
                cam = 0

            simgchnls[cam].append([])

            simgprms[cam].append(imgprms[i])
            simgprms[cam][-1]['padpix'] = padpix

            for c in range(nchnls):
                simgchnls[cam][-1].append(
                    pivlib.prjim2wrld( imgchnls[i][c], wicsp[cam] ) )

        # Pad the images.
        if ( padpix > 0 ):
            wpxdim  = wicsp[0]['wpxdim']
            pado    = array([padpix,padpix])
            pwpxdim = wpxdim +2*pado

            for cam in range(ncam):
                for frm in range( len(simgchnls[0]) ):
                    for c in range(nchnls):
                        timg = zeros(pwpxdim,dtype=float)
                        timg[padpix:(padpix+wpxdim[0]),
                             padpix:(padpix+wpxdim[1])] = simgchnls[cam][frm][c]
                        simgchnls[cam][frm][c] = timg
			#Xiyuan:rotate cal
			if (self.m_config.has_key('rotate_cal')):
				if (self.m_config['rotate_cal']):
					simgchnls[cam][frm][c] = rot90(timg,2)
        carriage['imgchnls'] = simgchnls
        carriage['imgprms']  = simgprms


#################################################################
#
class recordtime(spivetstep):
    """
    STEP: Saves the planar timestamp for Frame 0 (in [s]) to a 
    PIVVar.  Also saves the the time between Frame 0 and Frame 1
    ([s]) to a PIVVar.
    
    NOTE: Only handles data on a per-plane basis (ie, use with 
    loop_plane step).

    ---- carriage inputs ----
    pivdata -- (M) PIVData object in which the PLNR-TIME and 
               PLNR-DLTATIME variables should be stored.
    imgprms -- (M) List of image parameters sorted as:
                   imgprms[CAM][Frame]
               Must contain:
                   'TS' ----- Timestamp [msec].  Note the units.  

    ---- carriage outputs ----
    pivdata -- PIVData object, modified in place.  PLNR-TIME and
               PLNR-DLTATIME variables will be added to the object.

    ---- config dictionary contents ----
    None
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        pd       = carriage['pivdata']
        imgprms  = carriage['imgprms']
        
        eshape = pd[0].eshape
        
        # Get time of Frame 0.
        t0 = imgprms[0][0]['TS']/1000.
        t1 = imgprms[0][1]['TS']/1000.
        deltat = (t1 -t0)

        tv = pivlib.PIVVar([1,1,eshape[1],eshape[2]],
                           "PLNR-TIME","S",dtype=float,vtype="E")
        tv[:,...] = t0

        dtv = pivlib.PIVVar([1,1,eshape[1],eshape[2]],
                            "PLNR-DLTATIME","S",dtype=float,vtype="E")
        dtv[:,...] = deltat
        

        pd[0].addVars([tv,dtv])


#################################################################
#
def _flowstats(ofDisp,ofINAC,label='OFCOMP STATISTICS'):
    """
    Displays flow stats.
    """
    ofinac = ofINAC.squeeze()

    imsk = ofinac > 0
    inac = imsk.sum()

    print "--- %s ---" % label
    print "INAC CELLS:  " + str(inac)
    print "|"
    
    ofDisp.printStats()
    print "---"


#################################################################
#
class oflow2d(spivetstep):
    """
    STEP: 2D Optical flow

    Computes 2D optical flow from a set of image frames (representing a 
    single plane).  

    NOTE: Only handles data on a per-plane basis (ie, use with 
    loop_plane step).

    ---- carriage inputs ----
    imgchnls - (M) List of dewarped image channels for two or four images
               ordered according to:  
                   imgchnls[CAM][Frame][Channel]
               Channel 2 will be used for flow computation (ie, the
               intensity channel for HSI or the blue channel for RGB).
    imgprms -- (M) List of image parameters sorted as:
                   imgprms[CAM][Frame]
               Must contain:
                   'padpix' - Number of pad pixels.
    pivdata -- (O) Planar PIVData object to which the optical flow vectors
               will be added.  If a valid PIVData object is not passed,
               one will be created.

    ---- carriage outputs ----
    pivdata -- PIVData object containing the optical flow vectors and INAC
               flag values.  The flow vectors will be 2D vectors 
               (PIVVar name: R*, where * corresponds to the camera 
               number).  INAC flag values will be stored in the 
               variables with name 'R*INAC'.  CMAX values will be stored as
               R*CMAX. 
               
               If a PIVData object is created, the PIVData world z-origin
               will be set to 0.0.  The world x,y-origin will be computed
               from 'gp_rbndx', and 'padpix'.  Cell size will be determined 
               from the pivdict Global parameters and zcellsz.  Otherwise, 
               the output PIVData object origin and cellsz will be the same 
               as the input.

               NOTE:  Any variables in the input PIVData object named
               R*, R*INAC, or R*CMAX will be overwritten.

    ---- config dictionary contents ----
    pivdict -- (M) The dictionary containing PIV tuning parameters.  Must
               contain the Global, Image Registration, and Optical Flow
               parameters defined in the pivlib module documentation.
    zcellsz -- (O) Cell size along the z-axis.  Should be in units of
               pixels.  Defaults to 1.0.  
               
               NOTE:  If an input PIVData obect is available on the 
               carriage, zcellsz will be ignored.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        import time

        # Initialization.
        carriage = self.m_carriage
        imgchnls = carriage['imgchnls']
        imgprms  = carriage['imgprms']

        pivdict = self.m_config['pivdict']

        if ( self.m_config.has_key('zcellsz') ):
            zcellsz = self.m_config['zcellsz']
        else:
            zcellsz = 1.0

        ncam = len(imgchnls)

        # Dump out the pivdict.
        keys = pivdict.keys()
        keys.sort()
        print
        print "--- pivdict ---"
        for k in keys:
            if ( k.startswith("pg_wicsp") 
                or k.startswith("pg_camcal") 
                or k.startswith("tc_tlc") ):
                continue
            v = pivdict[k]
            print "%-15s : %s" % (k,str(v))
        print "---"        

        # Setup the PIVEpoch and PIVData objects.
        if ( carriage.has_key('pivdata') ):
            pd = carriage['pivdata']
            pe = pd[0]
        else:
            pe  = pivlib.PIVEpoch(0)
            pad = imgprms[0][0]['padpix']
            
            [cellsz,origin] = csznorgn(zcellsz,pivdict,pad,False)        

            #Xiyuan
            print "cellsz,origin",cellsz,origin 
            pd = pivlib.PIVData(cellsz,origin,"PIVDATA")
            pd.append(pe)

            carriage['pivdata'] = pd
        
        # Compute two-component flow for each camera.
        for cam in range(ncam):
            a    = time.time()
            rslt = pivlib.ofcomp(imgchnls[cam][0][2],
                                 imgchnls[cam][1][2],
                                 pivdict)
            a    = time.time() -a
            print "CAM %i OFCOMP RUN TIME: %g" % (cam,a)
        
            rslt[0].setAttr("R%i" % cam,rslt[0].units)
            rslt[1].setAttr("R%iINAC" % cam,"NA")
            rslt[2].setAttr("R%iCMAX" % cam,"NA")

            _flowstats(rslt[0],rslt[1])

            pe.addVars(rslt)
        

#################################################################
#
class refine2dof(spivetstep):
    """
    STEP: Refine 2D Optical flow

    Refines 2D optical flow results using an iterative implementation
    of thin plate splines.
    
    Some flow regimes with strong velocity gradients can cause image
    registration to fail or produce inaccurate results if the image block 
    deforms excessively between frames or for other reasons.  
    
    The refine2dof step takes the results from the oflow2d step and 
    interatively improves those results.  The oflow2d results are used
    to warp Frame 0 using thin plate splines and then optical flow
    is recomputed to form a displacement correction.

    Running the refine2dof step is computationally expensive.  Although
    the method can certainly be run to refine all flow vectors for all images
    in a dataset, the computation time and memory footprint required to do so 
    will be extreme.  The reasons for these burdens are twofold: 1) at each
    iteration, the optical flow is recomputed which has a non-negligible
    computational cost, and 2) each iteration requires the fitting and
    evaluation of thin plate splines which takes a lot of memory and is
    extremely costly in terms of computation time.  Consequently, flow
    refinement should be limited to those problematic regions of a dataset.
    
    refine2dof should be called immediately after oflow2d (or following
    any desired filtering steps).
    
    NOTE: Only handles data on a per-plane basis (ie, use with 
    loop_plane step).

    ---- carriage inputs ----
    imgchnls - (M) List of dewarped image channels for two or four images
               ordered according to:  
                   imgchnls[CAM][Frame][Channel]
               Channel 2 will be used for flow computation (ie, the
               intensity channel for HSI or the blue channel for RGB).
    imgprms -- (M) List of image parameters sorted as:
                   imgprms[CAM][Frame]
               Must contain:
                   'padpix' - Number of pad pixels.
    pivdata -- (M) Planar PIVData object to which the corrected optical 
               flow vectors will be added.  The pivdata object must contain
               2D flow results from a previous call to oflow2d.

    ---- carriage outputs ----
    pivdata -- PIVData object containing the updated optical flow vectors.
               The output variable names will be appended with '-TR' (eg,
               R0-TR).  Two additional variables will be added to the 
               PIVData object to contain the normalized cross-correlation
               coefficients (recorded at the last iteration for the 
               particular cell).  These variables will be named R*CMAX-TR,
               where * represents the camera number.
    
    ---- config dictionary contents ----
    pivdict -- (M) The dictionary containing PIV tuning parameters.  Must
               contain the Global, Image Registration, and Optical Flow
               parameters defined in the pivlib module documentation.
    varnm ---- (M) List of variable names to refine, one for each camera.  
               The length of varnm must be equal to the number of cameras.  
               If the results for a given camera are not to be processed, 
               set the varnm entry for that camera to None.
    crbndx --- (O) rbndx array for planr cells to refine.  crbndx is a 2x2 
               array of the same format as the pivdict parameter 'gp_rbndx'.
               The rectangular region of cells for refinement will be
               automatically reduced if a group of cells has a correction
               below eps for two consecutive iterations.  
               
               Defaults to all cells (which is computationally expensive).  
    planes --- (O) List of z-planes to process.  If the Plane number, as
               contained within the carriage input imgprms, is listed in
               planes, then the flow results for that plane will be refined.
               Otherwise, the results for the varnm variables will be copied
               with new names and otherwise unaltered.  Defaults to processing 
               all planes (which is computationally expensive).
    rfactor -- (O) Relaxation factor.  The update will be scaled by rfactor
               to speed convergence.  A value of 1.0 will disable convergence
               acceleration.  Be careful of large values as they can cause
               the method to diverge.  Defaults to 1.
    its ------ (O) Maximum number of iterations.  If eps is not reached, 
               iterative loop will terminate after its iterations.  Defaults to 
               10.
    eps ------ (O) Maximum acceptable correction magnitude.  If the maximum
               correction magnitude during an iteration is less than eps, the
               iterative loop will terminate.  Defaults to 0.5.
    cfltrprm - (O) List of argument dictionaries for filtering operations.
               These parameters control optional smoothing operations.
               The list can contain any number of operations and each 
               operation will be applied in order.  The set of operations will
               be applied each iteration to the flow correction before it 
               applied to the flow vectors.  
               
               Note: No pre-processing of input flow vectors will be performed
               by the refine2dof step during initialization.
    
               Each dictionary must contain a key named 'filter' specifying
               the filter to run.  Currently two filters are supported
                   'medfltr'
                   'gsmooth'           
               The details for these filters can be found in the pivpost
               documentation.
    
               In addition to the 'filter' key, each dictionary must contain
               additional keys depending on the type of filter.  The keys are:
                   medfltr keys: 'fdim', 'rthsf', 'reps', 'planar', 'nit',
                                 'cndx'
                   gsmooth keys: 'gbsd', 'planar', 'nit', 'cndx'
               All keys must be available for the given filter.     

               Defaults to None.
    ifltr ---- (O) List of frequency domain image filters, one for each camera.
               If ifltr for a particular camera is not set to None, the raw 
               images used for refinement will be filtered.  The filters will 
               be applied in the frequency domain.  It is recommended that the 
               user set 'dbgpath' and design the filter according to the FFT
               images produced.  The filter must be a real or complex floating
               point array having dimensions of
                   rrsize +2*(of_maxdisp +of_rmaxdisp)
               where rrsize is image size corresponding to crbndx.  
               
               Application of the filter proceeds as follows.  First the image
               and filter are padded.  The filter padding is performed by taking
               the user-specified ifltr, inverse Fourier transforming to spatial 
               coordinates, taking the real part of the resulting spatial filter,
               padding, forward transforming, and then taking the absolute value
               of the resulting frequency filter to ensure the filter is a 
               zero-phase shift filter.  Image padding is done using the method 
               specified by the config parameter ifpad.  The padded image/filter
               dimensions are 2X the originals to prevent wrap-around induced 
               errors during filtering. 
    
               Defaults to None for all cameras.  Note: If ifltr is specified,
               the list must have an entry (None or a valid filter/URL) for
               each camera.
    
               Note: ifltr can be set to a list of URL's that point to filters.
               The filters will then be retrieved.  URL's must point to files
               pickled with pivlib.pkldump().
    lbpath --- (O) Path to store the filters of ifltr if necessary.  Only needed
               if the ifltr filters are retrieved.  Defaults to 'R2DIFILTER'.
    ifpad ---- (O) Specifies how ifltr image padding should be handled prior
               to application of the filter.  The following values are
               possible
                   0 ---- Images will be padded with zeros.
                   1 ---- Images will be padded with the mean value of the
                          particular image.  Can reduce edge darkening or
                          ringing depending on the filter.
               Defaults to 0.
    dbgpath -- (O) Path in which to store debug images.  Debug images can be
               very helpful during tuning.  If dgbpath is set to a valid path,
               then a series of images will be dumped for each camera.  All
               images will have dimensions of
                   rrsize +2*(of_maxdisp +of_rmaxdisp)
               where rrsize is image size corresponding to crbndx.  The images
               are
                   RAW-C*_F*.png -------- Raw, unfiltered image region that will
                                          be used during the refinement process.  
                   FFT-C*_F*.png -------- Log of the FFT magnitude.  Useful for 
                                          designing filters.
                   FLTRD-C*_F*.png ------ Raw images filtered using the specified
                                          ifltr.
                   WARPED-C*_F0_IT*.png - Warped version of FRAME 0 at the start
                                          of iteration IT.
               Only set dgbpath when working on a single z-plane for a single
               Epoch.  Otherwise images will be overwritten.  dgbpath must
               be set to an absolute path.  Defaults to None (which disables
               output).
               
               Note: dgbpath is only effective when steps are run locally (ie,
               when the loop_epoch step configuration parameter 'parallel' is
               set to False).
    fltronly - (O) If True, then the images will be filtered and the step will
               return.  Primarily intended for use with filter tuning. Defaults
               to False.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def __regionprms__(self,pivdict,crbndx,pxorgn):
        """
        Computes region bounding indices and block center coords.
        """
        crsize   = crbndx[:,1] -crbndx[:,0]
        tnrcells = crsize.prod()
        
        # Get pivdict parameters.
        rbndx = array( pivdict['gp_rbndx'] )
        bsize = array( pivdict['gp_bsize'] )
        bolap = array( pivdict['gp_bolap'] )
        bsdiv = pivdict['gp_bsdiv']
        
        maxdisp  = array( pivdict['of_maxdisp'] )
        rmaxdisp = array( pivdict['of_rmaxdisp'] )
        
        # Compute image region rbndx corresponding to crbndx.
        rrbndx = rbndx.copy()
        ebsize = (bsize -bolap)/bsdiv
            
        rrbndx[:,0] = rbndx[:,0] +crbndx[:,0]*ebsize -pxorgn
        rrbndx[:,1] = rbndx[:,0] +crbndx[:,1]*ebsize -pxorgn
        
        # Compute extended rrbndx.  The extended region contains a border of
        # of_maxdisp +of_rmaxdisp pixels.  
        mxdsp   = maxdisp +rmaxdisp
        xrrbndx = rrbndx.copy()
        
        xrrbndx[:,0] = xrrbndx[:,0] -mxdsp
        xrrbndx[:,1] = xrrbndx[:,1] +mxdsp
        
        xrrsize = xrrbndx[:,1] -xrrbndx[:,0]

        # Setup image indices and block center coordinates (in pixels) for the
        # extended region.  Coordinates are relative to the extended region.
        xrrfn = indices(xrrsize,dtype=float)
        xrrfn = xrrfn.reshape([2,xrrfn[0].size]).transpose()
        xrrfn = xrrfn +xrrbndx[:,0]

        wxrrfn = zeros(xrrfn.shape,dtype=float)

        bco    = bsize/(2.*bsdiv)
        bccrds = indices(crsize,dtype=float)
        for i in xrange(2):
            bccrds[i,...] = ebsize[i]*bccrds[i,...] +rrbndx[i,0] +bco[i]

        bccrds  = bccrds.reshape([2,tnrcells]).transpose()

        # Build the dictionary.
        rprms = {'crsize':crsize,
                 'tnrcells':tnrcells,
                 'xrrbndx':xrrbndx,
                 'xrrsize':xrrsize,
                 'rrbndx':rrbndx,
                 'xrrfn':xrrfn,
                 'wxrrfn':wxrrfn,
                 'bccrds':bccrds}
        
        return rprms
    
    def __cfilter__(self,crc,cfltrprm):
        """
        Applies the filter schedule from cfltrprm to the vector correction.
        """
        for fltr in cfltrprm:
            ftype = fltr['filter']
            if ( ftype == 'medfltr' ):
                [crc,fflg] = pivlib.medfltr(crc,
                                            fltr['fdim'],
                                            rthsf=fltr['rthsf'],
                                            reps=fltr['reps'],
                                            planar=fltr['planar'],
                                            nit=fltr['nit'],
                                            cndx=fltr['cndx'])                        
            elif ( ftype == 'gsmooth' ):
                crc = pivlib.gsmooth(crc,
                                     fltr['gbsd'],
                                     planar=fltr['planar'],
                                     nit=fltr['nit'],
                                     cndx=fltr['cndx'])
            else:
                print "refine2dof(): Unknown filter %s" % (ftype) 

        return crc
    
    def __ifilter__(self,imgs,ifltr,ifpad,pifltr=None):
        """
        Applies the filters of ifltr to the intensity images in imgchnls.
        
        imgs must be a list.
        
        Returns [fimgs,pifltr].
        """
        # Initialization.
        ncam   = len(imgs)
        imdim  = array(imgs[0][0].shape)
        pimdim = 2*imdim

        if ( compat.checkNone(pifltr) ):
            pifltr = []
        
        # Main loop.
        fimgs  = []
        for c in xrange(ncam):
            fimgs.append([])
            
            # Pad the filter.
            if ( len(pifltr) != ncam ):
                PF = pivlib.padfltr(ifltr[c],pimdim,decntr=True)
                PF = abs( PF )  # Force zero phase shift.
                pifltr.append( PF )
                
            for f in xrange(2):
                pimg = pivlib.padimg(imgs[c][f],pimdim,ptype=ifpad)
                PIMG = pifltr[c]*fft.fft2(pimg)
                pimg = real( fft.ifft2(PIMG) )
                pimg = pimg -pimg.min()
                pimg = pimg/pimg.max()                
                fimgs[c].append( pimg[0:imdim[0],0:imdim[1]] )
                
        return [fimgs,pifltr]

    def __xtrctimgs__(self,imgchnls,pivdict,crbndx):
        """
        Extracts image regions of interest corresponding to crbndx.  ROI
        will be larger than equivalent crbndx by 2*(of_maxdisp +of_rmaxdisp).

        Returns imgs.
        """
        # Initialization.
        ncam  = len(imgchnls)

        rprms   = self.__regionprms__(pivdict,crbndx,[0,0])
        xrrbndx = rprms['xrrbndx']
        
        # Get image regions.
        imgs = []
        for c in xrange(ncam):
            imgs.append([])
            for f in xrange(2):
                img = imgchnls[c][f][2][xrrbndx[0,0]:xrrbndx[0,1],
                                        xrrbndx[1,0]:xrrbndx[1,1]]
                imgs[c].append(img)

        return imgs
        
    def execute(self):
        from scipy import fftpack, stats, ndimage
        import time, pylab
        a = time.time()

        # Initialization.
        carriage = self.m_carriage
        imgchnls = carriage['imgchnls']
        imgprms  = carriage['imgprms']
        pd       = carriage['pivdata']

        pivdict = self.m_config['pivdict']
        varnm   = self.m_config['varnm']

        ncam = len(imgchnls)
                
        if ( self.m_config.has_key('crbndx') ):
            crbndx = array(self.m_config['crbndx'])
        else:
            nvcells = pd[0].eshape[1::]
            crbndx  = array([[0,nvcells[0]],[0,nvcells[1]]])
        
        crsize   = crbndx[:,1] -crbndx[:,0]
        tnrcells = crsize.prod()

        if ( self.m_config.has_key('planes') ):
            planes = self.m_config['planes']
        else:
            planes = None
            
        if ( self.m_config.has_key('rfactor') ):
            rfactor = self.m_config['rfactor']
        else:
            rfactor = 1.
        
        if ( self.m_config.has_key('its') ):
            its = self.m_config['its']
        else:
            its = 10
            
        if ( self.m_config.has_key('eps') ):
            eps = self.m_config['eps']
        else:
            eps = 0.5
            
        if ( self.m_config.has_key('cfltrprm') ):
            cfltrprm = self.m_config['cfltrprm']
        else:
            cfltrprm = []

        if ( self.m_config.has_key('lbpath') ):
            lbpath = self.m_config['lbpath']
        else:
            lbpath = 'R2DIFILTER'
            
        runfltr = False
        if ( self.m_config.has_key('ifltr') ):
            ifltr   = self.m_config['ifltr']
            runfltr = True

            mflg = False
            for f in xrange(ncam):
                fltr = ifltr[f]
                if ( isinstance(fltr,basestring) ):
                    pth      = url2local(fltr,lbpath)
                    fltr     = pivlib.pklload(pth[0])
                    ifltr[f] = fltr
                    mflg     = True

            if ( mflg ):
                self.m_config['ifltr'] = ifltr
            
        if ( self.m_config.has_key('ifpad' ) ):
            ifpad = self.m_config['ifpad']
        else:
            ifpad = 0

        if ( self.m_config.has_key('dbgpath') ):
            try:
                # If running parallel, this global is available.  We don't
                # want to dump debug images if parallel.
                rnk = SleighRank
                rnk = 0

                dbgpath = None
            except:
                dbgpath = self.m_config['dbgpath']
                if ( not compat.checkNone(dbgpath) and not os.path.exists(dbgpath) ):
                    os.mkdir(dbgpath)
        else:
            dbgpath = None
            
        if ( self.m_config.has_key('fltronly') ):
            fltronly = self.m_config['fltronly']
        else:
            fltronly = False
            
        # This config parameter will be added after the step has been run
        # once if filtering is enabled.
        if ( self.m_config.has_key('pifltr') ):
            pifltr = self.m_config['pifltr']
        else:
            pifltr = None

        # Check that the current plane is in planes, otherwise just copy the
        # input variables.
        if ( not compat.checkNone(planes) ):
            cplane = imgprms[0][0]['S']
            if ( not planes.__contains__(cplane) ):
                for cam in xrange(ncam):
                    v = varnm[cam]
                    if ( compat.checkNone(v) ):
                        continue

                    rvar = pd[0][v].copy()
                    rvar.setAttr("%s-TR" % rvar.name,rvar.units,rvar.vtype) 

                    cmaxvar = pivlib.PIVVar([1,1,rvar.shape[2],rvar.shape[3]],
                                            "R%iCMAX-TR" % cam,"NA")

                    pd[0].addVars([rvar,cmaxvar])
                return None

        # Extract image regions corresponding to crbndx.
        imgs = self.__xtrctimgs__(imgchnls, pivdict, crbndx)
                                                                    
        # Filter the images.
        if ( runfltr ):
            [fimgs,pifltr] = self.__ifilter__(imgs,ifltr,ifpad,pifltr=pifltr)
            self.m_config['pifltr'] = pifltr
        else:
            fimgs = imgs

        # Dump debug images.
        if ( not compat.checkNone(dbgpath) ):
            for c in xrange(ncam):
                for f in xrange(2):
                    pivlib.imwrite(imgs[c][f],
                                   "%s/RAW-C%i_F%i.png" % (dbgpath,c,f),
                                   vmin=0.,vmax=1.)

                    pivlib.imwrite(fimgs[c][f],
                                   "%s/FLTRD-C%i_F%i.png" % (dbgpath,c,f),
                                   vmin=0.,vmax=1.)

                    IMG = fft.fftshift( fft.fft2(imgs[c][f]) )
                    IMG = log10( abs(IMG) +1.E-12 )
                    pivlib.imwrite(IMG,
                                   "%s/FFT-C%i_F%i.png" % (dbgpath,c,f),
                                   cmap=pylab.cm.jet)
            del IMG
            
        if ( fltronly ):
            return            
                
        # Grab a copy of Frame 0.  This frame will be warped.
        pf0 = []
        for cam in xrange(ncam):
            pf0.append(fimgs[cam][0].copy())
        
        # Get a reference to the original rbndx.  We'll need to restore this
        # later.
        orbndx = pivdict['gp_rbndx']
        
        # Grab the pixel origin for the extended region in relation to
        # the full image.
        rprms   = self.__regionprms__(pivdict,crbndx,[0,0])
        xrrbndx = rprms['xrrbndx']
        pxorgn  = xrrbndx[:,0]
        
        ocrbndx = crbndx

        print "-v-v-v-v-v- REFINE2DOF -v-v-v-v-v-"
        print "EXT REGION: [%i:%i,%i:%i] PIX" % tuple(xrrbndx.reshape(4))

        # Camera loop.
        for cam in xrange(ncam):
            print ">>>>> CAM %i <<<<<" % cam

            if ( compat.checkNone(varnm[cam]) ):
                continue

            # Get region parameters.
            pivdict['gp_rbndx'] = orbndx

            crbndx = ocrbndx
            rprms  = self.__regionprms__(pivdict,crbndx,pxorgn)

            crsize   = rprms['crsize']
            tnrcells = rprms['tnrcells']
            xrrbndx  = rprms['xrrbndx']
            xrrsize  = rprms['xrrsize']
            rrbndx   = rprms['rrbndx']
            xrrfn    = rprms['xrrfn']
            wxrrfn   = rprms['wxrrfn']
            bccrds   = rprms['bccrds']

            # Due to the memory intensive nature of TPS, the image region
            # to be warped needs to be broken into chunks.
            csz = 10000
            ncit = int(ceil(xrrfn.shape[0]/float(csz)))
    
            # Setup image rbndx.
            pivdict['gp_rbndx'] = rrbndx

            # The candidate deletion mask.  In order for the window to be
            # reduced in size, the block updates must be less than eps for
            # two iterations.
            cmsk = zeros(crsize,dtype=bool)

            # Updated variable.
            rvar = pd[0][varnm[cam]].copy()
            var  = rvar[:,:,crbndx[0,0]:crbndx[0,1],crbndx[1,0]:crbndx[1,1]]
            
            cmaxvar = pivlib.PIVVar([1,1,rvar.shape[2],rvar.shape[3]],
                                    "R%iCMAX-TR" % cam,"NA")
            
            # Main loop.
            for it in xrange(its):
                # Create TPS object.  Do not use regularization.  Regularization
                # creates a disconnect between the vector correction and the
                # corresponding image warp.  As a result, vector corrections can
                # accumulate without ever causing a corresponding change in the
                # image warp.  Smooth the correction instead. 
                dbccrds = bccrds +var[1:3,0,...].reshape([2,tnrcells]).transpose()
                tps     = pivlib.pivutil.tpswarp(dbccrds,bccrds,lmda=0.)
                
                # Get warped coordinates (ie, coordinates to warp Frame 0).  
                # Loop over chunks.
                for cit in xrange(ncit):
                    bndx = cit*csz
                    endx = bndx +csz
                    wxrrfn[bndx:endx,:] = tps.xfrm(xrrfn[bndx:endx,:])
            
                p = xrrfn -wxrrfn

                # Warp the image.
                wimg = pivlib.pivutil.pxshift(fimgs[cam][0],
                                              xrrfn.astype(int),p)
                wimg = wimg.reshape(xrrsize)
                pf0[cam][xrrbndx[0,0]:xrrbndx[0,1],
                         xrrbndx[1,0]:xrrbndx[1,1]] = wimg

                if ( not compat.checkNone(dbgpath) ):
                    pivlib.imwrite(pf0[cam],
                                   "%s/WARPED-C%i_F0_IT%i.png" % (dbgpath,cam,it),
                                   vmin=0.,vmax=1.)
                                
                # Compute flow.
                [crc,inac,cmax] = pivlib.ofcomp(pf0[cam],
                                                fimgs[cam][1],
                                                pivdict)

                # Filter the correction and apply.
                crc = self.__cfilter__(crc, cfltrprm)

                cmag = crc[1:3,...].reshape([2,tnrcells])
                cmag = sqrt( (cmag*cmag).sum(0) )
                cmmx = cmag.max()
                print "IT%i MAX |CRC|: %f" % (it,cmmx)
                                
                var = var +rfactor*crc

                # Update the variable.  This needs to be in the loop so
                # the analysis region can be reduced as necessary below.
                rvar[:,:,crbndx[0,0]:crbndx[0,1],crbndx[1,0]:crbndx[1,1]] = var
                cmaxvar[:,:,crbndx[0,0]:crbndx[0,1],crbndx[1,0]:crbndx[1,1]] = cmax

                if ( cmmx < eps ):
                    break

                # Reduce size of analysis region if possible.
                cmag  = cmag.reshape([crsize[0],crsize[1]])
                tmsk  = cmag < eps
                msk   = tmsk*cmsk
                cmsk  = tmsk
                elim  = [ zeros(crsize[0],dtype=bool),
                          zeros(crsize[1],dtype=bool) ]

                for ndx in xrange(crsize[0]):
                    if ( msk[ndx,:].all() ):
                        elim[0][ndx] = True 
                
                for ndx in xrange(crsize[1]):
                    if ( msk[:,ndx].all() ):
                        elim[1][ndx] = True
                        
                eadj = zeros([2,2],dtype=int)
                for c in xrange(2):
                    # Need to account for bsdiv, otherwise window will
                    # not contain a whole number of coarse blocks.
                    bsdiv = pivdict['gp_bsdiv']
                    
                    bo =  (elim[c].argmin()/bsdiv)*bsdiv
                    eo = -(elim[c][::-1].argmin()/bsdiv)*bsdiv

                    eadj[c,:] = [bo,eo]
                        
                if ( not ( eadj == 0 ).all() ):
                    pivdict['gp_rbndx'] = orbndx

                    tcrbndx   = crbndx +eadj
                    rprms     = self.__regionprms__(pivdict,tcrbndx,pxorgn)
                    ttnrcells = rprms['tnrcells']

                    # Ensure that TPS lmat isn't singular.
                    if ( ttnrcells >= 3 ):
                        cmsk = cmsk[eadj[0,0]:crsize[0]+eadj[0,1],
                                    eadj[1,0]:crsize[1]+eadj[1,1]]
                        
                        crbndx   = tcrbndx
                        tnrcells = ttnrcells
                        crsize   = rprms['crsize']
                        xrrbndx  = rprms['xrrbndx']
                        xrrsize  = rprms['xrrsize']
                        rrbndx   = rprms['rrbndx']
                        xrrfn    = rprms['xrrfn']
                        wxrrfn   = rprms['wxrrfn']
                        bccrds   = rprms['bccrds']
                        
                        csz = 10000
                        ncit = int(ceil(xrrfn.shape[0]/float(csz)))
                
                        var = rvar[:,:,
                                   crbndx[0,0]:crbndx[0,1],
                                   crbndx[1,0]:crbndx[1,1]]

                    pivdict['gp_rbndx'] = rrbndx
                print 

            # Store the variable.
            rvar.setAttr("%s-TR" % rvar.name,rvar.units,rvar.vtype)
            
            pd[0].addVars([rvar,cmaxvar])

        print "RUN TIME: %f" % ( time.time() -a )
        print "-^-^-^-^-^- REFINE2DOF -^-^-^-^-^-"

        pivdict['gp_rbndx'] = orbndx


#################################################################
#
class oflow3d(spivetstep):
    """
    STEP: 3D Optical flow

    Computes 3D optical flow from the output of the oflow2D step.
    Resulting displacement vectors will have units of mm.  All Epochs
    in the input PIVData object will be processed.

    NOTE: oflow3d only supports two cameras.

    ---- carriage inputs ----
    pivdata -- (M) Planar PIVData object to which the 3D optical flow 
               vectors will be added.  Must contain the 2D results from
               the oflow2d step.
    imgprms -- (O) List of image parameters sorted as:
                   imgprms[CAM][Frame]
               Either imgprms must contain
                   'padpix' - Number of pad pixels.
               or the padpix configuration parameter must be specified.

    ---- carriage outputs ----
    *pivdata - PIVData object containing the 3D optical flow vectors and 
               INAC flag values.  The PIVVar containing the 3D flow
               vectors will have the variable name 'U'.  INAC flag values 
               will be stored in the variable with name 'UINAC'.  
               
               The PIVData cellsz and origin will be updated to have
               units of mm.  The world x,y-origin will be computed
               from 'gp_rbndx', 'padpix', and 'pg_wicsp' parameters.  Cell
               size will be determined from the pivdict Global parameters
               and zcellsz.  
               
               NOTE:  Any variables in the input PIVData object named
               U or UINAC will be overwritten.

    ---- config dictionary contents ----
    pivdict -- (M) The dictionary containing PIV tuning parameters.  Must
               contain the Global, Optical Flow, and photogrammetric 
               parameters defined in the pivlib module documentation.
    pdname --- (O) Name of the PIVData object in the carriage to convert.
               Eg, 'epivdata'.  Defaults to 'pivdata'.
    varnms --- (O) List of variable names in the PIVData object containing
    		   the 2D flow results.  Defaults to ['R0','R1','R0INAC','R1INAC'].
    zcellsz -- (O) Cell size along the z-axis.  Should be in units of mm. 
               Defaults to 1.0.  
    lbpath --- (O) Path to store WICSP and CAMCAL object(s) if necessary.  
               Only needed if these objects are retrieved.  Defaults to
               'CALIBRATION'.
    padpix --- (O) Number of pad pixels placed around the image border
               by a previous call to dewarpimg.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        import time

        # Initialization.
        carriage = self.m_carriage

        if ( self.m_carriage.has_key('imgprms') ):
            imgprms = self.m_carriage['imgprms']
            
        pivdict = self.m_config['pivdict']

        if ( self.m_config.has_key('lbpath') ):
            lbpath = self.m_config['lbpath']
        else:
            lbpath = 'CALIBRATION'
        
        chkpivdict(pivdict,lbpath)
 
        if ( pivdict.has_key('pg_wicsp') ):
            wicsp = pivdict['pg_wicsp']
        else:
            wicsp = None

        if ( pivdict.has_key('pg_camcal') ):
            camcal = pivdict['pg_camcal']
        else:
            camcal = None

        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'pivdata'

        if ( self.m_config.has_key('varnms') ):
            varnms = self.m_config['varnms']
        else:
            varnms = ['R0','R1','R0INAC','R1INAC']
 
        if ( self.m_config.has_key('zcellsz') ):
            zcellsz = self.m_config['zcellsz']
        else:
            zcellsz = 1.0

        pd = carriage[pdname]

        try:
            padpix = imgprms[0][0]['padpix']        
        except:
            if ( self.m_config.has_key('padpix') ):
                padpix = self.m_config['padpix']
            
        # Update cellsz and origin.
        [cellsz,origin] = csznorgn(zcellsz,pivdict,padpix)        
        pd.setCellsz(cellsz)
        pd.setOrigin(origin)
        
        # Build the flowstats string.
        nepc  = len(pd)
        nplns = pd[0][varnms[0]].shape[1]
        if ( nepc > 1 or nplns > 1 ):
            usexstr = True
            fsstr   = 'THREE-COMPONENT FLOW STATISTICS (E%i,S%i)'
        else:
            usexstr = False
            fsstr   = 'THREE-COMPONENT FLOW STATISTICS'

        sshape = [ pd[0][varnms[0]].shape[0],
                   1,
                   pd[0][varnms[0]].shape[2],
                   pd[0][varnms[0]].shape[3] ]
        
        # Compute three-component flow and store results.
        for e in range( nepc ):
            pe = pd[e]
    
            ofdisp2d0 = pe[varnms[0]]
            ofdisp2d1 = pe[varnms[1]]
            ofinac2d0 = pe[varnms[2]]
            ofinac2d1 = pe[varnms[3]]

            ofinac3d = ofinac2d0 +ofinac2d1
            ofinac3d.setAttr("UINAC","NA")
            
            pe.addVars(ofinac3d)
                    
            for s in range(nplns):                
                ofdisp3d = pivlib.tcfrecon(
                              ofdisp2d0[:,s,:,:].reshape(sshape),
                              ofdisp2d1[:,s,:,:].reshape(sshape),
                              pivdict )
                
                sofi = ofinac3d[:,s,:,:]        
                if ( usexstr ):
                    _flowstats(ofdisp3d,sofi,fsstr % (e,s))
                else:
                    _flowstats(ofdisp3d,sofi,fsstr)
        
                if ( s == 0 ):
                    ovar = pivlib.PIVVar(ofdisp2d0.shape,
                                         "U",ofdisp3d.units,
                                         vtype=ofdisp3d.vtype)
                    
                    pe.addVars(ovar)
                    
                pe['U'][:,s,:,:] = ofdisp3d[:,0,:,:]
                

#################################################################
#
class disp2vel(spivetstep):
    """
    STEP: Convert displacements to velocities.

    Takes an optical flow PIVVar from a PIVData object and 
    divides the displacement by the time between CAM0 Frames 0 
    and 1 to create a velocity.  All Epochs in the PIVData object 
    will be processed.

    This step is primarily intended to be used immediately following
    oflow.

    ---- carriage inputs ----
    *pivdata - (M) PIVData object containing the optical flow 
               results in a variable name 'U'.  Will be 
               modified in place.
    imgprms -- (O) List of image parameters sorted as:
                   imgprms[CAM][Frame]
               Either imgprms must contain
                   'TS' ----- Timestamp [msec].  Note the units.  
               or the PIVData object must contain a variable named
               PLNR-DLTATIME with units of [sec] (eg, as stored by the 
               recordtime step).  If present, PLNR-DLTATIME will be used 
               instead of imgprms.
                
    ---- carriage outputs ----
    *pivdata - PIVData object, modified in place, containing the
               'U' values expressed as velocities.

    ---- config dictionary contents ----
    pdname --- (O) Name of the PIVData object in the carriage
               to convert.  Eg, 'epivdata'.  Defaults to 
               'pivdata'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'pivdata'

        pd = carriage[pdname]
        try:
            deltat  = pd[0]['PLNR-DLTATIME']
            havepdt = True
        except KeyError:
            imgprms = carriage['imgprms']
            havepdt = False
        
        if ( not havepdt ):            
            # Get time difference between frames.
            t0 = imgprms[0][0]['TS']
            t1 = imgprms[0][1]['TS']
            deltat = (t1 -t0)/1000.
            print "deltat [s]: %f" % deltat 

        for e in range( len(pd) ):
            ofdisp = pd[e]['U']
            if ( havepdt ):
                deltat = pd[e]['PLNR-DLTATIME']

            ofdisp[:,...] = ofdisp/deltat

            ofdisp.setUnits("%s_S" % (ofdisp.units))
            

#################################################################
#
class bimedfltr(spivetstep):
    """
    STEP: Remove salt and pepper noise from an image.

    Takes an image, breaks it into blocks, and applies a median filter 
    to remove salt and pepper noise from the image.  Applying this 
    step to the hue channel prior to extracting temperatures from TLC 
    colors is very useful.
    
    For more information on the technique used, see the documentation
    for pivlib.bimedfltr().

    ---- carriage inputs ----
    imgchnls - (M) List of dewarped hue, saturation, and intensity
               channels for each image ordered according to camera:
                   imgchnls[CAM][Frame][Channel]

    ---- carriage outputs ----
    imgchnls - Filtered image channels.  Channels will be filtered in
               place.
               
               Depending on the config options prccam, prcfrm, and prcchnl,
               some images may be ignored.  In this case, no modifications
               to those channels will be made

    ---- config dictionary contents ----
    rbndx ---- (M) 2x2 array giving the bounding indices for the region 
               to be analyzed.  rbndx is specified in the same format as 
               the pivdict parameter gp_rbndx.
    bsize ---- (M) Two element list specifying the size of the blocks to
               use during filtering.  ([y,x] pixels)
    rthsf ---- (O) Median threshold scale factor.  Defaults to 2.
    prccam --- (O) Boolean flag list indicating whether images for
               respective camera should be processed.  Defaults to 
               True for all cameras.
    prcfrm --- (O) Boolean flag list indicating whether images for
               respective Frame should be processed.  Defaults to True 
               for all Frames.
    prcchnl -- (O) Boolean flag list indicating whether a particular
               channel should be processed.  Defaults to True for all
               channels.                              
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage

        imgchnls = carriage['imgchnls']

        ncam  = len(imgchnls)
        nfrm  = len(imgchnls[0])
        nchnl = len(imgchnls[0][0])
        
        rbndx = self.m_config['rbndx']
        bsize = self.m_config['bsize']
        
        if ( self.m_config.has_key('rthsf') ):
            rthsf = self.m_config['rthsf']
        else:
            rthsf = 2.
            
        if ( self.m_config.has_key('prccam') ):
            prccam = self.m_config['prccam']
        else:
            prccam = ones(ncam,dtype=bool)
            
        if ( self.m_config.has_key('prcfrm') ):
            prcfrm = self.m_config['prcfrm']
        else:
            prcfrm = ones(nfrm,dtype=bool)            

        if ( self.m_config.has_key('prcchnl') ):
            prcchnl = self.m_config['prcchnl']
        else:
            prcchnl = ones(nchnl,dtype=bool)
            
        # Apply the filter.
        for c in range(ncam):
            if ( not prccam[c] ):
                continue
            for f in range(nfrm):
                if ( not prcfrm[f] ):
                    continue
                for ch in range(nchnl):
                    if ( not prcchnl[ch] ):
                        continue
                    imgchnls[c][f][ch] = pivlib.bimedfltr(imgchnls[c][f][ch],
                                                          rbndx,
                                                          bsize,
                                                          rthsf)
            
            
#################################################################
#
class tlcmask(spivetstep):
    """
    STEP: Find valid TLC particles.

    Takes the dewarped hue, saturation, and intensity channels
    and computes a mask identifying 'valid' TLC paricles.

    ---- carriage inputs ----
    imgchnls - (M) List of dewarped hue, saturation, and intensity
               channels for each image ordered according to camera:
                   imgchnls[CAM][Frame][Channel]

    ---- carriage outputs ----
    tlcmask -- A list of TLC masks for each image in imgchnls.  Masks
               will be stored as
                   tlcmask[CAM][Frame]
               Each tlcmask is a 2D array of the same size as the input
               image with valid particle pixels having a mask value of 
               1.0 and invalid particle pixels having mask values of 0.0.
               
               Depending on the config option prccam, the images from
               some cameras may be ignored.  In this case, the tlcmask
               for those images will be set to None.

    ---- config dictionary contents ----
    pivdict -- (M) The dictionary containing at least the Global Parameters
               as defined in the pivlib module documentation.  
    prccam --- (O) Boolean flag list indicating whether images for
               respective camera should be processed.  If True, then
               a TLC mask for the cameras will be created, otherwise the
               the mask will be set to None.  Defaults to True for all
               cameras.
    bgth ----- (O) Background threshold for intensity channel.  All TLC
               pixels with an intensity less than bgth will have a mask
               value of 0.0.  Defaults to 0.01.
    sth ------ (O) Background threshold for saturation channel.  All TLC
               pixels with a saturation less than sth will have a mask
               value of 0.0.  Defaults to 0.01.
    bcth ----- (O) Bubble convolution threshold.  Bubble convolution removes
               regions of high, contiguous intensity which are often bubbles
               or other artifacts.  0. <= bcth <= 22.  Smaller values result
               in smaller regions being identified as bubbles with TLC mask
               values set to 0.0.  A value of 22 will disable bubble removal.  
               Defaults to 15.
    hxrng ---- (O) Two element list specifying range of hue values to exclude 
               from the TLC mask.  Any pixel with a hue such that 
                   hxrng[0] <= hue <= hxrng[1]
               will have a TLC mask value of 0.0.  Defaults to None (which
               disables hue exclusion).
    hxbpft --- (O) Limiting fraction for hue exclusion.  If during hue 
               exclusion the number of pixels in a block having a hue 
               within hxrng exceeds hxbpft*blkpix (where blkpix is the 
               total number of pixels in the block), then the mask for the
               pixels in that block will not be set to 0.0 based on hue.  
               This parameter overides hue exclusion when it appears that
               the entire block is composed predominantly of pixels within
               the hue exclusion range.  Defaults to 0.5.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage

        imgchnls = carriage['imgchnls']

        ncam  = len(imgchnls)
        nfrm  = len(imgchnls[0])

        pivdict = self.m_config['pivdict']
        bsize   = pivdict['gp_bsize']
        rbndx   = pivdict['gp_rbndx']

        if ( self.m_config.has_key('prccam') ):
            prccam = self.m_config['prccam']
        else:
            prccam = ones(ncam,dtype=bool)

        if ( self.m_config.has_key('bgth') ):
            bgth = self.m_config['bgth']
        else:
            bgth = 0.01
            
        if ( self.m_config.has_key('sth') ):
            sth = self.m_config['sth']
        else:
            sth = 0.01
            
        if ( self.m_config.has_key('bcth') ):
            bcth = self.m_config['bcth']
        else:
            bcth = 15.
            
        if ( self.m_config.has_key('hxrng') ):
            hxrng = self.m_config['hxrng']
        else:
            hxrng = None
            
        if ( self.m_config.has_key('hxbpft') ):
            hxbpft = self.m_config['hxbpft']
        else:
            hxbpft = 0.5

        # Get the mask.
        tlcmask = []
        for c in range( ncam ):
            tlcmask.append([])
            for f in range( nfrm ):
                if ( prccam[c] ):
                    pmsk = tlclib.tlcmask(imgchnls[c][f],
                                          rbndx,
                                          bsize,
                                          bgth,
                                          sth,
                                          bcth,
                                          hxrng,
                                          hxbpft,
                                          False)
                else:
                    pmsk = None
                    
                tlcmask[c].append(pmsk)
                
        carriage['tlcmask'] = tlcmask
        
        
#################################################################
#
class sctfcomp(spivetstep):
    """
    STEP: Temperature field computation.

    Computes the temperature field for a particular z-plane based on 
    TLC hue using the images from a single camera.
    
    The z-axis cell size, zcellsz, will be multipled by the image plane
    number and added to tlcp0swz to compute the stationary world 
    z-coordinate for the imaged plane currently under consideration.  If 
    the TLC calibration has no z-dependence, then these concepts can be
    ignored.
    
    Both frames will be processed and the average temperature from the
    two planes will be stored.
    
    ---- carriage inputs ----
    imgchnls - (M) List of dewarped hue, saturation, and intensity channels
               for two or four images ordered according to:  
                   imgchnls[CAM][Frame][Channel]
               Only the hue channel, Channel 0,  from a single camera
               will be used to extract temperature.
    imgprms -- (M) List of image parameters sorted as:
                   imgprms[CAM][Frame]
               Must contain:
                   'S' ------ Plane number.
                   'padpix' - Number of pad pixels.
    pivdata -- (M) PIVData object to which the temperature field for the
               plane will be added.  If a valid PIVData object is not passed,
               one will be created.
    tlcmask -- (O) A list of TLC masks for each image in imgchnls.  Masks
               should be ordered as
                   tlcmask[CAM][Frame]
               Each tlcmask is a 2D array of the same size as the input
               image with valid particle pixels having a mask value of 
               1.0 and invalid particle pixels having mask values of 0.0.
               A convenience step, tlcmask, is provided to compute 
               appropriate masks.  Defaults to None.               
               
    ---- carriage outputs ----
    pivdata -- PIVData object augmented with the temperature field and 
               tfINAC flag values.  The scalar temperature values will 
               be stored with the variable name 'T', and the tfINAC
               values will be stored under 'TINAC'.  
               
               If no issues were encountered during hue to temperature
               conversion, the tfINAC value for the cell will be zero,
               otherwise it will be non-zero.  For more on the tfINAC flag,
               consult the help for spivet.tlclib.tfcomp().  A non-zero
               tfINAC flag will be multiplied by Frame*128.0.
               
               If a PIVData object is created, the PIVData world z-origin
               will be set to 0.0.  The world x,y-origin will be computed
               from 'gp_rbndx', 'padpix', and 'pg_wicsp' parameters.  Cell
               size will be determined from the pivdict Global parameters
               and zcellsz.  Otherwise, the output PIVData object origin
               and cellsz will be the same as the input.

               NOTE:  Any variables in the input PIVData object named
               T or TINAC will be overwritten.               

    ---- config dictionary contents ----
    pivdict -- (M) The PIV dictionary containing at least the Global,
               Photogrammetric, and Thermochromic parameters as defined 
               in the pivlib module documentation.  
               
               The 'tc_tlccal' parameter can also be a URL to the TLCCAL
               object.
    zcellsz -- (O) Cell size along the z-axis.  Should be in units of mm 
               if 3D vectors are computed, pixels otherwise.  Defaults to 
               1.0.  
               
               NOTE:  If an input PIVData obect is available on the 
               carriage, zcellsz will be ignored.
    tlcp0swz - (O) Coordinates for Plane 0 of the current dataset expressed
               in the stationary world coordinates (ie, fixed with respect
               to the laboratory) used during TLC calibration.  tlcp0swz can
               be ignored if TLC calibration has no depth dependence.  See 
               documentation for spivet.tlclib.tfcomp for more details.
               Defaults to 0.0.  Should be in units consistent with the 
               TLC calibration (generally mm).
    lbpath --- (O) Path to store TLCCAL object if necessary.  Only needed 
               if this object is retrieved.  Defaults to 'CALIBRATION'.               
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        import time

        # Initialization.
        carriage = self.m_carriage
        imgchnls = carriage['imgchnls']
        imgprms  = carriage['imgprms']

        ncam = len(imgchnls)
        nfrm = len(imgchnls[0])

        if ( carriage.has_key('tlcmask') ):
            tlcmask = carriage['tlcmask']
        else:
            tlcmask = None

        pivdict = self.m_config['pivdict']
        
        if ( self.m_config.has_key('lbpath') ):
            lbpath = self.m_config['lbpath']
        else:
            lbpath = 'CALIBRATION'         

        chkpivdict(pivdict,lbpath)
        
        tlccam = pivdict['tc_tlccam']
        
        if ( self.m_config.has_key('tlcp0swz') ):
            tlcp0swz = self.m_config['tlcp0swz']
        else:
            tlcp0swz = 0.0
 
        # Setup the PIVEpoch and PIVData objects.
        if ( carriage.has_key('pivdata') ):
            pd      = carriage['pivdata']
            zcellsz = pd.cellsz[0]
        else:
            if ( self.m_config.has_key('zcellsz') ):
                zcellsz = self.m_config['zcellsz']
            else:
                zcellsz = 1.0        

            pe  = pivlib.PIVEpoch(0)
            pad = imgprms[0][0]['padpix']
            
            [cellsz,origin] = csznorgn(zcellsz,pivdict,pad)        
        
            pd = pivlib.PIVData(cellsz,origin,"PIVDATA")
            pd.append(pe)
                    
            carriage['pivdata'] = pd
                    
        # Compute the stationary world z-coordinate.
        swz = zcellsz*imgprms[tlccam][0]['S'] +tlcp0swz
        
        for f in range( nfrm ):
            if ( compat.checkNone(tlcmask) ):
                pmsk = None
            else:
                pmsk = tlcmask[tlccam][f]
            
            [tf,tfINAC] = tlclib.tfcomp(imgchnls[tlccam][f][0],
                                        swz,
                                        pivdict,
                                        pmsk)
            
            if ( f == 0 ):
                atf     = tf.copy()
                atfINAC = tfINAC.copy()
            else:
                atf     = atf +tf
                atfINAC = atfINAC +f*128.*tfINAC
                
        atf = atf/float(nfrm)
        
        atf.setAttr(tf.name,tf.units)
        atfINAC.setAttr(tfINAC.name,tfINAC.units)
        pd[0].addVars([atf,atfINAC])


#################################################################
#
class loop_plane(spivetstep):
    """
    STEP: Loop over planes.

    Reads the contents of a single EFRMLST file from the bldsched
    step and performs the steps specified in the config dictionary.
    
    NOTE: The loop_plane step expects the substeps to produce a
    valid PIVData object.  The loop_plane step will then gather
    the results from each plane PIVData and store them in a single 
    PIVData object valid for the Epoch as a whole.

    ---- carriage inputs ----
    lfpath --- (M) Local path to EFRMLST file.  In lieu of passing
               a file path, the user can read in the file elsewhere
               and simply pass in the contents as a list of lines 
               (ie, from a call to readlines() on the file object).

    ---- carriage outputs ----
    epivdata - PIVData object containing the aggregate results for 
               the Epoch.

    ---- config dictionary contents ----
    steps ---- (M) List of rank 2 containing the step class names,
               fully qualified package name, and the config dictionaries 
               for each step.  For example:
                   [ ['step1','pkg.subpkg.module',config1], 
                     ['step2','pkg.subpkg.module',config2], ...]
                     
               A local module containing user-defined steps would have
               a fully qualified package name equal to the module file
               name minus the '.py'.  For example, suppose a module file
               usersteps.py contains the step procdata.  The correct 
               entry in the steps list would be
                   ['procdata','usersteps',myconfig] 
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        lfpath   = carriage['lfpath']

        steps = self.m_config['steps']
	#Xiyuan:print steps
	#print "stepsplane:",steps
	
        # Create the step objects and store the config dictionaries.
        steplst = []
        for step in steps:
            cname   = step[0]
            modname = step[1]
            config  = step[2]

            __import__(modname)
            module = sys.modules[modname]

            cobj = module.__dict__[cname]()
            cobj.setConfig(config)

            steplst.append(cobj)

        # Parse the EFRMLST file.
        if ( isinstance( lfpath, list ) ):
            epc     = int( lfpath[0].strip() )
            ncams   = int( lfpath[1].strip() ) 
            nfrms   = int( lfpath[2].strip() )
            nplns   = int( lfpath[3].strip() )
            bimgurl = lfpath[4].strip()
            frmprog = lfpath[5].strip()

            flines = lfpath[6::]
        else:
            ffh      = open(lfpath,'r')
            epc      = int(ffh.readline().strip())
            ncams    = int(ffh.readline().strip())
            nfrms    = int(ffh.readline().strip())
            nplns    = int(ffh.readline().strip())
            bimgurl  = ffh.readline().strip()
            frmprog  = ffh.readline().strip()

            flines = ffh.readlines()
            ffh.close()

        if ( nfrms > 2 ):
            raise ValueError("EFRMLST file can only support two frames.")

        # Get Epoch time.
        prms = flines[0].strip().split(' ')
        tsv  = float( prms[4] )/1000.

        print "Epoch time: %f" % tsv

        # Initialize the Epoch.
        pe = pivlib.PIVEpoch(tsv,"SEC")

        # Process the images.
        cline = 0
        for s in range( nplns ):
            prms  = flines[cline].strip().split(' ')
            cline = cline +1

            for i in range( len(prms) -1 ):
                prms[i] = "%s/%s" % (bimgurl,prms[i])            

            carriage['imgurls'] = prms[0:-1]

	    ii=0

            for step in steplst:
                step.setCarriage(carriage)
                step.execute()
		#Xiyuan
		#print "finished step:", steps[ii][0] 
		ii=ii+1
	#Xiyuan:skip the following part to get gcrg
'''	
            tpd    = carriage['pivdata']
            varnms = tpd[0].keys()
            if ( s == 0 ):
                epd = pivlib.PIVData(tpd.cellsz,tpd.origin,tpd.desc)

                evars = []
                for vn in varnms:
                    tvar = tpd[0][vn]
                    evar = pivlib.PIVVar([tvar.shape[0],
                                          nplns,
                                          tvar.shape[2],
                                          tvar.shape[3]],
                                          tvar.name,
                                          tvar.units,
                                          vtype=tvar.vtype)
                    evars.append(evar)

                pe.addVars(evars)
                epd.append(pe)

            for vn in varnms:
                pe[vn][:,s,:,:] = tpd[0][vn][:,0,:,:]

        carriage['epivdata'] = epd
'''

#################################################################
#
class medfltr(spivetstep):
    """
    STEP: Median filter PIVData object.

    Applies a median filter to the contents of a PIVData object.
    All Epochs in the PIVData object will be processed.

    Five parameters control the behavior of the medfltr step: mfdim,
    rthsf, reps, planar, and mfits.  For more information on the effects 
    of the configuration parameters, see the documentation on 
    pivlib.medfltr().
         
    ---- carriage inputs ----
    *pivdata - (M) PIVData object containing the flow results for the
               Epoch.  

    ---- carriage outputs ----
    *pivdata - The input PIVData object with new filtered variables 
               appended.  Filtered variable names will be appended with
               '-MF', (eg, U-MF).

               An additional variable indicating which cells have been
               filtered will also be added.  This filter flag variable will
               be named the same as the unfiltered variable with '-MFFLG'
               appended (eg, U-MFFLG).

    ---- config dictionary contents ----
    varnm ---- (M) Name of variable or list of names to filter.  
    mfdim ---- (O) Median filter size.  Should be an odd integer. Defaults
               to 5.
    rthsf ---- (O) Median threshold scale factor.  Defaults to 2.0.
    reps ----- (O) Residual epsilon.  Defaults to 0.0. 
    planar --- (O) Boolean flag indicating whether a planar filter should
               be applied or a 3D filter should be applied.  Defaults to
               False.
    mfits ---- (O) Median filter iterations.  Defaults to 3.
    cndx ----- (O) Component of variable to filter.  Set to None to filter
               all components, otherwise set to the integer index of the
               desired component.  Defaults to None. 
    pdname --- (O) Name of the PIVData object in the carriage to convert.
               Eg, 'epivdata'.  Defaults to 'pivdata'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'pivdata'

        pd = carriage[pdname]

        varnm = self.m_config['varnm']
        if ( not isinstance(varnm,list) ):
            varnm = [ varnm ]

        if ( self.m_config.has_key('mfdim') ):
            mfdim = self.m_config['mfdim']
        else:
            mfdim = 5

        if ( self.m_config.has_key('rthsf') ):
            rthsf = self.m_config['rthsf']
        else:
            rthsf = 2.
            
        if ( self.m_config.has_key('reps') ):
            reps = self.m_config['reps']
        else:
            reps = 0.

        if ( self.m_config.has_key('planar') ):
            planar = self.m_config['planar']
        else:
            planar = False

        if ( self.m_config.has_key('mfits') ):
            mfits = self.m_config['mfits']
        else:
            mfits = 3
            
        if ( self.m_config.has_key('cndx') ):
            cndx = self.m_config['cndx']
        else:
            cndx = None
        
        # Apply the median filter.
        for e in range( len(pd) ):
            for vn in varnm:
                [fv,fltrd] = pivlib.medfltr(pd[e][vn],
                                            mfdim,rthsf,reps,planar,
                                            nit=mfits,
                                            cndx=cndx)

                nfltrd = fltrd > 0
                nfltrd = nfltrd.sum()

                print "MEDIAN FILTERED E%i (%s): %i" % (e,vn,nfltrd)
                print "MEDIAN FILTERED E%i (%s): %f%%" % \
                    (e,vn,(100.*nfltrd)/fltrd.size)

                pd[e].addVars([fv,fltrd])
                
                
#################################################################
#
class divfltr(spivetstep):
    """
    STEP: Filter PIVData object using a zero divergence criteria.

    Applies a zero divergence filter to the contents of a PIVData 
    object.  All Epochs in the PIVData object will be processed.
    
    For more information on the effects of the configuration parameters,
    see the documentation on pivlib.divfltr().

    ---- carriage inputs ----
    *pivdata - (M) PIVData object containing the flow results for the
               Epoch.  

    ---- carriage outputs ----
    *pivdata - The input PIVData object with new filtered variables 
               appended.  Filtered variable names will be appended with
               '-DF', (eg, U-DF).

               An additional variable indicating which cells have been
               filtered will also be added.  This filter flag variable will
               be named the same as the unfiltered variable with '-DFFLG'
               appended (eg, U-DFFLG).

    ---- config dictionary contents ----
    varnm ---- (M) Name of variable or list of names to filter.  Each
               variable must be a 3 component vector.
    divp ----- (O) Divergence percentile for filtering.  Any cell with
               a divergence exceeding the divp percentile will be flagged
               as having excessive divergence. Defaults to 99.
    planar --- (O) Boolean flag indicating whether a planar filter should
               be applied or a 3D filter should be applied.  See 
               documentation on pivlib.divfltr() for more details.  Defaults
               to False.
    mfits ---- (O) Maximum number of filter iterations.  Defaults to 100.
    pdname --- (O) Name of the PIVData object in the carriage to convert.
               Eg, 'epivdata'.  Defaults to 'pivdata'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'pivdata'

        pd     = carriage[pdname]
        cellsz = pd.cellsz

        varnm = self.m_config['varnm']
        if ( not isinstance(varnm,list) ):
            varnm = [ varnm ]

        if ( self.m_config.has_key('divp') ):
            divp = self.m_config['divp']
        else:
            divp = 99

        if ( self.m_config.has_key('planar') ):
            planar = self.m_config['planar']
        else:
            planar = False

        if ( self.m_config.has_key('maxits') ):
            maxits = self.m_config['maxits']
        else:
            maxits = 100
        
        # Apply the filter.
        for e in range( len(pd) ):
            for vn in varnm:
                [fv,fltrd] = pivlib.divfltr(pd[e][vn],cellsz,
                                            divp,planar,maxits)

                nfltrd = fltrd > 0
                nfltrd = nfltrd.sum()

                print "DIVERGENCE FILTERED E%i (%s): %i" % (e,vn,nfltrd)
                print "DIVERGENCE FILTERED E%i (%s): %f%%" % \
                    (e,vn,(100.*nfltrd)/fltrd.size)

                pd[e].addVars([fv,fltrd])                
                
                
#################################################################
#
class gsmooth(spivetstep):
    """
    STEP: Gaussian smooth PIVData object.

    Applies a symmetric Gaussian smoothing filter to the contents of a 
    PIVData object.  All Epochs in the PIVData object will be processed.
    
    For more information on the effects of the configuration parameters,
    see the documentation on pivlib.gsmooth().

    ---- carriage inputs ----
    *pivdata - (M) PIVData object containing the flow results for the
               Epoch.  

    ---- carriage outputs ----
    *pivdata - The input PIVData object with new filtered variables 
               appended.  Filtered variable names will be appended with
               '-GS', (eg, T-GS).

    ---- config dictionary contents ----
    varnm ---- (M) Name of variable or list of names to filter.  
    gbsd ----- (O) Standard deviation, in units of cells (ie, not mm),
               for the Gaussian smoothing kernel.  Default is 0.5. 
    planar --- (O) Boolean flag indicating whether a planar filter should
               be applied or a 3D filter should be applied.  Defaults to
               False.
    mfits ---- (O) Filter iterations.  Defaults to 1.
    cndx ----- (O) Component of variable to filter.  Set to None to filter
               all components, otherwise set to the integer index of the
               desired component.  Defaults to None. 
    pdname --- (O) Name of the PIVData object in the carriage to convert.
               Eg, 'epivdata'.  Defaults to 'pivdata'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'pivdata'

        pd = carriage[pdname]

        varnm = self.m_config['varnm']
        if ( not isinstance(varnm,list) ):
            varnm = [ varnm ]

        if ( self.m_config.has_key('gbsd') ):
            gbsd = self.m_config['gbsd']
        else:
            gbsd = 0.5

        if ( self.m_config.has_key('planar') ):
            planar = self.m_config['planar']
        else:
            planar = False

        if ( self.m_config.has_key('mfits') ):
            mfits = self.m_config['mfits']
        else:
            mfits = 1
            
        if ( self.m_config.has_key('cndx') ):
            cndx = self.m_config['cndx']
        else:
            cndx = None            
        
        # Apply the gaussian filter.
        for e in range( len(pd) ):
            for vn in varnm:
                fv = pivlib.gsmooth(pd[e][vn],gbsd,planar,nit=mfits,cndx=cndx)

                pd[e].addVars(fv)


#################################################################
#
class dltae0(spivetstep):
    """
    STEP: Computes the change in a variable from its value at 
    Epoch 0.  In a thermal convection experiment, for example, the
    excess temperature above ambient is what drives the flow.
    Assuming the fluid temperature was uniform at the start of the 
    experiment (Epoch 0), the dltae0 step would then subtract the 
    Epoch 0 temperature from all other Epochs.
      
    All Epochs will be processed.

    ---- carriage inputs ----
    *pivdata - (M) Input PIVData object containing the variable.
               
    ---- carriage ouptuts ----
    *pivdata - Input PIVData object with DLTAE0-VAR variable added.
    
    ---- config dictionary contents ----
    varnm ---- (M) Name of variable or list of names to process.  
    pdname --- (O) Name of the PIVData object in the carriage to convert.
               Defaults to 'dpivdata'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)
        
    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        
        varnm = self.m_config['varnm']
        if ( not isinstance(varnm,list) ):
            varnm = [varnm]
        
        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'dpivdata'
        
        pd = carriage[pdname]

        # Compute the delta.
        for vn in varnm:        
            dvn   = "DLTAE0-%s" % vn
            var0  = pd[0][vn]
            units = var0.units
            vtype = var0.vtype
            for e in range(0,len(pd)):
                var  = pd[e][vn]

                dvar = var -var0
                dvar.setAttr(dvn,units,vtype)
                
                pd[e].addVars(dvar)



#################################################################
# Helper functions for the eqtime and synchronize steps.  These
# functions are wrappers around interpolation function that enable
# the steps to call using map().
#
# These function could be included in a generic interpolating step
# class that eqtime and synchronize are then derived from.
def _splrep(x,y,s):
    """ 
    ----
    
    x              # splrep() x argument.
    y              # splrep() y argument.
    s              # splrep() s argument (the smoothness). 
    
    ----
    
    Helper function for synchronize step.  _splrep() is a wrapper
    around Scipy's splrep() function that the synchronize step can
    call using map().
    
    Returns spl, the spline object. 
    """
    spl = interpolate.splrep(x,y,s=s)
    return spl

def _splev(x,spl):
    """ 
    ----
    
    x              # splev() x argument.
    spl            # splev() tck argument (the spline object).
    
    ----
    
    Helper function for synchronize step.  _splev() is a wrapper
    around Scipy's splev() function that the synchronize step can
    call using map().
    
    Returns val, the interpolated value. 
    """
    val = interpolate.splev(x,spl)
    return val

                

#################################################################
#
class eqtime(spivetstep):
    """
    STEP: Equipartitions total elapsed time between Epochs.  
    While holding the time of the first and last Epochs fixed, each 
    variable in varnm is temporally interpolated using cubic splines 
    such that the remaining Epochs within the PIVDATA object are 
    equally spaced in time.  The primary motivation for this step is 
    to correct for jitter in elapsed time between Epochs. 

    The eqtime step should only be run after all Epochs are available 
    in the PIVData object since data will be splined across those 
    Epochs.

    ---- carriage inputs ----
    *pivdata - (M) Input PIVData object containing the variable(s) to
               be temporally adjusted.  
                                 
    ---- carriage ouptuts ----
    *pivdata - A new PIVData object containing only the adjusted 
               variables.  Adjusted variable names will be the
               same as input.  The new PIVData object on the carriage
               will be named 
                   "%s-eq" % pdname
    
    ---- config dictionary contents ----
    varnm ---- (M) Name of variable or list of variables to adjust.    
    pdname --- (O) Name of the PIVData object in the carriage containing
               the variables.  Defaults to 'dpivdata'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)
        
    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        
        varnm = self.m_config['varnm']
        if ( not isinstance(varnm,list) ):
            varnm = [ varnm ]

        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'dpivdata'
        
        pd   = carriage[pdname]        
        nepc = len(pd)
        if ( nepc < 4 ):
            raise ValueError("PIVData must contain at least 4 Epochs")

        # Setup the new PIVData object.
        npdname = "%s-eq" % pdname
        npd     = pivlib.PIVData(pd.cellsz,pd.origin)

        # Get Epoch times and equipartition.
        times  = array(pd.getTimes())
        dt     = (times[-1] -times[0])/(nepc -1)
        qtimes = dt*arange(nepc,dtype=float) +times[0]
        
        ncells  = array(pd[0].eshape)
        tncells = ncells.prod()

        atimes = times.reshape([1,nepc])
        atimes = atimes.repeat(tncells,0)
        
        aqtimes = qtimes.reshape([1,nepc])
        aqtimes = aqtimes.repeat(tncells,0)
        
        # Create the smoothing vector.  Set svec to zero for all cells
        # so that splrep does interpolation.
        svec = zeros(tncells)

        # Adjust.
        for vn in varnm:
            var = []
            for e in pd:
                var.append(e[vn])
            var  = array(var)
            avar = empty(var.shape,dtype=float)
            
            # var has shape [nepochs,ncmp,nz,ny,nx]
            ncmp = var.shape[1]
            for c in xrange(ncmp):
                cmp = var[:,c,:,...]
                cmp = cmp.reshape((nepc,tncells)).transpose()
                
                spl = map(_splrep,atimes,cmp,svec)
                for e in xrange(nepc):
                    acmp = array( map(_splev,aqtimes[:,e],spl) )
                    avar[e,c,...] = acmp.reshape((ncells[0],ncells[1],ncells[2]))
            
            for e in xrange(nepc):
                pivvar = pivlib.PIVVar((ncmp,ncells[0],ncells[1],ncells[2]),
                                       vn,
                                       pd[e][vn].units,
                                       dtype=float,vtype=pd[e][vn].vtype,
                                       data=avar[e,...])
                npd.addVars(e,pivvar)
                
        # Store the results.
        npd.setTimes(qtimes)
        carriage[npdname] = npd


#################################################################
#
class synchronize(spivetstep):
    """
    STEP: Interpolates a variable or list of variables in time using 
    splines and adjusts the variables as though the planes were all
    taken simultaneously.  This step is primarily aimed at eliminating
    the time difference between acquisition separate z-planes that
    is characteristic of scanning-type setups similar to the University
    of Michigan system.

    The variables will be synchronized by evolving planar data forward
    in time.  That is, planes acquired earlier will be evolved as though
    they were taken later.  This approach is meant to leverage existing
    data instead of extrapolating backward in time at the start of the
    experiment.  As a result, however, the final PIVData object will
    contain one less Epoch (the last Epoch will be discared).

    Epoch time values will also be updated.

    The synchronize step should only be run after all Epochs are
    available.

    ---- carriage inputs ----
    *pivdata - (M) Input PIVData object containing the variable(s) to
               be synchronized.  PIVData object must contain the PLNR-TIME
               variable (created with recordtime step).  
               
               NOTE: All variables being synchronized must have the same 
               vtype as PLNR-TIME.  No checks are performed.
               
    ---- carriage ouptuts ----
    *pivdata - A new PIVData object with synchronized variables appended.
               Synchronized variable names will be appended with '-SN'.
               The new PIVData object will replace the old pivdata
               object named pdname on the carriage.
    
    ---- config dictionary contents ----
    varnm ---- (M) Name of variable or list of variables to synchronize.    
    pdname --- (O) Name of the PIVData object in the carriage containing
               the variables.  Defaults to 'dpivdata'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)
        
    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        
        varnm = self.m_config['varnm']
        if ( not isinstance(varnm,list) ):
            varnm = [ varnm ]

        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'dpivdata'
        
        pd   = carriage[pdname]        
        nepc = len(pd)
        if ( nepc < 4 ):
            raise ValueError("PIVData must contain at least 4 Epochs")

        # Extract the PLNR-TIME variable and adjust it.
        plnrt = []
        for e in pd:
            plnrt.append(e['PLNR-TIME'])
        plnrt = array(plnrt)
        
        ncells   = plnrt.shape[2::]
        tnpcells = ncells[1]*ncells[2]   # Planar cells.
        tncells  = ncells[0]*tnpcells    # All cells.
        
        aplnrt = empty(plnrt.shape,dtype=float)
        for e in range(nepc):
            aplnrt[e,...] = plnrt[e,...].max()

        plnrt  = plnrt.reshape((nepc,tncells)).transpose()
        aplnrt = aplnrt.reshape((nepc,tncells)).transpose()

        # Create the smoothing vector.  Set svec to zero for all cells
        # so that splrep does interpolation.
        svec = zeros(tncells)

        # Synchronize.
        for vn in varnm:
            var = []
            for e in pd:
                var.append(e[vn])
            var  = array(var)
            avar = empty(var.shape,dtype=float)
            
            # var has shape [nepochs,ncmp,nz,ny,nx]
            ncmp = var.shape[1]
            for c in range(ncmp):
                cmp = var[:,c,:,...]
                cmp = cmp.reshape((nepc,tncells)).transpose()
                
                spl = map(_splrep,plnrt,cmp,svec)
                for e in range(nepc -1):
                    acmp = array( map(_splev,aplnrt[:,e],spl) )
                    avar[e,c,...] = acmp.reshape((ncells[0],ncells[1],ncells[2]))
            
            for e in range(nepc -1):
                pivvar = pivlib.PIVVar((ncmp,ncells[0],ncells[1],ncells[2]),
                                       "%s-SN" % vn,
                                       pd[e][vn].units,
                                       dtype=float,vtype=pd[e][vn].vtype,
                                       data=avar[e,...])
                pd[e].addVars(pivvar)
                
        # Store the results.
        pd.setTimes(aplnrt)
        pd = pd[0:(nepc -1)]
        carriage[pdname] = pd
        
        
#################################################################
#
class prune(spivetstep):
    """
    STEP: Prunes a variable, or list of variables, from a PIVData
    object.  This step simply removes unused variables from
    the PIVData object.
    
    All Epochs will be processed.

    ---- carriage inputs ----
    *pivdata - (M) PIVData object for variable pruning.

    ---- carriage outputs ----
    *pivdata - The input PIVData object with variables pruned.

    ---- config dictionary contents ----
    varnm ---- (M) Name of variable or list of names to delete from the
               PIVData object.  
    pdname --- (O) Name of the PIVData object in the carriage to convert.
               Eg, 'epivdata'.  Defaults to 'pivdata'.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'pivdata'

        pd = carriage[pdname]

        varnm = self.m_config['varnm']
        if ( not isinstance(varnm,list) ):
            varnm = [ varnm ]
        
        # Prune.
        for e in range( len(pd) ):
            for vn in varnm:
                pd[e].pop(vn)
                

#################################################################
#
class assemble(spivetstep):
    """
    STEP: Assemble individual Epochs.

    This step is generally not needed by the user unless the user
    wants to manually assemble a group of individual Epochs into a
    large, aggregate PIVData file.  The individual files can contain
    any number of Epochs, however, the full suite of Epochs must have
    monotonically increasing time.  The aggregate PIVData object created
    will consist of the individual Epochs ordered by increasing time.
    
    ---- carriage inputs ----
    asybpath - (M) Base path or list of base paths containing the 
               ExodusII files to be assembled.  

    ---- carriage outputs ----
    *pivdata - The aggregate PIVData object.  Will be stored as 
              'apivdata' on the carriage unless pdname is set.

    ---- config dictionary contents ----
    pdname --- (O) Name to use when storing aggregate PIVData data 
               object on the carriage.  Defaults to apivdata.
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        asybpath = carriage['asybpath']
       
        if ( self.m_config.has_key('pdname') ):
            pdname = self.m_config['pdname']
        else:
            pdname = 'apivdata' 
        
        if ( not isinstance(asybpath,list) ):
            asybpath = [asybpath]

        # First trip through files to find out how many Epochs need
        # to be processed.
        vdl = []
        ept = []
        for bpth in asybpath:
            dl = os.listdir(bpth)
            for f in dl:
                if ( not f.startswith('PIVDATA') ):
                    continue
                else:
                    pth = "%s/%s" % (bpth,f)
                    pd  = pivlib.loadpivdata(pth)
                    tms = pd.getTimes()
                    del pd
                    
                    for t in tms:
                        vdl.append(pth)
                        ept.append(t)
                        
        # Assemble the results.
        ept  = array(ept)
        ordr = ept.argsort()
        
        cdl = ""
        for i in range( ordr.size ):
            pth = vdl[ ordr[i] ]
            if ( pth != cdl ):
                pd  = pivlib.loadpivdata( pth )
                cdl = pth
                cnt = 0
            else:
                cnt = cnt +1

            if ( i == 0 ):
                apd = pivlib.PIVData(pd.cellsz,pd.origin,pd.desc)

            print "Appending ( %s )( E%i )" % (pth,cnt)            
            apd.append(pd[cnt])
        
        carriage[pdname] = apd


#################################################################
#
# loop_epoch helpers.
#
# These functions are run on the remote machine.  As such they
# apparently need to be in the namespace of the module, otherwise
# access to SleighUserNws and friends won't work.
#
def _loop_epoch_setupgws(steps,wrkrstepsvn,nwsmodvn,wrkrtdirvn):
    """
    Sets up global workspace environment on each worker node.
    """
    import tempfile, tarfile

    # Initialization.
    ltarfn = "tarmod.tgz"
    
    # Bring in the steps and store them in the global namespace.
    globals()[wrkrstepsvn] = steps

    # Create a temporary directory.
    tdir = tempfile.mkdtemp(dir="./")
    globals()[wrkrtdirvn] = tdir
    os.chdir(tdir)

    # Download the tar file and extract the contents.
    try:
        fh = open(ltarfn,'wb')
        SleighUserNws.findFile(nwsmodvn,fh)
        fh.close()
    except:
        # Must be running locally.  Need to implement means to
        # handle user steps.
        fh.close()

    try:
        tar = tarfile.open(ltarfn)
        tar.extractall()
        tar.close()
    except:
        print "Extracting tar file failed."


def _loop_epoch_cleanupgws(wrkrtdirvn):
    """
    Cleans up the worker global workspace environment.
    """
    import shutil
    os.chdir("../")
    shutil.rmtree(globals()[wrkrtdirvn])


def _loop_epoch_worker(carriage,wrkrstepsvn):
    """
    Performs the loop work. sflg will be set if an exception is thrown.
    If an exception is thrown, then sepdvn or epivdata will be set
    to the traceback text.
    
    If running in parallel, stores a PIVDATA object file directly
    to the NWS.  Returns [sflg,sepdvn,stdout,stderr] where
        sflg ------ Success flag.  0 if successfull, non-zero otherwise.
        sepdvn ---- The variable name on the NWS where the PIVDATA
                    is stored.
        stdout ---- The stdout transcript.
        stderr ---- The stderr transcript.
        
    If running locally, returns [sflg,epivdata,stdout,stderr] where
        sflg ------ Success flag.  0 if successfull, non-zero otherwise.
        epivdata -- A valid PIVData object.
        stdout ---- The stdout transcript.
        stderr ---- The stderr transcript.        
    """
    import socket, traceback

    # Initialization.
    epdfn = "PIVDATA.ex2"
    #Xiyuan: commented 
    #lsofn      = "stdout"
    #sys.stdout = open(lsofn,'w', 0)

    #lsefn      = "stderr"
    #sys.stderr = open(lsefn,'w', 0)

    rhost = socket.gethostname()
    print "Running on: %s" % rhost
    
    steps = globals()[wrkrstepsvn] 
    #	Xiyuan:print steps
    #print "stepsepoch:",steps


    try:
        rnk = SleighRank
        parallel = True
        print "Running remotely."
    except:
        print "Running locally."
        parallel = False
    
    try:        
        if ( parallel ):
            svnext = "%i-%i" % (SleighRank,random.randint(0,1000000))
    
        # Create the step objects and store the config dictionaries.
        steplst = []
        for step in steps:
            cname   = step[0]
            modname = step[1]
            config  = step[2]
    
            __import__(modname)
            module = sys.modules[modname]
    
            cobj = module.__dict__[cname]()
            cobj.setConfig(config)
    
            steplst.append(cobj)
    
        # Execute the steps.
        for step in steplst:
            step.setCarriage(carriage)
            step.execute()
   	    #Xiyuan:save current plane last raw carriage
	    gcrg = carriage
	    print "epochcrg"
 
        epd = carriage['epivdata']
        if ( parallel ):
            # Write the PIVDATA object to a file and then load in directly
            # into the network space.  This is necessary to prevent Python 
            # ASCII serialization of the PIVDATA object (which can cause the 
            # process to run out of memory).
            epd.save(epdfn)
        
            sepdvn = "PIVDATA-%s-%s" % (rhost,svnext) 
            SleighUserNws.declare(sepdvn,'single')
            fh = open(epdfn,'rb')
            SleighUserNws.storeFile(sepdvn,fh)
            fh.close()
        
        sflg = 0        
        if ( parallel ):
            pdo = sepdvn
        else:
            pdo = epd

    except KeyboardInterrupt:
        # Need to handle Ctrl-C explicitly and die.
        raise 
            
    except:
        sflg = 1
        traceback.print_exc(file=sys.stderr)
        pdo  = traceback.format_exc()
       
    # Close out the local logs.
    #sys.stdout.close()
    #sys.stdout = sys.__stdout__
    
    #sys.stderr.close()
    #sys.stderr = sys.__stderr__
    
    # Load the log files.
    #fh      = open(lsofn,'r')
    #stdoutf = fh.read()
    #fh.close()

    #fh      = open(lsefn,'r')
    #stderrf = fh.read()
    #fh.close()
    
    #return [sflg,pdo,stdoutf,stderrf]
    
    #Xiyuan: uncomment this to assgin gcrg
    #gcrg=0
    return [sflg,pdo,gcrg]

#################################################################
#
class loop_epoch(spivetstep):
    """
    STEP: Loop over Epochs.

    Reads the contents of the lbfpath directory where the EFRMLST
    files are stored and performs the steps specified in the 
    config dictionary.
    
    If NetworkSpaces is available and properly configured, then
    loop_epoch will launch workers on remote nodes.  Each worker
    will process one Epoch.  The config option, parallel, will
    override the use of NetworkSpaces.  A word of caution:  A
    persistent workspace will be established on each remote worker
    that is reused for all Epochs processed on that node.  Care
    must be taken in substeps to ensure that memory leaks are not
    exposed (eg, via calls to pylab.figure() without closing the
    figure).

    loop_epoch will inject into the carriage a list of lines from each 
    EFRMLST file contained in lbfpath.  Therefore, at least one of the 
    config steps (eg, loop_plane) must act upon the contents of the 
    EFRMLST file.  The EFRMLST lines will be stored on the carriage under 
    'lfpath'.
    
    NOTE: The EFRMLST files must be named using the same format as
    produced by the bldsched step.  Namely:
  
        EFRMLST-Ex-F0_y

    where:
        x ---- Epoch number.  0 <= x <= x_max.  Format: %i
        y ---- Upper frame number.  Generally 1.  Format: %i
    
    NOTE: The loop_epoch step expects the substeps to produce a
    valid PIVData object.  The loop_epoch step will then gather
    the results from each Epoch PIVData and, if the config parameter 
    assemble is True, store them in a single PIVData object valid for the 
    entire dataset as a whole.
    
    NOTE: Periodically, a problem will occur with the processing of
    a dataset.  This can be particularly problematic when running
    a parallel case, since a remote machine can drop a network connection
    or have other issues.  In such cases, loop_epoch tries to fail as 
    gracefully as possible.  If during execution an exception is thrown, 
    loop_epoch will catch the exception and try to continue operating.  
    However, the final results will not be assembled into an aggregate 
    PIVData obect.  Instead, the user should inspect the logs and correct 
    the problem.  The Epochs that did not get processed can then be fed 
    back into a new instance of the loop_epoch step with obpath set to 
    another directory (so that those Epochs that were successfully 
    processed don't get clobbered with the rerun).  Once the missing 
    Epochs are available, the user can execute the assemble step to 
    assemble the two groups of epochs into the desired aggregate PIVData 
    object.  

    ---- carriage inputs ----
    lbfpath -- (M) Local path to directory containing EFRMLST files.

    ---- carriage outputs ----
    lfpath --- Path to a particular EFRMLST file.  Refreshed during
               loop for each EFRMLST file in lbfpath.
    dpivdata - List of PIVData objects containing the aggregate results 
               for the dataset of each frame group.  Only placed on 
               carriage before returning from loop_epoch.execute().  If
               only one frame group is used (ie, Frames 0 and 1), then
               the list will contain one entry.  Otherwise, the list
               will contain an entry for each frame group (eg, Frame 0
               to 1, Frame 0 to 2, etc.).  See documentation on the
               bldsched step for more information on how multiple frames
               are handled.  Note that dpivdata will be an empty list if
               an exception is thrown (described above) or the config
               parameter assemble is False.

    ---- config dictionary contents ----
    steps ---- (M) List of rank 2 containing the step class names,
               fully qualified package name, and the config dictionaries 
               for each step.  For example:
                   [ ['step1','pkg.subpkg.module',config1], 
                     ['step2','pkg.subpkg.module',config2], ...]
                     
               A local module containing user-defined steps would have
               a fully qualified package name equal to the module file
               name minus the '.py'.  For example, suppose a module file
               usersteps.py contains the step procdata.  The correct 
               entry in the steps list would be
                   ['procdata','usersteps',myconfig]                 
    parallel - (O) Boolean flag indicating whether the loop_epoch step 
               should be run in parallel.  Flag will be ignored if 
               Network spaces is not installed and configured.  Defaults 
               to True.
    obpath --- (O) Base path in which to store output files.  During a 
               parallel run, stdout and stderr are redirected to files.
               These files will be stored in obpath.  Defaults to OUTPUT.
    assemble - (O) Flag indicating whether results should be assembled
               into one aggregate PIVData object that will be placed
               on the carriage under the name dpivdata as described
               above.  If assemble is False, the results won't be 
               aggregated.  NOTE: The assemble flag will be overridden and
               set to False if an error occurs during execution.  
               Defaults to True.  
    """
    def __init__(self,carriage={},config={}):
        spivetstep.__init__(self,carriage,config)

        self.m_nwsmodvn  = "tar_mod_payload" 

        self.m_wrkrstepsvn = "spivet_wrkr_steps"
        self.m_wrkrtdirvn  = "spivet_wrkr_tdir"
        
        self.m_accobpath = ""    # Output path used by accumulator.
        
        self.m_sleigh = None 
        
        self.m_assemble = True   # False when assembling should be skipped.
	#Xiyuan: add temporary carriage external port
	self.m_crg = {}
        
    def __accumulator__(self,acc):
        """
        This function stores results from eachElem() as they arrive.
        NOTE: The NWS documentation (for 1.6.3) is incorrect.  The
        first argument to the accumulator is the index and the second
        is the result.
        """
        # Initialization.
        sl = self.m_sleigh
        if ( compat.checkNone(sl) ):
            parallel = False
        else:
            parallel = True
        
        obpath = self.m_accobpath
        
        # Process the results.  The result is expected to be a list as
        # returned by _loop_epoch_worker().
        for a in acc:
            tn = a[0]
            r  = a[1]

            if ( r[0] > 0 ):
                # Had an exception.
                print "----- FAILED: E%i -----" % tn
                print r[1]
                print "-----"
                self.m_assemble = False
            else:
                oepdp = "%s/PIVDATA-E%i.ex2" % (obpath,tn)
                if ( parallel ):
                    print "ACCUMULATING: E%i <-- NWS:%s" % (tn,r[1])
                    fh = open(oepdp,'wb')
                    sl.userNws.fetchFile(r[1],fh)
                    fh.close()
                else:
                    print "ACCUMULATING: E%i" % tn                
                    r[1].save(oepdp)
            
            fh = open("%s/STDOUT-E%i" % (obpath,tn),'w')
            fh.write(r[2])
            fh.close()

            fh = open("%s/STDERR-E%i" % (obpath,tn),'w')
            fh.write(r[3])
            fh.close()           
        
    def __assemble__(self,accobpath):
        """
        Assembles the individual PIVDATA files that have been accumulated
        in accobpath into one PIVData object.  Each PIVDATA file is
        expected to hole a single Epoch.  Returns the object.
        """
        # Find out how many Epochs need to be processed.
        dl = os.listdir(accobpath)
        dl.sort()
        cp = 0
        for f in range( len(dl) ):
            if ( not dl[cp].startswith('PIVDATA') ):
                dl.pop(cp)
            else:
                cp = cp +1

        # Expects file names to have the form
        #     PIVDATA-E%i.ex2
        _cmprule = lambda x,y: cmp( int(x[0:-4].split('-')[1][1::]),
                                    int(y[0:-4].split('-')[1][1::]) ) 
        dl.sort(_cmprule)
        
        # Assemble the results.
        for i in range(len(dl)):
            pd = pivlib.loadpivdata( "%s/%s" % (accobpath,dl[i]) )
            
            if ( i == 0 ):
                epd = pivlib.PIVData(pd.cellsz,pd.origin,pd.desc)
            
            epd.append(pd[0])
        
        return epd

    def __tarmods__(self,sl):
        """
        Creates an archive of python modules in the current directory
        as well those in ~/.spivet/usersteps and uploads it to the
        NWS.  Contents stored under the variable named as in 
        m_nwsmodvn.

        Returns the path to the gzipped tar file.
        """
        import tarfile, tempfile

        # Create a temporary file.
        tfh = tempfile.mkstemp()
        os.close(tfh[0])

        tar = tarfile.open(tfh[1],'w:gz')
        
        # Get python files in current directory.
        try:
            dl = os.listdir('./')
            for f in dl:
                if ( f[-3::] == '.py' ):
                    tar.add("%s/%s" % ("./",f),recursive=False)
        except:
            print "Cannot get local directory listing."

        # Get python files in usersteps directory.
        usd = spivetconf.get_user_steps_dir()
        if ( not compat.checkNone(usd) ):
            dl = os.listdir(usd)
            for f in dl:
                if ( f[-3::] == '.py' ):
                    ufp = "%s/%s" % (usd,f)
                    ti  = tar.gettarinfo(ufp,f)
                    tar.addfile(ti,open(ufp,'rb'))                

        tar.close()

        # Upload the tar file to the NWS server.
        sl.userNws.declare(self.m_nwsmodvn,'single')
        fh = open(tfh[1],'rb')
        sl.userNws.storeFile(self.m_nwsmodvn,fh)
        fh.close()
        
        # Cleanup.
        os.remove(tfh[1])


    def execute(self):
        # Initialization.
        carriage = self.m_carriage
        lbfpath  = carriage['lbfpath']

        steps = self.m_config['steps']

        if ( self.m_config.has_key('parallel') ):
            parallel = self.m_config['parallel']
        else:
            parallel = True

        if ( self.m_config.has_key('obpath') ):
            obpath = self.m_config['obpath']
        else:
            obpath = "OUTPUT"
            
        if ( self.m_config.has_key('assemble') ):
            self.m_assemble = self.m_config['assemble']

        # Need to get obpath in absolute form.  If not run in parallel,
        # this machine is the sole worker.  As such, it will change
        # to and work in a temporary directory and won't come back out
        # until all Epochs are processed.
        if ( obpath[0] != '/' ):
            # Have relative path.
            obpath = "%s/%s" % (os.getcwd(),obpath)

        if ( not os.path.exists(obpath) ):
            os.mkdir(obpath)

        # Check if NWS is available and configured.
        if ( parallel ):
            try:
                from nws.sleigh import Sleigh, sshcmd
                #import imp

                nwsconf = spivetconf.get_nwsconf()
                if ( compat.checkNone(nwsconf) ):
                    raise ImportError("No nwsconf")

                launch = nwsconf.sleighLaunch
                if ( launch == 'sshcmd' ):
                    launch = sshcmd

                sl = Sleigh(launch=launch,
                            nodeList=nwsconf.nodeList,
                            nwsHost=nwsconf.nwsHost,
                            user=nwsconf.remoteUser,
                            workingDir=nwsconf.workingDir)
                
                self.m_sleigh = sl 
            except:
                print "Running in serial mode."
                parallel = False
        
        # Setup work schedule.
        dl = os.listdir(lbfpath)
        dl.sort()
        cp = 0
        for f in range( len(dl) ):
            if ( not dl[cp].startswith('EFRMLST') ):
                dl.pop(cp)
            else:
                cp = cp +1
                
        # Since the entries were sorted above, frmgrp's should be
        # encountered in order (ie, 3 should not precede 2).
        frmgrplst = [[]]
        mxfrmgrp  = 1
        for f in dl:
            frmgrp = int(f[-1])
            if ( frmgrp > mxfrmgrp ):
                frmgrplst.append([])
                mxfrmgrp = mxfrmgrp +1
            
            frmgrplst[frmgrp-1].append(f)

        # Expects EFRMLST file names to have the form
        #     EFRMLST-E%i-F0_%i
        _cmprule = lambda x,y: cmp( int(x.split('-')[1][1::]),
                                    int(y.split('-')[1][1::]) )        

        crggrplst = []
        for fgrp in frmgrplst:
            fgrp.sort(_cmprule)
            
            crglst = []
            for ef in fgrp:  
                # Each ef represents an EFRMLST.
                fh     = open("%s/%s" % (lbfpath,ef),'r')
                flines = fh.readlines()
                fh.close()
                
                # eachElem() expects a list of arguments, so we need
                # send each worker its own copy of the carriage.
                crg = carriage.copy()
                crg['lfpath'] = flines 
                crglst.append(crg)  

            crggrplst.append(crglst)
            
        # Setup for parallel work.
        wargs = [steps,
                 self.m_wrkrstepsvn,
                 self.m_nwsmodvn,
                 self.m_wrkrtdirvn]

        if ( parallel ):
            self.__tarmods__(sl)

            rslt = sl.eachWorker(_loop_epoch_setupgws,*wargs)
            if ( not compat.checkNone(rslt[0]) ):
                print "Global workspace setup may have failed ..."
                print rslt
                print
        else:
            _loop_epoch_setupgws(*wargs)

        # Run the loop.
        fgrp = 0
        dpdl = []
        for grp in crggrplst:
            # Setup accumulation directory.
            self.m_accobpath = "%s/FG%i" % (obpath,fgrp)
            if ( not os.path.exists(self.m_accobpath) ):
                os.mkdir(self.m_accobpath)                
            
            # Generate Epoch PIVData objects for the Frame group.
            if ( parallel ):
                sl.eachElem(_loop_epoch_worker,
                            [grp],
                            self.m_wrkrstepsvn,
                            loadFactor=nwsconf.loadFactor,
                            accumulator=self.__accumulator__)
            else:
		#Xiyuan, start of STDOUT-Ecnt
		cnt = 0
                #cnt = 0
                for crg in grp:   
                    erslt = _loop_epoch_worker(crg,self.m_wrkrstepsvn)
		    #Xiyuan
		    self.m_crg = erslt[2]
		    erslt = erslt[:2]

                    self.__accumulator__([[cnt,erslt]])
                    
                    cnt = cnt +1

            if ( self.m_assemble ):
                epd = self.__assemble__(self.m_accobpath)
                dpdl.append(epd)
                
            fgrp = fgrp +1

        if ( parallel ):
            sl.eachWorker(_loop_epoch_cleanupgws,self.m_wrkrtdirvn)
        else:
            _loop_epoch_cleanupgws(self.m_wrkrtdirvn)

        carriage['dpivdata'] = dpdl
