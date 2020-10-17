"""
Filename:  pivof.py
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
  Module containing optical flow computation routines.

Contents:
  blkflow()
  ofcomp()
  tcfrecon()
"""

from numpy import *
from scipy import linalg

from pivdata import *
import pivutil
import pivir
from spivet import compat

#################################################################
#
def blkflow(f1,f2,rbndx,pivdict,ip=None):
    """
    ----
    
    f1             # Frame 1 (greyscale mxn array).
    f2             # Frame 2 (greyscale mxn array).
    rbndx          # Block region boundary index (2x2 array).
    pivdict        # Config dictionary.
    ip=None        # Displacement vector to be used as an initial guess.
    
    ----
        
    Computes flow for a given block using a three stage process.  

    Stage I registration is based on the Sequential Similarity Detection 
    Algorithm, and provides a fast registration estimate that has a 
    precision of 1 pixel.  Stage I will only be run if ip = None.
    
    Stage II registration uses normalized cross-correlation for a 
    sub-pixel refinement of Stage I results (or ip if passed).  

    In general, if the Stage II NCC coefficient is below 
    pivdict['of_nccth'], then the frames are registered using irsctxt().
    A sufficiently small NCC coefficient indicates that a good match
    using a correlation type measure could not be found.  Such conditions
    result from: a) excessive particle pattern deformation between frames 
    (eg, in regions of high velocity gradients), or b) the presence of 
    stationary bubbles visible along with moving tracers.  Instead of 
    using a correlation type measure (NCC or SSDA), irsctxt() tries to 
    match every particle in the f1 block to its corresponding particle in 
    the f2 block.  Although the techniques employed in irsctxt() (and 
    other similar approaches) can often produce a reasonable estimate of 
    a displacement vector when correlation type measures fail, use of 
    irsctxt() has two principal drawbacks. First and foremost, irsctxt() 
    results are generally not as accurate as those produced using irncc()
    or irssda().  Nevertheless, when faced with the choice between a 
    completely erroneous displacement vector or a reasonable but somewhat
    inaccurate estimate, blkflow() takes the reasonable estimate path.  
    The second drawback to irsctxt() is the method's computational cost.  
    irsctxt() inner workings are complex and iterative in nature.  As a 
    result, applying irsctxt() to a handful of blocks can consume as much
    computation time as the irssda()/irncc() duo applied to an entire 
    image.  Because of these considerations, the use of irsctxt() is 
    mainly intended to be a lifeline of sorts used only for problematic 
    blocks that fail to register with the primary Stage I and Stage II 
    methods.  Stage III will not be run if irsctxt() is successful.  For 
    more details on the inner workings of irsctxt(), see the irsctxt() 
    documentation.
    
    The pivdict parameter 'of_hrsc' if True will enable irsctxt() to be
    used when ip is set (ie, when ip is not None).  When blkflow() is called by 
    ofcomp(), ip is only set when a block has been subdivided.  Hence, setting
    of_hrsc = True will enable the use of irsctxt() for subdivided blocks.
    
    Stage III registration using the Lucas-Kanade algorithm and the
    registration results from Stage II is run if of_highp is set.  
    If the inac flag is set during Stage III, but was not set with 
    Stage II, then the Stage II results will be kept.

    Returns [p,inac,cmax] where 
        p ----- The computed displacements, p=[dy,dx], in moving 
                from f1 to f2.
        inac -- Inaccuracy flag.  Set > 0 when flow computation fails.
        cmax -- Normalized cross correlation coefficient.  Indicates
                quality of irncc() registration.  Set to -1.0 if
                irsctxt() is called or of_highp is set.
    """

    # Initialization.
    maxdisp   = pivdict['of_maxdisp']
    rmaxdisp  = pivdict['of_rmaxdisp']
    highp     = pivdict['of_highp']
    nccth     = pivdict['of_nccth']
    hrsc     = pivdict['of_hrsc']
    #highp=True
    # Stage 1. 
    haveip = True 
    if ( compat.checkNone(ip) ): 
        [ip,tinac,cmin] = pivir.irssda(f1,f2,rbndx,maxdisp,pivdict)
        haveip = False
	iip=ip;iinac=tinac;icmax=cmin

    pmaxdisp = rmaxdisp
    # Stage 2.
    [ip,tinac,cmax] = pivir.irncc(f1,f2,rbndx,pmaxdisp,pivdict,ip)
#Xiyuan:skip everything after piv (no PTV, no irlk)
    return [ip,tinac,cmax]
'''
    if ( compat.checkNone(ip) ):
        pmaxdisp = maxdisp
    else:
        pmaxdisp = rmaxdisp
    if ( ( cmax < nccth ) and ( hrsc or not haveip ) ):
        [p,inac] = pivir.irsctxt(f1,f2,rbndx,maxdisp,pivdict)
        if ( not compat.checkNone(p) ):
            #return [p,inac,-1.]
	
	    if (haveip == False and 1-icmax/100.>nccth and any(abs(array(iip)-array(p))>array(rmaxdisp)*1.0)):
		return [iip,iinac,1-icmax/100.]
	    else:
		return [p,inac,-1.]
        
    # Stage 3.
    if ( highp ):
        [p,inac] = pivir.irlk(f1,f2,rbndx,pmaxdisp,pivdict,1.,ip)
        if ( ( compat.checkNone(p) ) or ( ( inac > 0 ) and ( tinac == 0 ) ) ):
            if ( not compat.checkNone(ip) ):
                p    = ip
                inac = tinac
            else:
                p = zeros(2,dtype=float)

        return [p,inac,-1.]
    else:
        #return [ip,tinac,cmax]
	
	if (tinac > 0 and haveip == False):
		#print "ssda,icmax=",icmax
		return [iip,iinac,1-icmax/100.]
	else:
		return [ip,tinac,cmax]
'''	
'''
#Xiyuan:using a parallel version higher order blkflow inside a new ofcomp
import ctypes as c
from multiprocessing import Array
import multiprocessing
from multiprocessing.pool import Pool
import numpy as np
#modify multiprocessing.Process to non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# modify multiprocessing.Pool class
class myPool(Pool):
    Process = NoDaemonProcess



def parallel_blkflow(m,n,ydim,xdim,f1,f2,pivdict,bsdiv,hbso,bsdcmx,ltnblks,lnblks):
    global s_ofdy,s_ofdx,s_ofinac,s_ofcmax
#     singlerun=time.time()
    ofdy=np.frombuffer(s_ofdy.get_obj()).reshape(ydim,xdim)
    ofdx=np.frombuffer(s_ofdx.get_obj()).reshape(ydim,xdim)
    ofinac=np.frombuffer(s_ofinac.get_obj()).reshape(ydim,xdim)
    ofcmax=np.frombuffer(s_ofcmax.get_obj()).reshape(ydim,xdim)
    
    
    bn=(n+1)+(m+1)*lnblks[1]
    prbndx  = zeros((2,2), dtype=int)
    hprbndx = zeros((2,2), dtype=int)
    
    prbndx[0,0] = rbndx[0,0] +m*lbso[0]
    prbndx[0,1] = prbndx[0,0] +bsize[0]
    prbndx[1,0] = rbndx[1,0] +n*lbso[1]
    prbndx[1,1] = prbndx[1,0] +bsize[1]
    
    [p,inac,cmax] = blkflow(f1,f2,prbndx,pivdict)
    
    path=""
    if ( bsdiv > 1 ):
        for hm in range(bsdiv):
            hprbndx[0,0] = prbndx[0,0] +hm*hbso[0]
            hprbndx[0,1] = hprbndx[0,0] +hbso[0]

            mndx = m*bsdiv +hm
            for hn in range(bsdiv):
                hprbndx[1,0] = prbndx[1,0] +hn*hbso[1]
                hprbndx[1,1] = hprbndx[1,0] +hbso[1]

                nndx = n*bsdiv +hn

                if ( inac == 0 ):
                    [hp,hinac,hcmax] = blkflow(f1,f2,hprbndx,pivdict,p)

                    lowcmx = ( cmax -hcmax ) > bsdcmx
                    if ( hinac > 0 or lowcmx ):
                        ofdy[mndx, nndx]   = p[0]
                        ofdx[mndx, nndx]   = p[1]
                        if ( lowcmx ):
                            ofinac[mndx, nndx] = -100. -hinac
                        else:
                            ofinac[mndx, nndx] = 100. +hinac
                        ofcmax[mndx, nndx] = cmax
                    else:
                        ofdy[mndx, nndx]   = hp[0]
                        ofdx[mndx, nndx]   = hp[1]
                        ofinac[mndx, nndx] = hinac
                        ofcmax[mndx, nndx] = hcmax
                else:
                    ofdy[mndx, nndx]   = p[0]
                    ofdx[mndx, nndx]   = p[1]
                    ofinac[mndx, nndx] = inac
                    ofcmax[mndx, nndx] = cmax
    else:
        ofdy[m,n]   = p[0]
        ofdx[m,n]   = p[1]
        ofinac[m,n] = inac
        ofcmax[m,n] = cmax
#     print "singlerun:",time.time()-singlerun
    if ( int(10*bn/ltnblks) > tpc ):
        tpc = tpc +1
        print " |  " + str(tpc*10) + "% complete"
    
def initpool(a,b,c,d):
    global s_ofdy,s_ofdx,s_ofinac,s_ofcmax
    s_ofdy=a;s_ofdx=b;s_ofinac=c;s_ofcmax=d


def ofcomp(f1,f2,pivdict):
    print "STARTING: ofcomp"

    # Initialization.
    rbndx     = pivdict['gp_rbndx']
    bsize     = pivdict['gp_bsize']
    bolap     = pivdict['gp_bolap']
    bsdiv     = pivdict['gp_bsdiv']
    bsdcmx    = pivdict['of_bsdcmx']

    imdim = f1.shape
    rbndx = array(rbndx)

    [rsize,lbso,hbso,lnblks,hnblks] = pivutil.getblkprm(rbndx,bsize,
                                                        bolap,bsdiv)
    ltnblks = lnblks[0]*lnblks[1]

    ofDisp = PIVVar([3,1,hnblks[0],hnblks[1]],'U','PIX')
    ofINAC = PIVVar([1,1,hnblks[0],hnblks[1]],'UINAC','NA',dtype=int)

    ofCMAX = PIVVar([1,1,hnblks[0],hnblks[1]],'CMAX','NA')

    print ' | nyblks %i' % hnblks[0]
    print ' | nxblks %i' % hnblks[1]

    ofdy   = ofDisp[1,0,:,:]
    # ofdx   = ofDisp[2,0,:,:]
    # ofinac = ofINAC[0,0,:,:]

    # ofcmax = ofCMAX[0,0,:,:]
    ydim,xdim=ofdy.shape
    #convert to shared!
    size1d=ofdy.size
    s_ofdy=Array(c.c_double, ofdy.size)
    s_ofdx=Array(c.c_double, size1d)
    s_ofinac=Array(c.c_double, size1d)
    s_ofcmax=Array(c.c_double, size1d)

    subp=[]
    #def initpool(a,b,c,d):
    #    global s_ofdy,s_ofdx,s_ofinac,s_ofcmax
    #    s_ofdy=a;s_ofdx=b;s_ofinac=c;s_ofcmax=d
    pool = myPool(initializer=initpool,initargs=(s_ofdy,s_ofdx,s_ofinac,s_ofcmax),processes=4)


    # prbndx  = zeros((2,2), dtype=int)
    # hprbndx = zeros((2,2), dtype=int)

    # Main loop.
    print " | Flow computation ..."
    for m in range(lnblks[0]):
        for n in range(lnblks[1]):
            proc=pool.apply_async(parallel_blkflow,(m,n,ydim,xdim,f1,f2,pivdict,bsdiv,hbso,bsdcmx,ltnblks,lnblks))
            subp.append(proc)
    pool.close()
    pool.join()
    for proc in subp:
    	proc.get()
    print "max ofdy:",np.max(s_ofdy)
    ofDisp[1,0,:,:]=np.asarray(s_ofdy).reshape(ydim,xdim)
    ofDisp[2,0,:,:]=np.asarray(s_ofdx).reshape(ydim,xdim)
    ofINAC[0,0,:,:]=np.asarray(s_ofinac).reshape(ydim,xdim)
    ofCMAX[0,0,:,:]=np.asarray(s_ofcmax).reshape(ydim,xdim)
    ofcmax = ofCMAX[0,0,:,:]
    cmax = ofcmax[ofcmax > 0.]
    print " | NCC CMAX MEAN: %f, STDDEV %f" % (cmax.mean(),cmax.std())

    print " | EXITING: ofcomp"
    return [ofDisp,ofINAC,ofCMAX]
    #print "CAM %i OFCOMP RUN TIME: %g" % (cam,a)
'''        
#################################################################
#
def ofcomp(f1,f2,pivdict):
    """
    ----
    
    f1             # Frame 1 (greyscale mxn array).
    f2             # Frame 2 (greyscale mxn array).
    pivdict        # Config dictionary.
    
    ----
        
    Primary driving routine to compute optical flow between two image
    frames.  Operation proceeds as follows:
        1) ofcomp() subdivides the images into blocks.
        2) Loop start.
        3) Flow for each block is computed using blkflow().
        4) If of_bsdiv > 1 and blkflow() did not return inac for the
           coarse block, the block is subdivided into of_bsdiv sub-blocks
           along each axis.  Flow for each of these sub-blocks is then
           computed, using the results from Step 3 as initialization.
           If blkflow() returns inac for the sub-block, then the flow 
           vector computed from Step 3 will be stored for the sub-block.
           Similarly, if the difference between the normalized cross-correlation
           coefficient from Step 3 and the the subdivided block is greater than 
           of_bsdcmx, then the results of Step 3 will be stored for the 
           sub-block.

    ofcomp() returns [ofDisp, ofINAC, ofCMAX], three PIVVar objects (s,t below 
    are the number of blocks in the y- and x-direction respectively):  
        ofDisp ---- 3 x 1 x s x t PIVVar object containing the displacement
                    vector in moving from f1 to f2.  ofDisp.data[:,:,0] = 0.
                    The variable name will be set to U.
        ofINAC ---- 1 x 1 x s x t PIVVar object containing the inaccuracy flag 
                    (0 indicates a good value, > 0 indicates that the 
                    value should be treated with suspicion).  NOTE: If 
                    of_bsdcmx > 0.0 and of_bsdiv > 1, then INAC will be 
                    negative for cells where coarse results were retained
                    based on of_bsdcmx.  Variable name will be set to UINAC. 
        ofCMAX ---- 1 x 1 x s x t PIVVar object containing the maximum value
                    of the normalized cross-correlation coefficient.  ofCMAX
                    gives a direct measure of the quality of a flow vector with
                    0.0 indicating no match between image frames was found and 
                    1.0 indicating that the computed displacement vector 
                    produced a perfect image match.  ofCMAX will be negative
                    if of_highp is set or irsctxt() has been run.  See
                    blkflow() for more details.  Variable name will be set to
                    CMAX.
    """

    print "STARTING: ofcomp"

    # Initialization.
    rbndx     = pivdict['gp_rbndx']
    bsize     = pivdict['gp_bsize']
    bolap     = pivdict['gp_bolap']
    bsdiv     = pivdict['gp_bsdiv']
    bsdcmx    = pivdict['of_bsdcmx']

    imdim = f1.shape
    rbndx = array(rbndx)

    [rsize,lbso,hbso,lnblks,hnblks] = pivutil.getblkprm(rbndx,bsize,
                                                        bolap,bsdiv)
    ltnblks = lnblks[0]*lnblks[1]

    ofDisp = PIVVar([3,1,hnblks[0],hnblks[1]],'U','PIX')
    ofINAC = PIVVar([1,1,hnblks[0],hnblks[1]],'UINAC','NA',dtype=int)
    
    ofCMAX = PIVVar([1,1,hnblks[0],hnblks[1]],'CMAX','NA')

    print ' | nyblks %i' % hnblks[0]
    print ' | nxblks %i' % hnblks[1]

    ofdy   = ofDisp[1,0,:,:]
    ofdx   = ofDisp[2,0,:,:]                
    ofinac = ofINAC[0,0,:,:]

    ofcmax = ofCMAX[0,0,:,:]

    prbndx  = zeros((2,2), dtype=int)
    hprbndx = zeros((2,2), dtype=int)

    # Main loop.
    print " | Flow computation ..."
    tpc = 0
    bn  = 0
    for m in range(lnblks[0]):
        prbndx[0,0] = rbndx[0,0] +m*lbso[0]
        prbndx[0,1] = prbndx[0,0] +bsize[0]
        for n in range(lnblks[1]):
            bn = bn +1

            prbndx[1,0] = rbndx[1,0] +n*lbso[1]
            prbndx[1,1] = prbndx[1,0] +bsize[1]

            # Register the block.
            [p,inac,cmax] = blkflow(f1,f2,prbndx,pivdict)
	    path=""
            if ( bsdiv > 1 ):
                for hm in range(bsdiv):
                    hprbndx[0,0] = prbndx[0,0] +hm*hbso[0]
                    hprbndx[0,1] = hprbndx[0,0] +hbso[0]

                    mndx = m*bsdiv +hm
                    for hn in range(bsdiv):
                        hprbndx[1,0] = prbndx[1,0] +hn*hbso[1]
                        hprbndx[1,1] = hprbndx[1,0] +hbso[1]
                    
                        nndx = n*bsdiv +hn

                        if ( inac == 0 ):
                            [hp,hinac,hcmax] = blkflow(f1,f2,hprbndx,pivdict,p)
                            
                            lowcmx = ( cmax -hcmax ) > bsdcmx     
                            if ( hinac > 0 or lowcmx ):
                                ofdy[mndx, nndx]   = p[0]
                                ofdx[mndx, nndx]   = p[1]
                                if ( lowcmx ):
                                    ofinac[mndx, nndx] = -100. -hinac
				    path = "path1"
                                else:
                                    ofinac[mndx, nndx] = 100. +hinac
                                    path = "path2"
                                ofcmax[mndx, nndx] = cmax
                            else:
                                ofdy[mndx, nndx]   = hp[0]
                                ofdx[mndx, nndx]   = hp[1]
                                ofinac[mndx, nndx] = hinac                            
                                ofcmax[mndx, nndx] = hcmax
				path = "path3"
                        else:
                            ofdy[mndx, nndx]   = p[0]
                            ofdx[mndx, nndx]   = p[1]
                            ofinac[mndx, nndx] = inac
                            ofcmax[mndx, nndx] = cmax
			    path = "path4"
			#if ofinac[mndx, nndx]>0:
                		#print "inaccruate! inac,cmax,path=",ofinac[mndx, nndx],ofcmax[mndx, nndx],path
            else:
                ofdy[m,n]   = p[0]
                ofdx[m,n]   = p[1]
                ofinac[m,n] = inac
                ofcmax[m,n] = cmax
		path = "path5"
            if ( int(10*bn/ltnblks) > tpc ):
                tpc = tpc +1
                print " |  " + str(tpc*10) + "% complete"
    cmax = ofcmax[ofcmax > 0.]
    print " | NCC CMAX MEAN: %f, STDDEV %f" % (cmax.mean(),cmax.std())

    print " | EXITING: ofcomp"
    return [ofDisp,ofINAC,ofCMAX]

#################################################################
#
def tcfrecon(ofDisp0,ofDisp1,pivdict):
    """
    ----
    
    ofDisp0        # PIVVar object with CAM0 flow.
    ofDisp1        # PIVVar object with CAM1 flow.
    pivdict        # Config dictionary.
    
    ----
        
    Reconstructs three component flow using 2D flow fields from
    two seperate cameras (ie, by stereo vision).

    The following geometry is used.  Let dy' and dx', the apparent 
    displacements computed in 2D, be

        dy' = dy +ey                      Eq. 1
        dx' = dx +ex

    where dy and dx are the true displacements, and ey and ex are the
    displacement errors (aka out of plane errors) caused by a non-zero dz.  
    The camera pinhole coordinates in world coordinates are represented 
    as the vector _wcc.  The starting location of the particle at time 
    t=0 will be the vector _xs, with the true final location of the 
    particle given by _xf.  The apparent final location of the particle in
    the xy_world plane will be given by _xf'.  Then,

        yf' = ys +dy'
        xf' = xs +dx'

    Let _r be the vector from (z_w, y_w, x_w) = (0,wcc[1],wcc[2]) to _xf' 
    (_r lies in the xy_world plane).

        r = sqrt( (ys +dy' -wcc[1])^2 +(xs +dx' -wcc[2])^2 )

    Two angles can then be defined such that

        tan(theta) = r/wcc[0]
        sin(phi)   = ys +dy' -wcc[1]
        cos(phi)   = xs +dx' -wcc[2]
    
    The errors due to out of plane displacements can then be represented
    as

        ey = dz*tan(theta)*sin(phi)       # Eq. 2
        ex = dz*tan(theta)*cos(phi)

    Subsituting Eq's 2 into Eq's 1 for each camera yields a set of
    four equations in three unknowns that can be solved by least squares.

    Returns (s,t below are the number of blocks in the y- and x-direction 
    respectively):  
        ofDisp ---- 3 x 1 x s x t PIVVar object containing the three
                    component displacement vector in moving from f1 to f2.
                    Variable name will be set to U3D.  
    """
    # Initialization.
    rbndx   = pivdict['gp_rbndx']
    bsize   = pivdict['gp_bsize']
    bolap   = pivdict['gp_bolap']
    bsdiv   = pivdict['gp_bsdiv']
    camcal0 = pivdict['pg_camcal'][0]
    camcal1 = pivdict['pg_camcal'][1]
    mmpx    = pivdict['pg_wicsp'][0]['mmpx']
    wos     = pivdict['pg_wicsp'][0]['wos']

    nzbolap = ( bolap[0] > 0 ) or ( bolap[1] > 0 )
    if ( ( nzbolap ) and ( bsdiv > 1 ) ):
        print "ERROR: bolap > 0 and bsdiv > 1 cannot be used simultaneously."
        return None

    nyblks = ofDisp0.shape[2]
    nxblks = ofDisp0.shape[3]
    tnblks = nyblks*nxblks

    ofDisp = PIVVar([3,1,nyblks,nxblks],'U3D','MM')

    # Compute coordinates of block centers in world coordinates.
    ndxmat = indices((nyblks,nxblks))
    yv = ndxmat[0,:,:]
    xv = ndxmat[1,:,:]

    absizey = bsize[0]/bsdiv
    absizex = bsize[1]/bsdiv

    yv = (yv*(absizey -bolap[0]) +rbndx[0,0] +absizey/2.)*mmpx +wos[0]
    xv = (xv*(absizex -bolap[1]) +rbndx[1,0] +absizex/2.)*mmpx +wos[1]

    yv = yv.reshape(tnblks)
    xv = xv.reshape(tnblks)

    # Compute camera center in world coordinates (wcc) as well as
    # tan(theta)/r, r*cos(phi), and r*sin(phi) for CAM0.
    wcc = linalg.solve(camcal0['Rmat'],-camcal0['T'])

    ofdy0 = mmpx*ofDisp0[1,0,:,:]
    ofdx0 = mmpx*ofDisp0[2,0,:,:]

    ofdy0 = ofdy0.reshape(tnblks)
    ofdx0 = ofdx0.reshape(tnblks)

    ry    = yv +ofdy0 -wcc[1]
    rx    = xv +ofdx0 -wcc[2]
    ttsp0 = ry/wcc[0]
    ttcp0 = rx/wcc[0]

    # Compute camera center in world coordinates (wcc) as well as
    # tan(theta)/r, r*cos(phi), and r*sin(phi) for CAM1.
    wcc = linalg.solve(camcal1['Rmat'],-camcal1['T'])

    ofdy1 = mmpx*ofDisp1[1,0,:,:]
    ofdx1 = mmpx*ofDisp1[2,0,:,:]

    ofdy1 = ofdy1.reshape(tnblks)
    ofdx1 = ofdx1.reshape(tnblks)

    ry    = yv +ofdy1 -wcc[1]
    rx    = xv +ofdx1 -wcc[2]
    ttsp1 = ry/wcc[0]
    ttcp1 = rx/wcc[0]

    # Compute the three components.
    ofdz    = ofDisp[0,0,:,:].reshape(tnblks)
    ofdy    = ofDisp[1,0,:,:].reshape(tnblks)
    ofdx    = ofDisp[2,0,:,:].reshape(tnblks)

    for i in range(tnblks):
        smat      = zeros((4,3),dtype=float)
        smat[0,0] = ttsp0[i]
        smat[0,1] = 1.
        smat[1,0] = ttcp0[i]
        smat[1,2] = 1.
        smat[2,0] = ttsp1[i]
        smat[2,1] = 1.
        smat[3,0] = ttcp1[i]
        smat[3,2] = 1.

        bvec    = zeros(4,dtype=float)
        bvec[0] = ofdy0[i]
        bvec[1] = ofdx0[i]
        bvec[2] = ofdy1[i]
        bvec[3] = ofdx1[i]
    
        smat = matrix(smat)
        bvec = matrix(bvec)
        bvec = array(smat.transpose()*bvec.transpose()).squeeze()
        smat = array(smat.transpose()*smat)

        soln = linalg.solve(smat,bvec)

        ofdz[i] = soln[0]
        ofdy[i] = soln[1]
        ofdx[i] = soln[2]

    return ofDisp
