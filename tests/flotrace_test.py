"""
Filename:  flotrace_test.py
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
    Runs various validation tests on the flotrace module.
"""

from numpy import *
from spivet import flolib, pivlib

import unittest, StringIO, sys

class test_psvtrace(unittest.TestCase):
    def setUp(self):
        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr

        # Problem setup.
        nt      = 25
        nc      = 21
        csz     = 4.
        cellsz  = (csz,csz,csz)  
        origin  = (0.,0.,0.)
        ncells  = array([nc,nc,nc])
        tncells = nc**3
        
        # Initialize the velocity field.
        omgad = 4.*pi/(nt -1.)
        
        indxmat  = indices((nc,nc,nc))
        clc      = (nc -1)/2.
        zndx     = (indxmat[0,:,:,:] -clc)*csz
        yndx     = (indxmat[1,:,:,:] -clc)*csz
        self.clc = clc
        
        fvar = pivlib.PIVVar((3,nc,nc,nc),name="OFDISP",units="MM_S")
        fvar[0,...] = omgad*yndx   # About the x-axis
        fvar[1,...] = -omgad*zndx
        
        pd = pivlib.PIVData(cellsz,origin,"TESTFLOW",fvar)
        for i in range(1,nt):
            pd.addVars(i,fvar)
            pd[i].setTime(i)
        
        self.pd = pd
        
        # Advection parameters.
        self.tssdiv = 3
        self.ntrcpc = 10
        
        # Initialize composition.
        icomp = pivlib.PIVVar((1,nc,nc,nc),
                             name="COMPOSITION",
                             units="NA",
                             vtype=pd[0]["OFDISP"].vtype,
                             dtype=int16)
        
        icomp[0,:,nc/2,:] = 1
        
        self.icomp = icomp

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__

    def checkError(self,tclst):
        eps = 1.e-3
        
        # Compute error.
        otcrd = tclst[0][0]
        for i in range(0,len(tclst)):
            otcrd = otcrd[tclst[i][2]]
    
        tcrd = tclst[-1][0]
    
        errz = tcrd[:,0] -otcrd[:,0]
        erry = tcrd[:,1] -otcrd[:,1]
        errx = tcrd[:,2] -otcrd[:,2]
    
        # The outermost cell can be trashy due to edge extension.
        rad = sqrt( (otcrd[:,0] -self.clc)**2 +(otcrd[:,1] -self.clc)**2 )
        msk = rad < ( self.clc - 1. )
    
        ezmin = errz[msk].min()
        ezmax = errz[msk].max()
        ezrms = sqrt((errz[msk]**2).mean())
    
        eymin = erry[msk].min()
        eymax = erry[msk].max()
        eyrms = sqrt((erry[msk]**2).mean())
    
        exmin = errx.min()
        exmax = errx.max()
        exrms = sqrt((errx**2).mean())
   
        self.assertTrue(abs(ezmin) < eps)
        self.assertTrue(abs(ezmax) < eps)
        self.assertTrue(ezrms < eps)

        self.assertTrue(abs(eymin) < eps)
        self.assertTrue(abs(eymax) < eps)
        self.assertTrue(eyrms < eps)

        self.assertTrue(abs(exmin) < eps)
        self.assertTrue(abs(exmax) < eps)
        self.assertTrue(exrms < eps)

    def testLinearInterpolation(self):
        # Advect tracers using linear methods.
        tclst = flolib.psvtrace(self.pd,
                                "OFDISP",
                                self.icomp,
                                self.ntrcpc,
                                self.tssdiv,
                                hist=2,
                                interp=['L','L'])
        self.checkError(tclst)

    def testCubicInterpolation(self):
        # Advect tracers using cubic methods.
        tclst = flolib.psvtrace(self.pd,
                                "OFDISP",
                                self.icomp,
                                self.ntrcpc,
                                self.tssdiv,
                                hist=2,
                                interp=['C','C'])
        self.checkError(tclst)


def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( test_psvtrace ) )
    
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())


