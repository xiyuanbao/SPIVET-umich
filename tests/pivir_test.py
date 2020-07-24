"""
Filename:  pivir_test.py
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
    Runs various validation tests on the pivir module and
    underlying pivlibc module.

    NOTE: Any changes to imread() that affect computation of
    intensity will break these tests.
"""
from spivet import pivlib
from PIL import Image
from numpy import *
import hashlib, os, unittest, StringIO, sys
from os import path

class test_pivir(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Setup the problem.
        self.sv = array([[0.0,0.0],
                    [0.2,0.2],
                    [-0.2,-0.2],
                    [0.56,0.2],
                    [-1.2,1.2],
                    [-1.2,1.56],
                    [-1.2,-1.56],
                    [-2.3,-2.3],
                    [2.3,2.3]])
        #self.rbndx = [16,32,16,32]
        rbndx      = array([[16,32],[16,32]])
        self.rbndx = rbndx
        
        [bch,bcs,bci] = pivlib.imread("%s/baseimg.png" % self.dfp,0)
        self.bci = bci

        # Load a shifted image.
        self.maxdisp = (4,4) 
        self.pivdict = {
            'ir_eps':0.003,
            'ir_maxits':100, 
            'ir_mineig':0.05,
            'ir_imthd':'C',
            'ir_iedge':0
        }
        bsci = pivlib.imread("%s/shftimg.png" % self.dfp,0)
        bsci = bsci[2]
        
        bsi = 0.5*ones([64,64])
        bsi[rbndx[0,0]:rbndx[0,1],rbndx[1,0]:rbndx[1,1]] = bsci
        self.bsi = bsi

        # Load known translations.
        self.kr = pivlib.pklload("%s/pivir-ir-knowns" % self.dfp)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__       

    def test_imsblicore(self):
        for i in range(self.sv.shape[0]):        
            kfn = "%s/pivir-imsblicore-known-%i" % (self.dfp,i)
            ksi = pivlib.pklload(kfn)
            si  = pivlib.pivlibc.imsblicore(self.bci,
                                            self.rbndx.reshape(4),
                                            self.sv[i])
            
            d = abs(si -ksi)
            d = d < self.eps
            
            self.assertTrue( d.all(), "Point %i" % i )

    def test_imsblicore_EdgeTreatment0(self):
        for i in range(self.sv.shape[0]):                
            kfn = "%s/pivir-imsbcicore-e0-known-%i" % (self.dfp,i)
            ksi = pivlib.pklload(kfn)
            si  = pivlib.pivlibc.imsbcicore(self.bci,
                                            self.rbndx.reshape(4),
                                            self.sv[i],0)
        
            d = abs(si -ksi)
            d = d < self.eps
            self.assertTrue( d.all(), "Point %i" % i )
                
    def test_imsblicore_EdgeTreatment1(self):
        for i in range(self.sv.shape[0]):                
            kfn = "%s/pivir-imsbcicore-e1-known-%i" % (self.dfp,i)
            ksi = pivlib.pklload(kfn)
            si  = pivlib.pivlibc.imsbcicore(self.bci,
                                            self.rbndx.reshape(4),
                                            self.sv[i],1)
        
            d = abs(si -ksi)
            d = d < self.eps
            
            self.assertTrue( d.all(), "Point %i" % i )

    def test_pxsbcicore(self):
        ksi  = pivlib.pklload("%s/pivir-pxsbcicore-known" % self.dfp)
        px   = [13,14]
        pxca = px*ones([9,2],dtype=int)
        si   = pivlib.pivlibc.pxsbcicore(self.bci,pxca,self.sv)
        
        d = abs(si -ksi)
        d = d < self.eps
        self.assertTrue( d.all() )

    def test_irncc(self):
        tp1  = pivlib.pivir.irncc(self.bci,
                                  self.bsi,
                                  self.rbndx,
                                  self.maxdisp,
                                  self.pivdict)
        tp1b = pivlib.pivir.irncc(self.bci,
                                  self.bsi,
                                  self.rbndx,
                                  self.maxdisp,
                                  self.pivdict,
                                  self.sv[6])
    
        d = abs(tp1[0] -self.kr[2])
        d = d < self.eps        
        self.assertTrue( d.all() )
        
        d = abs(tp1b[0] -self.kr[3])
        d = d < self.eps
        self.assertTrue( d.all() )  # With preshift.
        
    def test_irlk(self):
        tp2  = pivlib.pivir.irlk(self.bci,
                                 self.bsi,
                                 self.rbndx,
                                 self.maxdisp,
                                 self.pivdict)
        tp2b = pivlib.pivir.irlk(self.bci,
                                 self.bsi,
                                 self.rbndx,
                                 self.maxdisp,
                                 self.pivdict,
                                 pinit=self.sv[6])
        tp2c = pivlib.pivir.irlk(self.bci,
                                 self.bsi,
                                 self.rbndx,
                                 self.maxdisp,
                                 self.pivdict,
                                 1.4,
                                 pinit=self.sv[6])

        d = abs(tp2[0] -self.kr[4])
        d = d < self.eps
        self.assertTrue( d.all() )
        
        d = abs(tp2b[0] -self.kr[5])
        d = d < self.eps
        self.assertTrue( d.all() )  # With preshift.
        
        d = abs(tp2c[0] -self.kr[6])
        d = d < self.eps
        self.assertTrue( d.all() )  # With relaxation.

    def test_irssda(self):
        tp3  = pivlib.pivir.irssda(self.bci,
                                   self.bsi,
                                   self.rbndx,
                                   self.maxdisp,
                                   self.pivdict)
        tp3b = pivlib.pivir.irssda(self.bci,
                                   self.bsi,
                                   self.rbndx,
                                   self.maxdisp,
                                   self.pivdict,
                                   self.sv[6])
        
        d = abs(tp3[0] -self.kr[7])
        d = d < self.eps
        self.assertTrue( d.all() )
        
        d = abs(tp3b[0] -self.kr[8])
        d = d < self.eps
        self.assertTrue( d.all() )  # With preshift.


def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( test_pivir ) )
    
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
