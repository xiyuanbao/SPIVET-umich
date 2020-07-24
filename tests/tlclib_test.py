"""
Filename:  tlclibc_test.py
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
    Runs various validation tests on the tlclib functions.
"""

from spivet import tlclib
from spivet.tlclib import tlclibc
from numpy import *

import unittest, StringIO, sys

class test_tlclib(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr

        # Build coefficient array.
        zcoeff = array([1.,2.])
        ycoeff = array([3.,4.,5.])
        xcoeff = array([6.,7.,8.])
        
        coeff = []
        for z in range(2):
            for y in range(3):
                for x in range(3):
                    coeff.append( zcoeff[z]*ycoeff[y]*xcoeff[x] )
        
        self.coeff = array(coeff)
        self.order = array([1,2,2]) 
        
        self.idv = array([[0.,0.,0.],[1.,2.,3.]])
        
    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__

    def test_evalmpoly(self):        
        # Check that evalmpoly returns the correct values.
        kva = array([18.,9207.])
        
        rva = tlclibc.evalmpoly(self.idv,self.coeff,self.order)
        self.assertEqual( (abs(rva -kva) < self.eps).all(), True )
        
    def test_dndiT(self):        
        # Check derivatives from tlctccal.
        tlccal = {'pcoeff':self.coeff,'porder':self.order}
        rva = tlclib.tlctccal.dn_diT(self.idv,1,0,tlccal)
        kva = array([36.,6138.])
        self.assertEqual( (abs(rva -kva) < self.eps).all(), True )
        
        rva = tlclib.tlctccal.dn_diT(self.idv,2,0,tlccal)
        kva = array([0.,0.])
        self.assertEqual( (abs(rva -kva) < self.eps).all(), True )
        
        rva = tlclib.tlctccal.dn_diT(self.idv,2,1,tlccal)
        kva = array([60.,2970.])
        self.assertEqual( (abs(rva -kva) < self.eps).all(), True )

        rva = tlclib.tlctccal.dn_diT(self.idv,2,2,tlccal)
        kva = array([48.,1488.])
        self.assertEqual( (abs(rva -kva) < self.eps).all(), True )

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( test_tlclib ) )
    
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())


