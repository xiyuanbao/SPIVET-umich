"""
Filename:  pivdata_test.py
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
    Runs various validation tests on the pivdata module.
"""

from spivet.pivlib import pivdata as p
from spivet.pivlib import pkldump, pklload
from numpy import *
import hashlib, os, unittest, StringIO, sys, uuid
from os import path

#################################################################
#
class testPIVVar(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create a test object.
        self.var = p.PIVVar([1,1,2,2])

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__ 
        
    def testBadShape(self):
        self.assertRaises(ValueError,
                          p.PIVVar, [1,1,2,2,1] )

    def testBadDimCount(self):
        self.assertRaises(ValueError,
                          p.PIVVar, [4,1,1,1] )

    def testShapeSize(self):
        self.assertEqual(len(self.var.shape),4)
    
    def testDefaultName(self):
        self.assertEqual(self.var.name,"NOTSET")

    def testDefaultUnits(self):
        self.assertEqual(self.var.units,"NOTSET")
        
    def testDefault_vtype(self):
        self.assertEqual(self.var.vtype,"E")

    def testInitializationValue(self):
        self.assertTrue( ( abs(self.var) < self.eps ).all() )
        
    def testDefault_dtype(self):
        self.assertEqual( self.var.dtype, float )

    def testPreventDotAttrSet(self):
        try:
            self.var.attr = 4
            rval = 1
        except AttributeError:
            rval = 0

        self.assertEqual( rval, 0 )

    def testDerived_vtype(self):
        vv   = self.var +1
        n    = vv.name
        rval = 0

        self.assertEqual(vv.vtype,"E")

    def testSetName(self):
        name = "ANAME"
        self.var.setName(name)
        
        self.assertEqual(self.var.name,name)
        
    def testSetUnits(self):
        units = "UNIT"
        self.var.setUnits(units)
        
        self.assertEqual(self.var.units,units)

    def testSet_vtype(self):
        val = "N"
        self.var.setVtype(val)
        
        self.assertEqual(self.var.vtype,val)

    def testSetLowerCase_vtype(self):
        val = "n"
        self.var.setVtype(val)
        
        self.assertEqual(self.var.vtype,val.upper())

    def testInvalid_vtype(self):
        self.assertRaises(ValueError,
                          self.var.setVtype, "Q")

    def testPickle(self):
        vv = self.var +4.
        pkldump(vv,"%s/pivvar-pkl-testfile-out" % self.ofp)
        lvv = pklload("%s/pivvar-pkl-testfile-out" % self.ofp)
        
        self.assertEqual(vv.name,lvv.name)
        self.assertEqual(vv.units,lvv.units)
        self.assertEqual(vv.vtype,lvv.vtype)

        tst = vv == lvv
        self.assertTrue( tst.all() )

    def testSetAttr(self):
        attr = ["NAME","MM"]
        self.var.setAttr(attr[0],attr[1])
        
        self.assertEqual(self.var.name,attr[0])
        self.assertEqual(self.var.units,attr[1])

    def testInitialization(self):
        v = p.PIVVar([1,1,2,2],"VARNAME","VARUN",vtype="N")

        self.assertEqual(v.name, "VARNAME")
        self.assertEqual(v.units, "VARUN")
        self.assertEqual(v.vtype, "N")

    def testVectorDim(self):
        v = p.PIVVar([3,1,2,2])
        self.assertEqual(v.shape[0],3)
        
    def testTensorDim(self):
        v = p.PIVVar([9,1,2,2])
        self.assertEqual(v.shape[0],9)


#################################################################
#
class testPIVCache(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create a test object.
        self.pc = p.PIVCache()

        # Some objects to track deactivated variables.
        self.deactivated = None

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__ 
        
    def __deactivate__(self,ouuid,obj):
        self.deactivated = ouuid
        
    def testSetCacheSize(self):
        pc = self.pc
        
        pc.setCacheSize(10)
        self.assertEqual(pc.csize,10)
        
        pc.setCacheSize(12)
        self.assertEqual(pc.csize,12)
        
    def testCurrentCacheSize(self):
        pc = self.pc
        
        self.assertEqual(pc.ccsize,0)
        
        pc.store(None)
        self.assertEqual(pc.ccsize,1)
                
    def testStore(self):
        pc = self.pc
        
        csize = 6
        pc.setCacheSize(csize)
        
        ids = []
        for i in xrange(10):
            ids.append( pc.store(str(uuid.uuid4()),self.__deactivate__) )
 
            if ( i < csize ):
                self.assertEqual(pc.ccsize,i+1)
                self.assertEqual(self.deactivated,None)
            else:
                self.assertEqual(pc.ccsize,csize)
                self.assertNotEqual(self.deactivated,None)
                self.deactivated = None

        self.assertTrue(isinstance(ids[0],uuid.UUID))

    def testLRU(self):
        # Ensure that old objects are the ones purged from the cache.
        pc = self.pc
        
        csize = 6
        pc.setCacheSize(csize)
        
        ids  = []
        aobj = None
        for i in xrange(10):
            if ( i == 0 ):
                # Take a reference to object 0.  This one should not be
                # removed from the cache.
                aobj = str(uuid.uuid4())
                ids.append( pc.store(aobj,self.__deactivate__) )
            else:            
                ids.append( pc.store(str(uuid.uuid4()),self.__deactivate__) )
 
            if ( i >= csize ):
                self.assertEqual(self.deactivated,ids[1])
                ids.pop(1)
        
    def testMostRecentlyUsed(self):
        # Ensure that used objects are moved to the end of the cache.
        pc = self.pc
        
        csize = 6
        pc.setCacheSize(csize)
        
        ids  = []
        aobj = None
        for i in xrange(10):
            ids.append( pc.store(str(uuid.uuid4()),self.__deactivate__) )
 
        pc.retrieve(ids[-3])
        for i in xrange(5):
            pc.store(str(uuid.uuid4()),self.__deactivate__)
            self.assertNotEqual(self.deactivated,ids[-3])
        pc.store(str(uuid.uuid4()),self.__deactivate__)
        self.assertEqual(self.deactivated,ids[-3])

    def testRetrieveMethod(self):
        pc = self.pc
        
        csize = 6
        pc.setCacheSize(csize)
        
        ids  = []
        aobj = None
        for i in xrange(10):
            if ( i < 9 ):
                ids.append( pc.store(str(uuid.uuid4()),self.__deactivate__) )
            else:
                aobj = str(uuid.uuid4())
                ids.append( pc.store(aobj,self.__deactivate__) )
                
        obj = pc.retrieve(ids[-1])
        self.assertEqual(obj,aobj)
        
        self.assertRaises(KeyError,pc.retrieve,"DUMMY")
        self.assertRaises(KeyError,pc.retrieve,ids[0])
        
    def testRetrieveBracket(self):
        pc = self.pc
        
        csize = 6
        pc.setCacheSize(csize)
        
        ids  = []
        aobj = None
        for i in xrange(10):
            if ( i < 9 ):
                ids.append( pc.store(str(uuid.uuid4()),self.__deactivate__) )
            else:
                aobj = str(uuid.uuid4())
                ids.append( pc.store(aobj,self.__deactivate__) )
                
        obj = pc[ids[-1]]
        self.assertEqual(obj,aobj)

    def testAssimilateCache(self):
        apc = p.PIVCache()
        pc  = self.pc

        csize = 6
        apc.setCacheSize(csize)
        pc.setCacheSize(csize)

        aids = []
        objs = []
        for i in xrange(csize):
            objs.append( str(uuid.uuid4()) )
            aids.append( apc.store(objs[-1],self.__deactivate__) )
        
        ids  = []
        for i in xrange(10):
            ids.append( pc.store(str(uuid.uuid4()),self.__deactivate__) )

        pc.assimilateCache(apc)

        for i in xrange(csize):
            self.assertEqual(pc[aids[i]],objs[i])
            
    def testPop(self):
        pc  = self.pc

        csize = 6
        pc.setCacheSize(csize)
        
        ids  = []
        for i in xrange(10):
            if ( i == 8 ):
                aobj = str(uuid.uuid4())
                ids.append( pc.store(aobj,self.__deactivate__) )
            else:
                ids.append( pc.store(str(uuid.uuid4()),self.__deactivate__) )
        
        pc.pop(ids[-1])
        
        obj = pc.retrieve(ids[-2])
        self.assertEqual(obj,aobj)
        

#################################################################
#
class testPIVEpoch(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create a test object.
        self.pe = p.PIVEpoch(1)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__ 
        
    def testMandatoryTime(self):
        self.assertRaises(TypeError,
                          p.PIVEpoch)
        
    def testDefaultUnits(self):
        self.assertEqual(self.pe.units,"NOTSET")
        
    def testDefaultTime_dtype(self):
        self.assertEqual(type(self.pe.time), float)
        
    def testDefaultTime(self):
        self.assertTrue( abs(self.pe.time -1.) < self.eps )
        
    def testPreventDotAttrSet(self):
        try:
            self.pe.attr = 4
            rval = 1
        except AttributeError:
            rval = 0

        self.assertEqual( rval, 0 )

    def testSetTime(self):
        self.pe.setTime(4)
       
        self.assertTrue( abs(self.pe.time -4.) < self.eps )

    def testSetUnits(self):
        self.pe.setUnits("HOUR")
        
        self.assertEqual(self.pe.units,"HOUR")

    def testStoresPIVVarOnly(self):
        a = array([1,2])
        
        self.assertRaises(ValueError,
                          self.pe.addVars,a)

    def testAddVars(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        self.pe.addVars([v1,v2])
        
        self.assertEqual(len(self.pe),2)
        self.assertEqual(self.pe['V1'].name,"V1")
        self.assertEqual(self.pe['V2'].name,"V2")

    def test_eshape(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        self.pe.addVars([v1,v2])

        es = self.pe.eshape
        es = es == array([1,2,2])
        self.assertTrue(es.all())

    def test_nshape(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        self.pe.addVars([v1,v2])

        ns = self.pe.nshape
        ns = ns == array([2,3,3])
        self.assertTrue(ns.all())

    def testImmutability_eshape(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        self.pe.addVars([v1,v2])

        es    = self.pe.eshape
        es[0] = 5
        es    = self.pe.eshape
        es    = es == array([1,2,2])
        self.assertTrue(es.all())

    def testImmutability_nshape(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        self.pe.addVars([v1,v2])

        ns    = self.pe.nshape
        ns[0] = 5
        ns    = self.pe.nshape
        ns    = ns == array([2,3,3])
        self.assertTrue(ns.all())
        
    def testShapeCompatibility(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")
        v3 = p.PIVVar([3,2,2,2],"V3","MM")

        self.pe.addVars([v1,v2])

        self.assertRaises(ValueError,
                          self.pe.addVars,v3)

    def testGetVars(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        self.pe.addVars([v1,v2])

        vars = self.pe.getVars()
        self.assertEqual(len(vars),2)

    def testInitialization(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        e = p.PIVEpoch(2,"SEC",[v2,v1])

        self.assertEqual(e.units,"SEC")

        es = e.eshape
        es = es == array([1,2,2])
        self.assertTrue(es.all())
        
        ns = array(e.nshape)
        ns = ns == array([2,3,3])
        self.assertTrue(ns.all())
        
        vars = e.getVars()
        self.assertEqual(len(vars),2)

    def testBracketAddNameMismatch(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        e = p.PIVEpoch(2,"SEC",[v2,v1])

        v4 = v1 +8.
        v4.setName("V4")
                           
        try:
            e["NEWVAR"] = v4
            rval = 1
        except KeyError:
            rval = 0

        self.assertEqual(rval,0)

    def testBracketAddName(self):
        v1 = p.PIVVar([1,1,2,2],"V1","MM")
        v2 = p.PIVVar([3,2,3,3],"V2","MM",vtype="N")

        e = p.PIVEpoch(2,"SEC",[v2,v1])

        v4 = v1 +8.
        v4.setName("V4")
                           
        e["V4"] = v4

        self.assertEqual(len(e.getVars()),3)
        
    def testCache(self):
        vlst = []
        for i in xrange(10):
            random.seed(i)
            vals = random.rand(9)
            var = p.cpivvar(vals.reshape([1,1,3,3]),"VAR%i" % i)
            vlst.append(var)
        
        pe = p.PIVEpoch(0,vars=vlst,csize=6)
        vlst = []
        
        random.seed(10)
        vals = random.rand(9)
        var  = p.cpivvar(vals.reshape([1,1,3,3]),"VAR10")
        pe.addVars(var)
        
        for i in xrange(10):
            avar = pe['VAR%i' % i]
            
            random.seed(i)
            kvar = random.rand(9)
            random.seed(None)        
            
            tst = avar.reshape(avar.size) -kvar
            self.assertTrue( ( abs(tst) < self.eps ).all() )

    def testCache1(self):
        vlst = []
        for i in xrange(10):
            random.seed(i)
            vals = random.rand(9)
            var = p.cpivvar(vals.reshape([1,1,3,3]),"VAR%i" % i)
            vlst.append(var)
        
        pe = p.PIVEpoch(0,vars=vlst,csize=1)
        vlst = []
        
        random.seed(10)
        vals = random.rand(9)
        var  = p.cpivvar(vals.reshape([1,1,3,3]),"VAR10")
        pe.addVars(var)
        
        for i in xrange(10):
            avar = pe['VAR%i' % i]
            
            random.seed(i)
            kvar = random.rand(9)
            random.seed(None)        
            
            tst = avar.reshape(avar.size) -kvar
            self.assertTrue( ( abs(tst) < self.eps ).all() )
            
    def testVariableRename(self):
        pe = p.PIVEpoch(0,csize=6)
        kvar = None
        for i in xrange(3):
            random.seed(i)
            vals = random.rand(9)
            var = p.cpivvar(vals.reshape([1,1,3,3]),"VAR%i" % i)
            if ( i == 1 ):
                kvar = var
            
            pe.addVars(var)
            
        self.assertEqual(len(pe),3)
        
        var = pe['VAR1']
        var.setName('AVAR')
        
        var = pe['AVAR']
        
        self.assertTrue((var == kvar).all())
        
    def testBadVariableName(self):
        pe = p.PIVEpoch(0,csize=6)
        for i in xrange(3):
            random.seed(i)
            vals = random.rand(9)
            var = p.cpivvar(vals.reshape([1,1,3,3]),"VAR%i" % i)
            pe.addVars(var)
            
        self.assertEqual(len(pe),3)
        
        self.assertRaises(KeyError,pe.__getitem__,'VARX')
        
        
#################################################################
#
class testPIVData(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create a test object.
        self.cellsz = (1.,2,3.)
        self.origin = (4.,5,6.)
        self.pd = p.PIVData(self.cellsz,self.origin,"SOMEDATA")

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__ 
        
    def test_cellsz_dtype(self):
        self.assertEqual(type(self.pd.cellsz[1]),float64)
        
    def test_origin_dtype(self):
        self.assertEqual(type(self.pd.origin[1]),float64)

    def testDescription(self):
        self.assertEqual(self.pd.desc,"SOMEDATA")
        
    def testInitializedQA(self):
        self.assertEqual(len(self.pd.getQA()), 1)
        self.assertEqual(len(self.pd.getQA()[0]), 4)

    def testSetCellsz(self):
        self.pd.setCellsz([2,4,5])
        kc = array([2.,4.,5.])
        tc = abs(self.pd.cellsz -kc)
        tc = (tc < self.eps).all()
        
        self.assertTrue(tc)

    def testSetOrigin(self):
        kc = array([2.,4.,5.])
        kc = kc +1.
        kc = kc.round()
        self.pd.setOrigin(kc)
        tc = abs(self.pd.origin -kc)
        tc = (tc < self.eps).all()

        self.assertTrue(tc)

    def testSetDesc(self):
        self.pd.setDesc("ADESC")
        self.assertEqual(self.pd.desc,"ADESC")

    def testPreventDotAttrSet(self):
        try:
            self.pd.attr = 4
            rval = 1
        except AttributeError:
            rval = 0

        self.assertEqual(rval,0)

    def testEmptyTimes(self):
        self.assertEqual(len(self.pd.getTimes()),0)
        
    def testSetTimeInvalidEpoch(self):
        self.assertRaises(ValueError,
                          self.pd.setTime,1,2.)
        
    def testSetTimesInvalidEpochs(self):
        self.assertRaises(ValueError,
                          self.pd.setTimes,[4.,5.])
        
    def testSetTimeUnitsWhileEmpty(self):
        self.assertRaises(ValueError,
                          self.pd.setTimeUnits,"HOUR")

    def testAddVars(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        self.pd.addVars(0,[v1,v2])
        
        self.assertEqual(len(self.pd[0]), 2)
        self.assertEqual(self.pd[0]["VARB"].name,"VARB")

    def testSkipEpochAddVars(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        self.pd.addVars(0,[v1,v2])

        self.pd.addVars(2,[v1,v2])

        self.assertEqual(len(self.pd[0]), 2)
        self.assertEqual(len(self.pd[1]), 0)
        self.assertEqual(len(self.pd[2]), 2)
        
    def testAppend(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        self.pd.addVars(0,[v1,v2])

        e1 = p.PIVEpoch(4.,"DAYS")
        self.pd.append(e1)
        
        self.assertEqual(self.pd[1].units,"DAYS")

    def testExtend(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        self.pd.addVars(0,[v1,v2])

        e1 = p.PIVEpoch(4.,"DAYS")
        e2 = p.PIVEpoch(1.,"WEEKS")
        self.pd.extend([e1,e2])
        
        self.assertEqual(len(self.pd),3)
        self.assertEqual(self.pd[2].units,"WEEKS")
 
    def testEnforceMonotonicTime(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        self.pd.addVars(0,[v1,v2])

        e1 = p.PIVEpoch(4.,"DAYS")
        e2 = p.PIVEpoch(1.,"WEEKS")
        self.pd.extend([e1,e2])

        self.assertRaises(ValueError,
                          self.pd.dump2ex2,
                          "%s/pivdata-testfile-out" % self.ofp)

    def testSetTimes(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        self.pd.addVars(0,[v1,v2])

        e1 = p.PIVEpoch(4.,"DAYS")
        e2 = p.PIVEpoch(1.,"WEEKS")
        self.pd.extend([e1,e2])

        ktms = array([0,1,2])
        self.pd.setTimes(ktms)
        tms = self.pd.getTimes()
        
        tc = abs( tms -ktms ) < self.eps
        
        self.assertTrue( tc.all() )

    def testInit1DList(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        v3 = v1 +8.
        v3.setName("VARA")
        v4 = v2 +8.
        v4.setName("VARB")
        
        cellsz = (1.,2,3.)
        origin = (4.,5,6.)
        d = p.PIVData(cellsz,origin,"SOMEDATA",vars=[v1,v2])

        self.assertEqual(len(d),1)
        self.assertEqual(len(d[0]),2)

    def testInit2DList(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        v3 = v1 +8.
        v3.setName("VARA")
        v4 = v2 +8.
        v4.setName("VARB")
        
        cellsz = (1.,2,3.)
        origin = (4.,5,6.)
        d = p.PIVData(cellsz,origin,"SOMEDATA",vars=[[v1,v2],[v3,v4]])

        self.assertEqual(len(d),2)
        self.assertEqual(len(d[0]),2)
        self.assertEqual(len(d[1]),2)

    def testWrite(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        v3 = v1 +8.
        v3.setName("VARA")
        v4 = v2 +8.
        v4.setName("VARB")
        
        cellsz = (1.,2,3.)
        origin = (4.,5,6.)
        d = p.PIVData(cellsz,origin,"SOMEDATA",vars=[[v1,v2],[v3,v4]])

        d[1]["VARA"].setUnits(d[0]["VARA"].units)
        d.dump2ex2("%s/pivdata-testfile-out" % self.ofp)
        ld = p.loadpivdata("%s/pivdata-testfile-out.ex2" % self.ofp)

        self.assertEqual(len(ld),             len(d),            "PIVDATA LEN")
        self.assertEqual(len(ld[0]),          len(d[0]),         "EPOCH LEN")
        self.assertEqual(ld.desc,             d.desc,            "DESC")
        self.assertEqual(ld[0].units,         d[0].units,        "TIME UNITS")
        self.assertEqual(ld[0].time,          d[0].time,         "TIME")
        self.assertEqual(ld[1]["VARA"].name,  d[1]["VARA"].name, "VAR NAME")
        self.assertEqual(ld[1]["VARA"].vtype, d[1]["VARA"].vtype,"VAR VTYPE")
        self.assertEqual(ld[1]["VARA"].units, d[1]["VARA"].units,"VAR UNITS")

        self.assertEqual(ld[1]["VARA"].m_max_str_len,
                         d[1]["VARA"].m_max_str_len,"MAX_STR_LEN")
        
        self.assertTrue( (ld[0]["VARA"] == d[0]["VARA"]).all(), "DATA" )
        self.assertTrue( (ld.cellsz == d.cellsz).all(),         "CELLSZ" )
        self.assertTrue( (ld.origin == d.origin).all(),         "ORIGIN" )

    def testLoad(self):
        v1 = p.PIVVar([1,1,2,2],"VARA","MM")
        v2 = v1 +2.
        v2.setName("VARB")
        v3 = v1 +8.
        v3.setName("VARA")
        v4 = v2 +8.
        v4.setName("VARB")
        
        cellsz = (1.,2,3.)
        origin = (4.,5,6.)
        d = p.PIVData(cellsz,origin,"SOMEDATA",vars=[[v1,v2],[v3,v4]])

        d[1]["VARA"].setUnits(d[0]["VARA"].units)
        ld = p.loadpivdata("%s/pivdata-nonewprops-known.ex2" % self.dfp)

        self.assertEqual(len(ld),             len(d),            "PIVDATA LEN")
        self.assertEqual(len(ld[0]),          len(d[0]),         "EPOCH LEN")
        self.assertEqual(ld.desc,             d.desc,            "DESC")
        self.assertEqual(ld[0].units,         d[0].units,        "TIME UNITS")
        self.assertEqual(ld[0].time,          d[0].time,         "TIME")
        self.assertEqual(ld[1]["VARA"].name,  d[1]["VARA"].name, "VAR NAME")
        self.assertEqual(ld[1]["VARA"].vtype, d[1]["VARA"].vtype,"VAR VTYPE")
        self.assertEqual(ld[1]["VARA"].units, d[1]["VARA"].units,"VAR UNITS")

        self.assertEqual(ld[1]["VARA"].m_max_str_len,
                         d[1]["VARA"].m_max_str_len,"MAX_STR_LEN")
        
        self.assertTrue( (ld[0]["VARA"] == d[0]["VARA"]).all(), "DATA" )
        self.assertTrue( (ld.cellsz == d.cellsz).all(),         "CELLSZ" )
        self.assertTrue( (ld.origin == d.origin).all(),         "ORIGIN" )
        
    def testSimpleSetCache(self):
        pe1 = p.PIVEpoch(1)
        pe2 = p.PIVEpoch(2)
        pe3 = p.PIVEpoch(3)
        
        pd = self.pd
        
        self.assertNotEqual(id(pe1.getCache()),id(pe2.getCache()))
        self.assertNotEqual(id(pe1.getCache()),id(pd.getCache()))
        self.assertNotEqual(id(pe3.getCache()),id(pd.getCache()))
        
        pd.extend([pe1,pe2])
        
        self.assertEqual(id(pe1.getCache()),id(pd.getCache()))
        self.assertEqual(id(pe2.getCache()),id(pd.getCache()))
        
        pd.append(pe3)
        self.assertEqual(id(pe3.getCache()),id(pd.getCache()))

        acache = p.PIVCache()
        pd.setCache(acache)
        self.assertEqual(id(pe1.getCache()),id(pd.getCache()))
        self.assertEqual(id(pe2.getCache()),id(pd.getCache()))
        self.assertEqual(id(pe3.getCache()),id(pd.getCache()))

    def testSetCache(self):
        vlst = []
        epcl = []
        for e in xrange(3):
            epcl.append(p.PIVEpoch(e,csize=6))
        
            for i in xrange(10):
                random.seed(i)
                vals = random.rand(9)
                var = p.cpivvar(vals.reshape([1,1,3,3]),"VAR%i" % i)
                
                epcl[e].addVars(var)
                vlst.append(var.copy())
        
        # Should have deactivated some variables.
        for e in xrange(3):        
            self.assertTrue( epcl[e].getCache().ccsize < 10 )
        
        pd = self.pd
        pd.getCache().setCacheSize(6)
        pd.extend(epcl)
        for e in xrange(3):
            for i in xrange(10):
                self.assertTrue((pd[e]["VAR%i" % i] == vlst[e*10 +i]).all())
        
        pd.setCache(p.PIVCache(6))
        self.assertTrue( pd.getCache().ccsize <= 6 )
        for e in xrange(3):
            for i in xrange(10):
                self.assertTrue((pd[e]["VAR%i" % i] == vlst[e*10 +i]).all())

        pd.getCache().setCacheSize(1)
        for e in xrange(3):
            for i in xrange(10):
                self.assertTrue((pd[e]["VAR%i" % i] == vlst[e*10 +i]).all())

        self.assertTrue( pd.getCache().ccsize <= 1 )

            
def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( testPIVVar   ) )
    suite.addTest( unittest.makeSuite( testPIVCache ) )
    suite.addTest( unittest.makeSuite( testPIVEpoch ) )
    suite.addTest( unittest.makeSuite( testPIVData  ) )
    
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())



