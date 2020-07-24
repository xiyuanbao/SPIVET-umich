"""
Filename:  exodusII_test.py
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
    Performs testing on exodusII functions.
"""

from spivet.pivlib import exodusII as ex2
from numpy import *
import commands
import os
from os import path

dfp = "data"

ofp = "test-output"
if ( not path.exists(ofp) ):
    os.mkdir(ofp)

tfname = "%s/test.ex2" % ofp
kfname = "%s/known.ex2" % dfp

ncdump_ex = "ncdump"

ttcnt = 0  # Total test count.
ftcnt = 0  # Failed test count.

def excheck(idstr, rval):
    global ttcnt
    global ftcnt
    
    ttcnt = ttcnt +1
    if ( isinstance(rval,int) ):
        prval = rval
        if ( rval < 0 ):
            ss    = "FAILED <<"
            ftcnt = ftcnt +1
        else:
            ss = "PASSED"
    else:
        ss = "PASSED"
        prval = 0

    print "%-30s%5i : %s" % (idstr, prval, ss)

    return rval

def check(idstr, xval, rval):
    global ttcnt
    global ftcnt
    
    ttcnt = ttcnt +1
    if ( rval != xval ):
        ss    = "FAILED (EXPECTED %s, GOT %s)" % (str(xval),str(rval))
        ftcnt = ftcnt +1
    else:
        ss = "PASSED"

    print "%-34s- : %s" % (idstr, ss)

    return rval


fh = ex2.ex_create(tfname,ex2.exc['EX_CLOBBER'],8,8)
excheck("ex_create", 
      fh[0])

#####

excheck("ex_close", 
      ex2.ex_close(fh))

#####

l = ex2.ex_open(tfname,ex2.exc["EX_READ"],8)
excheck("ex_open", 
      l[0])

#####

excheck("ex_update", 
      ex2.ex_update(l))

#####

excheck("ex_close", 
      ex2.ex_close(l))

#####

fh = ex2.ex_create(tfname,ex2.exc['EX_CLOBBER'],8,8)
excheck("ex_create", 
      fh[0])

#####

excheck("ex_put_init", 
      ex2.ex_put_init(fh,"This is a test file",3,8,1,1,1,1))

ex2.ex_update(fh)
rval = excheck("ex_get_init",
               ex2.ex_get_init(fh))
check("ex_get_init - title", "This is a test file", rval[0])
check("ex_get_init - ndim",3,rval[1])
check("ex_get_init - nnodes",8,rval[2])
check("ex_get_init - nelem",1,rval[3])
check("ex_get_init - neblk",1,rval[4])
check("ex_get_init - nns",1,rval[5])
check("ex_get_init - nss",1,rval[6])

#####

rval = excheck("ex_inquire",
               ex2.ex_inquire(fh,ex2.exc["EX_INQ_NODES"]))

check("ex_inquire - nnodes",8,rval[0])

#####

qa = [["ABC","DEF","GHI","JKL"],["MNO","PQR","STU","VWX"]]
excheck("ex_put_qa", 
      ex2.ex_put_qa(fh,2,qa))

ex2.ex_update(fh)
rval = excheck("ex_get_qa",ex2.ex_get_qa(fh))
oqa = array(qa)
nqa = array(rval)
check("ex_get_qa - recs",True,( oqa == nqa ).all())

#####

info = ["INFO A", "INFO B", "INFO C"]
excheck("ex_put_info", 
      ex2.ex_put_info(fh,3,info))

ex2.ex_update(fh)
rval = excheck("ex_get_info",ex2.ex_get_info(fh))
oi = array(info)
ni = array(rval)
check("ex_get_info - recs",True,(oi == ni).all())

#####

xnc = [0.1,1.,0.,1.,0.,1.,0.,1.]
ync = [0.,0.,1.,1.,0.,0.,1.,1.]
znc = [0.,0.,0.,0.,1.,1.,1.,1.]
excheck("ex_put_coord",
      ex2.ex_put_coord(fh,xnc,ync,znc))

ex2.ex_update(fh)
rval = excheck("ex_get_coord",ex2.ex_get_coord(fh))
oxnc = array(xnc)
oync = array(ync)
oznc = array(znc)
check("ex_get_coord - xnc",True,(oxnc == rval[0]).all())
check("ex_get_coord - ync",True,(oync == rval[1]).all())
check("ex_get_coord - znc",True,(oznc == rval[2]).all())

#####

cn = ["X", "Y", "Z"]
excheck("ex_put_coord_names",
      ex2.ex_put_coord_names(fh,cn))

ex2.ex_update(fh)
rval = excheck("ex_get_coord_names",ex2.ex_get_coord_names(fh))
ocn = array(cn)
ncn = array(rval)
check("ex_get_coord_names - cn",True,(ocn == ncn).all())

#####

nnmap = [10,20,30,40,50,60,70,80]
excheck("ex_put_node_num_map",
      ex2.ex_put_node_num_map(fh,nnmap))

ex2.ex_update(fh)
rval = excheck("ex_get_node_num_map",ex2.ex_get_node_num_map(fh))
onnmap = array(nnmap)
nnnmap = array(rval)
check("ex_get_node_num_map - nmap",True,(onnmap == nnnmap).all())

#####

enmap = [100]
excheck("ex_put_elem_num_map",
      ex2.ex_put_elem_num_map(fh,enmap))

ex2.ex_update(fh)
rval = excheck("ex_get_elem_num_map",ex2.ex_get_elem_num_map(fh))
oenmap = array(enmap)
nenmap = array(rval)
check("ex_get_elem_num_map - enmap",True,(oenmap == nenmap).all())

#####

eomap = [1]
excheck("ex_put_map",
      ex2.ex_put_map(fh,eomap))

ex2.ex_update(fh)
rval = excheck("ex_get_map",ex2.ex_get_map(fh))
oeomap = array(eomap)
neomap = array(rval)
check("ex_get_map - map",True,(oeomap == neomap).all())

#####

excheck("ex_put_elem_block",
      ex2.ex_put_elem_block(fh,1,"HEX",1,8,3))

ex2.ex_update(fh)
rval = excheck("ex_get_elem_block",ex2.ex_get_elem_block(fh,1))
check("ex_get_elem_block - etype","HEX",rval[0])
check("ex_get_elem_block - nelem",1,rval[1])
check("ex_get_elem_block - nnpe",8,rval[2])
check("ex_get_elem_block - nattr",3,rval[3])

#####

rval = excheck("ex_get_elem_blk_ids",ex2.ex_get_elem_blk_ids(fh))
check("ex_get_elem_blk_ids - len",1,rval.size)
check("ex_get_elem_blk_ids - id",1,rval[0])

#####

conn = [1,2,4,3,5,6,8,7]
excheck("ex_put_elem_conn",
      ex2.ex_put_elem_conn(fh,1,conn))

ex2.ex_update(fh)
rval = excheck("ex_get_elem_conn",ex2.ex_get_elem_conn(fh,1))
oconn = array(conn)
nconn = array(rval)
check("ex_get_elem_conn - conn",True,(oconn == nconn).all())

#####

eattr = [1.0,2.0,3.2]
excheck("ex_put_elem_attr",
      ex2.ex_put_elem_attr(fh,1,eattr))

ex2.ex_update(fh)
rval = excheck("ex_get_elem_attr",ex2.ex_get_elem_attr(fh,1))
oeattr = array(eattr)
neattr = array(rval)
check("ex_get_elem_attr - attr",True,(oeattr == neattr).all())

#####

excheck("ex_put_node_set_param",
      ex2.ex_put_node_set_param(fh,1,2,2))

ex2.ex_update(fh)
rval = excheck("ex_get_node_set_param",ex2.ex_get_node_set_param(fh,1))
check("ex_get_node_set_param - nninset",2,rval[0])
check("ex_get_node_set_param - ndinset",2,rval[1])

#####

excheck("ex_put_node_set",
      ex2.ex_put_node_set(fh,1,[1,4]))

ex2.ex_update(fh)
rval = excheck("ex_get_node_set",ex2.ex_get_node_set(fh,1))
ons = array([1,4])
nns = array(rval)
check("ex_get_node_set - nodeset",True,(ons == nns).all())

#####

excheck("ex_put_node_set_dist_fact",
      ex2.ex_put_node_set_dist_fact(fh,1,[1.2,3.4]))

ex2.ex_update(fh)
rval = excheck("ex_get_node_set_dist_fact",
               ex2.ex_get_node_set_dist_fact(fh,1))
odf = array([1.2,3.4])
ndf = array(rval)
check("ex_get_node_set_dist_fact - df",True,(odf == ndf).all())

#####

rval = excheck("ex_get_node_set_ids",ex2.ex_get_node_set_ids(fh))
check("ex_get_node_set_ids - len",1,rval.size)
check("ex_get_node_set_ids - id",1,rval[0])

#####

excheck("ex_put_side_set_param",
      ex2.ex_put_side_set_param(fh,1,1,0))

ex2.ex_update(fh)
rval = excheck("ex_get_side_set_param",ex2.ex_get_side_set_param(fh,1))
check("ex_get_side_set_param - nsinset",1,rval[0])
check("ex_get_side_set_param - ndinset",0,rval[1])

#####

excheck("ex_put_side_set",
      ex2.ex_put_side_set(fh,1,[1],[3]))

ex2.ex_update(fh)
rval = excheck("ex_get_side_set",ex2.ex_get_side_set(fh,1))
check("ex_get_side_set - elist",1,rval[0][0])
check("ex_get_side_set - slist",3,rval[1][0])

#####

excheck("ex_put_prop_names",
      ex2.ex_put_prop_names(fh,ex2.exc['EX_ELEM_BLOCK'],1,["PROP A"]))

ex2.ex_update(fh)
rval = excheck("ex_get_prop_names",
              ex2.ex_get_prop_names(fh,ex2.exc['EX_ELEM_BLOCK']))

check("ex_get_prop_names - name","PROP A",rval[1])

#####

excheck("ex_put_prop",
      ex2.ex_put_prop(fh,ex2.exc['EX_ELEM_BLOCK'],1,"PROP A",4))

ex2.ex_update(fh)
rval = excheck("ex_get_prop",
              ex2.ex_get_prop(fh,ex2.exc['EX_ELEM_BLOCK'],1,"PROP A"))
check("ex_get_prop - prop",4,rval)

#####

excheck("ex_put_var_param - elem",
      ex2.ex_put_var_param(fh,"e",2))

ex2.ex_update(fh)
rval = excheck("ex_get_var_param - elem",ex2.ex_get_var_param(fh,"e"))
check("ex_get_var_param - elem(vars)",2,rval)

#####

excheck("ex_put_var_names - elem",
      ex2.ex_put_var_names(fh,"e",2,["EVAR1","EVAR2"]))

ex2.ex_update(fh)
rval = excheck("ex_get_var_names - elem",ex2.ex_get_var_names(fh,"e"))
oname = array(["EVAR1","EVAR2"])
nname = array(rval)
check("ex_get_var_names - elem(names)",True,(oname == nname).all())

#####

excheck("ex_put_var_param - glob",
      ex2.ex_put_var_param(fh,"g",1))

ex2.ex_update(fh)
rval = excheck("ex_get_var_param - glob",
              ex2.ex_get_var_param(fh,"g"))
check("ex_get_var_param - glob(param)",1,rval)

#####

excheck("ex_put_var_names - glob",
      ex2.ex_put_var_names(fh,"g",1,["GVAR"]))

ex2.ex_update(fh)
rval = excheck("ex_get_var_names - glob",
              ex2.ex_get_var_names(fh,"g"))
check("ex_get_var_names - glob(name)","GVAR",rval[0])

#####

excheck("ex_put_var_param - node",
      ex2.ex_put_var_param(fh,"n",2))
    
ex2.ex_update(fh)
rval = excheck("ex_get_var_param - node",ex2.ex_get_var_param(fh,"n"))
check("ex_get_var_param - node(param)",2,rval)

#####

excheck("ex_put_var_names - node",
      ex2.ex_put_var_names(fh,"n",2,["VAR1","VAR2"]))

ex2.ex_update(fh)
rval = excheck("ex_get_var_names - node",ex2.ex_get_var_names(fh,"n"))
oname = array(["VAR1","VAR2"])
nname = array(rval)
check("ex_get_var_names - node(name)",True,(oname == nname).all())

#####

excheck("ex_put_time",
      ex2.ex_put_time(fh,1,0.3))

ex2.ex_update(fh)
rval = excheck("ex_get_time",ex2.ex_get_time(fh,1))
check("ex_get_time - time",0.3,rval)

#####

excheck("ex_put_elem_var_tab",
      ex2.ex_put_elem_var_tab(fh,1,2,[1,1]))

ex2.ex_update(fh)
rval = excheck("ex_get_elem_var_tab",ex2.ex_get_elem_var_tab(fh))
otab = array([1,1])
ntab = array(rval)
check("ex_get_elem_var_tab - tab",True,(otab == ntab).all())

#####

excheck("ex_put_elem_var",
      ex2.ex_put_elem_var(fh,1,1,1,1,[2.4]))

ex2.ex_update(fh)
rval = excheck("ex_get_elem_var",ex2.ex_get_elem_var(fh,1,1,1))
check("ex_get_elem_var - var1",2.4,rval[0])

#####

excheck("ex_put_elem_var",
      ex2.ex_put_elem_var(fh,1,2,1,1,[2.7]))

ex2.ex_update(fh)
rval = excheck("ex_get_elem_var",ex2.ex_get_elem_var(fh,1,2,1))
check("ex_get_elem_var - var2",2.7,rval[0])

#####

excheck("ex_put_glob_vars",
      ex2.ex_put_glob_vars(fh,1,1,[12.4]))

ex2.ex_update(fh)
rval = excheck("ex_get_glob_vars",ex2.ex_get_glob_vars(fh,1))
check("ex_get_glob_vars - var",12.4,rval[0])

#####

var = [0.2,1.2,2.2,3.2,4.2,5.2,6.2,7.2]
excheck("ex_put_nodal_var",
      ex2.ex_put_nodal_var(fh,1,1,8,var))

ex2.ex_update(fh)
rval = excheck("ex_get_nodal_var",ex2.ex_get_nodal_var(fh,1,1))
check("ex_get_nodal_var - var1",True,(array(var) == rval).all())

#####

var = [10.2,11.2,12.2,13.2,14.2,15.2,16.2,17.2]
excheck("ex_put_nodal_var",
      ex2.ex_put_nodal_var(fh,1,2,8,var))

ex2.ex_update(fh)
rval = excheck("ex_get_nodal_var",ex2.ex_get_nodal_var(fh,1,2))
check("ex_get_nodal_var - var2",True,(array(var) == rval).all())

#####

excheck("ex_close",
      ex2.ex_close(fh))

#####
    
cmd = "%s %s" % (ncdump_ex,kfname)
kft = commands.getoutput(cmd)
ndx = kft.find("{")
kft = kft[(ndx+1):len(kft)]

cmd = "%s %s" % (ncdump_ex,tfname)
tft = commands.getoutput(cmd)
ndx = tft.find("{")
tft = tft[(ndx+1):len(tft)]

ocv = 0
if ( kft != tft ):
    ocv = -1
    print "ERROR: Library produced unexpected output."

excheck("output comparison", ocv)
