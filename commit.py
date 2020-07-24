#!/usr/bin/env python
"""
Filename:  commit.py
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
    Updates build_revision prior to svn commit and then executes svn.
"""

import os, commands, sys, re

rfpath = "lib/spivet/spivetrev.py"

rev = commands.getoutput("svnversion")
rev = rev.split(":")
rev = rev.pop()
rev = rev.strip("MS")

nrev = int(rev) +1

print "Current Rev: %s New Rev: %s" % (rev,nrev)
print "Continue (y/N)"
line = sys.stdin.readline()

if (line.strip().lower() != "y"):
    print "Exiting"
    sys.exit()

fh   = open(rfpath,'r')
data = fh.read()
fh.close()

nrevstr = "spivet_bld_rev = %i" % nrev
rex     = re.compile(r"spivet_bld_rev[\s]*=[\s]*[\d]+")
data    = re.sub(rex,nrevstr,data)

fh = open(rfpath,'w')
fh.write(data)
fh.close()

print "Be sure to run install after commit."


