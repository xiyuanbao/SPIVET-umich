"""
Filename:  tlctccdrv.py
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
  Driver file for thermochromic calibration.
"""

from spivet import pivlib, tlclib
from numpy import *
from scipy import stats
import pylab
import os, sys

# >>>>> ANALYSIS SETUP <<<<<
s0zcoord  = 0.                            # z-coordinate of sequence 0. [mm]
zcellsz   = -52.2                         # z Cell size [mm].
porder    = [1,1,7]                       # Polynomial order for calibration.
lmda      = 0.002                         # Smoothing parameter.
btcpdata  = 'TLCCAL-09282008_TC_OUT_CAM1' # Calibration points file.

# >>>>> END USER MODIFIABLE CODE <<<<<

# Turn on interactive pylab.
pylab.ion()

# Load the calibration points.
tcpa = tlclib.loadtcpa(btcpdata)

# Calibrate.
[tlccal,err] = tlclib.calibrate(tcpa,porder,lmda)
tlclib.savetlccal(tlccal,"TLCCAL_CAM1")

# Form some diagnostic plots.  First need to build dictionaries that
# contain a list of tcpa points corresponding to each value of temperature
# z position, theta, and hue.
hmin = tcpa[:,3].min()
hmax = tcpa[:,3].max()
dh   = (hmax -hmin)/9.
ha   = arange(10)*dh +hmin
hca  = ha[0:9] +dh/2.       # Bin centers.

tmpnd = {}
znd   = {}
thnd  = {}
hnl   = []
for i in range( tcpa.shape[0] ):
    tmp = str(tcpa[i,0])
    z   = str(tcpa[i,1])
    th  = str(round(180.*tcpa[i,2]/pi,0))
    hue = tcpa[i,3]

    if ( not tmpnd.has_key(tmp) ):
        tmpnd[tmp] = [i]
    else:
        tmpnd[tmp].append(i)

    if ( not znd.has_key(z) ):
        znd[z] = [i]
    else:
        znd[z].append(i)

    if ( not thnd.has_key(th) ):
        thnd[th] = [i]
    else:
        thnd[th].append(i)

    hndx = argmin( abs( hue -hca ) )
    if ( hndx >= len(hnl) ):
        for k in range(len(hnl),hndx+1):
            hnl.append([])      

    hnl[hndx].append(i)

# Construct plots.

# Build aggregate error plot.
pylab.figure()
pylab.boxplot(err)
ax = pylab.gcf().gca()
ax.set_xticks([])
pylab.ylabel("Err [degC]")
pylab.title("Calibration Error")
pylab.savefig("TLCCAL-ERR.png")
print "Calibration error [degC] at percentile (5%%, 95%%): (%g, %g)" %\
    (stats.scoreatpercentile(err,5),stats.scoreatpercentile(err,95))

# Build error vs. temp plot.
pylab.figure()
tmpel = []
skeys = array(tmpnd.keys())
ordr  = skeys.astype(float).argsort()
skeys = skeys[ordr]
for key in skeys:
    ndxl = tmpnd[key]
    tmpel.append( err[ndxl] )

pylab.boxplot(tmpel)
ax = pylab.gcf().gca()
ax.set_xticklabels(skeys)
pylab.xlabel("Temperature [degC]")
pylab.ylabel("Err [degC]")
pylab.title("Calibration Error vs. Temperature")
pylab.savefig("TLCCAL-ERR_VS_TEMP.png")

# Build computed vs. measured temp plot.
pylab.figure()
tmpel = []
skeys = array(tmpnd.keys())
ordr  = skeys.astype(float).argsort()
skeys = skeys[ordr]
for key in skeys:
    ndxl = tmpnd[key]
    tmpel.append( tcpa[ndxl,0] +err[ndxl] )

pylab.boxplot(tmpel)
ax = pylab.gcf().gca()
ax.set_xticklabels(skeys)
pylab.xlabel("Measured Temperature [degC]")
pylab.ylabel("Computed Temperature [degC]")
pylab.title("Computed Temperature vs. Measured Temperature")
pylab.savefig("TLCCAL-CTEMP_VS_MTEMP.png")

# Build error vs. z plot.
pylab.figure()
zel = []
skeys = array(znd.keys())
ordr  = skeys.astype(float).argsort()
skeys = skeys[ordr]
for key in skeys:
    ndxl = znd[key]
    zel.append( err[ndxl] )

pylab.boxplot(zel)
ax = pylab.gcf().gca()
ax.set_xticklabels(skeys)
pylab.xlabel("Z [mm]")
pylab.ylabel("Err [degC]")
pylab.title("Calibration Error vs. Z-Coordinate")
pylab.savefig("TLCCAL-ERR_VS_Z.png")

# Build error vs. hue plot.
pylab.figure()
hel = []
skeys = hca.round(2).astype('S4')
for l in hnl:
    hel.append( err[l] )

pylab.boxplot(hel)
ax = pylab.gcf().gca()
ax.set_xticklabels(skeys)
pylab.xlabel("Hue")
pylab.ylabel("Err [degC]")
pylab.title("Calibration Error vs. Hue")
pylab.savefig("TLCCAL-ERR_VS_HUE.png")

# Build error vs. theta plot.
pylab.figure()
thel = []
skeys = array(thnd.keys())
ordr  = skeys.astype(float).argsort()
skeys = skeys[ordr]
for key in skeys:
    ndxl = thnd[key]
    thel.append( err[ndxl] )

pylab.boxplot(thel)
ax = pylab.gcf().gca()
ax.set_xticklabels(skeys)
pylab.xlabel("Theta [deg]")
pylab.ylabel("Err [degC]")
pylab.title("Calibration Error vs. Theta")
pylab.savefig("TLCCAL-ERR_VS_THETA.png")

# Build sample curves.
hmin = tcpa[:,3].min()
hmax = tcpa[:,3].max()
dh   = (hmax -hmin)/99.
ha   = arange(100)*dh +hmin

thmin = tcpa[:,2].min()
thmax = tcpa[:,2].max()
dth   = (thmax -thmin)/4.
tha   = arange(5)*dth +thmin

data = zeros((100,3),dtype=float)
data[:,0] = -105.
data[:,2] = ha
pylab.figure()
for i in range(5):
    data[:,1]     = tha[i]
    [tmpa,tcinac] = tlclib.hue2tmp(data,tlccal)
    
    pylab.plot(ha,tmpa,
               label=r"$\theta = %3.1f^{\circ}$" % round(180.*tha[i]/pi,1),
               linewidth=2)

pylab.legend(loc='upper left')
pylab.title(r"Thermochromic Calibration P(%i,%i,%i), $\lambda = $%g" % 
            (porder[0],porder[1],porder[2],lmda) )
pylab.xlabel("Hue")
pylab.ylabel(r"Temperature [$^{\circ}$C]")
pylab.savefig("TLCCAL-EXAMPLE-CURVES.png",dpi=600)

# Build sample curves with points.
hmin = tcpa[:,3].min()
hmax = tcpa[:,3].max()
dh   = (hmax -hmin)/99.
ha   = arange(100)*dh +hmin

thmin = tcpa[:,2].min()
thmax = tcpa[:,2].max()
dth   = (thmax -thmin)/4.
tha   = arange(5)*dth +thmin

rad  = []
msk  = abs(tcpa[:,1] +105.) < 1.E-6
mta  = tcpa[msk,:]
 
for th in tha:
    rad.append( abs(mta[:,2] -th) )
rad  = array(rad).transpose()
bin  = rad.argsort(1)[:,0]
tht  = []
thh  = []
for i in range( len(tha) ):
    msk = bin == i
    tht.append( mta[msk,0] )
    thh.append( mta[msk,3] )

data = zeros((100,3),dtype=float)
data[:,0] = -105.
data[:,2] = ha
pylab.figure()
colors = ['b','g','r','c','m']
for i in range(5):
    data[:,1]     = tha[i]
    [tmpa,tcinac] = tlclib.hue2tmp(data,tlccal)
    
    pylab.plot(ha,tmpa,label=r"$\theta = %3.1f^{\circ}$" % round(180.*tha[i]/pi,1),
               linewidth=2)
    pylab.plot(thh[i],tht[i],"%s+" % (colors[i]),label="_nolegend_")

pylab.legend(loc='upper left')
pylab.title(r"Thermochromic Calibration P(%i,%i,%i), $\lambda = $%g" % 
            (porder[0],porder[1],porder[2],lmda) )
ax = pylab.gca()
ax.yaxis.set_minor_locator(pylab.MultipleLocator(0.1))
pylab.xlabel("Hue")
pylab.ylabel(r"Temperature [$^{\circ}$C]")
pylab.savefig("TLCCAL-EXAMPLE-CURVES-WPTS.png",dpi=600)

# Build single hue graphic.
pylab.figure()
h    = 0.63
dth  = (thmax -thmin)/99.
tha  = arange(100)*dth +thmin

data      = zeros((100,3),dtype=float)
data[:,0] = 105.
data[:,1] = tha
data[:,2] = h

[temp,inac] = tlclib.hue2tmp(data,tlccal)

temp = repeat(temp,100)
temp = temp.reshape((100,100)).transpose()

pylab.imshow(temp,interpolation='nearest')
pylab.setp(pylab.gca(),xticks=[],yticks=[])
c = pylab.colorbar(shrink=0.75)
c.set_label("Temp [degC]")
pylab.title("Computed Temperature for Hue=0.63 (z = 105 mm)")
pylab.savefig("TLCCAL-CONSTANT-HUE.png")
