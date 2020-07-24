/*
Filename:  flotracec.c
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
  These functions are for heavily numeric, tracer advection
  computations that are very slow if done through Python
  directly.

*/

#include "Python.h"
#include "numpy/arrayobject.h"
#include "svcismat.h"

#include <math.h>
#include <stdlib.h>
#include <float.h>

//
// WORKER FUNCTION PROTOTYPES
//
static void cleanup_po(int nvars, ...);
//~

/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   svinterp(var, crds)
//
// Returns ivar.
//
// This routine interpolates var at the coordinates specified by
// crds using trilinear interpolation.  NOTE:  Coordinates
// must be expressed in terms of indices (ie, not mm).
//
// If a coordinate is outside the perimeter of var, the interpolated
// value will be set to that of the cell immediately beneath the
// coordinate.  That is, svinterp implements edge extension for
// coordinates lying beyond the perimeter.
//
static PyObject *flotracec_svinterp(PyObject *self, PyObject *args) {

  PyObject      *ovar, *ocrds;
  PyArrayObject *var, *crds, *ivar;

  double   wt[8], ndx[3], shft[3], rshft[3], mxndx[3], eps, vval;
  npy_intp pndx[8], indx[3], ip1[3], ncells[3], nvdim;
  npy_intp varco, crdo, ivo, nvcmp, ncrds, c, i, j;
  npy_intp ivshape[2];

  double   *ptr_var, *ptr_crds, *ptr_ivar;
  npy_intp *vshape, *cshape;

  if (!PyArg_ParseTuple(
                        args,
                        "OO:svinterp",
                        &ovar,&ocrds
                        )
      )
    Py_RETURN_NONE;

  var  = (PyArrayObject *)PyArray_ContiguousFromAny(ovar,PyArray_DOUBLE,3,4);
  crds = (PyArrayObject *)PyArray_ContiguousFromAny(ocrds,PyArray_DOUBLE,2,2);

  if ( ( var == NULL ) ||
       ( crds == NULL ) ) {
    cleanup_po(2,var,crds);
    Py_RETURN_NONE;
  }

  // Initialization.
  nvdim  = PyArray_NDIM(var);
  vshape = PyArray_DIMS(var);
  cshape = PyArray_DIMS(crds);

  ncrds = cshape[0];

  if ( nvdim == 3 ) {
    ivshape[0] = ncrds;

    for (i = 0; i < 3; i++)
      ncells[i] = vshape[i];

    varco = vshape[0]*vshape[1]*vshape[2];

    nvcmp = 1;
  }
  else {
    ivshape[0] = ncrds;
    ivshape[1] = vshape[0];

    for (i = 0; i < 3; i++)
      ncells[i] = vshape[i +1];

    varco = vshape[1]*vshape[2]*vshape[3];

    nvcmp = vshape[0];
  }

  ivar = (PyArrayObject *)PyArray_EMPTY(nvdim-2,ivshape,PyArray_DOUBLE,
                                        PyArray_CORDER);
  if ( ivar == NULL ) {
    cleanup_po(3,ivar,var,crds);
    Py_RETURN_NONE;
  }

  ptr_var  = (double *)PyArray_DATA(var);
  ptr_crds = (double *)PyArray_DATA(crds);
  ptr_ivar = (double *)PyArray_DATA(ivar);

  eps = 2.*DBL_EPSILON;

  for ( i = 0; i < 3; i++)
    mxndx[i] = fmax(ncells[i]*(1. -eps) -1., eps);

  // Perform the interpolation.
  for (i = 0; i < ncrds; i++) {
    // Find the neighboring cells and shifts for the coordinate.
    crdo = 3*i;
    ivo  = nvcmp*i;
    for (j = 0; j < 3; j++) {
      ndx[j] = ptr_crds[crdo +j];
      ndx[j] = fmax(ndx[j],eps);
      ndx[j] = fmin(ndx[j],mxndx[j]);

      indx[j] = (npy_intp)floor(ndx[j]);
      ip1[j]  = indx[j] +1;
      ip1[j]  = (ip1[j] >= ncells[j] ? ncells[j] -1 : ip1[j]);

      shft[j]  = ndx[j] -indx[j];
      rshft[j] = 1. -shft[j];
    }

    // Compute the weights.
    wt[0] = rshft[0]*rshft[1]*rshft[2];
    wt[1] = rshft[0]*rshft[1]* shft[2];
    wt[2] = rshft[0]* shft[1]* shft[2];
    wt[3] = rshft[0]* shft[1]*rshft[2];
    wt[4] =  shft[0]*rshft[1]*rshft[2];
    wt[5] =  shft[0]*rshft[1]* shft[2];
    wt[6] =  shft[0]* shft[1]* shft[2];
    wt[7] =  shft[0]* shft[1]*rshft[2];

    // Compute the ptr_var indices corresponding to the weights.
    pndx[0] = indx[0]*ncells[1]*ncells[2] +indx[1]*ncells[2] +indx[2];
    pndx[1] = indx[0]*ncells[1]*ncells[2] +indx[1]*ncells[2] +ip1[2];
    pndx[2] = indx[0]*ncells[1]*ncells[2]  +ip1[1]*ncells[2] +ip1[2];
    pndx[3] = indx[0]*ncells[1]*ncells[2]  +ip1[1]*ncells[2] +indx[2];
    pndx[4] =  ip1[0]*ncells[1]*ncells[2] +indx[1]*ncells[2] +indx[2];
    pndx[5] =  ip1[0]*ncells[1]*ncells[2] +indx[1]*ncells[2] +ip1[2];
    pndx[6] =  ip1[0]*ncells[1]*ncells[2]  +ip1[1]*ncells[2] +ip1[2];
    pndx[7] =  ip1[0]*ncells[1]*ncells[2]  +ip1[1]*ncells[2] +indx[2];

    // Interpolate the variable components.
    for (c = 0; c < nvcmp; c++) {
      vval = 0.;
      for (j = 0; j < 8; j++)
        vval = vval +wt[j]*ptr_var[c*varco +pndx[j]];

      ptr_ivar[ivo +c] = vval;
    }
  }

  cleanup_po(2,var,crds);
  return PyArray_Return(ivar);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   svcismat()
//
// Returns smat, a 64x64 numpy array.
//
// This routine returns the inverse tricubic system matrix (matrix
// inv(B) in Lekien:2005).  svcismat simply packages the data in
// svcismat.h as a numpy array.
//
static PyObject *flotracec_svcismat(PyObject *self, PyObject *args) {
    PyArrayObject *smat;

    double *ptr_smat;

    npy_intp shape[2] = {64,64};

    npy_intp i;

    smat = (PyArrayObject *)PyArray_EMPTY(2,shape,PyArray_DOUBLE,
                                          PyArray_CORDER);

    if ( smat == NULL )
        Py_RETURN_NONE;

    ptr_smat = (double *)PyArray_DATA(smat);

    for ( i=0; i<4096; i++ )
        ptr_smat[i] = svcismat[i];

    return PyArray_Return(smat);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   svcinterp(crds, coeff)
//
// Returns ivar.
//
// This routine interpolates var at the coordinates specified by
// crds using tricubic interpolation as described in Lekien:2005.
// NOTE:  Coordinates must be expressed in terms of indices (ie, not mm).
//
// If a coordinate is outside the perimeter of var, the interpolated
// value will be set to that of the cell immediately beneath the
// coordinate.  That is, svinterp implements edge extension for
// coordinates lying beyond the perimeter.
//
static PyObject *flotracec_svcinterp(PyObject *self, PyObject *args) {

  PyObject      *ocrds, *ocoeff;
  PyArrayObject *crds, *coeff, *ivar;

  double   ndx[3], mxndx[3], shft[3], eps, vval, ps0[4], ps1[4], ps2[4];
  npy_intp indx[3], ncells[3], nvdim, crdo, coeffov[3], ivo;
  npy_intp nvcmp, ncrds, c, i, j, l, m, n, cnt, ivndim;

  npy_intp ivshape[2];

  double   *ptr_crds, *ptr_ivar, *ptr_coeff, *ccoeff;
  npy_intp *cfshape, *cshape;

  if (!PyArg_ParseTuple(
                        args,
                        "OO:svcinterp",
                        &ocrds,&ocoeff
                        )
      )
    Py_RETURN_NONE;

  crds  = (PyArrayObject *)PyArray_ContiguousFromAny(ocrds,PyArray_DOUBLE,2,2);
  coeff = (PyArrayObject *)PyArray_ContiguousFromAny(ocoeff,PyArray_DOUBLE,4,5);

  if ( ( crds == NULL ) || ( coeff == NULL ) ) {
    cleanup_po(2,crds,coeff);
    Py_RETURN_NONE;
  }

  // Initialization.
  nvdim  = PyArray_NDIM(coeff);
  if ( nvdim != 5 )
      return PyErr_Format(PyExc_ValueError,"coeff must have 5 dimensions.");

  cfshape = PyArray_DIMS(coeff);
  cshape  = PyArray_DIMS(crds);

  ncrds = cshape[0];

  for (i = 0; i < 3; i++)
      ncells[i] = cfshape[i +1] +1;

  nvcmp = cfshape[0];

  ivshape[0] = ncrds;
  ivshape[1] = nvcmp;
  if ( nvcmp == 1 ) {
      ivndim = 1;
  }
  else {
      ivndim = 2;
  }

  coeffov[2] = 64*cfshape[3];
  coeffov[1] = coeffov[2]*cfshape[2];
  coeffov[0] = coeffov[1]*cfshape[1];

  ivar = (PyArrayObject *)PyArray_EMPTY(ivndim,ivshape,PyArray_DOUBLE,
                                        PyArray_CORDER);
  if ( ivar == NULL ) {
    cleanup_po(2,crds,coeff);
    Py_RETURN_NONE;
  }

  ptr_crds  = (double *)PyArray_DATA(crds);
  ptr_coeff = (double *)PyArray_DATA(coeff);
  ptr_ivar  = (double *)PyArray_DATA(ivar);

  eps = 2.*DBL_EPSILON;

  for ( i = 0; i < 3; i++)
    mxndx[i] = fmax(ncells[i]*(1. -eps) -1., eps);

  ps0[0] = ps1[0] = ps2[0] = 1.;

  // Perform the interpolation.
  for (i = 0; i < ncrds; i++) {
    // Find the cell the the point is in.
    crdo = 3*i;
    ivo  = nvcmp*i;
    for (j = 0; j < 3; j++) {
      ndx[j] = ptr_crds[crdo +j];
      ndx[j] = fmax(ndx[j],eps);
      ndx[j] = fmin(ndx[j],mxndx[j]);

      indx[j] = (npy_intp)floor(ndx[j]);

      shft[j]  = ndx[j] -indx[j];
    }

    // Compute powers of the shifts.
    ps0[1] = shft[0];
    ps1[1] = shft[1];
    ps2[1] = shft[2];
    for ( j = 2; j < 4; j++ ){
        ps0[j] = pow(shft[0],j);
        ps1[j] = pow(shft[1],j);
        ps2[j] = pow(shft[2],j);
    }

    // Interpolate the variable components.
    for (c = 0; c < nvcmp; c++) {
      vval   = 0.;
      ccoeff = ptr_coeff
              +c*coeffov[0]
              +indx[0]*coeffov[1]
              +indx[1]*coeffov[2]
              +indx[2]*64;

      //PySys_WriteStdout("i %i, c %i | %i %i %i | %f %f %f | %f %f %f | %i %i %i | %i\n",i,c,coeffov[0],coeffov[1],coeffov[2],mxndx[0],mxndx[1],mxndx[2],ndx[0],ndx[1],ndx[2],indx[0],indx[1],indx[2],c*coeffov[0] +indx[0]*coeffov[1] +indx[1]*coeffov[2] +indx[2]*64);

      cnt = 0;
      for ( l=0; l<4; l++ ) {
          for ( m=0; m<4; m++ ) {
              for ( n=0; n<4; n++ ) {
                  vval = vval +ccoeff[cnt]*ps0[l]*ps1[m]*ps2[n];
                  cnt++;
              }
          }
      }

      ptr_ivar[ivo +c] = vval;
    }
  }

  cleanup_po(2,crds,coeff);
  return PyArray_Return(ivar);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   ctcorr(tcrd,ncells,rbndx)
//
// Returns [cta, ctandx, vtndx].
//
// This routine determines the correlation between tracers and
// the cells they occupy.  Tracers are evaluated for validity (ie,
// being within the perimeter of the region), and tcrd is compressed
// by moving valid tracers to the lower portion of tcrd.  tcrd is NOT
// resized here.
//
// The valid perimeter is denoted by rbndx, a 3x2 integer array.  The first
// index specifies (z,y,x), while the second gives:
//     rbndx[:,0] ----- Starting index
//     rbndx[:,1] ----- Ending index + 1
//
// cta and ctandx are valid for the modified tcrd upon return from
// ctcorr().
//
// NOTE: tcrd is modified in place.
//
// Three arrays are created:
//   cta ------ Array of same length as the number of tracers. cta
//              contains the valid tracer index for each tracer
//              contained in a cell.
//   ctandx --- (nz,ny,nx,2) element array that maps a cell index
//              into cta.  For each cell index, ctandx contains the
//              starting and ending indices (in Python slice format)
//              for cta.  For example, the tracers in cell [0,0,0]
//              are cta[ctandx[0,0,0,0]:ctandx[0,0,0,1]].
//   vtndx ---- Array containing the valid tracer index into the original,
//              uncompressed tcrd that was passed to ctcorr.  vtndx
//              can be used by the caller to compress additional arrays
//              such as trid.  The length of vtndx is equal to the number
//              of valid tracers.
//
static PyObject *flotracec_ctcorr(PyObject *self, PyObject *args) {

  PyObject      *otcrd, *orbndx, *rlst;
  PyArrayObject *tcrd,*rbndx,*cta,*ctandx,*vtndx;

  double tcmax[3], eps;
  int      ncells[3];
  npy_intp ctandxo[3];
  npy_intp i, tncells, dndx, izndx, iyndx, ixndx, dctandx, crdo, ocrdo;

  npy_intp ctandxs[4], ntr, vntr;

  double   *ptr_tcrd;
  npy_intp *ptr_cta, *ptr_ctandx, *zndx, *yndx, *xndx, *ptr_vtndx, *wvtndx;
  npy_intp *ptr_rbndx;

  if (!PyArg_ParseTuple(
                        args,
                        "O(iii)O:ctcorr",
                        &otcrd,&ncells[0],&ncells[1],&ncells[2],&orbndx
                        )
      )
    Py_RETURN_NONE;

  tcrd  = (PyArrayObject *)PyArray_ContiguousFromAny(otcrd,PyArray_DOUBLE,2,2);
  rbndx = (PyArrayObject *)PyArray_ContiguousFromAny(orbndx,NPY_INTP,2,2);

  if ( ( tcrd == NULL ) || ( rbndx == NULL ) ) {
    cleanup_po(2,tcrd,rbndx);
    Py_RETURN_NONE;
  }

  eps = 2.*DBL_EPSILON;

  // Initialization.
  ntr     = PyArray_DIMS(tcrd)[0];
  tncells = ncells[0]*ncells[1]*ncells[2];

  for (i = 0; i < 3; i++) {
    ctandxs[i] = ncells[i];
    tcmax[i]   = ncells[i] +0.5;
  }
  ctandxs[3] = 2;

  ctandx = (PyArrayObject *)PyArray_ZEROS(4,ctandxs,NPY_INTP,PyArray_CORDER);
  cta    = (PyArrayObject *)PyArray_EMPTY(1,&ntr,NPY_INTP,PyArray_CORDER);
  if ( ( ctandx == NULL ) || ( cta == NULL ) ) {
    cleanup_po(3, tcrd, ctandx, cta);
    Py_RETURN_NONE;
  }

  rlst = PyList_New(3);

  ptr_tcrd   = (double   *)PyArray_DATA(tcrd);
  ptr_rbndx  = (npy_intp *)PyArray_DATA(rbndx);
  ptr_ctandx = (npy_intp *)PyArray_DATA(ctandx);
  ptr_cta    = (npy_intp *)PyArray_DATA(cta);

  ctandxo[0] = ncells[1]*ncells[2]*2;
  ctandxo[1] = ncells[2]*2;
  ctandxo[2] = 2;

  zndx   = (npy_intp *)PyMem_Malloc(ntr*sizeof(npy_intp));
  yndx   = (npy_intp *)PyMem_Malloc(ntr*sizeof(npy_intp));
  xndx   = (npy_intp *)PyMem_Malloc(ntr*sizeof(npy_intp));
  wvtndx = (npy_intp *)PyMem_Malloc(ntr*sizeof(npy_intp));

  // Determine number of tracers in each cell.  Also store the tcrd indices
  // for valid tracers, and the corresponding cell indices.
  vntr = 0;
  for ( i = 0; i < ntr; i++ ) {
    crdo  = i*3;
    izndx = (npy_intp)floor(ptr_tcrd[crdo]    +0.5);
    iyndx = (npy_intp)floor(ptr_tcrd[crdo +1] +0.5);
    ixndx = (npy_intp)floor(ptr_tcrd[crdo +2] +0.5);

    if ( ( izndx < ptr_rbndx[0] )
         || ( iyndx < ptr_rbndx[2] )
         || ( ixndx < ptr_rbndx[4] )
         || ( izndx >= ptr_rbndx[1] )
         || ( iyndx >= ptr_rbndx[3] )
         || ( ixndx >= ptr_rbndx[5] ) )
      continue;

    // Have a valid tracer.
    ++ptr_ctandx[ izndx*ctandxo[0] +iyndx*ctandxo[1] +ixndx*ctandxo[2] +1 ];

    wvtndx[vntr] = i;

    zndx[vntr] = izndx;
    yndx[vntr] = iyndx;
    xndx[vntr] = ixndx;

    vntr = vntr +1;
  }

  // Allocate vtndx.  wvtndx will be copied to vtndx below.
  // There must be a better way to do this, but PyArray_Newshape segfaults.
  vtndx  = (PyArrayObject *)PyArray_EMPTY(1,&vntr,NPY_INTP,PyArray_CORDER);
  if ( vtndx == NULL ) {
    cleanup_po(4, tcrd, ctandx, cta, vtndx);
    Py_RETURN_NONE;
  }
  ptr_vtndx  = (npy_intp *)PyArray_DATA(vtndx);

  // Half build ctandx.  The first column will now contain the starting
  // index into cta.  The second column will be filled with the ending
  // index below.
  dndx = 0;
  for ( i = 0; i < tncells; i++ ) {
    ptr_ctandx[i*2]    = dndx;
    dndx               = dndx +ptr_ctandx[i*2 +1];
    ptr_ctandx[i*2 +1] = ptr_ctandx[i*2];
  }

  // Compress valid tracers and build cta.  We won't resize tcrd here.
  for ( i = 0; i < vntr; i++ ) {
    crdo  = i*3;
    ocrdo = wvtndx[i]*3;
    ptr_tcrd[crdo]    = ptr_tcrd[ocrdo];
    ptr_tcrd[crdo +1] = ptr_tcrd[ocrdo +1];
    ptr_tcrd[crdo +2] = ptr_tcrd[ocrdo +2 ];

    ptr_vtndx[i] = wvtndx[i];

    dctandx = zndx[i]*ctandxo[0] +yndx[i]*ctandxo[1] +xndx[i]*ctandxo[2] +1;
    dndx    = ptr_ctandx[dctandx];

    ptr_cta[dndx] = i;

    ++ptr_ctandx[dctandx];
  }

  PyList_SetItem(rlst,0,PyArray_Return(cta));
  PyList_SetItem(rlst,1,PyArray_Return(ctandx));
  PyList_SetItem(rlst,2,PyArray_Return(vtndx));

  PyMem_Free(zndx);
  PyMem_Free(yndx);
  PyMem_Free(xndx);
  PyMem_Free(wvtndx);
  cleanup_po(2,tcrd,rbndx);

  return rlst;
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   flotracec.chksrc(ctandx,scndx,sccomp,ntrpc)
//
// Returns [nstndx,nstid,nnst].
//
// chksrc scans source cells to determine if they have less than
// ntrpc particles.  If a deficient cell is found, the cell index
// and particle composition is added to nstndx and nstid,
// respectively.  nnst is the new number of source tracers.
//
// NOTE: nstndx and nstid are large enough to hold new source tracers
// for all source cells.  As a result, nnst should be used by the
// caller to determine the number of valid entries in nstndx and nstid.
//
//
static PyObject *flotracec_chksrc(PyObject *self, PyObject *args) {

  PyObject      *octandx, *oscndx, *osccomp, *rlst;
  PyArrayObject *ctandx, *scndx, *sccomp, *nstndx, *nstid;

  double   *ptr_sccomp, *ptr_nstid;
  int      ntrpc;
  npy_intp *ptr_ctandx, *ptr_scndx, *ptr_nstndx;
  npy_intp *ncells;

  npy_intp ctandxo[3], scndxo, nstndxo, ntric;
  npy_intp def, nnst, i, tnstr, dndx, nsc, sc, szndx, syndx, sxndx;

  npy_intp shape[2];

  if (!PyArg_ParseTuple(
                        args,
                        "OOOi:chksrc",
                        &octandx,&oscndx,&osccomp,&ntrpc
                        )
      )
    Py_RETURN_NONE;

  ctandx = (PyArrayObject *)PyArray_ContiguousFromAny(octandx,NPY_INTP,4,4);
  scndx  = (PyArrayObject *)PyArray_ContiguousFromAny(oscndx,NPY_INTP,2,2);
  sccomp = (PyArrayObject *)PyArray_ContiguousFromAny(osccomp,PyArray_DOUBLE,1,1);

  if ( ( ctandx == NULL ) ||
       ( scndx == NULL )  ||
       ( sccomp == NULL ) ) {
    cleanup_po(3,ctandx,scndx,sccomp);
    Py_RETURN_NONE;
  }

  // Initialization.
  nsc    = PyArray_DIMS(sccomp)[0];
  ncells = PyArray_DIMS(ctandx);

  ctandxo[0] = ncells[1]*ncells[2]*2;
  ctandxo[1] = ncells[2]*2;
  ctandxo[2] = 2;

  tnstr = nsc*ntrpc;

  shape[0] = tnstr;
  shape[1] = 3;
  nstndx   = (PyArrayObject *)PyArray_EMPTY(2,shape,NPY_INTP,PyArray_CORDER);
  nstid    = (PyArrayObject *)PyArray_EMPTY(1,&shape[0],PyArray_DOUBLE,
                                            PyArray_CORDER);

  if ( ( nstndx == NULL ) || ( nstid == NULL ) ) {
    cleanup_po(5,ctandx,scndx,sccomp,nstndx,nstid);
    Py_RETURN_NONE;
  }

  ptr_ctandx = (npy_intp *)PyArray_DATA(ctandx);
  ptr_scndx  = (npy_intp *)PyArray_DATA(scndx);
  ptr_sccomp = (double   *)PyArray_DATA(sccomp);
  ptr_nstndx = (npy_intp *)PyArray_DATA(nstndx);
  ptr_nstid  = (double   *)PyArray_DATA(nstid);

  rlst = PyList_New(3);

  // Main loop.
  nnst = 0;
  for ( sc = 0; sc < nsc; sc++ ) {
    scndxo = sc*3;
    szndx  = ptr_scndx[scndxo];
    syndx  = ptr_scndx[scndxo +1];
    sxndx  = ptr_scndx[scndxo +2];

    dndx  = szndx*ctandxo[0] +syndx*ctandxo[1] +sxndx*ctandxo[2];
    ntric = ptr_ctandx[dndx +1] -ptr_ctandx[dndx];

    def = ntrpc -ntric;
    for ( i = 0; i < def; i++ ) {
      nstndxo = nnst*3;
      ptr_nstndx[nstndxo]    = szndx;
      ptr_nstndx[nstndxo +1] = syndx;
      ptr_nstndx[nstndxo +2] = sxndx;

      ptr_nstid[nnst] = ptr_sccomp[sc];

      ++nnst;
    }
  }

  // NOTE: Py_BuildValue() should not be used with Numpy arrays as it
  // apparently causes a memory leak.
  PyList_SetItem(rlst,0,PyArray_Return(nstndx));
  PyList_SetItem(rlst,1,PyArray_Return(nstid));
  PyList_SetItem(rlst,2,Py_BuildValue("i",nnst));

  cleanup_po(3,ctandx,scndx,sccomp);

  return rlst;
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   flotracec.adjid(trid,cta,ctandx,scndx,sccomp)
//
// Adjusts the tracer identity for all source cells such that the
// source tracers are all of type sccomp.
//
// NOTE: trid is modified in place.
// NOTE: adjid(), if used, should be called immediately after ctcorr()
//       so that the cta and ctandx variables are accurate.
//
static PyObject *flotracec_adjid(PyObject *self, PyObject *args) {

  PyObject      *otrid, *octa, *octandx, *oscndx, *osccomp;
  PyArrayObject *trid, *cta, *ctandx, *scndx, *sccomp;

  double   *ptr_trid, *ptr_sccomp;
  npy_intp *ptr_cta, *ptr_ctandx, *ptr_scndx;
  npy_intp *ncells;

  npy_intp ctandxo[3], scndxo, ntric, szndx, syndx, sxndx;
  npy_intp i, dndx, nsc, sc, ttndx;

  if (!PyArg_ParseTuple(
                        args,
                        "OOOOO:adjid",
                        &otrid,&octa,&octandx,&oscndx,&osccomp
                        )
      )
    Py_RETURN_NONE;

  // otrid can have any dtype.  If the array isn't a double float, a copy
  // will be created and trid will point to it instead of otrid.  The flag
  // NPY_UPDATEIFCOPY ensures that the data in trid get type converted and
  // copied back into otrid if necessary.
  trid   = (PyArrayObject *)PyArray_FROMANY(otrid,PyArray_DOUBLE,1,1,
                                            NPY_DEFAULT | NPY_UPDATEIFCOPY );
  cta    = (PyArrayObject *)PyArray_ContiguousFromAny(octa,NPY_INTP,1,1);
  ctandx = (PyArrayObject *)PyArray_ContiguousFromAny(octandx,NPY_INTP,4,4);
  scndx  = (PyArrayObject *)PyArray_ContiguousFromAny(oscndx,NPY_INTP,2,2);
  sccomp = (PyArrayObject *)PyArray_ContiguousFromAny(osccomp,PyArray_DOUBLE,1,1);

  if ( ( trid == NULL ) ||
       ( cta == NULL ) ||
       ( ctandx == NULL ) ||
       ( scndx == NULL )  ||
       ( sccomp == NULL ) ) {
    cleanup_po(5,trid,cta,ctandx,scndx,sccomp);
    Py_RETURN_NONE;
  }

  // Initialization.
  nsc    = PyArray_DIMS(sccomp)[0];
  ncells = PyArray_DIMS(ctandx);

  ctandxo[0] = ncells[1]*ncells[2]*2;
  ctandxo[1] = ncells[2]*2;
  ctandxo[2] = 2;

  ptr_trid   = (double   *)PyArray_DATA(trid);
  ptr_cta    = (npy_intp *)PyArray_DATA(cta);
  ptr_ctandx = (npy_intp *)PyArray_DATA(ctandx);
  ptr_scndx  = (npy_intp *)PyArray_DATA(scndx);
  ptr_sccomp = (double   *)PyArray_DATA(sccomp);

  // Main loop.
  for ( sc = 0; sc < nsc; sc++ ) {
    scndxo = sc*3;
    szndx  = ptr_scndx[scndxo];
    syndx  = ptr_scndx[scndxo +1];
    sxndx  = ptr_scndx[scndxo +2];

    dndx  = szndx*ctandxo[0] +syndx*ctandxo[1] +sxndx*ctandxo[2];
    ntric = ptr_ctandx[dndx +1] -ptr_ctandx[dndx];
    for ( i = 0; i < ntric; i++ ) {
      ttndx = ptr_cta[ ptr_ctandx[dndx] +i ];
      ptr_trid[ ttndx ] = ptr_sccomp[sc];
    }
  }

  cleanup_po(5,trid,cta,ctandx,scndx,sccomp);
  Py_RETURN_NONE;
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   flotracec.tcseval(spl,tval)
//
// Returns ivar, the interpolated variable.
//
// Evaluates an interpolated variable at tval.  Wrapper around Scipy's
// splev_() of fitpack.
//
static PyObject *flotracec_tcseval(PyObject *self, PyObject *args) {

  PyObject      *ospl, *otval, *aspl, *mod, *splev, *dargs, *res, *k;
  PyArrayObject *t, *c, *tval, *ivar;

  double   *ptr_tval, *ptr_ivar;

  npy_intp tnval;
  npy_intp i;

  if (!PyArg_ParseTuple(
                        args,
                        "OO:tcseval",
                        &ospl,&otval
                        )
      )
    Py_RETURN_NONE;

  tval = (PyArrayObject *)PyArray_ContiguousFromAny(otval,PyArray_DOUBLE,1,1);

  if ( ( tval == NULL ) ) {
      Py_RETURN_NONE;
  }

  ptr_tval = PyArray_DATA(tval);

  // Initialization.
  tnval = PyArray_DIMS(tval)[0];

  ivar = (PyArrayObject *)PyArray_EMPTY(1,&tnval,PyArray_DOUBLE,PyArray_CORDER);

  if ( ivar == NULL ) {
     cleanup_po(2,tval,ivar);
     Py_RETURN_NONE;
  }

  ptr_ivar = (double *)PyArray_DATA(ivar);

  mod = PyImport_ImportModule("scipy.interpolate.dfitpack");
  if ( mod == NULL ) {
      Py_RETURN_NONE;
  }
  splev = PyObject_GetAttrString(mod,"splev");

  // Interpolate.
  for ( i = 0; i < tnval; i++ ) {
      aspl = PyList_GetItem(ospl,i);

      t = (PyArrayObject *)PyTuple_GetItem(aspl,0);
      c = (PyArrayObject *)PyTuple_GetItem(aspl,1);
      k = PyTuple_GetItem(aspl,2);

      dargs = Py_BuildValue("OOOd",t,c,k,ptr_tval[i]);
      res   = PyObject_CallObject(splev,dargs);
      ptr_ivar[i] = PyFloat_AsDouble(res);

      Py_DECREF(dargs);
      Py_DECREF(res);
  }

  cleanup_po(3,tval,mod,splev);
  return PyArray_Return(ivar);
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION cleanup_po
// Decrements references to PyObjects.
//
static void cleanup_po(int nvars, ...) {
  int i;
  va_list ap;
  PyObject *po;

  va_start(ap,nvars);
  for (i=0; i<nvars; i++) {
    po = va_arg(ap, PyObject *);
    if ( po == NULL )
      continue;
    else
      Py_DECREF(po);
  }
  va_end(ap);
}


static PyMethodDef flotracec_methods[] = {
  {"svinterp", flotracec_svinterp, METH_VARARGS,
   "Spatially interpolates a variable."
  },
  {"svcismat", flotracec_svcismat, METH_VARARGS,
   "Returns the system matrix for tricubic interpolation."
  },
  {"svcinterp", flotracec_svcinterp, METH_VARARGS,
   "Spatially interpolates a variable using tricubic interpolation."
  },
  {"ctcorr", flotracec_ctcorr, METH_VARARGS,
   "Computes cell to tracer correspondence."
  },
  {"chksrc", flotracec_chksrc, METH_VARARGS,
   "Computes cell to tracer correspondence."
  },
  {"adjid", flotracec_adjid, METH_VARARGS,
   "Forces all tracers in source cell to have source composition."
  },
  {"tcseval", flotracec_tcseval, METH_VARARGS,
   "Evaluates interpolated variable at tval."
  },
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initflotracec(void) {
  (void) Py_InitModule("flotracec",flotracec_methods);
  import_array();
}
