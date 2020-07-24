/*
Filename:  pivlibc.c
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
  These functions are for heavily numeric computations that
  are very slow if done through Python directly.

*/

#include "Python.h"
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdlib.h>

#include <mkl_lapack.h>
#include <mkl_cblas.h>
typedef MKL_INT lpk_int;
//#include <clapack.h>
//#include <cblas.h>

//typedef __CLPK_integer lpk_int;

//
// WORKER FUNCTION PROTOTYPES
//

static double wf_ddasum(PyArrayObject *in1,
                        npy_intp rbin1[4],
                        PyArrayObject *in2,
                        npy_intp rbin2[4]);

static void wf_dedco(PyArrayObject *in,
                     npy_intp rbin[4],
                     PyArrayObject *out,
                     npy_intp rbout[4]);

static double wf_dmsum(PyArrayObject *in1,
                       npy_intp rbin1[4],
                       PyArrayObject *in2,
                       npy_intp rbin2[4]);

static void bci_init(const double p[2], npy_intp offset[2], double fsm[16]);

static double bci_interp(double *ptr_imin,
                         double fsm[16],
                         npy_intp reimin,
                         npy_intp refy,
                         npy_intp reffy,
                         npy_intp reby,
                         npy_intp xndx,
                         npy_intp mci);

static int compare_doubles(const void *a, const void *b);

static void cleanup_po(int nvars, ...);
//~


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   bldex2msh(eshape)
//
// ----
//
// eshape      Element shape of mesh [z,y,x].
//
// ----
//
// Computes element connectivity for a structured, hexagonal mesh.
//
// Returns elcon, a concatenated, 1D array of node indices which make up
// each element.
//
static PyObject *pivlibc_bldex2msh(PyObject *self, PyObject *args) {

  PyObject      *oeshape;
  PyArrayObject *eshape, *elcon;

  npy_intp leshape[3], nshape[3], ecsz;
  npy_intp i, j, k, ecndx;
  npy_intp nnodpp, nnodpr, ndxcp, ndxfp, ndxcr, ndxfr;

  npy_intp *ptr_eshape, *ptr_elcon;

  // Big 'O' in the format is for an ndarray.
  if (!PyArg_ParseTuple(
                        args,
                        "O:bldex2msh",
                        &oeshape
                        )
      )
    return NULL;

  eshape = (PyArrayObject *)PyArray_ContiguousFromAny(oeshape,NPY_INTP,1,1);
  
  if ( eshape == NULL ) {
    cleanup_po(1,eshape);
    Py_RETURN_NONE;
  }

  // Initialization.
  ptr_eshape = (npy_intp *)PyArray_DATA(eshape);
  for ( i=0; i<3; i++ ) {
    leshape[i] = ptr_eshape[i];
    nshape[i]  = ptr_eshape[i] +1;
  }

  nnodpp = nshape[1]*nshape[2];
  nnodpr = nshape[2];

  ecsz  = 8*leshape[0]*leshape[1]*leshape[2];
  elcon = (PyArrayObject *)PyArray_EMPTY(1,&ecsz,NPY_INTP,PyArray_CORDER);

  if ( elcon == NULL )  {
    cleanup_po(2,eshape,elcon);
    Py_RETURN_NONE;
  }

  ptr_elcon = (npy_intp *)PyArray_DATA(elcon);

  // Get connectivity.  Exodus indices start at 1.
  ecndx = 0;
  for ( k=0; k<leshape[0]; k++ ) {
    ndxcp = k*nnodpp;
    ndxfp = ndxcp +nnodpp;
    for ( j=0; j<leshape[1]; j++ ) {
      ndxcr = j*nnodpr;
      ndxfr = ndxcr +nnodpr;
      for ( i=0; i<leshape[2]; i++ ) {
        ptr_elcon[ecndx]    = ndxcp +ndxcr +i +1;
        ptr_elcon[ecndx +1] = ndxcp +ndxcr +i +2;
        ptr_elcon[ecndx +2] = ndxcp +ndxfr +i +2;
        ptr_elcon[ecndx +3] = ndxcp +ndxfr +i +1;
        ptr_elcon[ecndx +4] = ndxfp +ndxcr +i +1;
        ptr_elcon[ecndx +5] = ndxfp +ndxcr +i +2;
        ptr_elcon[ecndx +6] = ndxfp +ndxfr +i +2;
        ptr_elcon[ecndx +7] = ndxfp +ndxfr +i +1;

        ecndx = ecndx +8;
      }
    }
  }

  cleanup_po(1,eshape);
  return PyArray_Return(elcon);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   gfitcore(pts, maxits, eps)
//
// ----
//
// pts         Three element array containing values to fit.
// maxits      Maximum iterations.
// eps         Desired accuracy.
//
// ----
//
// Fits a gaussian to three points using Newton-Raphson.
//
//    f = a exp( -(x-mu)^2/b^2 )
//
// The three points will be assigned coordinates [-1, 0, 1].
// The initial values of [a, mu, b] will be [pts[1], 0., 1.].
//
// Returns [a, mu, b].
//
static PyObject *pivlibc_gfitcore(PyObject *self, PyObject *args) {

  double jmat[9], pts[3], bvec[3], svec[3], l2hvec, eps, x;
  double arg, xmmu, xmmusq, bsq, bcb, vexp, apa;
  int    maxits, it, i, jri;

  lpk_int ipiv[3], info;
  static lpk_int sdim = 3;
  static lpk_int one  = 1;

  // Big 'O' in the format is for an ndarray.
  if (!PyArg_ParseTuple(
                        args,
                        "(ddd)id:gfitcore",
                        &pts[0],&pts[1],&pts[2],&maxits,&eps
                        )
      )
    return NULL;

  // Initialization.
  svec[0] = pts[1];
  svec[1] = 0.;
  svec[2] = 1.;

  // Main loop.
  l2hvec = 1.;
  it     = 0.;
  eps    = eps*eps;
  while ( ( l2hvec > eps ) && ( it < maxits ) ) {
    // Form the Jacobian and bvec.
    bsq  = svec[2]*svec[2];
    bcb  = bsq*svec[2];
    apa  = 2.*svec[0];
    for ( i = 0; i < 3; i++ ) {
      jri = i*3;
      x   = -1. +i;

      xmmu   = x -svec[1];
      xmmusq = xmmu*xmmu;

      arg  = -xmmusq/bsq;
      vexp = exp(arg);

      jmat[0 +i] = vexp;                // Column major.
      jmat[3 +i] = apa*xmmu/bsq*vexp;
      jmat[6 +i] = apa*xmmusq/bcb*vexp;

      bvec[i] = svec[0]*vexp -pts[i];
    }

    // Solve the system.
    dgesv_(&sdim,&one,jmat,&sdim,ipiv,bvec,&sdim,&info);

    if ( info != 0 ) {
      Py_RETURN_NONE;
    }

    l2hvec = 0.;
    for ( i = 0; i < 3; i++ ) {
      svec[i] = svec[i] -bvec[i];
      l2hvec  = l2hvec +bvec[i]*bvec[i];
    }

    ++it;
  }

  return Py_BuildValue("ddd",svec[0],svec[1],svec[2]);
}



/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   imsblicore(imin, rbndx, p)
//
// ----
//
// imin        Image to be shifted (greyscale mxn array).
// rbndx       Analyzed region boundary index (2x2 array).
// p           Array containing displacements ([dy,dx] pixels).
//
// ----
//
// Applies an image shift using bilinear interpolation.
//
// Returns bfr, a shifted version of the image region of interest
// specified by rbndx.
//
static PyObject *pivlibc_imsblicore(PyObject *self, PyObject *args) {

  PyObject      *oimin;
  PyArrayObject *imin, *bfr;

  double   alpha[4], p[2], fshift[2];
  double   cpy, cpx;
  int      rbndx[4];
  npy_intp offset[2];
  npy_intp rebfr, nxpix, reimin, refy, rndx, cndx;
  npy_intp i, j;
  npy_intp rsize[2];

  double *ptr_imin, *ptr_bfr;

  if (!PyArg_ParseTuple(
                        args,
                        "O(iiii)(dd):imsblicore",
                        &oimin,
                        &rbndx[0], &rbndx[1], &rbndx[2], &rbndx[3],
                        &p[0], &p[1]
                        )
      )
    return NULL;

  imin = (PyArrayObject *)PyArray_ContiguousFromAny(oimin,PyArray_DOUBLE,2,2);

  // Initialization.
  rsize[0] = rbndx[1] -rbndx[0];
  rsize[1] = rbndx[3] -rbndx[2];

  bfr = (PyArrayObject *)PyArray_SimpleNew(2,rsize,PyArray_DOUBLE);

  if ( (imin == NULL) || (bfr == NULL) ) {
    cleanup_po(2,imin,bfr);
    return NULL;
  }

  ptr_imin = (double *)PyArray_DATA(imin);
  ptr_bfr  = (double *)PyArray_DATA(bfr);

  nxpix = PyArray_DIMS(imin)[1];

  // Set the camera shifts and compute the weights.
  cpy = -p[0];
  cpx = -p[1];

  offset[0] = (npy_intp)floor(cpy);
  offset[1] = (npy_intp)floor(cpx);

  fshift[0] = cpy -offset[0];
  fshift[1] = cpx -offset[1];

  alpha[0] = (1. -fshift[0])*(1. -fshift[1]);
  alpha[1] = (1. -fshift[0])*fshift[1];
  alpha[2] = fshift[0]*(1. -fshift[1]);
  alpha[3] = fshift[0]*fshift[1];

  rbndx[0] = rbndx[0] +offset[0];
  rbndx[1] = rbndx[1] +offset[0];
  rbndx[2] = rbndx[2] +offset[1];
  rbndx[3] = rbndx[3] +offset[1];

  // Main loop.
  rndx = 0;
  for (i=rbndx[0]; i<rbndx[1]; i++) {
    reimin = i*nxpix;
    refy   = reimin +nxpix;
    rebfr  = rndx*rsize[1];

    cndx = 0;
    for (j=rbndx[2]; j<rbndx[3]; j++) {
      ptr_bfr[rebfr +cndx] =  alpha[0]*ptr_imin[reimin +j   ]
                             +alpha[1]*ptr_imin[reimin +j +1]
                             +alpha[2]*ptr_imin[refy   +j   ]
                             +alpha[3]*ptr_imin[refy   +j +1];

      cndx = cndx +1;
    }
    rndx = rndx +1;
  }

  cleanup_po(1,imin);
  return PyArray_Return(bfr);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   imsbcicore(imin, rbndx, p, edge)
//
// ----
//
// imin        Image to be shifted (greyscale mxn array).
// rbndx       Analyzed region boundary index (2x2 array).
// p           Array containing displacements ([dy,dx] pixels).
// edge        Edge treatment.
//
// ----
//
// This routine applies the image shift using bi-cubic
// interpolation.  It is configured to use the following
// convention for pixel ordering
//   A B
//   D C
// where A is the current pixel, B is in the direction of increasing
// x, and D is in the direction of increasing y.
//
// edge can be
//   0 - No edge treatment.  Image must be large enough for central
//       difference approximations to the first derivatives (i.e.,
//       1 pixel larger than rbndx in all directions AFTER shift is
//       applied).
//   1 - Use nearest neighbor.
//
// Returns bfr, a shifted version of the image region of interest
// specified by rbndx.
//
static PyObject *pivlibc_imsbcicore(PyObject *self, PyObject *args) {

  PyObject      *oimin;
  PyArrayObject *imin, *bfr;

  double   fsm[16], p[2];
  int      rbndx[4], edge;
  npy_intp offset[2];
  npy_intp rebfr, nxpix, reimin, reby, refy, reffy, rndx, cndx;
  npy_intp rstrt, cstrt, mxrpx, mci;
  npy_intp i, j;
  npy_intp rsize[2];

  double *ptr_imin, *ptr_bfr;

  if (!PyArg_ParseTuple(
                        args,
                        "O(iiii)(dd)i:imsbcicore",
                        &oimin,
                        &rbndx[0], &rbndx[1], &rbndx[2], &rbndx[3],
                        &p[0], &p[1],
                        &edge
                        )
      )
    return NULL;

  imin = (PyArrayObject *)PyArray_ContiguousFromAny(oimin,PyArray_DOUBLE,2,2);

  // Initialization.
  rsize[0] = rbndx[1] -rbndx[0];
  rsize[1] = rbndx[3] -rbndx[2];

  bfr = (PyArrayObject *)PyArray_SimpleNew(2,rsize,PyArray_DOUBLE);

  if ( (imin == NULL) || (bfr == NULL) ) {
    cleanup_po(2,imin,bfr);
    return NULL;
  }

  ptr_imin = (double *)PyArray_DATA(imin);
  ptr_bfr  = (double *)PyArray_DATA(bfr);

  nxpix = PyArray_DIMS(imin)[1];

  // Get the offsets, shift products, and handle edge pixels.
  bci_init(p,offset,fsm);

  rbndx[0] = rbndx[0] +offset[0];
  rbndx[1] = rbndx[1] +offset[0];
  rbndx[2] = rbndx[2] +offset[1];
  rbndx[3] = rbndx[3] +offset[1];

  if ( edge == 1 ) {
    // Do nearest neighbor for outer edge pixels.
    for (i=0; i<2; i++) {
      reimin = (rbndx[0] +i*(rsize[0] -1))*nxpix;
      rebfr  = i*(rsize[0] -1)*rsize[1];
      for (j=0; j<rsize[1]; j++) {
        ptr_bfr[rebfr +j] = ptr_imin[reimin +rbndx[2] +j];
      }
    }

    for (i=1; i<rsize[0]-1; i++) {
      reimin = (rbndx[0] +i)*nxpix;
      rebfr  = i*rsize[1];
      for (j=0; j<2; j++) {
        cndx = j*(rsize[1] -1);
        ptr_bfr[rebfr +cndx] = ptr_imin[reimin +rbndx[2] +cndx];
      }
    }

    rbndx[0] = rbndx[0] +1;
    rbndx[1] = rbndx[1] -1;
    rbndx[2] = rbndx[2] +1;
    rbndx[3] = rbndx[3] -1;

    rstrt = 1;
    cstrt = 1;
  }
  else {
    rstrt = 0;
    cstrt = 0;
  }

  // Main loop.
  rndx  = rstrt;
  mxrpx = ( imin->dimensions[0] -1 )*nxpix;
  for (i=rbndx[0]; i<rbndx[1]; i++) {
    reimin = i*nxpix;
    reby   = reimin -nxpix;
    refy   = reimin +nxpix;
    reffy  = refy +nxpix;
    reffy  = ( reffy < mxrpx ? reffy : mxrpx );

    rebfr = rndx*rsize[1];

    cndx = cstrt;
    for (j=rbndx[2]; j<rbndx[3]; j++) {
      mci = ( j+2 < nxpix-1 ? j+2 : nxpix-1 );

      ptr_bfr[rebfr +cndx] = bci_interp(ptr_imin,
                                        fsm,
                                        reimin,
                                        refy,
                                        reffy,
                                        reby,
                                        j,
                                        mci);

      cndx = cndx +1;
    }
    rndx = rndx +1;
  }

  cleanup_po(1,imin);
  return PyArray_Return(bfr);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   pxsbcicore(imin, pxca, pa)
//
// ----
//
// imin           # Image to be shifted (greyscale mxn array).
// pxca           # lx2 array of integer pixel coordinates (y,x).
// pa             # lx2 array of shift components (dy,dx).
//
// ----
//
// Applies a corresponding shift in pa to each pixel in pxls.
//
// Returns spx, an lx1 array of shifted pixel values.
//
static PyObject *pivlibc_pxsbcicore(PyObject *self, PyObject *args) {

  PyObject      *oimin, *opxca, *opa;
  PyArrayObject *imin, *pxca, *pa, *spx;

  double   fsm[16];
  double   *ptr_imin, *ptr_pa, *ptr_spx;
  npy_intp *ptr_pxca;
  npy_intp offset[2];
  npy_intp nxpix, repxca, reimin, reby, refy, reffy;
  npy_intp imys, imxs, mxrpx, mxys, mxxs, mci, xflg;
  npy_intp i;
  npy_intp npts;

  if (!PyArg_ParseTuple(
                        args,
                        "OOO:pxsbcicore",
                        &oimin, &opxca, &opa
                        )
      )
    return NULL;

  // Initialization.
  imin  = (PyArrayObject *)PyArray_ContiguousFromAny(oimin,PyArray_DOUBLE,2,2);
  pxca  = (PyArrayObject *)PyArray_ContiguousFromAny(opxca,NPY_INTP,2,2);
  pa    = (PyArrayObject *)PyArray_ContiguousFromAny(opa,PyArray_DOUBLE,2,2);

  if ( PyErr_Occurred() ) {
    cleanup_po(3,imin,pxca,pa);
    return NULL;
  }

  npts = pxca->dimensions[0];

  spx = (PyArrayObject *)PyArray_SimpleNew(1,&npts,PyArray_DOUBLE);

  if ( (imin == NULL) || (pxca == NULL) || (pa == NULL) || (spx == NULL) ) {
    cleanup_po(4,imin,pxca,pa,spx);
    return NULL;
  }

  ptr_imin = (double *)PyArray_DATA(imin);
  ptr_pxca = (npy_intp *)PyArray_DATA(pxca);
  ptr_pa   = (double *)PyArray_DATA(pa);
  ptr_spx  = (double *)PyArray_DATA(spx);

  nxpix = imin->dimensions[1];

  mxrpx = ( imin->dimensions[0] -1 )*nxpix;
  mxys  = imin->dimensions[0] -2;
  mxxs  = imin->dimensions[1] -2;

  // Main loop.
  for (i=0; i<npts; i++) {
    repxca = 2*i;

    // Get the offsets and shift products.
    bci_init(ptr_pa+repxca,offset,fsm);

    imys = ptr_pxca[repxca] +offset[0];
    imxs = ptr_pxca[repxca +1] +offset[1];

    // Do nearest neighbor if necessary.
    xflg = 0;
    if ( imys < 1 ) {
        xflg = 1;
        imys = 0;
    }
    else if ( imys > mxys ) {
        xflg = 1;
        imys = ( imys < imin->dimensions[0] ? imys : imin->dimensions[0] -1 );
    }

    if ( imxs < 1 ) {
        xflg = 1;
        imxs = 0;
    }
    else if ( imxs > mxxs ) {
        xflg = 1;
        imxs = ( imxs < imin->dimensions[1] ? imxs : imin->dimensions[1] -1 );
    }

    if ( xflg ) {
        ptr_spx[i] = ptr_imin[imys*nxpix +imxs];
        continue;
    }

    // Shift the pixel.
    reimin = imys*nxpix;
    reby   = reimin -nxpix;
    refy   = reimin +nxpix;
    reffy  = refy +nxpix;

    // Do additional nearest neighbor for the right and bottom sides to
    // reduce a loss of quality when the base pixel is on the second
    // row/column from either of these edges (bilinear interpolation uses
    // a 2x2 stencil).
    reffy  = ( reffy < mxrpx ? reffy : mxrpx );
    mci    = ( imxs+2 < nxpix-1 ? imxs+2 : nxpix-1 );

    ptr_spx[i] = bci_interp(ptr_imin,
                            fsm,
                            reimin,
                            refy,
                            reffy,
                            reby,
                            imxs,
                            mci);

  }

  cleanup_po(3,imin,pxca,pa);
  return PyArray_Return(spx);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   irinit(f2, bsndx, bsize)
//
// ----
//
// f2          Frame 2 (greyscale mxn array).
// bsndx       Region of interest starting indices (y,x).
// bsize       Region of interest size [pixels].
//
// ----
//
// Computes spatial gradients and builds associated matrices for
// irlk().  See documentation of pivir.irlk() for more details.
//
// Returns [apamat,apmat].
//
static PyObject *pivlibc_irinit(PyObject *self, PyObject *args) {
  PyObject      *of2, *rlst;
  PyArrayObject *f2, *apamat, *apmat;

  double fy, fx, sf;

  int bsndxy, bsndxx, ymax, xmax;

  npy_intp matdims[2];
  npy_intp pxndx, rndx, cndx;
  npy_intp terlf2, terlapmat, ref2, ref2fy, ref2ffy, ref2by, ref2bby;
  npy_intp i, j;

  double *ptr_f2, *ptr_apamat, *ptr_apmat;

  if (!PyArg_ParseTuple(
                        args,
                        "O(ii)(ii):irinit",
                        &of2,
                        &bsndxy, &bsndxx,
                        &ymax, &xmax
                        )
      )
    return NULL;

  // Initialization.
  f2 = (PyArrayObject *)PyArray_ContiguousFromAny(of2,PyArray_DOUBLE,2,2);

  matdims[0] = 2;
  matdims[1] = 2;
  apamat = (PyArrayObject *)PyArray_SimpleNew(2,matdims,PyArray_DOUBLE);

  matdims[1] = ymax*xmax;
  apmat = (PyArrayObject *)PyArray_SimpleNew(2,matdims,PyArray_DOUBLE);

  if ( (f2 == NULL) || (apamat == NULL) || (apmat == NULL) ) {
    cleanup_po(3,f2,apamat,apmat);
    return NULL;
  }

  ptr_f2     = (double *)PyArray_DATA(f2);
  ptr_apamat = (double *)PyArray_DATA(apamat);
  ptr_apmat  = (double *)PyArray_DATA(apmat);

  terlf2    = f2->strides[0]/sizeof(double);
  terlapmat = apmat->strides[0]/sizeof(double);

  ymax = bsndxy +ymax;
  xmax = bsndxx +xmax;

  sf = 1./12.;

  for (i=0; i<4; i++)
    ptr_apamat[i] = 0.;

  rlst = PyList_New(2);

  // Main loop.
  pxndx = 0;
  rndx  = 0;
  for (i=bsndxy; i<ymax; i++) {
    ref2    = i*terlf2;
    ref2by  = ref2   -terlf2;
    ref2bby = ref2by -terlf2;
    ref2fy  = ref2   +terlf2;
    ref2ffy = ref2fy +terlf2;

    cndx = 0;
    for (j=bsndxx; j<xmax; j++) {

      fy = sf*(-ptr_f2[ref2ffy +j]   +8.*ptr_f2[ref2fy +j]
               -8.*ptr_f2[ref2by +j] +ptr_f2[ref2bby +j]);

      fx = sf*(-ptr_f2[ref2 +j +2]    +8.*ptr_f2[ref2 +j +1]
               -8.*ptr_f2[ref2 +j -1] +ptr_f2[ref2 +j -2]);

      ptr_apamat[0] = ptr_apamat[0] +fy*fy;
      ptr_apamat[3] = ptr_apamat[3] +fx*fx;
      ptr_apamat[1] = ptr_apamat[1] +fx*fy;

      ptr_apmat[pxndx]            = fy;
      ptr_apmat[pxndx +terlapmat] = fx;

      pxndx = pxndx +1;
      cndx  = cndx  +1;

    }

    rndx = rndx +1;
  }
  ptr_apamat[2] = ptr_apamat[1];

  PyList_SetItem(rlst,0,PyArray_Return(apamat));
  PyList_SetItem(rlst,1,PyArray_Return(apmat));

  cleanup_po(1,f2);
  return rlst;
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   irncc_core(f1bfr, f2, rbndx, maxdisp)
//
// ----
//
// f1bfr       Frame 1 template (has shape provided by rbndx).
// f2          Frame 2 (greyscale mxn array).
// rbndx       Analyzed region boundary index (2x2 array).
// maxdisp     Max displacement [y,x].
//
// ----
//
// Computes the normalized cross correlation coefficient for
// irncc().  See documentation of pivir.irncc() for more details.
//
// Returns coeff, a 2D array of size 2*maxdisp +1.
//
static PyObject *pivlibc_irncc_core(PyObject *self, PyObject *args) {
  PyObject      *of1bfr, *of2;
  PyArrayObject *f1bfr, *f2, *f2bfr, *coeff;

  int rbndx[4];

  double   maxdispy, maxdispx, nf1bfr, nf2bfr, eps;
  npy_intp mrbndx[4], brbndx[4], mrsndx[2];
  npy_intp recoeff;
  npy_intp i, j;

  npy_intp rsize[2], prsize[2];

  double *ptr_coeff;

  if (!PyArg_ParseTuple(
                        args,
                        "OO(iiii)(dd):irncc_core",
                        &of1bfr,&of2,
                        &rbndx[0], &rbndx[1], &rbndx[2], &rbndx[3],
                        &maxdispy, &maxdispx
                        )
      )
    return NULL;

  f1bfr = (PyArrayObject *)PyArray_ContiguousFromAny(of1bfr,PyArray_DOUBLE,2,2);
  f2    = (PyArrayObject *)PyArray_ContiguousFromAny(of2,PyArray_DOUBLE,2,2);

  // Initialization.
  eps = 1.E-6;

  rsize[0] = rbndx[1] -rbndx[0];
  rsize[1] = rbndx[3] -rbndx[2];

  f2bfr = (PyArrayObject *)PyArray_SimpleNew(2,rsize,PyArray_DOUBLE);

  prsize[0] = 1 +2*maxdispy;
  prsize[1] = 1 +2*maxdispx;

  coeff = (PyArrayObject *)PyArray_SimpleNew(2,prsize,PyArray_DOUBLE);

  mrsndx[0] = rbndx[0] -maxdispy;
  mrsndx[1] = rbndx[2] -maxdispx;

  if ( (f1bfr == NULL)
       || (f2 == NULL)
       || (f2bfr == NULL)
       || (coeff == NULL) ) {
    cleanup_po(4,f1bfr,f2,f2bfr,coeff);
    return NULL;
  }

  ptr_coeff = (double *)PyArray_DATA(coeff);

  brbndx[0] = brbndx[2] = 0;
  brbndx[1] = rsize[0];
  brbndx[3] = rsize[1];

  // Setup the template block.
  wf_dedco(f1bfr,brbndx,f1bfr,brbndx);

  nf1bfr = wf_dmsum(f1bfr, brbndx, f1bfr, brbndx);
  nf1bfr = sqrt(nf1bfr);

  // Compute the correlation coefficient.
  for (i=0; i<prsize[0]; i++) {
    mrbndx[0] = mrsndx[0] +i;
    mrbndx[1] = mrbndx[0] +rsize[0];

    recoeff = i*prsize[1];

    for (j=0; j<prsize[1]; j++) {
      mrbndx[2] = mrsndx[1] +j;
      mrbndx[3] = mrbndx[2] +rsize[1];

      wf_dedco(f2,mrbndx,f2bfr,brbndx);

      nf2bfr = wf_dmsum(f2bfr, brbndx, f2bfr, brbndx);
      nf2bfr = fmax(sqrt(nf2bfr)*nf1bfr,eps);

      ptr_coeff[recoeff +j] =
        wf_dmsum(f1bfr, brbndx, f2bfr, brbndx)/nf2bfr;
    }
  }

  cleanup_po(3,f1bfr,f2,f2bfr);
  return PyArray_Return(coeff);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   irssda_core(f1bfr, f2, rbndx, maxdisp)
//
// ----
//
// f1bfr       Frame 1 template (has shape provided by rbndx).
// f2          Frame 2 (greyscale mxn array).
// rbndx       Analyzed region boundary index (2x2 array).
// maxdisp     Max displacement [y,x].
//
// ----
//
// This routine computes the sum of absolute differences similarity
// measure.
//
// Returns coeff, a 2D array of size 2*maxdisp +1.
//
static PyObject *pivlibc_irssda_core(PyObject *self, PyObject *args) {
  PyObject      *of1bfr, *of2;
  PyArrayObject *f1bfr, *f2, *coeff;

  int rbndx[4];

  double maxdispy, maxdispx;
  npy_intp mrbndx[4], brbndx[4], rsize[2], mrsndx[2];
  npy_intp recoeff;
  npy_intp i, j;
  npy_intp prsize[2];

  double *ptr_coeff;

  if (!PyArg_ParseTuple(
                        args,
                        "OO(iiii)(dd):irssda_core",
                        &of1bfr,&of2,
                        &rbndx[0], &rbndx[1], &rbndx[2], &rbndx[3],
                        &maxdispy, &maxdispx
                        )
      )
    return NULL;

  f1bfr = (PyArrayObject *)PyArray_ContiguousFromAny(of1bfr,PyArray_DOUBLE,2,2);
  f2    = (PyArrayObject *)PyArray_ContiguousFromAny(of2,PyArray_DOUBLE,2,2);

  // Initialization.
  rsize[0] = rbndx[1] -rbndx[0];
  rsize[1] = rbndx[3] -rbndx[2];

  prsize[0] = 1 +2*maxdispy;
  prsize[1] = 1 +2*maxdispx;

  coeff = (PyArrayObject *)PyArray_SimpleNew(2,prsize,PyArray_DOUBLE);

  mrsndx[0] = rbndx[0] -maxdispy;
  mrsndx[1] = rbndx[2] -maxdispx;

  if ( (f1bfr == NULL)
       || (f2 == NULL)
       || (coeff == NULL) ) {
    cleanup_po(3,f1bfr,f2,coeff);
    return NULL;
  }

  ptr_coeff = (double *)PyArray_DATA(coeff);

  brbndx[0] = brbndx[2] = 0;
  brbndx[1] = rsize[0];
  brbndx[3] = rsize[1];

  // Compute the similarity measure.
  for (i=0; i<prsize[0]; i++) {
    mrbndx[0] = mrsndx[0] +i;
    mrbndx[1] = mrbndx[0] +rsize[0];

    recoeff = i*prsize[1];

    for (j=0; j<prsize[1]; j++) {
      mrbndx[2] = mrsndx[1] +j;
      mrbndx[3] = mrbndx[2] +rsize[1];

      ptr_coeff[recoeff +j] = wf_ddasum(f1bfr,brbndx,f2,mrbndx);
    }
  }

  cleanup_po(2,f1bfr,f2);
  return PyArray_Return(coeff);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   crcht_core(rsize,eyndx,exndx,etheta,yco,xco,rd,surd)
//
// ----
//
// rsize       Two element array giving region size [y,x].
// eyndx       l-element array of edge y-indices.
// exndx       l-element array of edge x-indices.
// etheta      l-element array gradient angles for edge pixels.
// yco         Center y-offset array.
// xco         Center x-offset array.
// rd          Radii array.
// surd        Sorted, unique radii.
//
// ----
//
// Computes the Hough transform matrix for the circle Hough
// transform.  See documentation of pivutil.crcht() for more
// details.
//
// Returns htmat.
//
static PyObject *pivlibc_crcht_core(PyObject *self, PyObject *args) {
  PyObject      *oeyndx, *oexndx, *oetheta, *oyco, *oxco, *ord, *osurd;
  PyArrayObject *eyndx, *exndx, *etheta, *yco, *xco, *rd, *surd, *htmat;

  int rsize[2];

  double eps, phimx, gy, gx, arg, phi;
  npy_intp nht;
  npy_intp terlht, vyc, vxc;
  npy_intp nepts, nrd, nsurd;
  npy_intp i, j;
  npy_intp htsize[3];

  npy_intp *ptr_eyndx, *ptr_exndx, *ptr_yco, *ptr_xco, *ptr_htmat, *rdndx;
  double   *ptr_etheta, *ptr_rd, *ptr_surd, *ird;

  if (!PyArg_ParseTuple(
                        args,
                        "(ii)OOOOOOO:crcht_core",
                        &rsize[0],&rsize[1],
                        &oeyndx,&oexndx,&oetheta,&oyco,&oxco,&ord,&osurd
                        )
      )
    return NULL;

  eyndx  = (PyArrayObject *)PyArray_ContiguousFromAny(oeyndx,NPY_INTP,1,2);
  exndx  = (PyArrayObject *)PyArray_ContiguousFromAny(oexndx,NPY_INTP,1,2);
  etheta = (PyArrayObject *)PyArray_ContiguousFromAny(oetheta,PyArray_DOUBLE,1,2);
  yco    = (PyArrayObject *)PyArray_ContiguousFromAny(oyco,NPY_INTP,1,2);
  xco    = (PyArrayObject *)PyArray_ContiguousFromAny(oxco,NPY_INTP,1,2);
  rd     = (PyArrayObject *)PyArray_ContiguousFromAny(ord,PyArray_DOUBLE,1,2);
  surd   = (PyArrayObject *)PyArray_ContiguousFromAny(osurd,PyArray_DOUBLE,1,2);

  if ( (eyndx == NULL)
       || (exndx == NULL)
       || (etheta == NULL)
       || (yco == NULL)
       || (xco == NULL)
       || (rd == NULL)
       || (surd == NULL) ) {
    cleanup_po(7,eyndx,exndx,etheta,yco,xco,rd,surd);
    return NULL;
  }

  // Initialization.
  eps   = 0.00001;
  phimx = 3.14/12.;     // Keep small to prevent broad peaks.

  ptr_eyndx  = (npy_intp *)PyArray_DATA(eyndx);
  ptr_exndx  = (npy_intp *)PyArray_DATA(exndx);
  ptr_yco    = (npy_intp *)PyArray_DATA(yco);
  ptr_xco    = (npy_intp *)PyArray_DATA(xco);
  ptr_etheta = (double *)PyArray_DATA(etheta);
  ptr_rd     = (double *)PyArray_DATA(rd);
  ptr_surd   = (double *)PyArray_DATA(surd);

  nepts = PyArray_Size((PyObject *)eyndx);
  nrd   = PyArray_Size((PyObject *)rd);
  nsurd = PyArray_Size((PyObject *)surd);

  htsize[0] = rsize[0];
  htsize[1] = rsize[1];
  htsize[2] = nsurd;
  nht       = rsize[0]*rsize[1]*nsurd;
  terlht    = htsize[1]*htsize[2];
  htmat = (PyArrayObject *)PyArray_SimpleNew(3,htsize,NPY_INTP);
  if ( htmat == NULL ) {
    cleanup_po(8,eyndx,exndx,etheta,yco,xco,rd,surd,htmat);
    return NULL;
  }
  ptr_htmat = (npy_intp *)PyArray_DATA(htmat);
  for (i=0;i<nht;i++)
    ptr_htmat[i] = 0;

  ird   = (double *)PyMem_Malloc(nrd*sizeof(double));
  rdndx = (npy_intp *)PyMem_Malloc(nrd*sizeof(npy_intp));

  // Get the inverse radii.
  for (i=0;i<nrd;i++)
    ird[i] = 1./ptr_rd[i];

  // Build a map into the set of unique radii.
  for (i=0;i<nsurd;i++) {
    for (j=0;j<nrd;j++) {
      if ( abs(ptr_rd[j]-ptr_surd[i]) < eps)
        rdndx[j] = i;
    }
  }

   // Compute htmat.
  for (i=0; i<nepts; i++) {
    // Get gradient components.
    gy = sin(ptr_etheta[i]);
    gx = cos(ptr_etheta[i]);

    for (j=0; j<nrd; j++) {
      // Get angle between center offsets and gradient by dot product.
      arg = (ptr_yco[j]*gy +ptr_xco[j]*gx)*ird[j];
      if ( arg > 1. )
        arg = 1.;
      else if ( arg < -1. )
        arg = -1.;

      phi = acos(fabs(arg));
      phi = phi -phimx;
      if ( phi > eps )
        continue;

      // Compute valid center.
      vyc = ptr_eyndx[i] +ptr_yco[j];
      vxc = ptr_exndx[i] +ptr_xco[j];
      if ( ( vyc >= htsize[0] ) || ( vxc >= htsize[1] )
           || ( vyc < 0 ) || ( vxc < 0 ) )
        continue;

      // Vote for cells.
      ptr_htmat[vyc*terlht +vxc*htsize[2] +rdndx[j]] =
        ptr_htmat[vyc*terlht +vxc*htsize[2] +rdndx[j]] +1;

    }
  }

  PyMem_Free(ird);
  PyMem_Free(rdndx);
  cleanup_po(7,eyndx,exndx,etheta,yco,xco,rd,surd);

  return PyArray_Return(htmat);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   wcxmedian_core(var,fdim,planar)
//
// ----
//
// var         PIVVar to be filtered.
// fdim        Filter dimension.
// planar      Boolean indicating if 2D filter is applied to each z-plane.
//
// ----
//
// Computes the median of a scalar, 3D variable using a sliding
// window.  See documentation of pivpost.medfltr() for more
// details.
//
// var must have 3 dimensions.
//
// Returns mvar, the filtered variable.
//
static PyObject *pivlibc_wcxmedian_core(PyObject *self, PyObject *args) {
  PyObject      *ovar;
  PyArrayObject *var, *mvar;

  int fdim, hfdim, planar;
  int z, y, x, bz, by, bx, zsndx, ysndx, bzsndx, bysndx, mndx, mvarndx;
  int bzs, bze, bys, bye, bxs, bxe, bndx;

  npy_intp *vshape;
  double   *ptr_var, *ptr_mvar, *block;

  if (!PyArg_ParseTuple(
                        args,
                        "Oii:wcxmedian_core",
                        &ovar,&fdim,&planar
                        )
      )
    return NULL;

  var   = (PyArrayObject *)PyArray_ContiguousFromAny(ovar,PyArray_DOUBLE,3,3);
  if ( var == NULL ) {
    cleanup_po(1,var);
    return NULL;
  }

  // Initialization.
  vshape = PyArray_DIMS(var);
  mvar   = (PyArrayObject *)PyArray_ZEROS(3,vshape,PyArray_DOUBLE,PyArray_CORDER);
  if ( mvar == NULL ) {
    cleanup_po(2,var,mvar);
    return NULL;
  }

  hfdim = fdim/2;
  block = (double *)PyMem_Malloc((fdim*fdim*fdim)*sizeof(double));

  ptr_var  = (double *)PyArray_DATA(var);
  ptr_mvar = (double *)PyArray_DATA(mvar);

  // Compute the median.
  for (z=0; z<vshape[0]; z++){
    zsndx = z*vshape[1]*vshape[2];

    for (y=0; y<vshape[1]; y++) {
      ysndx = y*vshape[2];

      for (x=0; x<vshape[2]; x++) {

        // Inner loop.
        if ( planar ) {
          bzs = z;
          bze = z +1;
        }
        else {
          bzs = (-hfdim +z < 0 ? 0 : -hfdim +z);
          bze = (hfdim +z +1 > vshape[0] ? vshape[0] : hfdim +z +1);
        }

        bys = (-hfdim +y < 0 ? 0 : -hfdim +y);
        bye = (hfdim +y +1 > vshape[1] ? vshape[1] : hfdim +y +1);

        bxs = (-hfdim +x < 0 ? 0 : -hfdim +x);
        bxe = (hfdim +x +1 > vshape[2] ? vshape[2] : hfdim +x +1);

        bndx = 0;
        for (bz=bzs; bz<bze; bz++) {
          bzsndx = bz*vshape[1]*vshape[2];

          for (by=bys; by<bye; by++) {
            bysndx = by*vshape[2];

            for (bx=bxs; bx<bxe; bx++) {
              block[bndx] = ptr_var[bzsndx +bysndx +bx];
              bndx = bndx +1;
            }
          }
        }
        qsort((void *)block,bndx,sizeof(double),compare_doubles);

        mndx    = bndx/2;
        mvarndx = zsndx +ysndx +x;
        if ( ( bndx % 2 ) == 0 ) {
          ptr_mvar[mvarndx] = (block[mndx-1] +block[mndx])/2.;
        }
        else {
          ptr_mvar[mvarndx] = block[mndx];
        }

      }  // x
    }    // y
  }      // z

  PyMem_Free(block);
  cleanup_po(1,var);
  return PyArray_Return(mvar);
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION
//
// Sums the absolute value of the difference between two arrays.
// Data arrays are assumed contiguous.
//
static double wf_ddasum(PyArrayObject *in1,  // Input data array 1.
                         npy_intp rbin1[],         // [bsndxy, ymax, bsndxx, xmax].
                         PyArrayObject *in2,  // Input data array 2.
                         npy_intp rbin2[]          // [bsndxy, ymax, bsndxx, xmax].
                         ){
  double *ptr_in1, *ptr_in2;
  double s;
  npy_intp terlin1, terlin2, rein1, rein2, rsy, rsx;
  npy_intp i, j;

  ptr_in1 = (double *)PyArray_DATA(in1);
  ptr_in2 = (double *)PyArray_DATA(in2);

  terlin1 = in1->strides[0]/sizeof(double);
  terlin2 = in2->strides[0]/sizeof(double);

  rsy = rbin1[1] -rbin1[0];
  rsx = rbin1[3] -rbin1[2];

  s = 0.;
  for (i=0; i<rsy; i++) {
    rein1 = (i +rbin1[0])*terlin1 +rbin1[2];
    rein2 = (i +rbin2[0])*terlin2 +rbin2[2];

    for (j=0; j<rsx; j++) {
      s = s +fabs(ptr_in1[rein1 +j] -ptr_in2[rein2 +j]);
    }
  }

  return s;
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION
//
// Eliminates the dc offset from an image (ie: it subtracts the
// mean).  Data arrays are assumed contiguous.
//
static void wf_dedco(PyArrayObject *in,   // Input data array.
                     npy_intp rbin[],          // [bsndxy, ymax, bsndxx, xmax].
                     PyArrayObject *out,  // Output data array.
                     npy_intp rbout[]          // [bsndxy, ymax, bsndxx, xmax].
                     ){
  double *ptr_in, *ptr_out;
  double av;
  npy_intp terlin, terlout, rein, reout, rsy, rsx;
  npy_intp i, j;

  ptr_in  = (double *)PyArray_DATA(in);
  ptr_out = (double *)PyArray_DATA(out);

  terlin  = in->strides[0]/sizeof(double);
  terlout = out->strides[0]/sizeof(double);

  rsy = rbin[1] -rbin[0];
  rsx = rbin[3] -rbin[2];

  av = 0.;
  for (i=0; i<rsy; i++) {
    rein = (i +rbin[0])*terlin +rbin[2];
    for (j=0; j<rsx; j++) {
      av = av +ptr_in[rein +j];
    }
  }
  av = av/(rsy*rsx);

  for (i=0; i<rsy; i++) {
    rein  = (i +rbin[0])*terlin +rbin[2];
    reout = (i +rbout[0])*terlout +rbout[2];

    for (j=0; j<rsx; j++) {
      ptr_out[reout +j] = ptr_in[rein +j] -av;
    }
  }
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION
//
// This routine is essentially a dot product.  It multiplies two arrays
// together and sums the result.  Data arrays are assumed contiguous.
//
static double wf_dmsum(PyArrayObject *in1,  // Input data array 1.
                        npy_intp rbin1[],         // [bsndxy, ymax, bsndxx, xmax].
                        PyArrayObject *in2,  // Input data array 2.
                        npy_intp rbin2[]          // [bsndxy, ymax, bsndxx, xmax].
                        ){
  double *ptr_in1, *ptr_in2;
  double s;
  npy_intp terlin1, terlin2, rein1, rein2, rsy, rsx;
  npy_intp i, j;

  ptr_in1 = (double *)PyArray_DATA(in1);
  ptr_in2 = (double *)PyArray_DATA(in2);

  terlin1 = in1->strides[0]/sizeof(double);
  terlin2 = in2->strides[0]/sizeof(double);

  rsy = rbin1[1] -rbin1[0];
  rsx = rbin1[3] -rbin1[2];

  s = 0.;
  for (i=0; i<rsy; i++) {
    rein1 = (i +rbin1[0])*terlin1 +rbin1[2];
    rein2 = (i +rbin2[0])*terlin2 +rbin2[2];

    for (j=0; j<rsx; j++) {
      s = s +ptr_in1[rein1 +j]*ptr_in2[rein2 +j];
    }
  }

  return s;
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION
//
// Computes the camera offset and shift products for
// bicubic interpolation.
//
// Sets: offset[2], fsm[16]
//
static void bci_init(const double p[], npy_intp offset[], double fsm[]) {
  double cpy, cpx;
  double fshift[2];

  // Set the camera shifts.
  cpy = -p[0];
  cpx = -p[1];

  offset[0] = (npy_intp)floor(cpy);
  offset[1] = (npy_intp)floor(cpx);

  fshift[0] = cpy -offset[0];
  fshift[1] = cpx -offset[1];

  // Compute the products of the shifts.
  fsm[0]  = 1.;
  fsm[1]  = fshift[0];         // dy
  fsm[2]  = fshift[0]*fsm[1];  // dy^2
  fsm[3]  = fshift[0]*fsm[2];  // dy^3
  fsm[4]  = fshift[1];         // dx
  fsm[8]  = fshift[1]*fsm[4];  // dx^2
  fsm[12] = fshift[1]*fsm[8];  // dx^3
  fsm[5]  =  fsm[4]*fsm[1];    // dx*dy
  fsm[6]  =  fsm[4]*fsm[2];    // dx*dy^2
  fsm[7]  =  fsm[4]*fsm[3];    // dx*dy^3
  fsm[9]  =  fsm[8]*fsm[1];    // dx^2*dy
  fsm[10] =  fsm[8]*fsm[2];    // dx^2*dy^2
  fsm[11] =  fsm[8]*fsm[3];    // dx^2*dy^3
  fsm[13] = fsm[12]*fsm[1];    // dx^3*dy
  fsm[14] = fsm[12]*fsm[2];    // dx^3*dy^2
  fsm[15] = fsm[12]*fsm[3];    // dx^3*dy^3
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION
//
// Performs bicubic interpolation on the base pixel at reimin+xndx
// with shift products specified by fsm.
//
// Refer to Russell:1995 for more details.
//
// Returns: interpolated pixel.
//
static double bci_interp(double *ptr_imin,
                          double fsm[],
                          npy_intp reimin,
                          npy_intp refy,
                          npy_intp reffy,
                          npy_intp reby,
                          npy_intp xndx,
                          npy_intp mci) {

  static double bci_smat[256] = {
    1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
   -3., 0., 0., 3., 0., 0., 0., 0.,-2., 0., 0.,-1., 0., 0., 0., 0.,
    2., 0., 0.,-2., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    0., 0., 0., 0.,-3., 0., 0., 3., 0., 0., 0., 0.,-2., 0., 0.,-1.,
    0., 0., 0., 0., 2., 0., 0.,-2., 0., 0., 0., 0., 1., 0., 0., 1.,
   -3., 3., 0., 0.,-2.,-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0.,-3., 3., 0., 0.,-2.,-1., 0., 0.,
    9.,-9., 9.,-9., 6., 3.,-3.,-6., 6.,-6.,-3., 3., 4., 2., 1., 2.,
   -6., 6.,-6., 6.,-4.,-2., 2., 4.,-3., 3., 3.,-3.,-2.,-1.,-1.,-2.,
    2.,-2., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0., 0., 0., 2.,-2., 0., 0., 1., 1., 0., 0.,
   -6., 6.,-6., 6.,-3.,-3., 3., 3.,-4., 4., 2.,-2.,-2.,-2.,-1.,-1.,
    4.,-4., 4.,-4., 2., 2.,-2.,-2., 2.,-2.,-2., 2., 1., 1., 1., 1. };

  static double  cvec[16], fvec[16];
  static double  d_one  = 1.;
  static double  d_zero = 0.;
  static int     sdim   = 16;
  static int     one    = 1;
  double pix;

  fvec[0] = ptr_imin[reimin +xndx];
  fvec[1] = ptr_imin[reimin +xndx +1];
  fvec[2] = ptr_imin[refy +xndx +1];
  fvec[3] = ptr_imin[refy +xndx];

  // Compute the gradients.
  // fx
  fvec[4] = 0.5*(ptr_imin[reimin +xndx +1] -ptr_imin[reimin +xndx -1]);
  fvec[5] = 0.5*(ptr_imin[reimin +mci] -ptr_imin[reimin +xndx]);
  fvec[6] = 0.5*(ptr_imin[refy +mci] -ptr_imin[refy +xndx]);
  fvec[7] = 0.5*(ptr_imin[refy +xndx +1] -ptr_imin[refy +xndx -1]);

  // fy
  fvec[8]  = 0.5*(ptr_imin[refy +xndx] -ptr_imin[reby +xndx]);
  fvec[9]  = 0.5*(ptr_imin[refy +xndx +1] -ptr_imin[reby +xndx +1]);
  fvec[10] = 0.5*(ptr_imin[reffy +xndx +1] -ptr_imin[reimin +xndx +1]);
  fvec[11] = 0.5*(ptr_imin[reffy +xndx] -ptr_imin[reimin +xndx]);

  // fxy
  fvec[12] = 0.5*fvec[7] -0.25*(ptr_imin[reby +xndx +1] -ptr_imin[reby +xndx -1]);
  fvec[13] = 0.5*fvec[6] -0.25*(ptr_imin[reby +mci] -ptr_imin[reby +xndx]);
  fvec[14] = 0.25*(ptr_imin[reffy +mci] -ptr_imin[reffy +xndx]) -0.5*fvec[5];
  fvec[15] = 0.25*(ptr_imin[reffy +xndx +1] -ptr_imin[reffy +xndx -1]) -0.5*fvec[4];

  // Compute the coefficients.
  cblas_dgemv(CblasColMajor,CblasTrans,sdim,sdim,d_one,bci_smat,sdim,fvec,one,d_zero,cvec,one);

  // Compute the pixel value.
  pix = cblas_ddot(sdim,cvec,one,fsm,one);
  pix = fmax(pix,0.);
  pix = fmin(pix,1.);

  return pix;
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION
//
// Compares two double precision floating point values.
// Returns:
//   -1: a < b
//    0: a == b
//    1: a > b
//
static int compare_doubles(const void *a, const void *b) {
  double da, db;

  da = *( (const double *)a );
  db = *( (const double *)b );

  return (da > db) - (da < db);
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


static PyMethodDef pivlibcmethods[] = {
  {"bldex2msh", pivlibc_bldex2msh, METH_VARARGS, "Builds ExodusII mesh."},
  {"gfitcore", pivlibc_gfitcore, METH_VARARGS, "Fits gaussian to 3 points."},
  {"imsblicore", pivlibc_imsblicore, METH_VARARGS, "Applies image shift."},
  {"imsbcicore", pivlibc_imsbcicore, METH_VARARGS, "Applies image shift."},
  {"pxsbcicore", pivlibc_pxsbcicore, METH_VARARGS, "Applies image shift."},
  {"irinit", pivlibc_irinit, METH_VARARGS, "Builds gradient matrices."},
  {"irncc_core", pivlibc_irncc_core, METH_VARARGS,
   "Computes normalized cross-correlation coefficient."
  },
  {"irssda_core", pivlibc_irssda_core, METH_VARARGS,
   "Computes the sum of absolute differences similarity measure."
  },
  {"crcht_core", pivlibc_crcht_core, METH_VARARGS,
   "Computes the Hough transform matrix for circles."
  },
  {"wcxmedian_core", pivlibc_wcxmedian_core, METH_VARARGS,
   "Windowed, center-excluded median."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initpivlibc(void) {
  (void) Py_InitModule("pivlibc",pivlibcmethods);
  import_array();
}

