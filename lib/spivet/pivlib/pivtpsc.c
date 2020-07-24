/*
Filename:  pivtpsc.c
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
    Thin-plate spline image registration functions.  Also includes
    the LAPJV optimization method of Jonker:1987.
*/

#include "Python.h"
#include "numpy/arrayobject.h"

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
//   bldtpsmat(pts,lmda,fpts,kn)
//
// Computes the system matrix for thin plate splines.  The structure
// of lmat is taken from Bookstein:1989 (matrix L).
//
// Returns lmat.
//
//
static PyObject *pivtpsc_bldtpsmat(PyObject *self, PyObject *args) {

    PyObject      *opts, *ofpts;
    PyArrayObject *pts, *fpts, *lmat;

    double *ptr_pts, *ptr_fpts, *ptr_lmat;

    double   lmda, pyc, pxc, fpyc, fpxc, ryc, rxc, val, eps;
    int      kn;
    npy_intp npts, i, j;

    npy_intp lmdim[2];

    if ( !PyArg_ParseTuple(args,
            "OdOi:sccost",
            &opts, &lmda, &ofpts, &kn ) )
        return NULL;

    pts  = (PyArrayObject *)PyArray_ContiguousFromAny(opts,PyArray_DOUBLE,1,2);
    fpts = (PyArrayObject *)PyArray_ContiguousFromAny(ofpts,PyArray_DOUBLE,2,2);

    if ( ( pts == NULL )
            || ( fpts == NULL ) ) {

        cleanup_po(2,pts,fpts);
        return NULL;
    }

    // Initialization.
    npts = PyArray_Size((PyObject *)pts);
    if ( npts == 2 )
        npts = 1;
    else
        npts = PyArray_DIMS(pts)[0];

    lmdim[0] = npts +3;
    lmdim[1] = kn +3;

    lmat = (PyArrayObject *)PyArray_ZEROS(2,lmdim,PyArray_DOUBLE,PyArray_CORDER);
    if ( lmat == NULL ) {
        cleanup_po(2,pts,fpts);
        return NULL;
    }

    ptr_pts  = (double *)PyArray_DATA(pts);
    ptr_fpts = (double *)PyArray_DATA(fpts);
    ptr_lmat = (double *)PyArray_DATA(lmat);

    eps = 1.e-6;

    // Build the K-section of lmat.
    for ( i=0; i<npts; i++ ) {
        pyc = ptr_pts[i*2];
        pxc = ptr_pts[i*2 +1];

        for ( j=0; j<kn; j++ ) {
            fpyc = ptr_fpts[j*2];
            fpxc = ptr_fpts[j*2 +1];

            ryc = pyc -fpyc;
            rxc = pxc -fpxc;

            val = ryc*ryc +rxc*rxc;
            val = ( val < eps ? 1. : val );

            ptr_lmat[i*lmdim[1] +j] = 0.5*val*log10(val);
        }
    }

    // Regularize if necessary and construct the lower P-section of lmat.
    if ( npts == kn ) {
        for ( i=0; i<npts; i++ ) {
            ptr_lmat[i*lmdim[1] +i] = ptr_lmat[i*lmdim[1] +i] +lmda;
        }

        for ( i=0; i<npts; i++ ) {
            ptr_lmat[npts*lmdim[1] +i]     = 1.;
            ptr_lmat[(npts+1)*lmdim[1] +i] = ptr_pts[i*2];
            ptr_lmat[(npts+2)*lmdim[1] +i] = ptr_pts[i*2 +1];
        }
    }

    // Construct the upper P-Section of lmat.
    for ( i=0; i<npts; i++ ) {
        ptr_lmat[i*lmdim[1] +kn]    = 1.;
        ptr_lmat[i*lmdim[1] +kn +1] = ptr_pts[i*2];
        ptr_lmat[i*lmdim[1] +kn +2] = ptr_pts[i*2 +1];
    }

    cleanup_po(2,pts,fpts);
    return PyArray_Return(lmat);
}

/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   sccost(tcxa,scxa,nsdmy)
//
// Computes the cost matrix for point correspondence using shape
// contexts.  The technique is that of Belongie:2002.
//
// Expects tcxa and scxa to be integer arrays.  tcxa and scxa do
// not have to contain the same number of points.  Dummy points
// will be used in case the number of points isn't equal.  Matching
// to a dummy point has a low cost of 0.0.
//
// nsdmy specifies the number of synthetic dummy points in addition
// to those arising from unequal number of points.  If nsdmy is
// greater than 0, then nsdmy dummy points will be added to each
// axis of the cmat.
//
// Return cmat.
//
static PyObject *pivtpsc_sccost(PyObject *self, PyObject *args) {

    PyObject      *otcxa, *oscxa;
    PyArrayObject *tcxa, *scxa, *cmat;

    double   *ptr_cmat;
    npy_intp *ptr_tcxa, *ptr_scxa, *tcx, *scx;

    double   cst, cstn, cstd, eps, mnval;
    int      nsdmy;
    npy_intp ntfpts, nsfpts, mxfpts, nrho, nth, nhbns, tp, sp, i;

    npy_intp cmdim[2];

    if ( !PyArg_ParseTuple(args,
            "OOi:sccost",
            &otcxa, &oscxa, &nsdmy ) )
        return NULL;

    tcxa = (PyArrayObject *)PyArray_ContiguousFromAny(otcxa,NPY_INTP,3,3);
    scxa = (PyArrayObject *)PyArray_ContiguousFromAny(oscxa,NPY_INTP,3,3);

    if ( ( tcxa == NULL )
            || ( scxa == NULL ) ) {

        cleanup_po(2,tcxa,scxa);
        return NULL;
    }

    // Initialization.
    ntfpts = PyArray_DIMS(tcxa)[0];
    nsfpts = PyArray_DIMS(scxa)[0];

    nrho  = PyArray_DIMS(tcxa)[1];
    nth   = PyArray_DIMS(tcxa)[2];
    nhbns = nrho*nth;

    mxfpts   = ( ntfpts > nsfpts ? ntfpts : nsfpts );
    mxfpts   = mxfpts +nsdmy;
    cmdim[0] = cmdim[1] = mxfpts;

    cmat = (PyArrayObject *)PyArray_EMPTY(2,cmdim,PyArray_DOUBLE,PyArray_CORDER);
    if ( cmat == NULL ) {
        cleanup_po(2,tcxa,scxa);
        return NULL;
    }

    ptr_tcxa = (npy_intp *)PyArray_DATA(tcxa);
    ptr_scxa = (npy_intp *)PyArray_DATA(scxa);
    ptr_cmat = (double   *)PyArray_DATA(cmat);

    eps   = 1.e-3;
    mnval = 0.;

    // Build the cost matrix.
    for ( tp=0; tp<ntfpts; tp++ ) {
        tcx = ptr_tcxa +tp*nhbns;

        for ( sp=0; sp<nsfpts; sp++ ) {
            scx = ptr_scxa +sp*nhbns;

            cst = 0.;
            for ( i=0; i<nhbns; i++ ) {
                cstd = tcx[i] +scx[i];
                cstd = ( cstd < eps ? eps : cstd );

                cstn = tcx[i] -scx[i];
                cstn = cstn*cstn;

                cst = cst +cstn/cstd;
            }

            ptr_cmat[tp*mxfpts +sp] = cst;
        }
    }

    // Fill in dummy features.
    for ( tp=ntfpts; tp<mxfpts; tp++ ) {
        for ( sp=0; sp<mxfpts; sp++ ) {
            ptr_cmat[tp*mxfpts +sp] = mnval;
        }
    }

    if ( nsfpts < mxfpts ) {
        for ( tp=0; tp<mxfpts; tp++ ) {
            for ( sp=nsfpts; sp<mxfpts; sp++ ) {
                ptr_cmat[tp*mxfpts +sp] = mnval;
            }
        }
    }

    cleanup_po(2,tcxa,scxa);
    return PyArray_Return(cmat);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   iscost(trgn,tfpts,srgn,sfpts,nsdmy)
//
// Computes the cost matrix for appearance similarity using a 3x3
// window around each feature point.  Cost is taken as the sum of
// squares difference in intensity between the 3x3 windows.
//
// tfpts and sfpts do not need to contain an equal number of points.
// Null values (if number of tfpts != sfpts) will have a cost of zero.
// tfpts and sfpts must be *x2 integer arrays of points ordered as [y,x],
// where * is the number of points (and can vary between the two sets).
//
// nsdmy specifies the number of synthetic dummy points in addition
// to those arising from unequal number of points.  If nsdmy is
// greater than 0, then nsdmy dummy points will be added to each
// axis of the cmat.
//
// Returns icmat.
//
static PyObject *pivtpsc_iscost(PyObject *self, PyObject *args) {

    PyObject      *otrgn, *otfpts, *osrgn, *osfpts;
    PyArrayObject *trgn, *tfpts, *srgn, *sfpts, *icmat;

    double   *ptr_trgn, *ptr_srgn, *ptr_icmat;
    npy_intp *ptr_tfpts, *ptr_sfpts;

    double   ddot, dlta;
    int      nsdmy;
    npy_intp ntfpts, nsfpts, mxfpts, tp, sp, i, j, tro, tco;
    npy_intp sro, sco, tfcrdy, tfcrdx, sfcrdy, sfcrdx;

    npy_intp icmdim[2], *trsize, *srsize;

    if ( !PyArg_ParseTuple(args,
            "OOOOi:iscost",
            &otrgn, &otfpts, &osrgn, &osfpts, &nsdmy ) )
        return NULL;

    trgn  = (PyArrayObject *)PyArray_ContiguousFromAny(otrgn,PyArray_DOUBLE,2,2);
    tfpts = (PyArrayObject *)PyArray_ContiguousFromAny(otfpts,NPY_INTP,2,2);
    srgn  = (PyArrayObject *)PyArray_ContiguousFromAny(osrgn,PyArray_DOUBLE,2,2);
    sfpts = (PyArrayObject *)PyArray_ContiguousFromAny(osfpts,NPY_INTP,2,2);

    if ( ( trgn == NULL )
            || ( tfpts == NULL )
            || ( srgn == NULL )
            || ( sfpts == NULL ) ) {

        cleanup_po(4,trgn,tfpts,srgn,sfpts);
        return NULL;
    }

    // Initialization.
    ntfpts = PyArray_DIMS(tfpts)[0];
    nsfpts = PyArray_DIMS(sfpts)[0];

    trsize = PyArray_DIMS(trgn);
    srsize = PyArray_DIMS(srgn);

    mxfpts    = ( ntfpts > nsfpts ? ntfpts : nsfpts );
    mxfpts    = mxfpts +nsdmy;
    icmdim[0] = icmdim[1] = mxfpts;

    icmat = (PyArrayObject *)PyArray_ZEROS(2,icmdim,PyArray_DOUBLE,PyArray_CORDER);

    if ( icmat == NULL ) {
        cleanup_po(4,trgn,tfpts,srgn,sfpts);
        return NULL;
    }

    ptr_trgn  = (double   *)PyArray_DATA(trgn);
    ptr_tfpts = (npy_intp *)PyArray_DATA(tfpts);
    ptr_srgn  = (double   *)PyArray_DATA(srgn);
    ptr_sfpts = (npy_intp *)PyArray_DATA(sfpts);
    ptr_icmat = (double   *)PyArray_DATA(icmat);

    // Build the cost matrix.
    for ( tp=0; tp<ntfpts; tp++ ) {
        tfcrdy = ptr_tfpts[tp*2];
        tfcrdx = ptr_tfpts[tp*2 +1];

        for ( sp=0; sp<nsfpts; sp++ ) {
            sfcrdy = ptr_sfpts[sp*2];
            sfcrdx = ptr_sfpts[sp*2 +1];

            ddot = 0.;
            for ( i=-1; i<2; i++ ) {
                tco = tfcrdy +i;
                if ( ( tco < 0 ) || ( tco >= trsize[0] ) )
                    continue;

                sco = sfcrdy +i;
                if ( ( sco < 0 ) || ( sco >= srsize[0] ) )
                    continue;

                tco = tco*trsize[1];
                sco = sco*srsize[1];

                for ( j=-1; j<2; j++ ) {
                    tro = tfcrdx +j;
                    if ( ( tro < 0 ) || ( tro >= trsize[1] ) )
                        continue;

                    sro = sfcrdx +j;
                    if ( ( sro < 0 ) || ( sro >= srsize[1] ) )
                        continue;

                    dlta = ptr_trgn[tco +tro] -ptr_srgn[sco +sro];

                    ddot = ddot +dlta*dlta;
                }
            }
            ptr_icmat[tp*mxfpts +sp] = ddot;
        }
    }

    cleanup_po(4,trgn,tfpts,srgn,sfpts);
    return PyArray_Return(icmat);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   spcost(trmat,nbrndx,tfndx,rcmap,cwtfpts,sfpts,nsdmy)
//
// Computes a cost matrix based on force from a spring model
// of feature points connected to neighboring points.  The method
// is based on that of Okamoto:1995 and works as follows.  For
// each candidate template point, the warped neighbors of the point
// are extracted.  Then the total force on each search region point
// is computed using the template point's warped neighbors.
//
// trmat is a matrix of radii from one template feature point to all
// other template feature points.  The rows and columns should be
// ordered the same as tfpts.  trmat should not be sorted.  trmat
// should not be truncated to candidate points.
//
// nbrndx is a 2D array of indices into trmat specifying neighbors
// of a given feature point.  If the template region has ntfpts
// template feature points, and each point has 5 neighbors, then
// nbrndx is a ntfpts x 5 array.  nbrndx should not be truncated to
// candidate points.
//
// tfndx is a 1D array of indices into the rows of trmat providing the
// candidate feature points.  Truncated to candidate points.
//
// rcmap is the latest row to column map array from lapjv().  The length
// of tfndx and rcmap must be the same.  Truncated to candidate points.
//
// cwtfpts is the warped set of candidate template points that have been
// distorted using the latest warp estimate.  cwtfpts is an mx2 integer
// array of points ordered as [y,x], where m is the number of candidate
// points.  Truncated to candidate points.
//
// sfpts is the full array of search region feature points.  sfpts must be
// an lx2 integer arrays of points ordered as [y,x], where l is the number
// of points.
//
// nsdmy specifies the number of synthetic dummy points in addition
// to those arising from unequal number of points.  If nsdmy is
// greater than 0, then nsdmy dummy points will be added to each
// axis of the cmat.  Dummy points will have a cost of 0.0.
//
// Returns spmat.
//
static PyObject *pivtpsc_spcost(PyObject *self, PyObject *args) {

    PyObject      *otrmat, *onbrndx, *otfndx, *orcmap, *osfpts, *ocwtfpts;
    PyArrayObject *trmat, *nbrndx, *tfndx, *rcmap, *sfpts, *spmat, *cwtfpts;

    double   *ptr_trmat, *ptr_spmat;
    npy_intp *ptr_nbrndx, *ptr_tfndx, *ptr_rcmap, *ptr_sfpts, *t2cndx, *ptr_cwtfpts;

    double   trad, force, srad, srady, sradx;
    int      nsdmy;
    npy_intp ntfpts, nctfpts, nsfpts, nnbrs, mxfpts, tp, sp, i;
    npy_intp ptfndx, tnbr, ncnt;

    npy_intp spmdim[2];

    if ( !PyArg_ParseTuple(args,
            "OOOOOOi:spcost",
            &otrmat, &onbrndx, &otfndx, &orcmap, &ocwtfpts, &osfpts, &nsdmy ) )
        return NULL;

    trmat   = (PyArrayObject *)PyArray_ContiguousFromAny(otrmat,PyArray_DOUBLE,2,2);
    nbrndx  = (PyArrayObject *)PyArray_ContiguousFromAny(onbrndx,NPY_INTP,2,2);
    tfndx   = (PyArrayObject *)PyArray_ContiguousFromAny(otfndx,NPY_INTP,1,1);
    rcmap   = (PyArrayObject *)PyArray_ContiguousFromAny(orcmap,NPY_INTP,1,1);
    cwtfpts = (PyArrayObject *)PyArray_ContiguousFromAny(ocwtfpts,NPY_INTP,2,2);
    sfpts   = (PyArrayObject *)PyArray_ContiguousFromAny(osfpts,NPY_INTP,2,2);

    if ( ( trmat == NULL )
            || ( nbrndx == NULL )
            || ( tfndx == NULL )
            || ( rcmap == NULL )
            || ( cwtfpts == NULL )
            || ( sfpts == NULL ) ) {

        cleanup_po(6,trmat,nbrndx,tfndx,rcmap,cwtfpts,sfpts);
        return NULL;
    }

    // Initialization.
    ntfpts  = PyArray_DIMS(trmat)[0];           // Number of template points.
    nctfpts = PyArray_Size((PyObject *)tfndx);  // Number of candidate points.
    nsfpts  = PyArray_DIMS(sfpts)[0];
    nnbrs   = PyArray_DIMS(nbrndx)[1];

    mxfpts    = ( nctfpts > nsfpts ? nctfpts : nsfpts );
    mxfpts    = mxfpts +nsdmy;
    spmdim[0] = spmdim[1] = mxfpts;

    spmat = (PyArrayObject *)PyArray_ZEROS(2,spmdim,PyArray_DOUBLE,PyArray_CORDER);

    if ( spmat == NULL ) {
        cleanup_po(6,trmat,nbrndx,tfndx,rcmap,cwtfpts,sfpts);
        return NULL;
    }

    ptr_trmat   = (double   *)PyArray_DATA(trmat);
    ptr_nbrndx  = (npy_intp *)PyArray_DATA(nbrndx);
    ptr_tfndx   = (npy_intp *)PyArray_DATA(tfndx);
    ptr_rcmap   = (npy_intp *)PyArray_DATA(rcmap);
    ptr_cwtfpts = (npy_intp *)PyArray_DATA(cwtfpts);
    ptr_sfpts   = (npy_intp *)PyArray_DATA(sfpts);
    ptr_spmat   = (double   *)PyArray_DATA(spmat);

    t2cndx = (npy_intp *)PyMem_Malloc(ntfpts*sizeof(npy_intp));
    if ( t2cndx == NULL ) {
        cleanup_po(7,spmat,trmat,nbrndx,tfndx,rcmap,cwtfpts,sfpts);
        return NULL;
    }

    // Build a map from the full set of template feature points to
    // the candidate points.  This is the inverse of tfndx.
    for ( i=0; i<ntfpts; i++ )
        t2cndx[i] = -1.;

    for ( i=0; i<nctfpts; i++ ) {
        ptfndx         = ptr_tfndx[i];
        t2cndx[ptfndx] = i;
    }

    // Build the cost matrix.  The outer loop is over candidate template
    // points.
    for ( tp=0; tp<nctfpts; tp++ ) {
        ptfndx = ptr_tfndx[tp];   // Index into full set of template points.

        for ( sp=0; sp<nsfpts; sp++ ) {
            force = 0.;
            ncnt  = 0;
            for ( i=0; i<nnbrs; i++ ) {
                tnbr = ptr_nbrndx[ptfndx*nnbrs +i];  // Template neighbor index.
                trad = ptr_trmat[ptfndx*ntfpts +tnbr];

                tnbr = t2cndx[tnbr];  // tnbr candidate index.
                if ( tnbr < 0 )
                    continue;   // Not a candidate.

                srady = ptr_sfpts[sp*2] -ptr_cwtfpts[tnbr*2];
                sradx = ptr_sfpts[sp*2 +1] -ptr_cwtfpts[tnbr*2 +1];
                srad  = sqrt( srady*srady +sradx*sradx );

                force = force +fabs( srad -trad )/trad;
                ncnt++;
            }
            ptr_spmat[tp*mxfpts +sp] = force/fmax(1.,ncnt);
        }
    }

    PyMem_Free(t2cndx);
    cleanup_po(6,trmat,nbrndx,tfndx,rcmap,cwtfpts,sfpts);
    return PyArray_Return(spmat);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   shpctxt(pt,fpts,rmax,rbins,tbins)
//
// Computes the shape context for feature point pt in the context of
// the collection of points, fpts.  See documentation for pivutil.shpctxt()
// for more details.
//
// Returns sctxt, an rbins x tbins integer array.
//
static PyObject *pivtpsc_shpctxt(PyObject *self, PyObject *args) {

    PyObject      *ofpts;
    PyArrayObject *fpts, *sctxt;

    double   *ptr_fpts, *rad, *theta, *rbe, *tbe;
    npy_intp *ptr_sctxt;

    double   pt[2], rmax, rbsz, tbsz, ryc, rxc, rave, rval, tval;
    int      rbins, tbins;
    npy_intp npts, rcrd, tcrd, i, j;

    npy_intp scshp[2];

    if ( !PyArg_ParseTuple(args,
            "(dd)Odii:shpctxt",
            pt,pt+1,&ofpts,&rmax,&rbins,&tbins) )
        return NULL;

    fpts = (PyArrayObject *)PyArray_ContiguousFromAny(ofpts,PyArray_DOUBLE,1,2);
    if ( fpts == NULL )
        return NULL;

    // Initialization.
    npts = PyArray_Size((PyObject *)fpts);
    if ( npts == 2 )
        npts = 1;
    else
        npts = PyArray_DIMS(fpts)[0];

    scshp[0] = rbins;
    scshp[1] = tbins;
    sctxt    = (PyArrayObject *)PyArray_ZEROS(2,scshp,NPY_INTP,PyArray_CORDER);

    rad   = (double *)PyMem_Malloc(npts*sizeof(double));
    theta = (double *)PyMem_Malloc(npts*sizeof(double));
    rbe   = (double *)PyMem_Malloc((rbins +1)*sizeof(double));
    tbe   = (double *)PyMem_Malloc((tbins +1)*sizeof(double));

    if ( (sctxt == NULL)
            || (rad==NULL)
            || (theta==NULL)
            || (rbe == NULL)
            || (tbe == NULL) ) {
        cleanup_po(2,fpts,sctxt);

        if (rad!=NULL)
            PyMem_Free(rad);
        if (theta!=NULL)
            PyMem_Free(theta);
        if (rbe!=NULL)
            PyMem_Free(rbe);
        if (tbe!=NULL)
            PyMem_Free(tbe);

        return NULL;
    }

    ptr_fpts  = (double   *)PyArray_DATA(fpts);
    ptr_sctxt = (npy_intp *)PyArray_DATA(sctxt);

    tbsz = 2.*M_PI/tbins;

    for ( i=0; i<tbins +1; i++ )
        tbe[i] = -M_PI +i*tbsz;

    // Compute the radius and theta values.
    rave = 0.;
    for ( i=0; i<npts; i++ ) {
        ryc    = ptr_fpts[i*2] -pt[0];
        rxc    = ptr_fpts[i*2 +1] -pt[1];
        rad[i] = sqrt(ryc*ryc +rxc*rxc);

        rave = rave +rad[i];

        theta[i] = atan2(ryc,rxc);
    }
    rave = rave/npts;

    rbsz = log10(rmax)/rbins;
    for ( i=0; i<rbins +1; i++ )
        rbe[i] = i*rbsz;

    for ( i=0; i<npts; i++ )
        rad[i] = log10(rad[i]);

    // Compute the histogram.
    for ( i=0; i<npts; i++ ) {
        rval = rad[i];
        tval = theta[i];

        // Some log radii can be larger than the highest bin.
        // Just store them in the high bin.
        rcrd = rbins -1;
        for ( j=1; j<=rbins; j++ ) {
            if ( rval < rbe[j] ) {
                rcrd = j -1;
                break;
            }
        }

        tcrd = 0;
        for ( j=1; j<=tbins; j++ ) {
            if ( tval < tbe[j] ) {
                tcrd = j -1;
                break;
            }
        }

        ptr_sctxt[rcrd*tbins +tcrd]++;
    }


    PyMem_Free(rad);
    PyMem_Free(theta);
    PyMem_Free(rbe);
    PyMem_Free(tbe);
    cleanup_po(1,fpts);

    return PyArray_Return(sctxt);
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   lapjv(cmat)
//
// Solves the linear assignment problem for a bipartite graph.
// The solution method is taken directly from Jonker:1987.
//
// cmat is a 2D, floating point array of costs.
//
// Returns [optcst,rcmap,crmap].
//   optcst is the optimum cost for the arrangement.
//   rcmap yields the column minimum for each row.
//   crmap yields the row minimum for each column.
//
//
static PyObject *pivtpsc_lapjv(PyObject *self, PyObject *args) {

    PyObject      *ocmat, *rlst;
    PyArrayObject *cmat, *crmap, *rcmap;

    double   *ptr_cmat, *uvec, *vvec, *dvec;
    npy_intp *ptr_rcmap, *ptr_crmap, *frrws, *rcflg, *pred, *col;

    double   mnval, val, deps, dmax, u1, u2, optcst;
    npy_intp i, j, k, l, cmrndx, f, f0, cndx, cndx2, rndx, last, low, up, augflg;

    npy_intp n;

    if ( !PyArg_ParseTuple(args,"O:lapjv",&ocmat) )
        return NULL;

    cmat = (PyArrayObject *)PyArray_ContiguousFromAny(ocmat,PyArray_DOUBLE,2,2);
    if ( cmat == NULL )
        return NULL;

    // Initialization.
    n = PyArray_DIMS(cmat)[0];
    if ( n != PyArray_DIMS(cmat)[1] ) {
        cleanup_po(1,cmat);
        return PyErr_Format(PyExc_TypeError,"cmat must be square.");
    }

    rcmap = (PyArrayObject *)PyArray_ZEROS(1,&n,NPY_INTP,PyArray_CORDER);
    crmap = (PyArrayObject *)PyArray_ZEROS(1,&n,NPY_INTP,PyArray_CORDER);

    uvec  = (double   *)PyMem_Malloc(n*sizeof(double));
    vvec  = (double   *)PyMem_Malloc(n*sizeof(double));
    dvec  = (double   *)PyMem_Malloc(n*sizeof(double));
    rcflg = (npy_intp *)PyMem_Malloc(n*sizeof(npy_intp));
    frrws = (npy_intp *)PyMem_Malloc(n*sizeof(npy_intp));
    pred  = (npy_intp *)PyMem_Malloc(n*sizeof(npy_intp));
    col   = (npy_intp *)PyMem_Malloc(n*sizeof(npy_intp));

    if ( (rcmap == NULL)
            || (crmap==NULL)
            || (uvec==NULL)
            || (vvec==NULL)
            || (dvec==NULL)
            || (frrws==NULL)
            || (rcflg==NULL)
            || (pred==NULL)
            || (col==NULL) ) {
        cleanup_po(3,cmat,rcmap,crmap);
        if (uvec != NULL)
            PyMem_Free(uvec);
        if (vvec != NULL)
            PyMem_Free(vvec);
        if (dvec != NULL)
            PyMem_Free(dvec);
        if (frrws != NULL)
            PyMem_Free(frrws);
        if (rcflg != NULL)
            PyMem_Free(rcflg);
        if (pred != NULL)
            PyMem_Free(pred);
        if (col != NULL)
            PyMem_Free(col);
        return NULL;
    }

    ptr_cmat  = (double   *)PyArray_DATA(cmat);
    ptr_rcmap = (npy_intp *)PyArray_DATA(rcmap);
    ptr_crmap = (npy_intp *)PyArray_DATA(crmap);

    dmax = DBL_MAX;
    deps = DBL_EPSILON;

    // Keep the compiler happy.
    mnval = 0.;
    cndx2 = last = 0;

    // rcflg = 0 when no match, 1 when matched once, 2 if already matched.
    for ( i=0; i<n; i++ ) {
        rcflg[i] = 0;
        col[i]   = i;
    }

    // Column reduction step.
    for ( j=n-1; j>=0; j--) {
        mnval  = ptr_cmat[j];
        cmrndx = 0;

        // Find minimum cost in the column.
        for ( i=1; i<n; i++) {
            val = ptr_cmat[i*n +j];
            if ( val < mnval ) {
                mnval  = val;
                cmrndx = i;
            }
        }
        vvec[j] = mnval;

        // If a column hasn't been assigned to the cmrndx row, do so.
        // If the row already has a column, then set crmap to -1.
        if ( rcflg[cmrndx] == 0 ) {
            rcflg[cmrndx]     = 1;
            ptr_rcmap[cmrndx] = j;
            ptr_crmap[j]      = cmrndx;
        }
        else {
            rcflg[cmrndx] = 2;
            ptr_crmap[j]  = -1;
        }
    }

    // Reduction transfer step.
    f = 0;
    for ( i=0; i<n; i++ ) {
        if ( rcflg[i] == 0 ) {
            frrws[f] = i;
            f++;
        }
        else if ( rcflg[i] == 1 ) {
            cndx  = ptr_rcmap[i];
            mnval = dmax;
            for ( j=0; j<n; j++ ) {
                if ( j != cndx ) {
                    val = ptr_cmat[i*n +j] -vvec[j];
                    if ( val < mnval )
                        mnval = val;
                }
            }
            vvec[cndx] = vvec[cndx] -mnval;
        }
    }

    // Augmenting row reduction step.
    for ( l=0; l<2; l++ ) {
        k  = 0;
        f0 = f;
        f  = 0;
        while ( k < f0 ) {
            i    = frrws[k];
            k    = k +1;
            u1   = ptr_cmat[i*n] -vvec[0];
            cndx = 0;
            u2   = dmax;
            for ( j=1; j<n; j++ ) {
                mnval = ptr_cmat[i*n +j] -vvec[j];
                if ( mnval < u2 ) {
                    if ( mnval >= u1 ) {
                        u2    = mnval;
                        cndx2 = j;
                    }
                    else {
                        u2    = u1;
                        u1    = mnval;
                        cndx2 = cndx;
                        cndx  = j;
                    }
                }
            }
            rndx = ptr_crmap[cndx];
            if ( u1 < u2 )
                vvec[cndx] = vvec[cndx] -u2 +u1;
            else if ( rndx >= 0 ) {
                cndx = cndx2;
                rndx = ptr_crmap[cndx];
            }
            if ( rndx >= 0 ) {
                if ( u1 < u2 ) {
                    k        = k -1;
                    frrws[k] = rndx;
                }
                else {
                    frrws[f] = rndx;
                    f        = f +1;   // Typo in paper.  Must go here, or inf loop.
                }
            }
            ptr_rcmap[i]    = cndx;
            ptr_crmap[cndx] = i;
        }
    }

    // Augmentation step.
    f0 = f;
    for ( f=0; f<f0; f++ ) {
        rndx = frrws[f];
        low  = 0;
        up   = 0;
        for ( j=0; j<n; j++ ) {
            dvec[j] = ptr_cmat[rndx*n +j] -vvec[j];
            pred[j] = rndx;
        }

        augflg = 0;
        while (1) {
            if ( up == low ) {
                last  = low -1;
                mnval = dvec[col[up]];
                up++;

                for ( k=up; k<n; k++ ) {
                    j   = col[k];
                    val = dvec[j];
                    if ( val <= mnval ) {
                        if ( val < mnval ) {
                            up    = low;
                            mnval = val;
                        }
                        col[k]  = col[up];
                        col[up] = j;
                        up++;
                    }
                }

                for ( k=low; k<up; k++ ) {
                    j = col[k];
                    if ( ptr_crmap[j] < 0 ) {
                        augflg = 1;
                        break;
                    }
                }
            }

            if ( !augflg ) {
                cndx = col[low];
                i    = ptr_crmap[cndx];
                low++;

                u1 = ptr_cmat[i*n +cndx] -vvec[cndx] -mnval;
                for ( k=up; k<n; k++ ) {
                    j   = col[k];
                    val = ptr_cmat[i*n +j] -vvec[j] -u1;
                    if ( val < dvec[j] ) {
                        dvec[j] = val;
                        pred[j] = i;
                        if ( fabs(val -mnval) < fabs(mnval*deps) ) {
                            if ( ptr_crmap[j] < 0 ) {
                                augflg = 1;
                                break;
                            }
                            else {
                                col[k]  = col[up];
                                col[up] = j;
                                up++;
                            }
                        }
                    }
                }
            }

            if (augflg)
                break;
        } // while (1)

        for ( k=0; k<=last; k++ ) {
            cndx       = col[k];
            vvec[cndx] = vvec[cndx] +dvec[cndx] -mnval;
        }

        while (1) {
            i            = pred[j];
            ptr_crmap[j] = i;
            k            = j;
            j            = ptr_rcmap[i];
            ptr_rcmap[i] = k;

            if ( i == rndx )
                break;
        }
    } // for (f)

    val = 0;
    for ( i=0; i<n; i++ ) {
        j       = ptr_rcmap[i];
        uvec[i] = ptr_cmat[i*n +j] -vvec[j];
        val     = val +uvec[i] +vvec[j];
    }
    optcst = val;

    PyMem_Free(uvec);
    PyMem_Free(vvec);
    PyMem_Free(dvec);
    PyMem_Free(rcflg);
    PyMem_Free(frrws);
    PyMem_Free(pred);
    PyMem_Free(col);
    cleanup_po(1,cmat);

    rlst = PyList_New(3);
    PyList_SetItem(rlst,0,Py_BuildValue("d",optcst));
    PyList_SetItem(rlst,1,PyArray_Return(rcmap));
    PyList_SetItem(rlst,2,PyArray_Return(crmap));

    return rlst;
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


static PyMethodDef pivtpscmethods[] = {
    {"lapjv",pivtpsc_lapjv,METH_VARARGS,"Solves linear assignment problem."},
    {"shpctxt",pivtpsc_shpctxt,METH_VARARGS,"Computes the shape context."},
    {"iscost",pivtpsc_iscost,METH_VARARGS,"Computes cost based on intensity similarity."},
    {"sccost",pivtpsc_sccost,METH_VARARGS,"Computes cost matrix using shape contexts."},
    {"spcost",pivtpsc_spcost,METH_VARARGS,"Computes cost using a spring model"},
    {"bldtpsmat",pivtpsc_bldtpsmat,METH_VARARGS,"Computes the TPS matrix."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initpivtpsc(void) {
  (void) Py_InitModule("pivtpsc",pivtpscmethods);
  import_array();
}
