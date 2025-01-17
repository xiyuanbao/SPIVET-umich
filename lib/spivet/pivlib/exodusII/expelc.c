/*
 * Copyright (c) 2005 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Governement
 * retains certain rights in this software.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.  
 * 
 *     * Neither the name of Sandia Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */
/*  $Id: expelc.c,v 1.5 2007/10/08 15:01:45 gdsjaar Exp $ */

#include "exodusII.h"
#include "exodusII_int.h"
#include <stdlib.h> /* for free() */

/*!
 * writes the connectivity array for an element block
 */

int ex_put_elem_conn (int   exoid,
                      int   elem_blk_id,
                      const int  *connect)
{
   int numelbdim, nelnoddim, connid, elem_blk_id_ndx, iresult;
   long num_elem_this_blk, num_nod_per_elem, start[2], count[2]; 
   char errmsg[MAX_ERR_LENGTH];

   exerrval = 0; /* clear error code */

  /* Determine index of elem_blk_id in VAR_ID_EL_BLK array */
  elem_blk_id_ndx = ex_id_lkup(exoid,VAR_ID_EL_BLK,elem_blk_id);
  if (exerrval != 0) 
  {
    if (exerrval == EX_NULLENTITY)
     {
       sprintf(errmsg,
"Warning: connectivity array not allowed for NULL element block %d in file id %d",
               elem_blk_id,exoid);
       ex_err("ex_put_elem_conn",errmsg,EX_MSG);
       return (EX_WARN);
     }
     else
     {

      sprintf(errmsg,
        "Error: failed to locate element block id %d in %s array in file id %d",
              elem_blk_id,VAR_ID_EL_BLK, exoid);
      ex_err("ex_put_elem_conn",errmsg,exerrval);
      return (EX_FATAL);
    }
  }

/* inquire id's of previously defined dimensions  */

   if ((numelbdim = ncdimid (exoid, DIM_NUM_EL_IN_BLK(elem_blk_id_ndx))) == -1)
   {
     exerrval = ncerr;
     sprintf(errmsg,
     "Error: failed to locate number of elements in block %d in file id %d",
              elem_blk_id,exoid);
     ex_err("ex_put_elem_conn",errmsg, exerrval);
     return(EX_FATAL);
   }

   if (ncdiminq(exoid, numelbdim, NULL, &num_elem_this_blk) == -1)
   {
     exerrval = ncerr;
     sprintf(errmsg,
            "Error: failed to get number of elements in block %d in file id %d",
             elem_blk_id,exoid);
     ex_err("ex_put_elem_conn",errmsg,exerrval);
     return(EX_FATAL);
   }


   if ((nelnoddim = ncdimid (exoid, DIM_NUM_NOD_PER_EL(elem_blk_id_ndx))) == -1)
   {
     exerrval = ncerr;
     sprintf(errmsg,
       "Error: failed to locate number of nodes/elem in block %d in file id %d",
             elem_blk_id,exoid);
     ex_err("ex_put_elem_conn",errmsg,exerrval);
     return(EX_FATAL);
   }

   if (ncdiminq (exoid, nelnoddim, NULL, &num_nod_per_elem) == -1)
   {
     exerrval = ncerr;
     sprintf(errmsg,
       "Error: failed to get number of nodes/elem in block %d in file id %d",
             elem_blk_id,exoid);
     ex_err("ex_put_elem_conn",errmsg,exerrval);
     return(EX_FATAL);
   }


   if ((connid = ncvarid (exoid, VAR_CONN(elem_blk_id_ndx))) == -1)
   {
     exerrval = ncerr;
     sprintf(errmsg,
"Error: failed to locate connectivity array for element block %d in file id %d",
             elem_blk_id,exoid);
     ex_err("ex_put_elem_conn",errmsg, exerrval);
     return(EX_FATAL);
   }


   /* write out the connectivity array */
   start[0] = 0;
   start[1] = 0;

   count[0] = num_elem_this_blk;
   count[1] = num_nod_per_elem;

   iresult = ncvarput (exoid, connid, start, count, connect);

   if (iresult == -1)
   {
      exerrval = ncerr;
      sprintf(errmsg,
         "Error: failed to write connectivity array for block %d in file id %d",
                elem_blk_id,exoid);
      ex_err("ex_put_elem_conn",errmsg, exerrval);
      return(EX_FATAL);
   }


   return (EX_NOERR);

}
