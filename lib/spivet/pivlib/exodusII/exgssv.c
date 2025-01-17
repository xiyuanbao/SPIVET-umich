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
/*****************************************************************************
*
* exgssv - ex_get_sset_var
*
* entry conditions - 
*   input parameters:
*       int     exoid                   exodus file id
*       int     time_step               time step number
*       int     sset_var_index          sideset variable index
*       int     sset_blk_id             sideset id
*       int     num_side_this_sset       number of sides in this sideset
*       
*
* exit conditions - 
*       float*  sset_var_vals           array of sideset variable values
*
*
* revision history - 
*
*  $Id: exgssv.c,v 1.4 2007/10/08 15:01:43 gdsjaar Exp $
*
*****************************************************************************/

#include "exodusII.h"
#include "exodusII_int.h"

/*!
 * reads the values of a single sideset variable for one sideset at 
 * one time step in the database; assume the first time step and
 * sideset variable index is 1
 */

int ex_get_sset_var (int   exoid,
                     int   time_step,
                     int   sset_var_index,
                     int   sset_id, 
                     int   num_side_this_sset,
                     void *sset_var_vals)
{
   int varid, sset_id_ndx;
   long start[2], count[2];
   char errmsg[MAX_ERR_LENGTH];

   exerrval = 0; /* clear error code */

  /* Determine index of sset_id in VAR_SS_IDS array */
  sset_id_ndx = ex_id_lkup(exoid,VAR_SS_IDS,sset_id);
  if (exerrval != 0) 
  {
    if (exerrval == EX_NULLENTITY)
    {
      sprintf(errmsg,
              "Warning: no sideset variables for NULL sideset %d in file id %d",
              sset_id,exoid);
      ex_err("ex_get_sset_var",errmsg,EX_MSG);
      return (EX_WARN);
    }
    else
    {
      sprintf(errmsg,
     "Error: failed to locate sideset id %d in %s variable in file id %d",
              sset_id, VAR_ID_EL_BLK, exoid);
      ex_err("ex_get_sset_var",errmsg,exerrval);
      return (EX_FATAL);
    }
  }


/* inquire previously defined variable */

   if((varid=ncvarid(exoid,VAR_SS_VAR(sset_var_index,sset_id_ndx))) == -1)
   {
     exerrval = ncerr;
     sprintf(errmsg,
          "Error: failed to locate sideset variable %d for sideset %d in file id %d",
          sset_var_index,sset_id,exoid); /* this msg needs to be improved */
     ex_err("ex_get_sset_var",errmsg,exerrval);
     return (EX_FATAL);
   }

/* read values of sideset variable */

   start[0] = --time_step;
   start[1] = 0;

   count[0] = 1;
   count[1] = num_side_this_sset;

   if (ncvarget (exoid, varid, start, count,
        ex_conv_array(exoid,RTN_ADDRESS,sset_var_vals,num_side_this_sset)) == -1)
   {
     exerrval = ncerr;
     sprintf(errmsg,
        "Error: failed to get sset var %d for block %d in file id %d",
             sset_var_index,sset_id,exoid);/*this msg needs to be improved*/
     ex_err("ex_get_sset_var",errmsg,exerrval);
     return (EX_FATAL);
   }


   ex_conv_array( exoid, READ_CONVERT, sset_var_vals, num_side_this_sset );

   return (EX_NOERR);
}
