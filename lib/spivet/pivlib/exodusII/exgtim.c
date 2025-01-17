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
* exgtim - ex_get_time
*
* entry conditions - 
*   input parameters:
*       int     exoid                   exodus file id
*       int     time_step               time step number
*
* exit conditions - 
*       float   time_value              simulation time at specified step
*
* revision history - 
*
*  $Id: exgtim.c,v 1.5 2007/10/08 15:01:43 gdsjaar Exp $
*
*****************************************************************************/

#include <string.h>
#include "exodusII.h"
#include "exodusII_int.h"

/*!
 * reads the time value for a specified time step;
 * assume the first time step is 1
 */

int ex_get_time (int   exoid,
                 int   time_step,
                 void *time_value)
{
   int varid;
   long start[1];
   char var_name[MAX_VAR_NAME_LENGTH+1];
   char errmsg[MAX_ERR_LENGTH];

   exerrval = 0;        /* clear error code */

/* inquire previously defined dimensions  */

   strcpy (var_name, VAR_WHOLE_TIME);

/* inquire previously defined variable */

   if ((varid = ncvarid (exoid, var_name)) < 0)
   {
     exerrval = ncerr;
     sprintf(errmsg,
            "Error: failed to locate time variable in file id %d", exoid);
     ex_err("ex_get_time",errmsg,exerrval);
     return (EX_FATAL);
   }

/* read time value */

   start[0] = --time_step;

   if (ncvarget1 (exoid, varid, start,
              ex_conv_array(exoid,RTN_ADDRESS,time_value,1)) == -1)
   {
     exerrval = ncerr;
     sprintf(errmsg,
            "Error: failed to get time value in file id %d", exoid);
     ex_err("ex_get_time",errmsg,exerrval);
     return (EX_FATAL);
   }


   ex_conv_array( exoid, READ_CONVERT, time_value, 1 );

   return (EX_NOERR);
}
