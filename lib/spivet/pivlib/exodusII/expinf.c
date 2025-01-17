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
* expinf - ex_put_info
*
* entry conditions - 
*   input parameters:
*       int     exoid                   exodus file id
*       char*   info[]                  ptr array of info records
*
* exit conditions - 
*
* revision history - 
*
*  $Id: expinf.c,v 1.5 2007/10/08 15:01:46 gdsjaar Exp $
*
*****************************************************************************/

#include "exodusII.h"
#include "exodusII_int.h"
#include <string.h>

/*!
 * writes information records to the database
 */

int ex_put_info (int   exoid, 
                 int   num_info,
                 char *info[])
{
   int i, lindim, num_info_dim, dims[2], varid;
   long start[2], count[2];
   char errmsg[MAX_ERR_LENGTH];

   exerrval = 0; /* clear error code */

/* only do this if there are records */

   if (num_info > 0)
   {
/*   inquire previously defined dimensions  */

     if ((lindim = ncdimid (exoid, DIM_LIN)) == -1)
     {
       exerrval = ncerr;
       sprintf(errmsg,
              "Error: failed to get line string length in file id %d", exoid);
       ex_err("ex_put_info",errmsg,exerrval);
       return (EX_FATAL);
     }


/*   put file into define mode  */

     if (ncredef (exoid) == -1)
     {
       exerrval = ncerr;
       sprintf(errmsg,
              "Error: failed put file id %d into define mode", exoid);
       ex_err("ex_put_info",errmsg,exerrval);
       return (EX_FATAL);
     }


/*   define dimensions */

     if ((num_info_dim = ncdimdef (exoid, DIM_NUM_INFO, (long)num_info)) == -1)
     {
       if (ncerr == NC_ENAMEINUSE)      /* duplicate entry? */
       {
         exerrval = ncerr;
         sprintf(errmsg,
              "Error: info records already exist in file id %d", 
               exoid);
         ex_err("ex_put_info",errmsg,exerrval);
       }
       else
       {
         exerrval = ncerr;
         sprintf(errmsg,
              "Error: failed to define number of info records in file id %d",
               exoid);
         ex_err("ex_put_info",errmsg,exerrval);
       }

     goto error_ret;         /* exit define mode and return */
     }


/* define variable  */

     dims[0] = num_info_dim;
     dims[1] = lindim;

     if ((varid = ncvardef (exoid, VAR_INFO, NC_CHAR, 2, dims)) == -1)
     {
       exerrval = ncerr;
       sprintf(errmsg,
              "Error: failed to define info record in file id %d",
               exoid);
       ex_err("ex_put_info",errmsg,exerrval);
       goto error_ret;         /* exit define mode and return */
     }


/*   leave define mode  */

     if (ncendef (exoid) == -1)
     {
       exerrval = ncerr;
       sprintf(errmsg,
              "Error: failed to complete info record definition in file id %d",
               exoid);
       ex_err("ex_put_info",errmsg,exerrval);
       return (EX_FATAL);
     }


/* write out information records */

     for (i=0; i<num_info; i++)
     {
       start[0] = i;
       start[1] = 0;

       count[0] = 1;
       count[1] = strlen(info[i]) + 1;

       if (ncvarput (exoid, varid, start, count, (void*) info[i]) == -1)
       {
         exerrval = ncerr;
         sprintf(errmsg,
                "Error: failed to store info record in file id %d",
                 exoid);
         ex_err("ex_put_info",errmsg,exerrval);
         return (EX_FATAL);
       }
     }
   }

   return (EX_NOERR);

/* Fatal error: exit definition mode and return */
error_ret:
       if (ncendef (exoid) == -1)     /* exit define mode */
       {
         sprintf(errmsg,
                "Error: failed to complete definition for file id %d",
                 exoid);
         ex_err("ex_put_info",errmsg,exerrval);
       }
       return (EX_FATAL);
}
