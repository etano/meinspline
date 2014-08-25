/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Copyright (C) 2007 Kenneth P. Esler, Jr.                               //
//                                                                         //
//  This program is free software; you can redistribute it and/or modify   //
//  it under the terms of the GNU General Public License as published by   //
//  the Free Software Foundation; either version 2 of the License, or      //
//  (at your option) any later version.                                    //
//                                                                         //
//  This program is distributed in the hope that it will be useful,        //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//  GNU General Public License for more details.                           //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program; if not, write to the Free Software            //
//  Foundation, Inc., 51 Franklin Street, Fifth Floor,                     //
//  Boston, MA  02110-1301  USA                                            //
/////////////////////////////////////////////////////////////////////////////

#include "multi_nubspline_eval_d.h"

/************************************************************/
/* 1D double-precision, complex evaulation functions        */
/************************************************************/
void
eval_multi_NUBspline_1d_d (multi_NUBspline_1d_d *spline,
			  double x,
			  double* restrict vals)
{
  double a[4];

  int ix = get_NUBasis_funcs_d (spline->x_basis, x, a);
  int xs = spline->x_stride;

  double* restrict coefs0 = spline->coefs +(ix+0)*xs;
  double* restrict coefs1 = spline->coefs +(ix+1)*xs;
  double* restrict coefs2 = spline->coefs +(ix+2)*xs;
  double* restrict coefs3 = spline->coefs +(ix+3)*xs;
  for (int n=0; n<spline->num_splines; n++) 
    vals[n] = (a[0]*coefs0[n] + a[1]*coefs1[n] + 
	       a[2]*coefs2[n] + a[3]*coefs3[n]);
}


void
eval_multi_NUBspline_1d_d_vg (multi_NUBspline_1d_d *spline,
			     double x,
			     double* restrict vals,
			     double* restrict grads)
{
  double a[4], da[4];
  int ix = get_NUBasis_dfuncs_d (spline->x_basis, x, a, da);
  int xs = spline->x_stride;

  for (int n=0; n<spline->num_splines; n++) {
    vals[n]  = 0.0;
    grads[n] = 0.0;
  }

  for (int i=0; i<4; i++) { 
    double* restrict coefs = spline->coefs + ((ix+i)*xs);
    for (int n=0; n<spline->num_splines; n++) {
      vals[n]  +=   a[i] * coefs[n];
      grads[n] +=  da[i] * coefs[n];
    }
  }
}


void
eval_multi_NUBspline_1d_d_vgl (multi_NUBspline_1d_d *spline,
			       double x,
			       double* restrict vals,
			       double* restrict grads,
			       double* restrict lapl)	  
{
  double a[4], da[4], d2a[4];
  int ix = get_NUBasis_d2funcs_d (spline->x_basis, x, a, da, d2a);
  int xs = spline->x_stride;

  for (int n=0; n<spline->num_splines; n++) {
    vals[n]  = 0.0;
    grads[n] = 0.0;
    lapl[n]  = 0.0;
  }

  for (int i=0; i<4; i++) {      
    double* restrict coefs = spline->coefs + ((ix+i)*xs);
    for (int n=0; n<spline->num_splines; n++) {
      vals[n]  +=   a[i] * coefs[n];
      grads[n] +=  da[i] * coefs[n];
      lapl[n]  += d2a[i] * coefs[n];
    }
  }
}


void
eval_multi_NUBspline_1d_d_vgh (multi_NUBspline_1d_d *spline,
			      double x,
			      double* restrict vals,
			      double* restrict grads,
			      double* restrict hess)
{
  eval_multi_NUBspline_1d_d_vgl (spline, x, vals, grads, hess);
}


