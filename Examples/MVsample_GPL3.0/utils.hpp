//-------------------------------------------- LICENSE NOTICE -------------------------------------------------------------
// SParse Matrix-Vector multplication example code: A Tight-binding Hamiltonian stored with a Compressed Sparse Row format
//
// Copyright (C) 2021 Hoon Ryu (Korea Institute of Science and Technology Information / E: elec1020@gmail.com)
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with this program.
// If not, see <https://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------------------------------------------------------

#ifndef _UTILS
#define _UTILS

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "csr_complex.hpp"

void convert(double*, double*, int*, int*, double*, double*, int);
void convertNuma(double **dmatrix_R, double **dmatrix_I, int *index_i, int *index_j, double *nz_R, double *nz_I, int DIM);
void load(int*, int*, double*, double*, char*, char*, char*);
void dmv(double*, double*, double*, double*, double*, double*, int);
void dmvNuma(double **dmatrix_R, double **dmatrix_I, double *xR, double *xI, double *yR, double *yI, int DIM);
void spmv(CSRComplex*, double*, double*, double*, double*);
void dump_dmatrix(double*, double*, int, char*);

#endif
