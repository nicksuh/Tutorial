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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.hpp"
#include <numa.h>
#include <sched.h>
#include "csr_complex.hpp"


#define NODE_COUNT 2

int main(int argc, char **argv) {

	int nnz = 0;
	int *row_ptr, *col_ind;
	int localfirst, mysize, count, niter;
	int localend, globalsize, globalnnz, row_ptr_ind;
	int Nx, Ny, xcindfirst, xcindend, DIM;
	double *nzeroR, *nzeroI, *dmatrixR, *dmatrixI;
	double *VR, *VI, *WR, *WI, *YR, *YI;
	CSRComplex *smatrix;
	char *myfilename;
	FILE *fp;

	DIM = 5120;
	niter = 10;
	nnz = 175164;
	localfirst = 1;
	localend = DIM;
	mysize = localend - localfirst + 1;
	globalsize = mysize;
	printf("Dimension is initialized.\n");

	WR = new double[mysize]();
	WI = new double[mysize]();
	VR = new double[mysize]();
	VI = new double[mysize]();
	YR = new double[mysize]();
	YI = new double[mysize]();
	count = 0;

// first touch initialization.
	for(int ii = localfirst; ii <= localend; ii++)
	{
		VR[ii] = double(rand())/RAND_MAX;
		VI[ii] = double(rand())/RAND_MAX;
		YR[ii] = VR[ii];
		YI[ii] = VI[ii];
	}
	printf("Vectors are constructed.\n");

	row_ptr = new int[mysize+1]();
	col_ind = new int[nnz]();
	nzeroR  = new double[nnz](); 		 
	nzeroI  = new double[nnz]();
	
	load(row_ptr, col_ind, nzeroR, nzeroI, "./data/index_i.dat", "./data/index_j.dat", "./data/nonzeros.dat");
	
	smatrix = new CSRComplex(row_ptr, col_ind, nzeroR, nzeroI, globalsize, mysize, nnz, localfirst);
	printf("Matrix is constructed.\n");

	for(int ii = 0; ii < niter; ii++)
	{
		printf("SPMV at iteration %d\n", ii+1);
		spmv(smatrix, VR, VI, WR, WI);
		spmv(smatrix, WR, WI, VR, VI);
	}

	system("rm -rf ./result");
	system("mkdir ./result");  	

	smatrix->dumpCSR("./result/smatrix.dat");

	fp = fopen("./result/spmv_output.dat","wt");
	for(int ii = 0; ii < DIM; ii++)
		fprintf(fp, "%18.6e    %18.6e\n", VR[ii], VI[ii]);
	fclose(fp); 



	#ifdef NUMA
		double *dmatrixR_onNode[NODE_COUNT];
		double *dmatrixI_onNode[NODE_COUNT];
		#pragma omp parallel proc_bind(close) num_threads(NODE_COUNT)
		{
			int cpu = sched_getcpu();
			int node = numa_node_of_cpu(cpu);
			int chunkSize = (mysize*mysize) / NODE_COUNT;
			double * chunkR = (double * )numa_alloc_onnode(chunkSize * sizeof(double), node);
			double * chunkI = (double * )numa_alloc_onnode(chunkSize * sizeof(double), node);
			dmatrixR_onNode[node] = chunkR;
			dmatrixI_onNode[node] = chunkI;
		}
		convertNuma(dmatrixR_onNode, dmatrixI_onNode, row_ptr, col_ind, nzeroR, nzeroI, DIM);
	#else
		dmatrixR = new double[mysize*mysize]();
		dmatrixI = new double[mysize*mysize]();
		convert(dmatrixR, dmatrixI, row_ptr, col_ind, nzeroR, nzeroI, DIM);

	#endif

	//dump_dmatrix(dmatrixR, dmatrixI, DIM, "./result/dmatrix.dat");

	for(int ii = 0; ii < niter; ii++)
	{

	#ifdef NUMA
		printf("DMV NUMA at iteration %d\n", ii+1);
		double start = omp_get_wtime();
		dmvNuma(dmatrixR_onNode, dmatrixI_onNode, YR, YI, WR, WI, DIM);
		dmvNuma(dmatrixR_onNode, dmatrixI_onNode, WR, WI, YR, YI, DIM);
		double end = omp_get_wtime();	
		printf("DMV NUMA iter took %f seconds\n", end - start);
	#else
		printf("DMV at iteration %d\n", ii+1);
		double start = omp_get_wtime();
		dmv(dmatrixR, dmatrixI, YR, YI, WR, WI, DIM);
		dmv(dmatrixR, dmatrixI, WR, WI, YR, YI, DIM);
		double end = omp_get_wtime();	
		printf("DMV iter took %f seconds\n", end - start);
	#endif
	}

	fp = fopen("./result/dmv_output.dat","wt");
	for(int ii = 0; ii < DIM; ii++)
		fprintf(fp, "%18.6e    %18.6e\n", YR[ii], YI[ii]);
	fclose(fp);

	#ifdef NUMA
		int chunkSize = (mysize*mysize) / NODE_COUNT;
		for(int i = 0; i < NODE_COUNT; i++)
		{
			size_t size = chunkSize * sizeof(double);
			double * tmpR= dmatrixR_onNode[i];
			numa_free(tmpR,size);
			double * tmpI= dmatrixI_onNode[i];
			numa_free(tmpI,size);
		}
	#else
		delete [] dmatrixI;
		delete [] dmatrixR;
	#endif

	delete smatrix;	
	delete [] nzeroI;
	delete [] nzeroR;
	delete [] col_ind;
	delete [] row_ptr;
	delete [] YI;
	delete [] YR;
	delete [] VI;
	delete [] VR;
	delete [] WI;
	delete [] WR;

	printf("Memory: Deallocated.\n");

	return 0;
}
