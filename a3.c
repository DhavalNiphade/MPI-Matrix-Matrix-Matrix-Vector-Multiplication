#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  int rank,p,i, root = 0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&p);

  // Make the local vector size constant

  int global_vector_size = 10000;

  int local_vector_size = global_vector_size/p;

  double pi = 4.0*atan(1.0);
  
  // initialize the vectors
  double *a, *b;
  a = (double *) malloc(
       global_vector_size*sizeof(double));
  b = (double *) malloc(
       global_vector_size*sizeof(double));
  for (i=0;i<global_vector_size;i++) {
    a[i] = sqrt(i); 
    b[i] = sqrt(i); 
  }

  //
  double ascat[local_vector_size],bscat[local_vector_size];
  //ascat = (double*)malloc(local_vector_size*sizeof(double));
  //bscat = (double*)malloc(local_vector_size*sizeof(double));

  //Scatter the arrays
  MPI_Scatter(a,local_vector_size,MPI_DOUBLE,&ascat,local_vector_size,MPI_DOUBLE,root,MPI_COMM_WORLD);
  MPI_Scatter(b,local_vector_size,MPI_DOUBLE,&bscat,local_vector_size,MPI_DOUBLE,root,MPI_COMM_WORLD);

  // compute the dot product
  double partial_sum = 0.0;
  for (i=0;i<local_vector_size;i++) {
    partial_sum += ascat[i]*bscat[i];
  }

  //Compute the local dot product
  double sum = 0.0;
  MPI_Reduce(&partial_sum,&sum,1,MPI_DOUBLE, MPI_SUM,root,MPI_COMM_WORLD);
  

  if ( rank == root ) {
    printf("The dot product is %g.  Answer should be: %g\n",
           sum,0.5*global_vector_size*(global_vector_size-1));
  }

  free(a);
  free(b);
  MPI_Finalize();
  return 0;
}
