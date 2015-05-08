#include <iostream>
#include <mpi.h>

#include "cuda_functions.h"



int main(int argc, char ** argv)
{

	MPI_Init(&argc, &argv);
	cuda_knn();
	std::cout << "ok"; 

	MPI_Finalize();

}
