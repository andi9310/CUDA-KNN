#include <iostream>
#include <mpi.h>

#include "cuda_functions.h"



int main(int argc, char ** argv)
{
	MPI_Init(&argc, &argv);
	int myId, ierr;
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	if (myId == 0)
	{
		int dimensions;
		std::cin >> dimensions;
		
		int teachingCollectionCount;
		std::cin >> teachingCollectionCount;
		float *teachingCollection = new float[teachingCollectionCount*dimensions];
		int *teachedClasses = new int[teachingCollectionCount];
		for (int i=0; i< teachingCollectionCount; i++)
		{
			for (int j=0; j<dimensions; j++)
			{
				std::cin >> teachingCollection[i*dimensions+j];
			}
			std::cin >> teachedClasses[i];
		}
		// to this moment - definition of teaching collection
		
		int classifyCollectionCount;
		std::cin >> classifyCollectionCount;
		float *classifyCollection = new float[classifyCollectionCount*dimensions];
		int *classifiedClasses = new int[classifyCollectionCount];
		for (int i=0; i<classifyCollectionCount; i++)
		{
			for (int j=0; j<dimensions; j++)
			{
				std::cin >> classifyCollection[i*dimensions+j];
			}
		}
		// to this moment - deffinition of collection to classify
	
	
	cuda_knn(dimensions, teachingCollection, teachedClasses, teachingCollectionCount, classifyCollection, classifiedClasses, classifyCollectionCount);
	
	std::cout << "ok\n"; 

	delete[] teachingCollection;
	delete[] teachedClasses;
	}
	MPI_Finalize();

}
