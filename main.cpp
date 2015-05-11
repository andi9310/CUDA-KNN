#include <iostream>
#include <mpi.h>

#include "cuda_functions.h"
#include "variables.h"

#define CLASSIFY_DATA 1
#define RESULT 2

void mpiCheckErrors(int code);

int main(int argc, char ** argv)
{
	mpiCheckErrors(MPI_Init(&argc, &argv));
	
	int threadsPerBlock = 100;
	
	int myId, procCount;
	
	mpiCheckErrors(MPI_Comm_rank(MPI_COMM_WORLD, &myId));
	mpiCheckErrors(MPI_Comm_size(MPI_COMM_WORLD, &procCount));
	
	int pointsPerProc, myPointsCount;	
	int dimensions, teachingCollectionCount, K, classifyCollectionCount;
	
	int *classifiedClasses, *teachedClasses;
	float *teachingCollection, *classifyCollection;	
	
	if (myId == 0)
	{
		std::cin >> dimensions >> teachingCollectionCount;
		
		teachingCollection = new float[teachingCollectionCount*dimensions];
		teachedClasses = new int[teachingCollectionCount];
		
		for (int i=0; i< teachingCollectionCount; i++)
		{
			for (int j=0; j<dimensions; j++)
			{
				std::cin >> teachingCollection[i*dimensions+j];
			}
			std::cin >> teachedClasses[i];
		}
		// to this moment - definition of teaching collection
		
		std::cin >> classifyCollectionCount;
		
		classifyCollection = new float[classifyCollectionCount*dimensions];
		classifiedClasses = new int[classifyCollectionCount];
		
		pointsPerProc = classifyCollectionCount/procCount;
		
		for (int i=0; i<classifyCollectionCount; i++)
		{
			for (int j=0; j<dimensions; j++)
			{
				std::cin >> classifyCollection[i*dimensions+j];
			}
		}
		// to this moment - deffinition of collection to classify

		std::cin >> K;
	}
	
	mpiCheckErrors(MPI_Bcast((void*)&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD));
	mpiCheckErrors(MPI_Bcast((void*)&teachingCollectionCount, 1, MPI_INT, 0, MPI_COMM_WORLD));
	mpiCheckErrors(MPI_Bcast((void*)&pointsPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD));
	mpiCheckErrors(MPI_Bcast((void*)&K, 1, MPI_INT, 0, MPI_COMM_WORLD));
	
	if(myId != 0)
	{
		teachingCollection = new float[teachingCollectionCount*dimensions];
		teachedClasses = new int[teachingCollectionCount];
	}

	mpiCheckErrors(MPI_Bcast((void*)teachingCollection, teachingCollectionCount*dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD));
	mpiCheckErrors(MPI_Bcast((void*)teachedClasses, teachingCollectionCount, MPI_INT, 0, MPI_COMM_WORLD));

	if(myId == 0)
	{	
		myPointsCount = pointsPerProc+classifyCollectionCount%procCount;
		for(int i = 1; i < procCount; ++i)
		{
			mpiCheckErrors(MPI_Send(classifyCollection+(myPointsCount+(i-1)*pointsPerProc)*dimensions, pointsPerProc*dimensions, MPI_FLOAT, i, CLASSIFY_DATA, MPI_COMM_WORLD));
		}		
	}
	else
	{			
		classifyCollection = new float[pointsPerProc*dimensions];
		
		mpiCheckErrors(MPI_Recv((void*)classifyCollection, pointsPerProc*dimensions, MPI_FLOAT, 0, CLASSIFY_DATA, MPI_COMM_WORLD, NULL));
		
		classifiedClasses = new int[pointsPerProc];
		
		myPointsCount = pointsPerProc;
	}
	
	cuda_knn(K, dimensions, teachingCollection, teachedClasses, teachingCollectionCount, classifyCollection, classifiedClasses, myPointsCount, threadsPerBlock);
	
	if(myId == 0)
	{
		MPI_Status status;
		for(int i = 1; i < procCount; ++i)
		{
			mpiCheckErrors(MPI_Probe(MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status));
			int *location = classifiedClasses+myPointsCount+(status.MPI_SOURCE-1)*pointsPerProc;
			mpiCheckErrors(MPI_Recv((void*)location, pointsPerProc, MPI_INT, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status));
			
		}
		std::cout << "\n\nresult:\n";
		for(int i = 0; i < classifyCollectionCount; ++i)
		{
			std::cout << classifiedClasses[i] << '\n';
		}
	}
	else
	{
		mpiCheckErrors(MPI_Send((void*)classifiedClasses, pointsPerProc, MPI_INT, 0, RESULT, MPI_COMM_WORLD));
	}
	
	delete[] teachingCollection;
	delete[] teachedClasses;		
	delete[] classifiedClasses;
	delete[] classifyCollection;
	
	mpiCheckErrors(MPI_Finalize());
}

void mpiCheckErrors(int code)
{
	if(code)
		std::cerr << "MPI error no.: " << code << "\n";
}
