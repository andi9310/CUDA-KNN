#include <iostream>
#include <mpi.h>

#include "cuda_functions.h"
#include "utils.h"
#include "datatypes.h"

#define CLASSIFY_DATA 1
#define RESULT 2
#define DEVICES_INFO 3
#define FINISH 4
#define SUBRANGES 5

//#define DEBUG

struct computationPart
{
	int start;
	int count;
};

// Prints any MPI errors to stderr
// input:
// code - return code from MPI function
void mpiCheckErrors(int code);

int getTotalMultiprocessors(int n, GpuProperties **processesGpuProperties, int * gpusPerProcess);
unsigned long long getTotalMemory(int n, GpuProperties **processesGpuProperties, int * gpusPerProcess);

int main(int argc, char ** argv)
{
	mpiCheckErrors(MPI_Init(&argc, &argv));
	
	int myId, procCount;
	
	mpiCheckErrors(MPI_Comm_rank(MPI_COMM_WORLD, &myId));
	mpiCheckErrors(MPI_Comm_size(MPI_COMM_WORLD, &procCount));
	
	const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Datatype mpi_gpu_properties_type;
	MPI_Aint offsets[2];

    offsets[0] = offsetof(GpuProperties, memory);
    offsets[1] = offsetof(GpuProperties, multiprocessors);
	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_gpu_properties_type);
    MPI_Type_commit(&mpi_gpu_properties_type);
	
	GpuProperties **processesGpuProperties = NULL;
	GpuProperties *properties;
	int * gpusPerProcess = NULL;
	
	if (myId == 0)
	{		
		processesGpuProperties = new GpuProperties*[procCount];
		processesGpuProperties[0] = NULL;
		gpusPerProcess = new int[procCount];
		gpusPerProcess[0] = 0;
		
		MPI_Status status;
		for(int i = 1; i < procCount; ++i)
		{
			mpiCheckErrors(MPI_Probe(MPI_ANY_SOURCE, DEVICES_INFO, MPI_COMM_WORLD, &status));
			int num;
			mpiCheckErrors(MPI_Get_count(&status, mpi_gpu_properties_type, &num));
			processesGpuProperties[status.MPI_SOURCE] = new GpuProperties[num];
			gpusPerProcess[status.MPI_SOURCE] = num;
			mpiCheckErrors(MPI_Recv(processesGpuProperties[status.MPI_SOURCE], num, mpi_gpu_properties_type, status.MPI_SOURCE, DEVICES_INFO, MPI_COMM_WORLD, &status));			
		}
		#ifdef DEBUG
			for (int i=1; i<procCount;i++)
			{
				std::cout << "Process " << i << ": ";
				for (int j=0; j<gpusPerProcess[i]; j++)
				{
					std::cout << "card " << j << ": memory: " << processesGpuProperties[i][j].memory << " multiprocessors: " << processesGpuProperties[i][j].multiprocessors << std::endl;
				}
			}
		#endif
	}
	else
	{
		int numGpus = getNumberOfGpus();
		properties = new GpuProperties[numGpus];
		getGpusProperties(properties);
		mpiCheckErrors(MPI_Send(properties, numGpus, mpi_gpu_properties_type, 0, DEVICES_INFO, MPI_COMM_WORLD));		
	}
	
	int maxThreadsPerMultiProcessor = 500; //not a cuda parameter really - just a variable which will be used as multiplier for assignment of parts of work to processes
	
	int threadsPerBlock = 100;	
	
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
	
	// Root sends necessary parameters to all slaves
	mpiCheckErrors(MPI_Bcast((void*)&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD));
	mpiCheckErrors(MPI_Bcast((void*)&teachingCollectionCount, 1, MPI_INT, 0, MPI_COMM_WORLD));
	mpiCheckErrors(MPI_Bcast((void*)&pointsPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD));
	mpiCheckErrors(MPI_Bcast((void*)&K, 1, MPI_INT, 0, MPI_COMM_WORLD));
	
	if(myId != 0)
	{
		teachingCollection = new float[teachingCollectionCount*dimensions];
		teachedClasses = new int[teachingCollectionCount];
	}

	// Root sends input data to all slaves - space is allocated above
	mpiCheckErrors(MPI_Bcast((void*)teachingCollection, teachingCollectionCount*dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD));
	mpiCheckErrors(MPI_Bcast((void*)teachedClasses, teachingCollectionCount, MPI_INT, 0, MPI_COMM_WORLD));

	int sent = 0;
	
	// Root sends parts of classifyCollection to each slave
	if(myId == 0)
	{	
		computationPart * parts = new computationPart[procCount];
		for(int i = 1; i < procCount; ++i)
		{
			int toSend = getTotalMultiprocessors(i, processesGpuProperties, gpusPerProcess) * maxThreadsPerMultiProcessor;
			if (classifyCollectionCount-sent < toSend)
			{
				toSend = classifyCollectionCount-sent;
			}
			if (toSend == 0)
			{
				//TODO: jakoś to obsłużyć że mamy za małe dane do liczby procesów
			}
			bool memoryCriterion = false;
			unsigned long memToAlloc = memoryToAllocSize(dimensions, teachingCollectionCount, toSend, K);
			
			while(memToAlloc > getTotalMemory(i, processesGpuProperties, gpusPerProcess)/2)
			{
				memoryCriterion = true;
				toSend *= 0.9;
				memToAlloc = memoryToAllocSize(dimensions, teachingCollectionCount, toSend, K);
			}
			
			mpiCheckErrors(MPI_Send(classifyCollection+sent*dimensions, toSend*dimensions, MPI_FLOAT, i, CLASSIFY_DATA, MPI_COMM_WORLD));
			parts[i].start = sent;
			sent += toSend;
			parts[i].count = toSend;
			
			int sum = 0;
			int *subranges = new int[gpusPerProcess[i]];
			
			if(memoryCriterion)
			{
				for(int j = 0; j < gpusPerProcess[i]-1; ++j)
				{
					int temp = (processesGpuProperties[i][j].memory*toSend)/getTotalMemory(i, processesGpuProperties, gpusPerProcess);
					subranges[j] = temp;
					sum += temp;
				}
			}
			else
			{
				for(int j = 0; j < gpusPerProcess[i]-1; ++j)
				{
					int temp = (processesGpuProperties[i][j].multiprocessors*toSend)/getTotalMultiprocessors(i, processesGpuProperties, gpusPerProcess);
					subranges[j] = temp;
					sum += temp;
				}
			}
			
			subranges[gpusPerProcess[i]-1] = toSend - sum;
			mpiCheckErrors(MPI_Send(subranges, gpusPerProcess[i], MPI_INT, i, SUBRANGES, MPI_COMM_WORLD));
			delete[] subranges;
		}	
		
		MPI_Status status;
		while (sent < classifyCollectionCount)
		{
			mpiCheckErrors(MPI_Probe(MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status));
			int source = status.MPI_SOURCE;
			int *location = classifiedClasses + parts[source].start;
			mpiCheckErrors(MPI_Recv(location, parts[source].count, MPI_INT, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status));
			int toSend = getTotalMultiprocessors(source, processesGpuProperties, gpusPerProcess) * maxThreadsPerMultiProcessor;
			
			if (classifyCollectionCount-sent < toSend)
			{
				toSend = classifyCollectionCount-sent;
			}
			
			bool memoryCriterion = false;
			unsigned long memToAlloc = memoryToAllocSize(dimensions, teachingCollectionCount, toSend, K);
						
			while(memToAlloc > getTotalMemory(source, processesGpuProperties, gpusPerProcess)/2)
			{
				toSend *= 0.9;
				memToAlloc = memoryToAllocSize(dimensions, teachingCollectionCount, toSend, K);
			}
			
			mpiCheckErrors(MPI_Send(classifyCollection+(sent)*dimensions, toSend*dimensions, MPI_FLOAT, source, CLASSIFY_DATA, MPI_COMM_WORLD));
			parts[source].start = sent;
			sent += toSend;
			parts[source].count = toSend;
			
			int sum = 0;
			int *subranges = new int[gpusPerProcess[source]];
			
			if(memoryCriterion)
			{
				for(int j = 0; j < gpusPerProcess[source]-1; ++j)
				{
					int temp = (processesGpuProperties[source][j].memory*toSend)/getTotalMemory(source, processesGpuProperties, gpusPerProcess);
					sum += temp;
					subranges[j] = temp;
				}
			}
			else
			{
				for(int j = 0; j < gpusPerProcess[source]-1; ++j)
				{
					int temp = (processesGpuProperties[source][j].multiprocessors*toSend)/getTotalMultiprocessors(source, processesGpuProperties, gpusPerProcess);
					sum += temp;
					subranges[j] = temp;
				}
			}
			
			subranges[gpusPerProcess[source]-1] = toSend - sum;
			mpiCheckErrors(MPI_Send(subranges, gpusPerProcess[source], MPI_INT, source, SUBRANGES, MPI_COMM_WORLD));
			delete[] subranges;
		}
		
		for (int i=1; i<procCount; i++)
		{
			mpiCheckErrors(MPI_Probe(MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status));
			int source = status.MPI_SOURCE;
			int *location = classifiedClasses + parts[source].start;
			mpiCheckErrors(MPI_Recv(location, parts[source].count, MPI_INT, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status));
			mpiCheckErrors(MPI_Send(NULL, 0, MPI_INT, source, FINISH, MPI_COMM_WORLD));
		}
		
		std::cout << "\n\nresult:\n";
		for(int i = 0; i < classifyCollectionCount; ++i)
		{
			std::cout << classifiedClasses[i] << '\n';
		}
		
		delete[] classifiedClasses;
		delete[] classifyCollection;
	}
	else
	{			
		while (true)
		{
			MPI_Status status;
			mpiCheckErrors(MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
			
			if (status.MPI_TAG == FINISH)
			{
				mpiCheckErrors(MPI_Recv(NULL, 0, MPI_INT, 0, FINISH, MPI_COMM_WORLD, NULL));
				break;
			}
			else if (status.MPI_TAG == CLASSIFY_DATA)
			{
				int count;
				mpiCheckErrors(MPI_Get_count(&status, MPI_FLOAT, &count));
				
				int numOfGpus = getNumberOfGpus();
				classifyCollection = new float[count];
				int* subranges = new int[numOfGpus];
				
				mpiCheckErrors(MPI_Recv(classifyCollection, count, MPI_FLOAT, 0, CLASSIFY_DATA, MPI_COMM_WORLD, NULL));				
				classifiedClasses = new int[count/dimensions];
				
				mpiCheckErrors(MPI_Recv(subranges, numOfGpus, MPI_INT, 0, SUBRANGES, MPI_COMM_WORLD, NULL));
				
				// Calls function with actual computation on all nodes
				cuda_knn(K, dimensions, teachingCollection, teachedClasses, teachingCollectionCount, classifyCollection, classifiedClasses, count/dimensions, threadsPerBlock, numOfGpus, subranges);
			
				mpiCheckErrors(MPI_Send(classifiedClasses, count/dimensions, MPI_INT, 0, RESULT, MPI_COMM_WORLD));

				delete[] subranges;
				delete[] classifyCollection;
				delete[] classifiedClasses;
			}
		}

		delete[] properties;
	}
	
	// Freeing memory
	delete[] teachingCollection;
	delete[] teachedClasses;		

	
	mpiCheckErrors(MPI_Finalize());
}

void mpiCheckErrors(int code)
{
	if(code)
		std::cerr << "MPI error no.: " << code << "\n";
}

int getTotalMultiprocessors(int n, GpuProperties **processesGpuProperties, int *gpusPerProcess)
{
	int total = 0;
	for (int i=0; i<gpusPerProcess[n];i++)
	{
		total+= processesGpuProperties[n][i].multiprocessors;
	}
	return total;
}

unsigned long long getTotalMemory(int n, GpuProperties **processesGpuProperties, int *gpusPerProcess)
{
	unsigned long long total = 0;
	for (int i=0; i<gpusPerProcess[n];i++)
	{
		total+= processesGpuProperties[n][i].memory;
	}
	return total;
}
