//#define DEBUG
//#define DEBUG_KERNEL

#include <iostream>

#include "variables.h"
#include "datatypes.h"

#ifdef DEBUG
	#include <cstdio>
#endif

// Prints any CUDA errors to stderr
// input:
// code - return code from CUDA function
void cudaCheckErrors(int code)
{
	if(code)
		std::cerr << "CUDA error no.: " << code << "\n";
}

int getNumberOfGpus()
{
	int n;
	cudaCheckErrors(cudaGetDeviceCount(&n));
	return n;
}

void getGpusProperties(GpuProperties *properties)
{
	int n = getNumberOfGpus();
	for (int i=0; i<n; i++)
	{
		cudaDeviceProp prop;
		cudaCheckErrors(cudaGetDeviceProperties(&prop, i));
		
		properties[i].memory = prop.totalGlobalMem;
		properties[i].multiprocessors = prop.multiProcessorCount;
	}
}


void printCudaMem()
{
	    size_t free_byte ;

        size_t total_byte ;

        cudaCheckErrors(cudaMemGetInfo( &free_byte, &total_byte ));
        double free_db = (double)free_byte ;

        double total_db = (double)total_byte ;

        double used_db = total_db - free_db ;

        std::cout << "GPU memory usage: used = " << used_db/1024.0/1024.0 << ", free = " << free_db/1024.0/1024.0 << " MB, total = " << total_db/1024.0/1024.0 << " MB\n";
}

// Finds slots for numbers in given array (ascending) and puts them there (sorting)
// input:
// myDistances - pointer to an array of distances, which will be fitted into myNearestDistances array
// K - size of myNearestDistances and myNearestIndexes arrays
// i - size of myDistances array
// output:
// myNearestIndexes - array of indices from myDistances array, wich were chosen as the lowest
__device__ int findSlot(float *myDistances, float *myNearestDistances, int *myNearestIndexes, int K, int i)
{
	int j;
	for (j=0; j<i; j++)
	{
		if (myDistances[i]<myNearestDistances[j]) //terrible nesting here
		{
			for (int k=K-1; k>j; k--)
			{
				myNearestDistances[k]=myNearestDistances[k-1];
				myNearestIndexes[k]=myNearestIndexes[k-1];
			}
			myNearestDistances[j]=myDistances[i];
			myNearestIndexes[j]=i;
			break;
		}
	}
	return j;
}

// Counts distances between point from classifyCollection to points from teachingCollection
// input:
// dimensions - dimensionality of space
// teachingCollection - pointer to an array, which represents points coordinates - each "dimensions" of elements represents one point (like vector) - teaching collection
// teachingCollectionCount - number of points in arrays mentioned above
// classifyCollection - pointer to an array, which represents points coordinates - each "dimensions" of elements represents one point (like vector) - classify collection
// classifyCollectionCount - number of points in array mentioned above
// output:
// distances - pointer to an array, which will be populated with distances to every point
__global__ void countDistances(int dimensions, float *teachingCollection, int teachingCollectionCount, float *classifyCollection, int classifyCollectionCount, float *distances)
{
	int tId = blockIdx.x*blockDim.x+threadIdx.x;
	int pointId = tId*dimensions;	
		
	if(tId >= classifyCollectionCount)
		return;
		
	#ifdef DEBUG_KERNEL
		printf("%d watek %d %d\n", tId, classifyCollectionCount, teachingCollectionCount);
	#endif
	
	for(int i = 0; i < teachingCollectionCount; ++i)
	{
		float distance = 0.0f;
		for(int j = 0; j < dimensions; ++j)
		{
			distance += (classifyCollection[pointId+j]-teachingCollection[i*dimensions+j])*(classifyCollection[pointId+j]-teachingCollection[i*dimensions+j]);
		}
		
		#ifdef DEBUG_KERNEL
			printf("%f\n", distance);
		#endif
		
		distances[teachingCollectionCount*tId+i] = distance;
	}
}

// Selects N-nearest points to a point from distances array
// input:
// K - number of nearest points to get
// distances - pointer to an array, which is populated with distances to every point
// teachingCollectionCount - number of points in teaching collection
// classifyCollectionCount - number of points in classify collection
// output:
// nearestIndexes - array of indices from distances array, wich were chosen as the lowest
__global__ void selectN(int K, float *distances, int teachingCollectionCount, int classifyCollectionCount, int *nearestIndexes, float *nearestDistances)
{		
	int myRank = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(myRank >= classifyCollectionCount)
		return;
		
	int *myNearestIndexes = nearestIndexes+K*myRank;
	float *myNearestDistances = nearestDistances+K*myRank;
	float *myDistances = distances + myRank * teachingCollectionCount;
	
	for (int i=0; i<K; i++)
	{
		int j = findSlot(myDistances, myNearestDistances, myNearestIndexes, K, i);
		
		if (j==i)
		{
			myNearestDistances[j]=myDistances[i];
			myNearestIndexes[j]=i;
		}
	}
	for (int i=K; i<teachingCollectionCount;i++)
	{
		findSlot(myDistances, myNearestDistances, myNearestIndexes, K, i);
	}
	
}

// Selects N-nearest points to a point from distances array
// input:
// K - number of nearest points to get
// nearestIndexes - array of indices from distances array, wich were chosen as the lowest
// distances - pointer to an array, which is populated with distances to every point
// classifyCollectionCount - number of points in classify collection
// teachedClasses - pointer to an array, which represents classes of each point from teaching collection
// teachingCollectionCount - number of points in teaching collection
// output:
// result - pointer to an array, which will be populated with numbers of class each point was fitted into
__global__ void chooseClass(int K, int *nearestIndexes, int classifyCollectionCount, int *teachedClasses, int teachingCollectionCount, int *classCounters, int *result)
{
	int tId = blockIdx.x*blockDim.x+threadIdx.x;
	int pointId = tId*K;	
		
	if(tId >= classifyCollectionCount)
		return;

	for(int i = 0; i < K; ++i)
	{
		classCounters[tId*MAX_CLASS_NUMBER+teachedClasses[nearestIndexes[i+pointId]]]++;
	}

	int maxIndex = 0, maxValue = classCounters[tId*MAX_CLASS_NUMBER];
	for(int i = 1; i < MAX_CLASS_NUMBER; ++i)
	{
		if(classCounters[tId*MAX_CLASS_NUMBER+i] > maxValue)
		{
			maxIndex = i;
			maxValue = classCounters[tId*MAX_CLASS_NUMBER+i];
		}
	}
	result[tId] = maxIndex;
}

void cuda_knn(int K, int dimensions, float *h_teachingCollection, int *h_teachedClasses, int teachingCollectionCount, float *h_classifyCollection, int *h_classifiedClasses, int classifyCollectionCount, int threadsPerBlock)
{
	int blocksPerGrid = classifyCollectionCount / threadsPerBlock + 1;
	
	// Memory allocation block
	float *d_teachingCollection, *d_classifyCollection, *d_nearestDistances, *d_distances;
	cudaCheckErrors(cudaMalloc(&d_teachingCollection, teachingCollectionCount*dimensions*sizeof(float)));
	cudaCheckErrors(cudaMalloc(&d_classifyCollection, classifyCollectionCount*dimensions*sizeof(float)));
	cudaCheckErrors(cudaMalloc(&d_nearestDistances, classifyCollectionCount*K*sizeof(float)));
	cudaCheckErrors(cudaMalloc(&d_distances, teachingCollectionCount*classifyCollectionCount*sizeof(float)));
	
	int *d_classCounters, *d_teachedClasses, *d_result, *d_nearestIndexes;
	cudaCheckErrors(cudaMalloc(&d_teachedClasses, teachingCollectionCount*sizeof(int)));
	cudaCheckErrors(cudaMalloc(&d_result, classifyCollectionCount*sizeof(int)));	
	cudaCheckErrors(cudaMalloc(&d_nearestIndexes, classifyCollectionCount*K*sizeof(int)));
	cudaCheckErrors(cudaMalloc(&d_classCounters, classifyCollectionCount*MAX_CLASS_NUMBER*sizeof(int)));
	cudaCheckErrors(cudaMemset(d_classCounters, 0, classifyCollectionCount*MAX_CLASS_NUMBER*sizeof(int)));
	
	// Copying parameters for kernels
	cudaCheckErrors(cudaMemcpy(d_teachingCollection, h_teachingCollection, teachingCollectionCount*dimensions*sizeof(float), cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaMemcpy(d_classifyCollection, h_classifyCollection, classifyCollectionCount*dimensions*sizeof(float), cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaMemcpy(d_teachedClasses, h_teachedClasses, teachingCollectionCount*sizeof(int), cudaMemcpyHostToDevice));
	
	// Kernel launches
	countDistances<<<blocksPerGrid, threadsPerBlock>>>(dimensions, d_teachingCollection, teachingCollectionCount, d_classifyCollection, classifyCollectionCount, d_distances);

	#ifdef DEBUG
		float *h_distances = new float[teachingCollectionCount*classifyCollectionCount];
		cudaCheckErrors(cudaMemcpy(h_distances, d_distances, teachingCollectionCount*classifyCollectionCount*sizeof(float), cudaMemcpyDeviceToHost)); // copy calculated distances back to host
		printf("\n\n");
		for(int i = 0; i < teachingCollectionCount*classifyCollectionCount; ++i)
		{
			printf("distance %f\n", h_distances[i]);
		}
	#endif

	selectN<<<blocksPerGrid, threadsPerBlock>>>(K, d_distances, teachingCollectionCount, classifyCollectionCount, d_nearestIndexes, d_nearestDistances);

	#ifdef DEBUG
		float *h_nearestDistances = new float[classifyCollectionCount*K];
		int *h_nearestIndexes = new int[classifyCollectionCount*K];
		cudaCheckErrors(cudaMemcpy(h_nearestDistances, d_nearestDistances, classifyCollectionCount*K*sizeof(float), cudaMemcpyDeviceToHost));	
		cudaCheckErrors(cudaMemcpy(h_nearestIndexes, d_nearestIndexes, classifyCollectionCount*K*sizeof(int), cudaMemcpyDeviceToHost));
		printf("\n\n");
		for(int i = 0; i < classifyCollectionCount*K; ++i)
		{
			printf("nearest %d %f\n", h_nearestIndexes[i], h_nearestDistances[i]);
		}
	#endif	

	chooseClass<<<blocksPerGrid, threadsPerBlock>>>(K, d_nearestIndexes, classifyCollectionCount, d_teachedClasses, teachingCollectionCount, d_classCounters, d_result);

	// Copying result back to host memory
	cudaCheckErrors(cudaMemcpy(h_classifiedClasses, d_result, classifyCollectionCount*sizeof(int), cudaMemcpyDeviceToHost));
		
	#ifdef DEBUG
		int *h_classCounters = new int[classifyCollectionCount*MAX_CLASS_NUMBER];
		cudaCheckErrors(cudaMemcpy(h_classCounters, d_classCounters, classifyCollectionCount*MAX_CLASS_NUMBER*sizeof(int), cudaMemcpyDeviceToHost));
		for(int i = 0; i < classifyCollectionCount*MAX_CLASS_NUMBER; ++i)
		{
			printf("counter %d\n", h_classCounters[i]);
		}
	#endif
			
	// Freeing memory
	cudaCheckErrors(cudaFree(d_nearestDistances));
	cudaCheckErrors(cudaFree(d_nearestIndexes));
	cudaCheckErrors(cudaFree(d_teachingCollection));
	cudaCheckErrors(cudaFree(d_classifyCollection));
	cudaCheckErrors(cudaFree(d_distances));
	cudaCheckErrors(cudaFree(d_classCounters));	
	cudaCheckErrors(cudaFree(d_teachedClasses));
	cudaCheckErrors(cudaFree(d_result));
	
	#ifdef DEBUG
		delete[] h_distances;
		delete[] h_nearestDistances;
		delete[] h_nearestIndexes;	
		delete[] h_classCounters;
	#endif
}
