//#define DEBUG
//#define DEBUG_KERNEL

#include <iostream>

#include "utils.h"
#include "datatypes.h"

#ifdef DEBUG
	#include <cstdio>
#endif
#ifdef DEBUG_KERNEL
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
	for (int i=0; i<n; ++i)
	{
		cudaDeviceProp prop;
		cudaCheckErrors(cudaGetDeviceProperties(&prop, i));
		
		properties[i].memory = prop.totalGlobalMem;
		properties[i].multiprocessors = prop.multiProcessorCount;
	}
}

// Prints info about memory usage on selectes (active) GPU
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
__device__ int findSlot(int distanceToClassify, float *myNearestDistances, int *myNearestIndexes, int K, int i)
{
	int j;
	for (j=0; j<i; ++j)
	{
		if (distanceToClassify<myNearestDistances[j])
		{
			for (int k=K-1; k>j; --k)
			{
				myNearestDistances[k]=myNearestDistances[k-1];
				myNearestIndexes[k]=myNearestIndexes[k-1];
			}
			myNearestDistances[j]=distanceToClassify;
			myNearestIndexes[j]=i;
			break;
		}
	}
	return j;
}

// Classifies points from classifyCollection to classes given in teachedClasses using KNN algorithm
// input:
// dimensions - dimensionality of space
// teachingCollection - pointer to an array, which represents points coordinates - each "dimensions" of elements represents one point (like vector) - teaching collection
// teachingCollectionCount - number of points in arrays mentioned above
// classifyCollection - pointer to an array, which represents points coordinates - each "dimensions" of elements represents one point (like vector) - classify collection
// classifyCollectionCount - number of points in array mentioned above
// K - number of nearest points to get
// teachedClasses - pointer to an array, which represents classes of each point from teaching collection
// distances - pointer to an array, which will be populated with distances to every point - array for temp data
// nearestDistances - array of distances from distances array, which were chosen as the lowest - array for temp data
// nearestIndexes - array of indexes from distances array, which were chosen as the lowest - array for temp data
// classConters - temp array for counting how many of points chosen as nearest are in each class
// output:
// result - pointer to an array, which will be populated with numbers of class each point was fitted into
__global__ void KNN_kernel(int dimensions, float *teachingCollection, int teachingCollectionCount, float *classifyCollection, int classifyCollectionCount, float *distances, int *nearestIndexes, float *nearestDistances, int K, int *classCounters, int *teachedClasses, int *result)
{
	extern __shared__ float s_TeachingCollection[];
	int tId = blockIdx.x*blockDim.x+threadIdx.x;
	int pointId = tId*dimensions;	
	int l;
	for (l = 0; l*blockDim.x<teachingCollectionCount;++l) 
	{
		for (int i = 0; i < dimensions; ++i) //each thread fetches one point from teachingCollection to shared memory
		{
			s_TeachingCollection[dimensions*threadIdx.x+i] = teachingCollection[l*blockDim.x*dimensions+threadIdx.x*dimensions+i];
		}
		__syncthreads();
		if(tId < classifyCollectionCount)
		{
			for(int i = 0; i < blockDim.x; ++i)
			{
				float distance = 0.0f;
				for(int j = 0; j < dimensions; ++j)
				{
					distance += (classifyCollection[pointId+j]-s_TeachingCollection[i*dimensions+j])*(classifyCollection[pointId+j]-s_TeachingCollection[i*dimensions+j]);
				}
				
				#ifdef DEBUG_KERNEL
					//printf("%f\n", distance);
				#endif
				
				distances[teachingCollectionCount*tId+i+l*blockDim.x] = distance;
				
			}
		}
	}
	__syncthreads();
	
	if (l*blockDim.x>=teachingCollectionCount) //last part of teachingCollection
	{
		int pointsLeft = teachingCollectionCount-(l-1)*blockDim.x;
		if (threadIdx.x < pointsLeft)
		{
			for (int i = 0; i < dimensions; ++i) //each thread fetches one point from teachingCollection to shared memory
			{
				s_TeachingCollection[dimensions*threadIdx.x+i] = teachingCollection[l*blockDim.x*dimensions+threadIdx.x*dimensions+i];
			}
		}
		__syncthreads();
		if(tId < classifyCollectionCount)
		{
			for(int i = 0; i < pointsLeft; ++i)
			{
				float distance = 0.0f;
				for(int j = 0; j < dimensions; ++j)
				{
					distance += (classifyCollection[pointId+j]-s_TeachingCollection[i*dimensions+j])*(classifyCollection[pointId+j]-s_TeachingCollection[i*dimensions+j]);
				}
				
				#ifdef DEBUG_KERNEL
					//printf("%f\n", distance);
				#endif
				
				distances[teachingCollectionCount*tId+i+l*blockDim.x] = distance;
				
			}
		}
	}	  
	// Distances computed
	
	
	if(tId >= classifyCollectionCount)
	{
		return;
	}
	int *myNearestIndexes = nearestIndexes+K*tId;
	float *myNearestDistances = nearestDistances+K*tId;
	float *myDistances = distances + tId * teachingCollectionCount;
	
	int distanceToPut = myDistances[0];
	for (int i=0; i<K-1; ++i)
	{
		int nextDistanceToPut = myDistances[i+1];
		int j = findSlot(distanceToPut, myNearestDistances, myNearestIndexes, K, i);
		
		if (j==i)
		{
			myNearestDistances[j]=distanceToPut;
			myNearestIndexes[j]=i;
		}
		distanceToPut=nextDistanceToPut;
	}
	//one iteration more
	int j = findSlot(distanceToPut, myNearestDistances, myNearestIndexes, K, K-1);	
	if (j==K-1)
	{
		myNearestDistances[j]=distanceToPut;
		myNearestIndexes[j]=K-1;
	}
	
	distanceToPut = myDistances[K];
	for (int i=K; i<teachingCollectionCount-1;++i)
	{
		int nextDistanceToPut = myDistances[i+1];
		findSlot(distanceToPut, myNearestDistances, myNearestIndexes, K, i);
		distanceToPut = nextDistanceToPut;
	}
	findSlot(distanceToPut, myNearestDistances, myNearestIndexes, K, teachingCollectionCount-1);

	int classIndex = tId*MAX_CLASS_NUMBER+teachedClasses[nearestIndexes[tId*K]];
	for(int i = 0; i < K-1; ++i)
	{
		int nextClassIndex = tId*MAX_CLASS_NUMBER+teachedClasses[nearestIndexes[i+1+tId*K]];
		++classCounters[classIndex];
		classIndex = nextClassIndex;
	}
	++classCounters[classIndex];


	int maxIndex = 0, maxValue = classCounters[tId*MAX_CLASS_NUMBER];
	
	int counter = classCounters[tId*MAX_CLASS_NUMBER+1];
	for(int i = 1; i < MAX_CLASS_NUMBER-1; ++i)
	{
		int nextCounter = classCounters[tId*MAX_CLASS_NUMBER+i+1];
		if( counter > maxValue)
		{
			maxIndex = i;
			maxValue = counter;
		}
		counter = nextCounter;
	}
	if( counter > maxValue)
	{
		maxIndex = MAX_CLASS_NUMBER-1;
		maxValue = counter;
	}
	result[tId] = maxIndex;
}

void cuda_knn(int K, int dimensions, float *h_teachingCollection, int *h_teachedClasses, int teachingCollectionCount, float *h_classifyCollection, int *h_classifiedClasses, int classifyCollectionCount, int threadsPerBlock, int numOfGpus, int *subranges)
{
	cudaStream_t *streams = new cudaStream_t[numOfGpus];
	
	// host memory allocation
    float **d_teachingCollection = new float*[numOfGpus], 
		  **d_classifyCollection = new float*[numOfGpus], 
		  **d_nearestDistances = new float*[numOfGpus], 
		  **d_distances = new float*[numOfGpus];
			
	int **d_classCounters = new int*[numOfGpus], 
		**d_teachedClasses = new int*[numOfGpus], 
		**d_result = new int*[numOfGpus], 
		**d_nearestIndexes = new int*[numOfGpus];	
	
	for(int i = 0; i < numOfGpus; ++i)
	{
		cudaSetDevice(i);
		cudaStreamCreate(streams+i);		

		// Device memory allocation block

		cudaCheckErrors(cudaMalloc(&d_teachingCollection[i], teachingCollectionCount*dimensions*sizeof(float)));
		cudaCheckErrors(cudaMalloc(&d_classifyCollection[i], subranges[i]*dimensions*sizeof(float)));
		cudaCheckErrors(cudaMalloc(&d_nearestDistances[i], subranges[i]*K*sizeof(float)));
		cudaCheckErrors(cudaMalloc(&d_distances[i], teachingCollectionCount*subranges[i]*sizeof(float)));
		
		cudaCheckErrors(cudaMalloc(&d_teachedClasses[i], teachingCollectionCount*sizeof(int)));
		cudaCheckErrors(cudaMalloc(&d_result[i], subranges[i]*sizeof(int)));	
		cudaCheckErrors(cudaMalloc(&d_nearestIndexes[i], subranges[i]*K*sizeof(int)));
		cudaCheckErrors(cudaMalloc(&d_classCounters[i], subranges[i]*MAX_CLASS_NUMBER*sizeof(int)));
		cudaCheckErrors(cudaMemset(d_classCounters[i], 0, subranges[i]*MAX_CLASS_NUMBER*sizeof(int)));
	}
	unsigned long long subrangesSum = 0;
	for(int i = 0; i < numOfGpus; ++i)
	{
		int blocksPerGrid = subranges[i] / threadsPerBlock + 1;
		cudaSetDevice(i);
		
		// Copying parameters for kernels
		cudaCheckErrors(cudaMemcpyAsync(d_teachingCollection[i], h_teachingCollection, teachingCollectionCount*dimensions*sizeof(float), cudaMemcpyHostToDevice, streams[i]));
		cudaCheckErrors(cudaMemcpyAsync(d_classifyCollection[i], h_classifyCollection, subranges[i]*dimensions*sizeof(float), cudaMemcpyHostToDevice, streams[i]));
		cudaCheckErrors(cudaMemcpyAsync(d_teachedClasses[i], h_teachedClasses, teachingCollectionCount*sizeof(int), cudaMemcpyHostToDevice, streams[i]));
		
		// Kernel launches
		KNN_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float), streams[i]>>>(dimensions, d_teachingCollection[i], teachingCollectionCount, d_classifyCollection[i], subranges[i], d_distances[i], d_nearestIndexes[i], d_nearestDistances[i], K, d_classCounters[i], d_teachedClasses[i], d_result[i]);

		// Copying result back to host memory
		cudaCheckErrors(cudaMemcpyAsync(h_classifiedClasses+subrangesSum, d_result[i], subranges[i]*sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
		
		// Ranges which are left
		subrangesSum += subranges[i];

		#ifdef DEBUG
			float *h_distances = new float[teachingCollectionCount*subranges[i]];
			cudaCheckErrors(cudaMemcpy(h_distances, d_distances[i], teachingCollectionCount*subranges[i]*sizeof(float), cudaMemcpyDeviceToHost); // copy calculated distances back to host
			
			printf("\n\n");
			for(int i = 0; i < teachingCollectionCount*subranges[i]; ++i)
			{
				printf("distance %f\n", h_distances[i]);
			}
			
			float *h_nearestDistances = new float[subranges[i]*K];
			int *h_nearestIndexes = new int[subranges[i]*K];
			cudaCheckErrors(cudaMemcpy(h_nearestDistances, d_nearestDistances[i], subranges[i]*K*sizeof(float), cudaMemcpyDeviceToHost));	
			cudaCheckErrors(cudaMemcpy(h_nearestIndexes, d_nearestIndexes[i], subranges[i]*K*sizeof(int), cudaMemcpyDeviceToHost));
			
			printf("\n\n");
			
			for(int i = 0; i < subranges[i]*K; ++i)
			{
				printf("nearest %d %f\n", h_nearestIndexes[i], h_nearestDistances[i]);
			}
			
			int *h_classCounters = new int[subranges[i]*MAX_CLASS_NUMBER];
			cudaCheckErrors(cudaMemcpy(h_classCounters, d_classCounters[i], subranges[i]*MAX_CLASS_NUMBER*sizeof(int), cudaMemcpyDeviceToHost));
			
			for(int i = 0; i < subranges[i]*MAX_CLASS_NUMBER; ++i)
			{
				printf("counter %d\n", h_classCounters[i]);
			}
		#endif
	}
	for(int i = 0; i < numOfGpus; ++i)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
		cudaStreamDestroy(streams[i]);		

		// Freeing memory
		cudaCheckErrors(cudaFree(d_nearestDistances[i]));
		cudaCheckErrors(cudaFree(d_nearestIndexes[i]));
		cudaCheckErrors(cudaFree(d_teachingCollection[i]));
		cudaCheckErrors(cudaFree(d_classifyCollection[i]));
		cudaCheckErrors(cudaFree(d_distances[i]));
		cudaCheckErrors(cudaFree(d_classCounters[i]));	
		cudaCheckErrors(cudaFree(d_teachedClasses[i]));
		cudaCheckErrors(cudaFree(d_result[i]));

	}
	
	delete[] streams;
	delete[] d_classCounters;
	delete[] d_teachedClasses; 
	delete[] d_result; 
	delete[] d_nearestIndexes;
	delete[] d_teachingCollection;
	delete[] d_classifyCollection;
	delete[] d_nearestDistances; 
	delete[] d_distances;
	
	#ifdef DEBUG
		delete[] h_distances;
		delete[] h_nearestDistances;
		delete[] h_nearestIndexes;	
		delete[] h_classCounters;
	#endif
}
