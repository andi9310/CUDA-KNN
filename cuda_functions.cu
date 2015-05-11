//#define DEBUG
//#define DEBUG_KERNEL

#include <iostream>

#include "variables.h"

#ifdef DEBUG
	#include <cstdio>
#endif

void cudaCheckErrors(int code)
{
	if(code)
		std::cerr << "CUDA error no.: " << code << "\n";
}

__device__ int findSlot(float *myDistances, float *myNearestDistances, int *myNearestIndexes, int N, int i)
{
	int j;
	for (j=0; j<i; j++)
	{
		if (myDistances[i]<myNearestDistances[j]) //terrible nesting here
		{
			for (int k=N-1; k>j; k--)
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

__global__ void selectN(int N, float *distances, int teachingCollectionCount, int classifyCollectionCount, int *nearestIndexes, float *nearestDistances)
{		
	int myRank = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(myRank >= classifyCollectionCount)
		return;
		
	int *myNearestIndexes = nearestIndexes+N*myRank;
	float *myNearestDistances = nearestDistances+N*myRank;
	float *myDistances = distances + myRank * teachingCollectionCount;
	
	for (int i=0; i<N; i++)
	{
		int j = findSlot(myDistances, myNearestDistances, myNearestIndexes, N, i);
		
		if (j==i)
		{
			myNearestDistances[j]=myDistances[i];
			myNearestIndexes[j]=i;
		}
	}
	for (int i=N; i<teachingCollectionCount;i++)
	{
		findSlot(myDistances, myNearestDistances, myNearestIndexes, N, i);
	}
	
}

__global__ void chooseClass(int N, int *nearestIndexes, int classifyCollectionCount, int *teachedClasses, int teachingCollectionCount, int *classCounters, int *result)
{
	int tId = blockIdx.x*blockDim.x+threadIdx.x;
	int pointId = tId*N;	
		
	if(tId >= classifyCollectionCount)
		return;

	for(int i = 0; i < N; ++i)
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

void cuda_knn(int N, int dimensions, float *h_teachingCollection, int *h_teachedClasses, int teachingCollectionCount, float *h_classifyCollection, int *h_classifiedClasses, int classifyCollectionCount, int threadsPerBlock)
{
	int blocksPerGrid = classifyCollectionCount / threadsPerBlock + 1;
		
	float *d_teachingCollection, *d_classifyCollection, *d_nearestDistances, *d_distances;
	cudaCheckErrors(cudaMalloc(&d_teachingCollection, teachingCollectionCount*dimensions*sizeof(float)));
	cudaCheckErrors(cudaMalloc(&d_classifyCollection, classifyCollectionCount*dimensions*sizeof(float)));
	cudaCheckErrors(cudaMalloc(&d_nearestDistances, classifyCollectionCount*N*sizeof(float)));
	cudaCheckErrors(cudaMalloc(&d_distances, teachingCollectionCount*classifyCollectionCount*sizeof(float)));
	
	int *d_classCounters, *d_teachedClasses, *d_result, *d_nearestIndexes;
	cudaCheckErrors(cudaMalloc(&d_teachedClasses, teachingCollectionCount*sizeof(int)));
	cudaCheckErrors(cudaMalloc(&d_result, classifyCollectionCount*sizeof(int)));	
	cudaCheckErrors(cudaMalloc(&d_nearestIndexes, classifyCollectionCount*N*sizeof(int)));
	cudaCheckErrors(cudaMalloc(&d_classCounters, classifyCollectionCount*MAX_CLASS_NUMBER*sizeof(int)));
	cudaCheckErrors(cudaMemset(d_classCounters, 0, classifyCollectionCount*MAX_CLASS_NUMBER*sizeof(int)));
	
	cudaCheckErrors(cudaMemcpy(d_teachingCollection, h_teachingCollection, teachingCollectionCount*dimensions*sizeof(float), cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaMemcpy(d_classifyCollection, h_classifyCollection, classifyCollectionCount*dimensions*sizeof(float), cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaMemcpy(d_teachedClasses, h_teachedClasses, teachingCollectionCount*sizeof(int), cudaMemcpyHostToDevice));
	
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
	
	selectN<<<blocksPerGrid, threadsPerBlock>>>(N, d_distances, teachingCollectionCount, classifyCollectionCount, d_nearestIndexes, d_nearestDistances);
	
	#ifdef DEBUG
		float *h_nearestDistances = new float[classifyCollectionCount*N];
		int *h_nearestIndexes = new int[classifyCollectionCount*N];
		cudaCheckErrors(cudaMemcpy(h_nearestDistances, d_nearestDistances, classifyCollectionCount*N*sizeof(float), cudaMemcpyDeviceToHost));	
		cudaCheckErrors(cudaMemcpy(h_nearestIndexes, d_nearestIndexes, classifyCollectionCount*N*sizeof(int), cudaMemcpyDeviceToHost));
		printf("\n\n");
		for(int i = 0; i < classifyCollectionCount*N; ++i)
		{
			printf("nearest %d %f\n", h_nearestIndexes[i], h_nearestDistances[i]);
		}
	#endif	
	
	chooseClass<<<blocksPerGrid, threadsPerBlock>>>(N, d_nearestIndexes, classifyCollectionCount, d_teachedClasses, teachingCollectionCount, d_classCounters, d_result);
	
	cudaCheckErrors(cudaMemcpy(h_classifiedClasses, d_result, classifyCollectionCount*sizeof(int), cudaMemcpyDeviceToHost));
		
	#ifdef DEBUG
		int *h_classCounters = new int[classifyCollectionCount*MAX_CLASS_NUMBER];
		cudaCheckErrors(cudaMemcpy(h_classCounters, d_classCounters, classifyCollectionCount*MAX_CLASS_NUMBER*sizeof(int), cudaMemcpyDeviceToHost));
		for(int i = 0; i < classifyCollectionCount*MAX_CLASS_NUMBER; ++i)
		{
			printf("counter %d\n", h_classCounters[i]);
		}
	#endif
		
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
