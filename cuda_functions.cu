#include <cmath>
#include <cstdio> // for debug purpose
#define DEBUG

__global__ void countDistances(int dimensions, float *teachingCollection, int teachingCollectionCount, float *classifyCollection, int classifyCollectionCount, float *distances)
{
	int tId = blockIdx.x*blockDim.x+threadIdx.x;
	int pointId = tId*dimensions;	
		
	if(pointId >= classifyCollectionCount)
		return;
	
	for(int i = 0; i < teachingCollectionCount; i+=dimensions)
	{
		float distance = 0.0f;
		for(int j = 0; j < dimensions; ++j)
		{
			distance += (classifyCollection[pointId+j]-teachingCollection[i+j])*(classifyCollection[pointId+j]-teachingCollection[i+j]);
		}
		distance = (float)sqrt(distance);
		
		distances[teachingCollectionCount*pointId+i] = distance;
	}

}

__global__ void selectN(int N, float *distances, int teachingCollectionCount, int classifyCollectionCount, int *nearestIndexes, float *nearestDistances)
{
	int pointsInBlock;
	if (blockIdx.x == gridDim.x-1)
	{
		pointsInBlock =  (gridDim.x * blockDim.x) % teachingCollectionCount;	
	}
	else
	{
		pointsInBlock = blockDim.x;
	}
	if (threadIdx.x < pointsInBlock)
	{
		int myRank = blockDim.x * blockIdx.x + threadIdx.x;
		int *myNearestIndexes = nearestIndexes+N*myRank;
		float *myNearestDistances = nearestDistances+N*myRank;
		float *myDistances = distances + myRank * teachingCollectionCount;
		for (int i=0; i<N; i++)
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
			if (j==i)
			{
				myNearestDistances[j]=myDistances[i];
				myNearestIndexes[j]=i;
			}
		}
		for (int i=N; i<teachingCollectionCount;i++)
		{
			for (int j=0; j<i; j++)
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
		}
	}
}

void cuda_knn(int N, int dimensions, float *h_teachingCollection, int *h_teachedClasses, int teachingCollectionCount, float *h_classifyCollection, int *h_classifiedClasses, int classifyCollectionCount)
{
	cudaError_t ierr;
	float *d_teachingCollection, *d_classifyCollection;
	ierr = cudaMalloc(&d_teachingCollection, teachingCollectionCount*dimensions*sizeof(float));
	ierr = cudaMalloc(&d_classifyCollection, classifyCollectionCount*dimensions*sizeof(float));

	ierr = cudaMemcpy(d_teachingCollection, h_teachingCollection, teachingCollectionCount*dimensions*sizeof(float), cudaMemcpyHostToDevice);
	ierr = cudaMemcpy(d_classifyCollection, h_classifyCollection, classifyCollectionCount*dimensions*sizeof(float), cudaMemcpyHostToDevice);

	float *d_distances, *h_distances;
	cudaMalloc(&d_distances, teachingCollectionCount*classifyCollectionCount*sizeof(float));
	h_distances = new float[teachingCollectionCount*classifyCollectionCount];
	
	const int threadsPerBlock = 100;
	int blocksPerGrid = classifyCollectionCount / 100 + 1;
		
	countDistances<<<5, 5>>>(dimensions, d_teachingCollection, teachingCollectionCount, d_classifyCollection, classifyCollectionCount, d_distances);
	
	ierr = cudaMemcpy(h_distances, d_distances, teachingCollectionCount*classifyCollectionCount*sizeof(float), cudaMemcpyDeviceToHost); // copy calculated distances back to host
	
	#ifdef DEBUG
		printf("\n");
		for(int i = 0; i < teachingCollectionCount*classifyCollectionCount; ++i)
		{
			if(i%dimensions == 0)
			{
				printf("\n");
			}
			printf("%f\n", h_distances[i]);
		}
	#endif

	float *d_nearestDistances;
	int *d_nearestIndexes;
	ierr = cudaMalloc(&d_nearestDistances, classifyCollectionCount*N*sizeof(float));
	ierr = cudaMalloc(&d_nearestIndexes, classifyCollectionCount*N*sizeof(int));
	
	selectN<<<blocksPerGrid, threadsPerBlock>>>(N, d_distances, teachingCollectionCount, classifyCollectionCount, d_nearestIndexes, d_nearestDistances);
	
	ierr = cudaFree(d_nearestDistances);
	ierr = cudaFree(d_nearestIndexes);
	ierr = cudaFree(d_teachingCollection);
	ierr = cudaFree(d_classifyCollection);
	ierr = cudaFree(d_distances);
	delete[] h_distances;

}
