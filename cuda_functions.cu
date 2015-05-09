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

void cuda_knn(int dimensions, float *h_teachingCollection, int *h_teachedClasses, int teachingCollectionCount, float *h_classifyCollection, int *h_classifiedClasses, int classifyCollectionCount)
{
	int ierr;
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
	
	cudaFree(d_teachingCollection);
	cudaFree(d_classifyCollection);
	cudaFree(d_distances);
	delete[] h_distances;

}
