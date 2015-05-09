
__global__ void countDistances(int dimensions, float *teachingCollection, int teachingCollectionCount, float *classifyCollection, int classifyCollectionCount, float *distances)
{
	//compute :P
}

void cuda_knn(int dimensions, float *h_teachingCollection, int *h_teachedClasses, int teachingCollectionCount, float *h_classifyCollection, int *h_classifiedClasses, int classifyCollectionCount)
{
	int ierr;
	float *d_teachingCollection, *d_classifyCollection;
	ierr = cudaMalloc(&d_teachingCollection, teachingCollectionCount*dimensions*sizeof(float));
	ierr = cudaMalloc(&d_classifyCollection, classifyCollectionCount*dimensions*sizeof(float));

	ierr = cudaMemcpy(d_teachingCollection, h_teachingCollection, teachingCollectionCount*dimensions, cudaMemcpyDeviceToHost);
	ierr = cudaMemcpy(d_classifyCollection, h_classifyCollection, classifyCollectionCount*dimensions, cudaMemcpyDeviceToHost);

	float *d_distances;
	cudaMalloc(&d_distances, teachingCollectionCount*classifyCollectionCount*sizeof(float));
	
	const int threadsPerBlock = 100;
	int blocksPerGrid = classifyCollectionCount / 100 + 1;
		
	countDistances<<<blocksPerGrid, threadsPerBlock>>>(dimensions, d_teachingCollection, teachingCollectionCount, d_classifyCollection, classifyCollectionCount, d_distances);
	cudaFree(d_teachingCollection);
	cudaFree(d_classifyCollection);
}
