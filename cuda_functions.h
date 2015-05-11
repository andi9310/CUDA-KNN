#ifndef CUDA_KNN
#define CUDA_KNN

void cuda_knn(int N, int dimensions, float *h_teachingCollection, int *h_teachedClasses, int teachingCollectionCount, float *h_classifyCollection, int *h_classifiedClasses, int classifyCollectionCount, int threadsPerBlock);

#endif
