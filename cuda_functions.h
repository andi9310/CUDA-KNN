#ifndef CUDA_KNN
#define CUDA_KNN

// Classifies collection of points into classes using KNN algorithm
// input:
// K - number of nearest points to get
// dimensions - dimensionality of space
// h_teachingCollection - pointer to an array, which represents points coordinates - each "dimensions" of elements represents one point (like vector) - teaching collection
// h_teachedClasses - pointer to an array, which represents classes of each point from teaching collection
// teachingCollectionCount - number of points in arrays mentioned above
// h_classifyCollection - pointer to an array, which represents points coordinates - each "dimensions" of elements represents one point (like vector) - classify collection
// classifyCollectionCount - number of points in array mentioned above
// threadsPerBlock - threads per CUDA block
// output:
// h_classifiedClasses - pointer to an array, which will be populated with numbers of class each point was fitted into
void cuda_knn(int K, int dimensions, float *h_teachingCollection, int *h_teachedClasses, int teachingCollectionCount, float *h_classifyCollection, int *h_classifiedClasses, int classifyCollectionCount, int threadsPerBlock);

#endif
