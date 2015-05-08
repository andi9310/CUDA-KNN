KNN_NUM_PROC?=1

.PHONY: all
all:
	nvcc -c *.cu -o cuda_knn.o
	mpiCC -c *.cpp -o knn.o
	mpiCC knn.o cuda_knn.o -o knn.out -L/usr/local/cuda/lib64 -lcudart
	
.PHONY: run
run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64
	mpirun -np $(KNN_NUM_PROC) ./knn.out
