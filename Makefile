KNN_NUM_PROC?=1

CPP_FILES=$(wildcard *.cpp)
CU_FILES=$(wildcard *.cu)
H_FILES=$(wildcard *.h)

.PHONY: all
all: build

.PHONY: build
build: knn.out

knn.out: cuda_knn.o knn.o
	mpiCC knn.o cuda_knn.o -o knn.out -L/usr/local/cuda/lib64 -lcudart
	
cuda_knn.o: $(CU_FILES)
	nvcc -arch compute_20 -c *.cu -o cuda_knn.o

knn.o: $(CPP_FILES) $(H_FILES)
	mpiCC -c *.cpp -o knn.o

.PHONY: run
run: build
	LD_LIBRARY_PATH=/usr/local/cuda/lib64 mpirun -np $(KNN_NUM_PROC) ./knn.out
	
.PHONY: clean
clean: 
	rm -f knn.out knn.o cuda_knn.o
