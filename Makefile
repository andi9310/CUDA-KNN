KNN_NUM_PROC?=1

CPP_FILES=$(wildcard *.cpp)
CU_FILES=$(wildcard *.cu)
H_FILES=$(wildcard *.h)

.PHONY: all
all: build

.PHONY: build
build: knn.out

knn.out: cuda_knn.o knn.o utils.o
	mpiCC knn.o cuda_knn.o utils.o -o knn.out -L/usr/local/cuda/lib64 -lcudart
	
cuda_knn.o: $(CU_FILES) $(H_FILES)
	nvcc -arch sm_20 -c *.cu -o cuda_knn.o

knn.o: main.cpp $(H_FILES)
	mpiCC -c main.cpp -o knn.o

utils.o: utils.cpp $(H_FILES)
	mpiCC -c utils.cpp -o utils.o

.PHONY: run
run: build
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64/mpi/gcc/openmpi/lib64/ mpirun -np $(KNN_NUM_PROC) ./knn.out
	
.PHONY: clean
clean: 
	rm -f knn.out knn.o cuda_knn.o

.PHONY: run-machinefile
run-machinefile: build
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64/mpi/gcc/openmpi/lib64/ mpirun -np $(KNN_NUM_PROC) -machinefile Machinefile ./knn.out
