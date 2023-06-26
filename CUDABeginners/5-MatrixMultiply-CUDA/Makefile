SCC = gcc
CC = nvcc
CC_OPTS = -arch=native
LIB = -lm

PROGS = mm-serial mm-cuda-um mm-omp

all: $(PROGS)

mm-serial: mm-serial.c
	$(SCC) -std=c11 -o mm-serial mm-serial.c -fopenmp $(LIB)

mm-omp: mm-omp.c
	$(SCC) -std=c11 -o mm-omp mm-omp.c -fopenmp $(LIB)

mm-cuda-um: mm-um.cu
	$(CC) $(CC_OPTS) -o mm-cuda-um mm-um.cu 

clean:
	rm -f $(PROGS)

