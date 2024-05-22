PGCC = pgcc
GCC = gcc

# options for pgcc
FAST = -fast -mp
MOPTS = -Minfo=opt     #shows further optimizations at compile

# options for gcc
GOPTS = -Ofast -fopenmp -lm

all: laplace_omp_gcc laplace_omp_pgcc

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

# build executable with gcc using fast code optimizations
laplace_omp_gcc:  jacobi.c laplace2d.c getCommandLine.o
	$(GCC) $(GOPTS) jacobi.c laplace2d.c getCommandLine.o -o laplace_omp_gcc

# build sequential executable with the pgcc compiler from NVIDIA
laplace_omp_pgcc: jacobi.c laplace2d.c getCommandLine.o
	$(PGCC) $(FAST) $(MOPTS) jacobi.c laplace2d.c getCommandLine.o -o laplace_omp_pgcc 

clean:
	rm -f *.o laplace_omp_gcc laplace_omp_pgcc *.prof
