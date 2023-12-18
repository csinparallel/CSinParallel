# Compilers
GCC = gcc
PGCC = pgcc

#sequential options for pgcc
FAST = -fast

#sequential options for gcc
GOPTS = -Ofast -fopenmp -lm
GOPTSO3 = -O3 -fopenmp -lm

MOPTS = -Minfo=opt     #shows further optimizations at pgcc compile

all: vectorAdd_omp vectorAdd_omp_O3 vectorAdd_omp_pgcc vectorAdd_omp_fast_pgcc

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

# build executable with gcc using fast code optimizations
vectorAdd_omp:  vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(GCC) $(GOPTS) vectorAdd.c getCommandLine.o -o vectorAdd_omp

# build executable with gcc using level 3 code optimizations
vectorAdd_omp_O3:  vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(GCC) $(GOPTSO3) vectorAdd.c getCommandLine.o -o vectorAdd_omp_O3

# build openMP executable with the pgcc compiler from NVIDIA
vectorAdd_omp_pgcc: vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(PGCC) $(MOPTS) vectorAdd.c getCommandLine.o -o vectorAdd_omp_pgcc 

# build openMP executable with the pgcc compiler from NVIDIA using -fast
vectorAdd_omp_fast_pgcc: vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(PGCC) $(FAST) $(MOPTS) vectorAdd.c getCommandLine.o -o vectorAdd_omp_fast_pgcc 

clean:
	rm -f *.o vectorAdd_omp* *.prof
