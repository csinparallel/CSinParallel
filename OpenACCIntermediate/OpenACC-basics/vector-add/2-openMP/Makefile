# Compilers
GCC = gcc
PGCC = pgcc

# options for pgcc
FAST = -fast -mp

# options for gcc
# -fast on pgcc is roughly equivalent to -O2
GOPTS = -O2

LIBS = -fopenmp -lm 

MOPTS = -Minfo=opt     #shows further optimizations at pgcc compile

all: vectorAdd_omp_fast_gcc vectorAdd_omp_pgcc vectorAdd_omp_fast_pgcc

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

helper_add.o: ../utils/helper_add.c
	$(GCC) $(GOPTS) -c -o helper_add.o ../utils/helper_add.c $(LIBS)

# build executable with gcc using fast code optimizations
vectorAdd_omp_fast_gcc:  vectorAdd.c getCommandLine.o helper_add.o
	$(GCC) $(GOPTS) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_omp_fast_gcc $(LIBS)

# build openMP executable with the pgcc compiler from NVIDIA
vectorAdd_omp_pgcc: vectorAdd.c getCommandLine.o helper_add.o
	$(PGCC) $(MOPTS) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_omp_pgcc $(LIBS)

# build openMP executable with the pgcc compiler from NVIDIA using -fast
vectorAdd_omp_fast_pgcc: vectorAdd.c getCommandLine.o helper_add.o
	$(PGCC) $(FAST) $(MOPTS) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_omp_fast_pgcc $(LIBS)

clean:
	rm -f *.o vectorAdd_omp* *.prof
