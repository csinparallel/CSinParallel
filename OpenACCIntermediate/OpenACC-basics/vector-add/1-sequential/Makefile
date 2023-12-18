# Compilers
PGCC = pgcc
GCC = gcc

#sequential options for pgcc
FAST = -fast
MOPTS = -Minfo=opt     #shows further optimizations at compile

#sequential options for gcc
GOPTS = -Ofast -fopenmp -lm
GOPTSO3 = -O3 -fopenmp -lm

all: vectorAdd_gcc vectorAdd_gcc_O3 vectorAdd_pgcc

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

# build executable with gcc using fast code optimizations
vectorAdd_gcc:  vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(GCC) $(GOPTS) vectorAdd.c getCommandLine.o -o vectorAdd_gcc

# build executable with gcc using level 3 code optimizations
vectorAdd_gcc_O3:  vectorAdd.c getCommandLine.o ../utils/helper_add.h
	gcc $(GOPTSO3) vectorAdd.c getCommandLine.o -o vectorAdd_gcc_O3

# build sequential executable with the pgcc compiler from NVIDIA
vectorAdd_pgcc: vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(PGCC) $(FAST) $(MOPTS) vectorAdd.c getCommandLine.o -o vectorAdd_pgcc 

clean:
	rm -f *.o vectorAdd_pgcc vectorAdd_gcc vectorAdd_gcc_O3 *.prof
