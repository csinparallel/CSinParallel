# Compilers
GCC = gcc
PGCC = pgcc

#sequential options for pgcc
FAST = -fast

# options for pgcc to compile for multicore
MOPTS = -mp -ta=multicore -Minfo=opt

OMP = -fopenmp

all: vectorAdd_multicore_pgcc

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

# build multicore executable with the pgcc compiler from NVIDIA using -fast
vectorAdd_multicore_pgcc: vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(PGCC) $(FAST) $(MOPTS) $(OMP) vectorAdd.c getCommandLine.o -o vectorAdd_multicore_pgcc 

clean:
	rm -f *.o vectorAdd_multicore_pgcc *.prof
