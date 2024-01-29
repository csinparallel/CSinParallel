# Compilers
GCC = gcc
PGCC = pgcc

#sequential options for pgcc
FAST = -fast

# options for pgcc to compile for multicore
MOPTS = -acc=multicore -Minfo=opt

OMP = -fopenmp
MATH = -lm

all: vectorAdd_multicore_pgcc

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(PGCC) $(FAST) -c -o getCommandLine.o ../utils/getCommandLine.c

helper_add.o: ../utils/helper_add.c
	$(PGCC) $(FAST) -c -o helper_add.o ../utils/helper_add.c $(MATH)

# build multicore executable with the pgcc compiler from NVIDIA using -fast
vectorAdd_multicore_pgcc: vectorAdd.c getCommandLine.o helper_add.o
	$(PGCC) $(FAST) $(MOPTS) $(OMP) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_multicore_pgcc 

clean:
	rm -f *.o vectorAdd_multicore_pgcc *.prof
