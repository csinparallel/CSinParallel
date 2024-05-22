PGCC = pgcc
GCC = gcc
# options
FAST = -fast
# options
#MOPTS = -mp -ta=multicore -Minfo=opt
MOPTS = -mp -acc=multicore -Minfo=opt

all: laplace_multicore_pgcc

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

laplace_multicore_pgcc: jacobi.c laplace2d.parallel.c getCommandLine.o
	$(PGCC) $(FAST) $(MOPTS) jacobi.c laplace2d.parallel.c getCommandLine.o -o laplace_multicore_pgcc


multicore_profile: laplace_multicore_pgcc
	nvprof --cpu-profiling off --openacc-profiling on -f -o laplace_multicore.prof ./laplace_multicore_pgcc -t 4
	nvprof -f -i laplace_multicore.prof 

clean:
	rm -f *.o laplace_multicore_pgcc *.prof

