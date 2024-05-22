PGCC = pgcc
GCC = gcc

#sequential options for pgcc
FAST = -fast
MOPTS = -Minfo=opt     #shows further optimizations at compile

#sequential options for gcc
GOPTS = -Ofast -fopenmp -lm
GOPTSO3 = -O3 -fopenmp -lm

all: laplace_gcc laplace_gcc_O3 laplace_pgcc

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

# build executable with gcc using fast code optimizations
laplace_gcc:  jacobi.c laplace2d.c getCommandLine.o
	$(GCC) $(GOPTS) jacobi.c laplace2d.c getCommandLine.o -o laplace_gcc

# build executable with gcc using level 3 code optimizations
laplace_gcc_O3:  jacobi.c laplace2d.c getCommandLine.o
	gcc $(GOPTSO3) jacobi.c laplace2d.c getCommandLine.o -o laplace_gcc_O3

# build sequential executable with the pgcc compiler from NVIDIA
laplace_pgcc: jacobi.c laplace2d.c getCommandLine.o
	$(PGCC) $(FAST) $(MOPTS) jacobi.c laplace2d.c getCommandLine.o -o laplace_pgcc 

clean:
	rm -f *.o laplace_pgcc laplace_gcc laplace_gcc_O3 *.prof
