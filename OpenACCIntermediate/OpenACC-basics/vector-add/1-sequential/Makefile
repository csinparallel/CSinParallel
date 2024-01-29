# Compilers
PGCC = pgcc
GCC = gcc

#sequential options for pgcc. -fast is minimum level -O2
FAST = -fast -acc=host
MOPTS = -Minfo=opt     #shows further optimizations at compile

#sequential options for gcc
GOPTS = -O2
# Note -O2 or -O3 is not possible when using fmaxf from the math library for checking data
# so we removed the use of it in function checkForErrors() in helper_add.c

LIBS = -lm

PROGS = vectorAdd_gcc vectorAdd_pgcc

all: $(PROGS)

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) $(GOPTS) -c -o getCommandLine.o ../utils/getCommandLine.c

helper_add.o: ../utils/helper_add.c
	$(GCC) $(GOPTS) -c -o helper_add.o ../utils/helper_add.c $(LIBS)

# build executable with gcc using fast code optimizations
vectorAdd_gcc:  vectorAdd.c getCommandLine.o helper_add.o
	$(GCC) $(GOPTS) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_gcc

# build sequential executable with the pgcc compiler from NVIDIA
vectorAdd_pgcc: vectorAdd.c getCommandLine.o helper_add.o
	$(PGCC) $(FAST) $(MOPTS) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_pgcc 

clean:
	rm -f *.o $(PROGS) 
