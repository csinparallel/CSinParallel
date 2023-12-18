# Compilers
GCC = gcc
PGCC = pgcc

#option for pgcc
FAST = -fast

# initial accelerator options
AOPTS= -acc -ta=tesla:cc75 -Minfo=accel

# accelerator options for managed memory
AOPTSM= -acc -ta=tesla:cc75,managed -Minfo=accel

OMP = -fopenmp

# options for pgcc to compile for multicore
MOPTS = -mp -ta=multicore -Minfo=opt

all: vectorAdd_acc vectorAdd_acc_managed vectorAdd_cpu

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

# build GPU executable with the pgcc compiler from NVIDIA 
vectorAdd_acc: vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(PGCC) $(AOPTS) $(OMP) vectorAdd.c getCommandLine.o -o vectorAdd_acc 

# build GPU executable with the pgcc compiler using unified (mamanged) memory
vectorAdd_acc_managed: vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(PGCC) $(AOPTS) $(OMP) vectorAdd.c getCommandLine.o -o vectorAdd_acc_managed 

vectorAdd_cpu: vectorAdd.c getCommandLine.o ../utils/helper_add.h
	$(PGCC) $(FAST) $(MOPTS) $(OMP) vectorAdd.c getCommandLine.o -o vectorAdd_cpu 

clean:
	rm -f *.o vectorAdd_acc vectorAdd_acc_managed vectorAdd_cpu *.prof
