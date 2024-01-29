# Compiler
PGCC = pgcc

#option for pgcc
FAST = -fast

# initial accelerator options
AOPTS= -acc=gpu  -Minfo=accel

# accelerator options for managed memory
AOPTSM= -acc=gpu -gpu=managed -Minfo=accel

OMP = -fopenmp
MATH = -lm

# options for pgcc to compile for multicore
MOPTS = -acc=multicore -Minfo=opt

PROGS = vectorAdd_acc vectorAdd_acc_managed vectorAdd_mc

all: $(PROGS)

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(PGCC) -c -o getCommandLine.o ../utils/getCommandLine.c

helper_add.o: ../utils/helper_add.c
	$(PGCC) $(FAST) -c -o helper_add.o ../utils/helper_add.c $(MATH)

# build GPU executable with the pgcc compiler from NVIDIA 
vectorAdd_acc: vectorAdd.c getCommandLine.o helper_add.o
	$(PGCC) $(AOPTS) $(OMP) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_acc 

# build GPU executable with the pgcc compiler using unified (mamanged) memory
vectorAdd_acc_managed: vectorAdd.c getCommandLine.o helper_add.o
	$(PGCC) $(AOPTS) $(OMP) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_acc_managed 

# multicore version
vectorAdd_mc: vectorAdd.c getCommandLine.o helper_add.o
	$(PGCC) $(FAST) $(MOPTS) $(OMP) vectorAdd.c getCommandLine.o helper_add.o -o vectorAdd_mc 

clean:
	rm -f *.o $(PROGS) *.prof
