CC = pgcc
# accelerator options for managed memory
#AOPTSM= -acc -ta=tesla:cc75,managed -Minfo=accel
# newest HPC SDK uses this (figures out the GPU architecture):
#ACC_OPTS = -acc=gpu -Minfo=accel
# the following is for slightly older versions
ACC_OPTS = -acc -ta=tesla:managed -Minfo=accel

LIB = -lm

PROGS = mm-openacc

all: $(PROGS)

mm-openacc: mm-openacc.c
	$(CC) $(ACC_OPTS) -o mm-openacc mm-openacc.c -fopenmp $(LIB)

clean:
	rm -f $(PROGS)

