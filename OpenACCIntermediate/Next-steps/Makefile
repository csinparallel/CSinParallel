CC = pgcc
# accelerator options for managed memory
#AOPTSM= -acc -ta=tesla:cc75,managed -Minfo=accel
# newest HPC SDK uses this (figures out the GPU architecture):
#ACC_OPTS = -acc=gpu -Minfo=accel

ACC_OPTS = -acc=gpu -gpu=managed -Minfo=accel 
MC_OPTS = -acc=multicore -Minfo=opt
HOST_OPTS = -acc=host -Minfo=opt

LIB = -lm

PROGS = matrix-cell matrix-cell-mc matrix-cell-seq matrix-sum matrix-sum-mc

all: $(PROGS)

matrix-cell: matrix-cell.c
	$(CC) $(ACC_OPTS) -o matrix-cell matrix-cell.c -fopenmp $(LIB)

matrix-cell-mc: matrix-cell.c
	$(CC) $(MC_OPTS) -o matrix-cell-mc matrix-cell.c -fopenmp $(LIB)

matrix-cell-seq: matrix-cell.c
	$(CC) $(HOST_OPTS) -o matrix-cell-seq matrix-cell.c -fopenmp $(LIB)

matrix-sum: matrix-sum.c
	$(CC) $(ACC_OPTS) -o matrix-sum matrix-sum.c -fopenmp $(LIB)

matrix-sum-mc: matrix-sum.c
	$(CC) $(MC_OPTS) -o matrix-sum-mc matrix-sum.c -fopenmp $(LIB)

clean:
	rm -f $(PROGS)

