CC = pgcc

ACC_OPTS = -acc=gpu -gpu=managed -Minfo=accel 

LIB = -lm

PROGS = mm-gpu

all: $(PROGS)

mm-gpu: mm-openacc.c
	$(CC) $(ACC_OPTS) -o mm-gpu mm-openacc.c -fopenmp $(LIB)

clean:
	rm -f $(PROGS)

