CC = pgcc

# initial accelerator options
#AOPTS= -fast -ta=tesla:cc75 -Minfo=accel
AOPTS= -fast -acc=gpu -Minfo=accel

# accelerator options for managed memory
#AOPTSM= -fast -ta=tesla:cc75,managed -Minfo=accel
AOPTSM= -fast -acc=gpu -gpu=managed -Minfo=accel

all: laplace_acc1 laplace_acc laplace_managed

# separate file to be used with other versions of this code
getCommandLine.o: ../utils/getCommandLine.c
	$(CC) -c -o getCommandLine.o ../utils/getCommandLine.c

laplace_acc1: jacobi.c laplace2d.kernels.1.c getCommandLine.o
	${CC} ${AOPTS} -o laplace_acc1 jacobi.c laplace2d.kernels.1.c getCommandLine.o

laplace_acc: jacobi.c laplace2d.kernels.c getCommandLine.o
	${CC} ${AOPTS} -o laplace_acc jacobi.c laplace2d.kernels.c getCommandLine.o

laplace_managed: jacobi.c laplace2d.kernels.c getCommandLine.o
	${CC} ${AOPTSM} -o laplace_managed jacobi.c laplace2d.kernels.c getCommandLine.o

clean:
	rm -f laplace_acc1 laplace_acc laplace_managed
	
