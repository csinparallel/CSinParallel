CC = nvcc
ARCH = -arch=native   # detects the architecture of your GPU(s)


PROGS = vectorAdd vectorAdd_cudaTiming vectorAdd-1 vectorAdd-all

all: $(PROGS)

vectorAdd: vectorAdd.cu 
	$(CC) $(ARCH) -o vectorAdd vectorAdd.cu

vectorAdd-1: vectorAdd-1.cu 
	$(CC) $(ARCH) -o vectorAdd-1 vectorAdd-1.cu

vectorAdd-all: vectorAdd-all.cu 
	$(CC) $(ARCH) -o vectorAdd-all vectorAdd-all.cu

vectorAdd_cudaTiming: vectorAdd_cudaTiming.cu
	$(CC) $(ARCH) -o vectorAdd_cudaTiming vectorAdd_cudaTiming.cu
	
clean:
	rm -f $(PROGS)
