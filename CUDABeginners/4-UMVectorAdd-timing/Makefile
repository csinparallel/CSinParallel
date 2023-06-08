CC = nvcc
ARCH = -arch=native   # detects the architecture of your GPU(s)


PROGS = vectorAdd vectorAdd_cudaTiming

all: $(PROGS)

vectorAdd: vectorAdd.cu 
	$(CC) $(ARCH) -o vectorAdd vectorAdd.cu

vectorAdd_cudaTiming: vectorAdd_cudaTiming.cu
	$(CC) $(ARCH) -o vectorAdd_cudaTiming vectorAdd_cudaTiming.cu
	
clean:
	rm -f $(PROGS)
